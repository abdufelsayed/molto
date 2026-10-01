import { useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { PageTitle, QueryState } from "@/components/page-state"
import {
  managementQuery,
  managementRequest,
} from "@/features/management/request"
import { errorMessage } from "@/features/management/api"
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
import { DiagnosticForm } from "./form"
import { Results } from "./results"
import { active, labels, type Capabilities, type Kind, type Run } from "./types"
export function DiagnosticsPage() {
  const client = useQueryClient()
  const capabilities = useQuery(
    managementQuery<Capabilities>(
      ["diagnostics", "capabilities"],
      "diagnostics/capabilities"
    )
  )
  const history = useQuery(
    managementQuery<{ runs: Run[] }>(
      ["diagnostics", "runs"],
      "diagnostics/runs",
      2000
    )
  )
  const [selected, setSelected] = useState<string | null>(null)
  const detail = useQuery({
    ...managementQuery<Run>(
      ["diagnostics", "run", selected],
      `diagnostics/runs/${encodeURIComponent(selected ?? "")}`,
      2000
    ),
    enabled: !!selected,
  })
  const results = useQuery({
    ...managementQuery<{ results: unknown }>(
      ["diagnostics", "results", selected],
      `diagnostics/runs/${encodeURIComponent(selected ?? "")}/results`,
      2000
    ),
    enabled: !!selected,
  })
  const refresh = () => client.invalidateQueries({ queryKey: ["molto"] })
  const start = useMutation({
    mutationFn: ({
      kind,
      options,
    }: {
      kind: Kind
      options: Record<string, unknown>
    }) =>
      managementRequest<Run>("diagnostics/runs", {
        method: "POST",
        body: { kind, options },
      }),
    onSuccess: (run) => {
      setSelected(run.id)
      void refresh()
    },
  })
  const cancel = useMutation({
    mutationFn: (id: string) =>
      managementRequest<Run>(
        `diagnostics/runs/${encodeURIComponent(id)}/cancel`,
        { method: "POST" }
      ),
    onSuccess: () => {
      void refresh()
    },
  })
  const run = detail.data
  return (
    <>
      <PageTitle
        title="Diagnostics"
        description="Measure throughput, evaluation accuracy, context capacity and ANE candidates."
      />
      <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card>
          <CardHeader>
            <CardTitle>New diagnostic</CardTitle>
            <CardDescription>
              Choose a workload before starting inference.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <QueryState query={capabilities}>
              {capabilities.data && (
                <DiagnosticForm
                  capabilities={capabilities.data}
                  pending={start.isPending}
                  busy={
                    history.isError ||
                    history.isPending ||
                    !!history.data?.runs.some(active)
                  }
                  error={start.isError ? errorMessage(start.error) : undefined}
                  submit={(kind, options) => start.mutate({ kind, options })}
                />
              )}
            </QueryState>
          </CardContent>
        </Card>
        <div className="grid gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Run history</CardTitle>
              <CardDescription>
                Interrupted and failed runs remain here. Select a run to inspect
                its measurements.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <QueryState query={history}>
                {history.data && (
                  <div className="grid gap-3">
                    {!history.data.runs.length && (
                      <p className="text-sm text-muted-foreground">
                        No diagnostics have run yet.
                      </p>
                    )}
                    {history.data.runs.map((item) => (
                      <button
                        key={item.id}
                        type="button"
                        onClick={() => setSelected(item.id)}
                        className={`flex items-start justify-between gap-3 rounded-lg border p-3 text-left hover:bg-muted ${selected === item.id ? "border-primary" : ""}`}
                      >
                        <span className="grid min-w-0 gap-1">
                          <span className="text-sm font-medium">
                            {labels[item.kind]}
                          </span>
                          <span className="truncate text-xs text-muted-foreground">
                            {item.id}
                          </span>
                          <span className="text-xs text-muted-foreground">
                            {typeof item.created_at === "number"
                              ? new Date(
                                  item.created_at * 1000
                                ).toLocaleString()
                              : new Date(item.created_at).toLocaleString()}
                          </span>
                        </span>
                        <Badge
                          variant={
                            item.status === "error"
                              ? "destructive"
                              : "secondary"
                          }
                        >
                          {item.status.replaceAll("_", " ")}
                        </Badge>
                      </button>
                    ))}
                  </div>
                )}
              </QueryState>
            </CardContent>
          </Card>
          {selected && (
            <Card>
              <CardHeader>
                <CardTitle>Run details</CardTitle>
                <CardDescription>{selected}</CardDescription>
              </CardHeader>
              <CardContent className="grid gap-4">
                <QueryState query={detail}>
                  {run && (
                    <>
                      <div role="status" className="grid gap-2">
                        <Badge variant="secondary">{run.status}</Badge>
                        {run.progress && (
                          <div className="text-sm">
                            <pre className="font-sans break-words whitespace-pre-wrap">
                              {typeof run.progress === "number"
                                ? `${run.progress}%`
                                : Object.entries(run.progress)
                                    .filter(
                                      ([, value]) =>
                                        typeof value === "string" ||
                                        typeof value === "number"
                                    )
                                    .map(
                                      ([name, value]) =>
                                        `${name.replaceAll("_", " ")}: ${String(value)}`
                                    )
                                    .join(" · ")}
                            </pre>
                          </div>
                        )}
                        {["cancelling", "cancel_requested"].includes(
                          run.status
                        ) && (
                          <p className="text-sm text-muted-foreground">
                            Cancellation requested. Admission remains held until
                            native work stops.
                          </p>
                        )}
                      </div>
                      {run.error && (
                        <Alert variant="destructive">
                          <AlertTitle>Run failed</AlertTitle>
                          <AlertDescription>
                            {run.error} Inspect the request snapshot, adjust the
                            workload and start a new run.
                          </AlertDescription>
                        </Alert>
                      )}
                      {active(run) && (
                        <Button
                          variant="outline"
                          disabled={
                            cancel.isPending ||
                            ["cancelling", "cancel_requested"].includes(
                              run.status
                            )
                          }
                          onClick={() => cancel.mutate(run.id)}
                        >
                          {cancel.isPending
                            ? "Requesting cancellation…"
                            : "Cancel run"}
                        </Button>
                      )}
                      {cancel.isError && (
                        <Alert variant="destructive">
                          <AlertTitle>Cancellation failed</AlertTitle>
                          <AlertDescription>
                            {errorMessage(cancel.error)}
                          </AlertDescription>
                        </Alert>
                      )}
                      <QueryState query={results}>
                        {results.data && (
                          <Results run={run} results={results.data.results} />
                        )}
                      </QueryState>
                    </>
                  )}
                </QueryState>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </>
  )
}
