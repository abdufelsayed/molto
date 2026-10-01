import { createFileRoute } from "@tanstack/react-router"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useState } from "react"
import {
  managementQuery,
  managementRequest,
} from "@/features/management/request"
import { errorMessage } from "@/features/management/api"
import { PageTitle, QueryState } from "@/components/page-state"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card"
import { FieldGroup } from "@/components/ui/field"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
import { Table, TableBody, TableCell, TableRow } from "@/components/ui/table"
import { bytes, count } from "@/lib/format"
import {
  Choice,
  SummaryTable,
  UsageChart,
} from "@/features/monitoring/components"
import type { Activity, Usage, Versions } from "@/features/monitoring/types"
export const Route = createFileRoute("/monitoring")({
  component: MonitoringPage,
})
function MonitoringPage() {
  const [period, setPeriod] = useState("7d")
  const [model, setModel] = useState("")
  const [resetScope, setResetScope] = useState("session")
  const [confirmReset, setConfirmReset] = useState(false)
  const client = useQueryClient()
  const activity = useQuery(
    managementQuery<Activity>(["activity"], "monitoring/activity", 2000)
  )
  const usage = useQuery(
    managementQuery<Usage>(
      ["usage", period, model],
      `monitoring/usage?range=${period}&model=${encodeURIComponent(model)}&include_details=true`,
      30000
    )
  )
  const versions = useQuery(
    managementQuery<Versions>(["versions"], "monitoring/versions")
  )
  const reset = useMutation({
    mutationFn: () =>
      managementRequest("monitoring/stats/reset", {
        method: "POST",
        body: { scope: resetScope },
      }),
    onSuccess: () => {
      setConfirmReset(false)
      void client.invalidateQueries({ queryKey: ["molto"] })
    },
  })
  const modelIds = [
    ...new Set([
      ...(activity.data?.models.map((item) => item.id) ?? []),
      ...(usage.data?.models.map((item) => item.model_id) ?? []),
      ...(model ? [model] : []),
    ]),
  ]
  return (
    <>
      <PageTitle
        title="Monitoring"
        description="Live requests, retained usage, and installed engine provenance."
      />
      <QueryState query={activity}>
        {activity.data && (
          <Card>
            <CardHeader>
              <CardTitle>Live activity</CardTitle>
              <CardDescription>
                {count(activity.data.total_active_requests)} active ·{" "}
                {count(activity.data.total_waiting_requests)} queued · uptime{" "}
                {activity.data.uptime_seconds == null
                  ? "Not reported"
                  : Math.floor(activity.data.uptime_seconds)}{" "}
                s · pressure{" "}
                {activity.data.memory_pressure
                  ? Object.entries(activity.data.memory_pressure)
                      .filter(
                        ([, value]) =>
                          typeof value === "number" ||
                          typeof value === "string" ||
                          typeof value === "boolean"
                      )
                      .map(
                        ([key, value]) =>
                          `${key.replaceAll("_", " ")} ${String(value)}`
                      )
                      .join(" · ")
                  : "Not reported"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="mb-4 text-sm">
                Model memory {bytes(activity.data.model_memory_used)} /{" "}
                {bytes(activity.data.model_memory_max)}
              </p>
              {activity.data.models.length ? (
                activity.data.models.map((item) => (
                  <div key={item.id} className="border-b py-3">
                    <p className="font-medium break-all">{item.id}</p>
                    <p className="text-sm text-muted-foreground">
                      {item.stage ?? "Loaded"} · {item.active_requests} active ·{" "}
                      {item.waiting_requests} queued · {item.prefilling.length}{" "}
                      prefilling · {item.generating.length} generating
                      {item.loading_elapsed_seconds != null
                        ? ` · loading ${item.loading_elapsed_seconds.toFixed(1)} s`
                        : ""}
                    </p>
                    {[
                      ...item.prefilling.map((request) => ({
                        ...request,
                        stage: "Prefilling",
                      })),
                      ...item.generating.map((request) => ({
                        ...request,
                        stage: "Generating",
                      })),
                      ...item.waiting.map((request) => ({
                        ...request,
                        stage: "Queued",
                      })),
                      ...item.activities,
                    ].map((request, index) => (
                      <p className="text-sm" key={request.request_id ?? index}>
                        {request.request_id ?? `Request ${index + 1}`} ·{" "}
                        {request.stage ??
                          request.detail ??
                          request.kind ??
                          "Active"}
                        {request.queue_position != null
                          ? ` · position ${request.queue_position}`
                          : ""}
                        {request.processed != null
                          ? ` · ${request.processed}/${request.total ?? "?"} prompt tokens`
                          : ""}
                        {request.generated_tokens != null
                          ? ` · ${request.generated_tokens} generated tokens`
                          : ""}
                        {(request.elapsed_seconds ?? request.elapsed) != null
                          ? ` · ${(request.elapsed_seconds ?? request.elapsed)?.toFixed(1)} s`
                          : ""}
                        {request.tokens_per_second != null
                          ? ` · ${request.tokens_per_second.toFixed(1)} tokens/s`
                          : ""}
                      </p>
                    ))}
                  </div>
                ))
              ) : (
                <p>No loaded models or queued requests.</p>
              )}
            </CardContent>
          </Card>
        )}
      </QueryState>
      <Card>
        <CardHeader>
          <CardTitle>Historical usage</CardTitle>
          <CardDescription>
            Ranges follow the server's local calendar. Missing timing
            measurements appear as not recorded.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <FieldGroup className="mb-4 grid sm:grid-cols-2">
            <Choice
              label="Date range"
              value={period}
              onChange={setPeriod}
              options={[
                { value: "today", label: "Today" },
                { value: "yesterday", label: "Yesterday" },
                { value: "7d", label: "Last 7 days" },
                { value: "30d", label: "Last 30 days" },
                { value: "90d", label: "Last 90 days" },
                { value: "month", label: "This month" },
              ]}
            />
            <Choice
              label="Model"
              value={model}
              onChange={setModel}
              options={[
                { value: "", label: "All models" },
                ...modelIds.map((id) => ({ value: id, label: id })),
              ]}
            />
          </FieldGroup>
          <QueryState query={usage}>
            {usage.data &&
              (usage.data.enabled &&
              usage.data.available &&
              usage.data.totals ? (
                <>
                  <p className="mb-3 text-sm text-muted-foreground">
                    {usage.data.start} to {usage.data.end} ·{" "}
                    {usage.data.timezone}
                  </p>
                  <SummaryTable data={usage.data.totals} />
                  {usage.data.dropped_requests > 0 && (
                    <p role="status">
                      {count(usage.data.dropped_requests)} requests could not be
                      recorded.
                    </p>
                  )}
                  <UsageChart daily={usage.data.daily ?? []} />
                  {usage.data.models.map((item) => (
                    <details key={item.model_id} className="mt-3">
                      <summary className="cursor-pointer break-all">
                        {item.model_id} · {count(item.total_tokens)} tokens
                      </summary>
                      <SummaryTable data={item} />
                    </details>
                  ))}
                </>
              ) : (
                <Alert>
                  <AlertTitle>
                    {usage.data.state === "unavailable"
                      ? "Usage history unavailable"
                      : "Usage recording disabled"}
                  </AlertTitle>
                  <AlertDescription>
                    {usage.data.state === "unavailable"
                      ? "The retained usage store could not be read. Retry after checking the server logs."
                      : "Enable usage recording in server settings to retain future requests."}
                  </AlertDescription>
                </Alert>
              ))}
          </QueryState>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Statistics reset</CardTitle>
          <CardDescription>
            Reset session or persisted all-time counters. Retained historical
            usage remains available.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <FieldGroup className="mb-4">
            <Choice
              label="Counters to reset"
              value={resetScope}
              onChange={(value) => {
                if (!reset.isPending) {
                  setResetScope(value)
                  setConfirmReset(false)
                }
              }}
              options={[
                { value: "session", label: "Session counters" },
                { value: "alltime", label: "Persisted all-time counters" },
              ]}
            />
          </FieldGroup>
          {confirmReset ? (
            <div className="flex gap-2">
              <Button
                disabled={reset.isPending}
                variant="destructive"
                onClick={() => reset.mutate()}
              >
                {reset.isPending
                  ? "Resetting…"
                  : `Confirm ${resetScope === "session" ? "session" : "all-time"} reset`}
              </Button>
              <Button
                disabled={reset.isPending}
                variant="outline"
                onClick={() => setConfirmReset(false)}
              >
                Cancel
              </Button>
            </div>
          ) : (
            <Button
              variant="outline"
              onClick={() => {
                reset.reset()
                setConfirmReset(true)
              }}
            >
              Reset selected statistics
            </Button>
          )}
          {reset.isError && (
            <p role="alert" className="mt-2 text-destructive">
              {errorMessage(reset.error)}
            </p>
          )}
        </CardContent>
      </Card>
      <QueryState query={versions}>
        {versions.data && (
          <Card>
            <CardHeader>
              <CardTitle>Engine provenance</CardTitle>
              <CardDescription>
                Python {versions.data.python} · {versions.data.platform}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableBody>
                  {Object.entries(versions.data.engines).map(
                    ([name, engine]) => (
                      <TableRow key={name}>
                        <TableCell>{name}</TableCell>
                        <TableCell>
                          {engine.version ?? "Not installed"}
                          <p className="text-xs break-all text-muted-foreground">
                            {engine.commit ?? "No source commit reported"}
                          </p>
                          {engine.url && engine.url.startsWith("https://") && (
                            <a
                              href={engine.url}
                              target="_blank"
                              rel="noreferrer"
                              className="text-sm underline"
                            >
                              Source repository
                            </a>
                          )}
                          {engine.source && (
                            <p className="text-xs break-all">{engine.source}</p>
                          )}
                        </TableCell>
                      </TableRow>
                    )
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        )}
      </QueryState>
    </>
  )
}
