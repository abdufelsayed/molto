import type { components } from "@/lib/api-types"
import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { managementQuery } from "@/features/management/request"
import {
  useAction,
  Failure,
  Choice,
  TextField,
} from "@/features/acquisition/controls"
import { PageTitle, QueryState } from "@/components/page-state"
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Progress,
  ProgressLabel,
  ProgressValue,
} from "@/components/ui/progress"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog"
export type Operation = {
  id: string
  kind: string
  model_id?: string | null
  status: string
  stage: string
  progress: number
  error?: string | null
  created_at?: string | number | null
  actions?: { cancel?: boolean; retry?: boolean; delete?: boolean }
  diffusion?: boolean
  result?: {
    output?: string
    output_path?: string
    path?: string
    repo_id?: string
    message?: string
  } | null
}
export function ActivityPage() {
  const query = useQuery(
    managementQuery<{ operations: Operation[] }>(
      ["operations"],
      "operations",
      2000
    )
  )
  const diffusion = useQuery(
    managementQuery<components["schemas"]["DiffusionJobsResponse"]>(
      ["diffusion-jobs"],
      "diffusion/jobs",
      2000
    )
  )
  const records: Operation[] = [
    ...(query.data?.operations ?? []),
    ...(diffusion.data?.jobs ?? []).map((job) => ({
      id: job.id,
      kind: `diffusion ${job.kind}`,
      model_id: job.model_id,
      status: job.status,
      stage: job.phase,
      progress: job.progress * 100,
      error: job.error,
      diffusion: true,
      actions: {
        cancel: ["queued", "waiting", "running"].includes(job.status),
      },
    })),
  ]
  const action = useAction()
  const [retryToken, setRetryToken] = useState("")
  const [filter, setFilter] = useState("all")
  const [confirmation, setConfirmation] = useState<{
    operation: Operation
    action: string
  } | null>(null)
  return (
    <>
      <PageTitle
        title="Activity"
        description="Follow server operations and retained progress. Actions are available only when the server supports them."
      />
      <Choice
        label="Show operations"
        value={filter}
        onChange={setFilter}
        options={[
          { value: "all", label: "All operations" },
          { value: "active", label: "In progress" },
          { value: "failed", label: "Failed" },
        ]}
      />
      <QueryState query={query}>
        {query.data && (
          <div className="grid gap-4">
            {records
              .filter(
                (item) =>
                  filter === "all" ||
                  (filter === "active"
                    ? ["queued", "running", "waiting", "cancelling"].includes(
                        item.status
                      )
                    : item.status === "failed")
              )
              .map((operation) => (
                <Card key={operation.id}>
                  <CardHeader>
                    <div className="flex justify-between gap-4">
                      <CardTitle className="break-all">
                        {operation.model_id ?? operation.kind}
                      </CardTitle>
                      <Badge variant="outline">{operation.status}</Badge>
                    </div>
                    <CardDescription>
                      {operation.kind} · {operation.id}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="grid gap-4">
                    <Progress
                      value={Math.max(0, Math.min(100, operation.progress))}
                    >
                      <ProgressLabel>{operation.stage}</ProgressLabel>
                      <ProgressValue />
                    </Progress>
                    {operation.error && (
                      <p role="alert" className="text-sm text-destructive">
                        {operation.error}
                      </p>
                    )}
                    {operation.result && (
                      <p className="text-sm break-all">
                        {operation.result.output_path ??
                          operation.result.output ??
                          operation.result.path ??
                          operation.result.repo_id ??
                          operation.result.message}
                      </p>
                    )}
                    <div className="flex gap-2">
                      {["cancel", "retry", "delete"]
                        .filter(
                          (name) =>
                            operation.actions?.[
                              name as "cancel" | "retry" | "delete"
                            ]
                        )
                        .map((name) => (
                          <Button
                            key={name}
                            variant="outline"
                            disabled={action.isPending || query.isError}
                            onClick={() => {
                              action.reset()
                              setRetryToken("")
                              setConfirmation({ operation, action: name })
                            }}
                          >
                            {name === "delete"
                              ? "Remove record"
                              : name === "retry"
                                ? "Retry"
                                : "Cancel"}
                          </Button>
                        ))}
                    </div>
                  </CardContent>
                </Card>
              ))}
            {!records.length && (
              <p className="py-8 text-center text-muted-foreground">
                No operations recorded yet.
              </p>
            )}
          </div>
        )}
      </QueryState>
      <QueryState query={diffusion}>{null}</QueryState>
      <Dialog
        open={confirmation !== null}
        onOpenChange={(open) => {
          if (!open && !action.isPending) setConfirmation(null)
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {confirmation?.action === "delete"
                ? "Remove this activity record?"
                : confirmation?.action === "retry"
                  ? "Retry this operation?"
                  : "Cancel this operation?"}
            </DialogTitle>
            <DialogDescription>
              {confirmation?.action === "delete"
                ? "This removes the retained record. Model files are not removed."
                : confirmation?.action === "retry"
                  ? "The server will start another attempt with the retained inputs."
                  : "The server will request cancellation. Progress may continue until native work stops."}
            </DialogDescription>
          </DialogHeader>
          {confirmation?.action === "retry" &&
            confirmation.operation.kind.startsWith("download_") && (
              <TextField
                label="Download token, optional"
                type="password"
                value={retryToken}
                onChange={setRetryToken}
                description="Credentials are not retained with jobs. Enter a token again for a gated repository."
              />
            )}
          <Failure error={action.error} />
          <DialogFooter>
            <Button
              variant="outline"
              disabled={action.isPending}
              onClick={() => setConfirmation(null)}
            >
              Back
            </Button>
            <Button
              disabled={action.isPending}
              onClick={() => {
                if (confirmation)
                  action.mutate(
                    {
                      path: `${confirmation.operation.diffusion ? "diffusion/jobs" : "operations"}/${encodeURIComponent(confirmation.operation.id)}${confirmation.action === "delete" ? "" : `/${confirmation.action}`}`,
                      method:
                        confirmation.action === "delete" ? "DELETE" : "POST",
                      body:
                        confirmation.action === "retry"
                          ? { token: retryToken }
                          : undefined,
                    },
                    {
                      onSuccess: () => {
                        setConfirmation(null)
                        setRetryToken("")
                      },
                    }
                  )
              }}
            >
              {action.isPending ? "Submitting…" : "Confirm"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
