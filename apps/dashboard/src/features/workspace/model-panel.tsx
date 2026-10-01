import { useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  managementRequest,
  managementQuery,
} from "@/features/management/request"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog"
import { bytes } from "@/lib/format"
import { Choice, Failure, Notes } from "./primitives"
import { registryQuery, storageQuery } from "./queries"
import type { DeletePlan, LifecycleResult } from "./types"

export function WorkspaceModelPanel({ modelId }: { modelId: string }) {
  const client = useQueryClient()
  const base = `workspace/models/${encodeURIComponent(modelId)}`
  const storage = useQuery(storageQuery())
  const registry = useQuery(registryQuery())
  const record = registry.data?.models.find((m) => m.id === modelId)?.record
  const [mode, setMode] = useState("structural")
  const [root, setRoot] = useState("")
  const [revision, setRevision] = useState("")
  const [confirm, setConfirm] = useState<"move" | "revision" | "delete" | null>(
    null
  )
  const [deletion, setDeletion] = useState<DeletePlan | null>(null)
  const [typedId, setTypedId] = useState("")
  const action = useMutation({
    mutationFn: ({
      operation,
      body,
      method = "POST",
    }: {
      operation: string
      body?: unknown
      method?: "POST" | "DELETE"
    }) =>
      managementRequest<LifecycleResult>(`${base}/${operation}`, {
        method,
        body,
      }),
    onSuccess: () => {
      setConfirm(null)
      setDeletion(null)
      void client.invalidateQueries({ queryKey: ["omlx"] })
    },
  })
  const preview = useMutation({
    mutationFn: () => managementRequest<DeletePlan>(`${base}/delete-plan`, {}),
    onSuccess: (data) => {
      setDeletion(data)
      setTypedId("")
      setConfirm("delete")
    },
  })
  const operation = useQuery({
    ...managementQuery<NonNullable<LifecycleResult["operation"]>>(
      ["workspace", "operation", action.data?.operation?.id],
      `operations/${action.data?.operation?.id}`,
      2000
    ),
    enabled: !!action.data?.operation?.id,
  })
  const operationBusy =
    !!action.data?.operation &&
    (!operation.data || ["queued", "running"].includes(operation.data.status))
  const busy = action.isPending || preview.isPending || operationBusy
  if (record?.kind === "virtual")
    return (
      <Card>
        <CardHeader>
          <CardTitle>Exposed profile</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm">
            This profile shares its physical model files. Manage files and
            revisions on{" "}
            {record.lineage.parents.map((p) => p.id).join(", ") ||
              "the physical base model"}
            .
          </p>
        </CardContent>
      </Card>
    )
  return (
    <Card>
      <CardHeader>
        <CardTitle>Files and revisions</CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        <p className="text-sm">
          Active revision: {record?.source.revision ?? "No cached revision"}
        </p>
        {record?.source.cached_revisions.map((r) => (
          <p key={r.revision} className="text-sm break-all">
            {r.revision}
            {r.active ? " • active" : ""}
          </p>
        ))}
        {record?.lineage.parents.map((p) => (
          <p key={p.id} className="text-sm">
            Depends on {p.id} • {p.relation}
          </p>
        ))}
        {record?.lineage.children.map((p) => (
          <p key={p.id} className="text-sm">
            Related model {p.id} • {p.relation}
          </p>
        ))}
        <FieldGroup>
          <Field>
            <FieldLabel>Verification</FieldLabel>
            <Choice
              label="Verification mode"
              value={mode}
              onChange={setMode}
              options={[
                { value: "structural", label: "Structural file check" },
                { value: "smoke", label: "Smoke inference check" },
              ]}
            />
            <p className="text-sm text-muted-foreground">
              Structural checks inspect files. Smoke checks load the model and
              run inference.
            </p>
            <Button
              disabled={busy}
              onClick={() =>
                action.mutate({ operation: "verify", body: { mode } })
              }
            >
              Verify model
            </Button>
          </Field>
          <Field>
            <FieldLabel>Upstream revision</FieldLabel>
            <div className="flex flex-wrap gap-2">
              <Button
                variant="outline"
                disabled={busy}
                onClick={() => action.mutate({ operation: "check-update" })}
              >
                Check for update
              </Button>
              <Button
                variant="outline"
                disabled={busy || record?.update.status !== "update_available"}
                onClick={() => action.mutate({ operation: "stage-update" })}
              >
                Stage available update
              </Button>
            </div>
            <p className="text-sm text-muted-foreground">
              Staging downloads a separate revision. Switch only after reviewing
              it.
            </p>
            <Input
              aria-label="Revision to activate"
              placeholder="Revision identifier"
              value={revision}
              onChange={(e) => setRevision(e.target.value)}
            />
            <Button
              variant="outline"
              disabled={busy || !revision.trim()}
              onClick={() => setConfirm("revision")}
            >
              Review revision switch
            </Button>
          </Field>
          <Field>
            <FieldLabel>Storage location</FieldLabel>
            <Choice
              label="Destination storage root"
              value={root}
              onChange={setRoot}
              disabled={storage.isError}
              options={(storage.data?.roots ?? [])
                .filter((r) => r.writable !== false)
                .map((r) => ({
                  value: r.id,
                  label: `${r.path}${r.free_bytes !== undefined ? ` • ${bytes(r.free_bytes)} free` : ""}`,
                }))}
            />
            <Failure error={storage.error} />
            <Button
              variant="outline"
              disabled={busy || !root}
              onClick={() => setConfirm("move")}
            >
              Review move
            </Button>
          </Field>
        </FieldGroup>
        <Button
          variant="destructive"
          disabled={busy}
          onClick={() => {
            setDeletion(null)
            action.reset()
            preview.mutate()
          }}
        >
          Preview file deletion
        </Button>
        <Failure error={operation.error} />
        {operation.data && (
          <div role="status" className="text-sm">
            <p>
              {operation.data.stage} • {operation.data.status} •{" "}
              {operation.data.progress}%
            </p>
            {operation.data.error && (
              <p className="text-destructive">{operation.data.error}</p>
            )}
            {operation.data.result?.summary && (
              <p>{operation.data.result.summary}</p>
            )}
          </div>
        )}
        <Failure error={preview.error} />
        <Failure error={action.error} />
        {action.data && (
          <div role="status" className="rounded-lg border p-3 text-sm">
            <p>
              {action.data.message ??
                action.data.status ??
                (action.data.operation
                  ? "Operation queued. Progress appears above."
                  : "Request completed.")}
            </p>
            {action.data.current_revision && (
              <p>Current revision: {action.data.current_revision}</p>
            )}
            {action.data.latest_revision && (
              <p>Latest revision: {action.data.latest_revision}</p>
            )}
            {action.data.revisions?.map((r) => (
              <p key={r.revision}>
                {r.revision}
                {r.active ? " • active" : ""} {r.path}
              </p>
            ))}
            <Notes items={action.data.warnings} />
            <Notes items={action.data.blockers} />
          </div>
        )}
        <Dialog
          open={confirm !== null}
          onOpenChange={(open) => {
            if (!open && !busy) setConfirm(null)
          }}
        >
          <DialogContent>
            <DialogHeader>
              <DialogTitle>
                {confirm === "delete"
                  ? "Delete model files"
                  : confirm === "move"
                    ? "Move model files"
                    : "Switch revision"}
              </DialogTitle>
              <DialogDescription>
                {modelId}. Loaded models, active requests, pinned models and
                dependencies may block this action. Unload and resolve blockers
                before continuing.
              </DialogDescription>
            </DialogHeader>
            {confirm === "delete" && deletion && (
              <>
                <p className="text-sm break-all">
                  {deletion.path}
                  {deletion.physical_bytes !== undefined &&
                    ` • ${bytes(deletion.physical_bytes)}`}
                </p>
                <Notes items={deletion.protected_reasons} />
                <Field>
                  <FieldLabel htmlFor="delete-model-confirm">
                    Type the model identifier to confirm
                  </FieldLabel>
                  <Input
                    id="delete-model-confirm"
                    value={typedId}
                    onChange={(e) => setTypedId(e.target.value)}
                  />
                </Field>
                <Button
                  variant="destructive"
                  disabled={
                    busy ||
                    deletion.protected_reasons.length > 0 ||
                    !deletion.safe ||
                    !deletion.plan_token ||
                    typedId !== modelId ||
                    action.isError
                  }
                  onClick={() =>
                    action.mutate({
                      operation: "delete",
                      method: "DELETE",
                      body: { plan_token: deletion.plan_token, drain: false },
                    })
                  }
                >
                  Delete previewed files
                </Button>
              </>
            )}
            {confirm === "move" && (
              <>
                <p className="text-sm break-all">
                  Destination:{" "}
                  {storage.data?.roots.find((r) => r.id === root)?.path}
                </p>
                <Button
                  disabled={busy}
                  onClick={() =>
                    action.mutate({
                      operation: "move",
                      body: { root_id: root, drain: false },
                    })
                  }
                >
                  Confirm move
                </Button>
              </>
            )}
            {confirm === "revision" && (
              <>
                <p className="text-sm break-all">
                  Activate revision {revision}
                </p>
                <Button
                  disabled={busy}
                  onClick={() =>
                    action.mutate({
                      operation: "revision",
                      body: { revision: revision.trim(), drain: false },
                    })
                  }
                >
                  Confirm revision switch
                </Button>
              </>
            )}
            <Failure error={action.error} />
            <Button
              variant="outline"
              disabled={busy}
              onClick={() => setConfirm(null)}
            >
              Cancel
            </Button>
          </DialogContent>
        </Dialog>
      </CardContent>
    </Card>
  )
}
