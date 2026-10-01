import { useState } from "react"
import { Link, getRouteApi } from "@tanstack/react-router"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { managementRequest } from "@/features/management/request"
import {
  modelsQuery,
  stateQuery,
  useManagement,
} from "@/features/management/queries"
import { modelState } from "@/features/management/api"
import { ModelActions } from "@/features/models/actions"
import { PageTitle, QueryState } from "@/components/page-state"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Checkbox } from "@/components/ui/checkbox"
import { Field, FieldLabel, FieldTitle } from "@/components/ui/field"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog"
import { bytes } from "@/lib/format"
import { Choice, Failure, Notes } from "./primitives"
import { WorkspaceModelPanel } from "./model-panel"
import { CollectionPanel } from "./collections"
import { TransferPanel } from "./transfer"
import { registryQuery, storageQuery } from "./queries"
import type { Artifact, Plan } from "./types"
const libraryRoute = getRouteApi("/models")
export function LibraryPanel() {
  const { api } = useManagement()
  const client = useQueryClient()
  const registry = useQuery(registryQuery())
  const storage = useQuery(storageQuery())
  const models = useQuery(modelsQuery(api))
  const state = useQuery(stateQuery(api))
  const search = libraryRoute.useSearch()
  const navigate = libraryRoute.useNavigate()
  const query = search.q
  const task = search.task
  const health = search.health
  const lifecycle = search.state
  function updateSearch(patch: Partial<typeof search>) {
    void navigate({ search: { ...search, ...patch }, replace: true })
  }
  const [selected, setSelected] = useState<string[]>([])
  const [focus, setFocus] = useState<Artifact | null>(null)
  const rescan = useMutation({
    mutationFn: () => api.rescan(),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["molto"] }),
  })
  const plan = useMutation({
    mutationFn: (ids: string[]) =>
      managementRequest<Plan>("workspace/plan", {
        method: "POST",
        body: { model_ids: ids },
      }),
  })
  const all = registry.data?.models ?? []
  const rows = all.filter(
    (m) =>
      `${m.id} ${m.name ?? ""} ${m.path}`
        .toLowerCase()
        .includes(query.toLowerCase()) &&
      (task === "all" ||
        m.record.capabilities.tasks.includes(task) ||
        m.model_type === task) &&
      (health === "all" || m.health === health) &&
      (!search.type || m.model_type === search.type) &&
      (lifecycle === "all" ||
        modelState(
          models.data?.models.find((x) => x.id === (m.model_id ?? m.id)) ?? {
            loaded: m.record.runtime.loaded,
            is_loading: m.record.runtime.loading,
            is_unloading: m.record.runtime.unloading,
          }
        ).toLowerCase() === lifecycle)
  )
  const tasks = [
    ...new Set(
      all
        .flatMap((m) =>
          m.record.capabilities.tasks.length
            ? m.record.capabilities.tasks
            : [m.model_type]
        )
        .filter((x): x is string => !!x)
    ),
  ]
  const focusedModel = models.data?.models.find(
    (m) => m.id === (focus?.model_id ?? focus?.id)
  )
  return (
    <>
      <PageTitle
        title="Model library"
        description="Inspect physical files, exposed models, health and storage."
      >
        <Button
          variant="outline"
          disabled={rescan.isPending || !!state.data?.preparation_active}
          onClick={() => rescan.mutate()}
        >
          Rescan directories
        </Button>
      </PageTitle>
      <Failure error={rescan.error} />
      <QueryState query={registry}>
        <div className="grid gap-3 sm:grid-cols-4">
          <Field className="min-w-0">
            <FieldLabel htmlFor="library-search">Search</FieldLabel>
            <Input
              id="library-search"
              aria-label="Search library"
              placeholder="Search identifier or path"
              value={query}
              onChange={(e) => updateSearch({ q: e.target.value })}
            />
          </Field>
          <Field className="min-w-0" aria-labelledby="library-task-label">
            <FieldTitle id="library-task-label">Task</FieldTitle>
            <Choice
              label="Filter by task"
              value={task}
              onChange={(value) => updateSearch({ task: value, type: "" })}
              options={[
                { value: "all", label: "All tasks" },
                ...tasks.map((t) => ({ value: t, label: t })),
              ]}
            />
          </Field>
          <Field className="min-w-0" aria-labelledby="library-health-label">
            <FieldTitle id="library-health-label">Health</FieldTitle>
            <Choice
              label="Filter by health"
              value={health}
              onChange={(value) => updateSearch({ health: value })}
              options={[
                { value: "all", label: "All health states" },
                ...[...new Set(all.map((m) => m.health))].map((h) => ({
                  value: h,
                  label: h,
                })),
              ]}
            />
          </Field>
          <Field className="min-w-0" aria-labelledby="library-lifecycle-label">
            <FieldTitle id="library-lifecycle-label">Lifecycle</FieldTitle>
            <Choice
              label="Filter by lifecycle"
              value={lifecycle}
              onChange={(value) =>
                updateSearch({ state: value as typeof search.state })
              }
              options={[
                "all",
                "loaded",
                "unloaded",
                "loading",
                "unloading",
                "failed",
              ].map((s) => ({
                value: s,
                label: s === "all" ? "All lifecycle states" : s,
              }))}
            />
          </Field>
        </div>
        <p className="text-sm text-muted-foreground">
          {all.length} artifacts • {selected.length} selected
        </p>
        <div className="grid gap-3 lg:grid-cols-2">
          {rows.map((m) => {
            const live = models.data?.models.find(
              (x) => x.id === (m.model_id ?? m.id)
            )
            return (
              <Card key={m.id}>
                <CardContent className="flex items-start gap-3 pt-4">
                  <Checkbox
                    aria-label={`Select ${m.id}`}
                    checked={selected.includes(m.id)}
                    disabled={m.record.kind === "virtual"}
                    onCheckedChange={(checked) => {
                      setSelected((current) =>
                        checked
                          ? [...current, m.id]
                          : current.filter((id) => id !== m.id)
                      )
                      plan.reset()
                    }}
                  />
                  <div className="min-w-0 flex-1 space-y-2">
                    <button
                      className="text-left font-medium break-all hover:underline"
                      onClick={() => setFocus(m)}
                    >
                      {m.name || m.id}
                    </button>
                    <p className="text-xs break-all text-muted-foreground">
                      {m.path}
                    </p>
                    <div className="flex flex-wrap gap-2">
                      <Badge
                        variant={
                          m.health === "incomplete" || m.health === "failed"
                            ? "destructive"
                            : "secondary"
                        }
                      >
                        {m.health}
                      </Badge>
                      <Badge variant="outline">
                        {m.task ?? m.model_type ?? "Unknown task"}
                      </Badge>
                      <Badge variant="outline">
                        {m.record.kind === "incomplete"
                          ? "Incomplete artifact"
                          : m.exposed
                            ? "Exposed"
                            : m.record.kind === "artifact"
                              ? "Unsupported artifact"
                              : "Physical files"}
                      </Badge>
                      {live && <Badge>{modelState(live)}</Badge>}
                    </div>
                    <p className="text-sm">
                      {m.size_bytes === undefined
                        ? "Size unavailable"
                        : bytes(m.size_bytes)}
                      {m.revision && ` • ${m.revision}`}
                    </p>
                    <Notes items={m.blockers} />
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setFocus(m)}
                  >
                    Inspect
                  </Button>
                </CardContent>
              </Card>
            )
          })}
        </div>
        {!rows.length && (
          <p className="rounded-lg border p-6 text-sm">
            {all.length
              ? "No artifacts match these filters."
              : "No model artifacts found. Add a checkpoint to a configured root and rescan."}
          </p>
        )}
      </QueryState>
      <Card>
        <CardHeader>
          <CardTitle>Memory plan</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-sm text-muted-foreground">
            Select models to estimate their combined memory requirements before
            loading.
          </p>
          <Button
            disabled={!selected.length || plan.isPending}
            onClick={() => plan.mutate(selected)}
          >
            Estimate {selected.length} selected models
          </Button>
          <Failure error={plan.error} />
          {plan.data && (
            <div role="status" className="space-y-2 text-sm">
              <p>
                Additional memory:{" "}
                {plan.data.additional_bytes !== undefined
                  ? bytes(plan.data.additional_bytes)
                  : "Unavailable"}{" "}
                • Memory ceiling:{" "}
                {plan.data.ceiling_bytes !== undefined
                  ? bytes(plan.data.ceiling_bytes)
                  : "Unavailable"}
              </p>
              <p>
                {plan.data.fits === true
                  ? "Selection fits the reported memory budget."
                  : plan.data.fits === false
                    ? "Selection exceeds the reported memory budget."
                    : "The server did not report a fit decision."}
              </p>
              <p>Current usage: {bytes(plan.data.current_bytes)}</p>
              {plan.data.evictions.map((e) => (
                <p key={e.id}>
                  Would unload {e.id} to free {bytes(e.bytes)}
                </p>
              ))}
              <Notes items={plan.data.warnings} />
              {plan.data.models?.map((m) => (
                <p key={m.id}>
                  {m.id}:{" "}
                  {m.estimated_bytes !== undefined
                    ? bytes(m.estimated_bytes)
                    : "Unavailable"}
                </p>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
      <CollectionPanel selected={selected} />
      <Card>
        <CardHeader>
          <CardTitle>Storage roots</CardTitle>
        </CardHeader>
        <CardContent>
          <QueryState query={storage}>
            {storage.data?.roots.map((root) => (
              <div
                key={root.id}
                className="space-y-1 border-b py-3 last:border-0"
              >
                <p className="text-sm font-medium break-all">{root.path}</p>
                <p className="text-sm text-muted-foreground">
                  {root.free_bytes !== undefined
                    ? `${bytes(root.free_bytes)} free`
                    : "Free capacity unavailable"}
                  {root.used_bytes !== undefined
                    ? ` • ${bytes(root.used_bytes)} used`
                    : ""}
                  {root.writable === false ? " • Read only" : ""}
                </p>
              </div>
            ))}
          </QueryState>
        </CardContent>
      </Card>
      <TransferPanel />
      <Dialog
        open={focus !== null}
        onOpenChange={(open) => {
          if (!open) setFocus(null)
        }}
      >
        <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle className="break-all">{focus?.id}</DialogTitle>
            <DialogDescription className="break-all">
              {focus?.path}
            </DialogDescription>
          </DialogHeader>
          {focus && (
            <>
              {focusedModel && (
                <>
                  <Link
                    className="text-sm underline"
                    to="/models/$modelId"
                    params={{ modelId: focusedModel.id }}
                  >
                    Open model configuration
                  </Link>
                  <ModelActions
                    model={focusedModel}
                    disabled={!!state.data?.preparation_active}
                  />
                </>
              )}
              <WorkspaceModelPanel
                key={focus.id}
                modelId={focus.model_id ?? focus.id}
              />
            </>
          )}
        </DialogContent>
      </Dialog>
    </>
  )
}
