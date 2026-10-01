import { useState } from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import {
  managementRequest,
  managementQuery,
} from "@/features/management/request"
import {
  useManagement,
  managementKey,
  modelSettingsQuery,
} from "@/features/management/queries"
import { errorMessage, settingsMessage } from "@/features/management/api"
import type { SettingsResult } from "@/features/management/api"
import { modelOptionsQuery, optionFields } from "./model-options"
import { SettingsForm } from "@/features/settings/form"
import { QueryState } from "@/components/page-state"
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardDescription,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
type Template = {
  name: string
  display_name?: string
  description?: string | null
  settings: Record<string, unknown>
}
type OptimalCandidate = {
  benchmark_id: string
  pp_tps?: number
  tg_tps?: number
  quantization?: string
  omlx_version?: string
}
type OptimalCandidates = {
  found: boolean
  context_length: number
  device: { chip_name: string; chip_variant: string; memory_gb: number }
  by_pp: OptimalCandidate[]
  by_tg: OptimalCandidate[]
}
type Presets = {
  presets: Template[]
  source?: string
  error?: string
}
export function ModelHelpers({ modelId }: { modelId: string }) {
  const { api, queryClient } = useManagement()
  const path = `models/${encodeURIComponent(modelId)}`
  const options = useQuery(modelOptionsQuery(modelId))
  const settings = useQuery(modelSettingsQuery(api, modelId))
  const templates = useQuery(
    managementQuery<{ templates: Template[] }>(["templates"], "templates")
  )
  const presets = useQuery(managementQuery<Presets>(["presets"], "presets"))
  const optimal = useQuery({
    ...managementQuery<OptimalCandidates>(
      ["optimal", modelId],
      `${path}/settings/optimal`
    ),
    enabled: false,
  })
  const generation = useQuery(
    managementQuery<{ settings: Record<string, unknown> }>(
      ["generation-config", modelId],
      `${path}/generation-config`
    )
  )
  const [editor, setEditor] = useState<Template | "new" | null>(null)
  const [recipe, setRecipe] = useState("")
  const [benchmark, setBenchmark] = useState("")
  const [notice, setNotice] = useState("")
  const [refreshedPresets, setRefreshedPresets] = useState<Template[] | null>(
    null
  )
  const mutation = useMutation({
    mutationFn: ({
      target,
      method = "POST",
      body,
    }: {
      target: string
      method?: "POST" | "DELETE" | "PATCH"
      body?: unknown
    }) => managementRequest<SettingsResult | Presets>(target, { method, body }),
    onSuccess: async (result) => {
      if ("presets" in result) setRefreshedPresets(result.presets)
      setNotice(
        typeof result === "object" && "model_id" in result
          ? settingsMessage(result as SettingsResult)
          : "Action completed. Current settings and saved configurations have been refreshed."
      )
      await queryClient.invalidateQueries({ queryKey: managementKey })
    },
  })
  const act = (
    target: string,
    confirmation?: string,
    body?: unknown,
    method?: "POST" | "DELETE" | "PATCH"
  ) => {
    if (!confirmation || window.confirm(confirmation))
      mutation.mutate({ target, body, method })
  }
  return (
    <div className="mt-6 space-y-6">
      {mutation.isError && (
        <Alert variant="destructive">
          <AlertTitle>Action failed</AlertTitle>
          <AlertDescription>{errorMessage(mutation.error)}</AlertDescription>
        </Alert>
      )}
      {notice && (
        <Alert>
          <AlertTitle>Configuration result</AlertTitle>
          <AlertDescription>{notice}</AlertDescription>
        </Alert>
      )}
      <Card>
        <CardHeader>
          <CardTitle>Cross-model templates</CardTitle>
          <CardDescription>
            Templates share universal request settings across models. Engine and
            identity settings stay model-specific.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Button
            disabled={!options.data || !settings.data || mutation.isPending}
            onClick={() => setEditor("new")}
          >
            Create template from current settings
          </Button>
          {editor && options.data && (
            <TemplateEditor
              key={editor === "new" ? "new" : editor.name}
              template={editor === "new" ? undefined : editor}
              current={settings.data?.settings ?? {}}
              options={options.data}
              done={() => {
                setEditor(null)
                void queryClient.invalidateQueries({ queryKey: managementKey })
              }}
            />
          )}
          <QueryState query={templates}>
            <div className="space-y-3">
              {templates.data?.templates.map((template) => (
                <div key={template.name} className="rounded-md border p-3">
                  <p className="font-medium">
                    {template.display_name || template.name}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {template.description}
                  </p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <Button
                      disabled={mutation.isPending}
                      onClick={() =>
                        act(
                          `${path}/templates/${encodeURIComponent(template.name)}/apply`,
                          `Apply ${template.name}? Universal settings omitted by this template reset to defaults.`
                        )
                      }
                    >
                      Apply template
                    </Button>
                    <Button
                      variant="outline"
                      disabled={mutation.isPending}
                      onClick={() => setEditor(template)}
                    >
                      Edit template
                    </Button>
                    <Button
                      variant="ghost"
                      disabled={mutation.isPending}
                      onClick={() =>
                        act(
                          `templates/${encodeURIComponent(template.name)}`,
                          `Delete shared template ${template.name}?`,
                          undefined,
                          "DELETE"
                        )
                      }
                    >
                      Delete template
                    </Button>
                  </div>
                </div>
              ))}
              {templates.data?.templates.length === 0 && (
                <p className="text-sm text-muted-foreground">
                  No shared templates saved.
                </p>
              )}
            </div>
          </QueryState>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Configuration helpers</CardTitle>
          <CardDescription>
            These actions save configuration and may reload the engine. Finish
            or discard unsaved settings before applying a helper.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <QueryState query={generation}>
            <div>
              <p className="mb-2 text-sm">
                Checkpoint generation defaults:{" "}
                {generation.data && Object.keys(generation.data.settings).length
                  ? Object.entries(generation.data.settings)
                      .map(
                        ([key, value]) =>
                          `${key.replaceAll("_", " ")}: ${String(value)}`
                      )
                      .join(", ")
                  : "None available"}
              </p>
              <Button
                variant="outline"
                disabled={
                  mutation.isPending ||
                  !generation.data ||
                  !Object.keys(generation.data.settings).length
                }
                onClick={() =>
                  act(
                    `${path}/generation-config`,
                    "Import checkpoint generation defaults into this model's settings?"
                  )
                }
              >
                Import generation config
              </Button>
            </div>
          </QueryState>
          <div className="space-y-3">
            <p className="text-sm text-muted-foreground">
              Find published optimal snapshots for this model and device. This
              contacts omlx.ai and compares results at a 4,096 token context.
            </p>
            <Button
              variant="outline"
              disabled={optimal.isFetching || mutation.isPending}
              onClick={() => void optimal.refetch()}
            >
              Find optimal snapshots
            </Button>
            {optimal.isError && (
              <Alert variant="destructive">
                <AlertDescription>
                  {errorMessage(optimal.error)}
                </AlertDescription>
              </Alert>
            )}
            {optimal.data && (
              <div className="space-y-2">
                <p className="text-sm">
                  {optimal.data.device.chip_name}{" "}
                  {optimal.data.device.chip_variant},{" "}
                  {optimal.data.device.memory_gb} GB
                </p>
                {Array.from(
                  new Map(
                    [...optimal.data.by_pp, ...optimal.data.by_tg].map(
                      (row) => [row.benchmark_id, row]
                    )
                  ).values()
                ).map((row) => (
                  <div
                    key={row.benchmark_id}
                    className="flex flex-wrap items-center justify-between gap-2 rounded border p-3"
                  >
                    <span className="text-sm">
                      {row.benchmark_id} · Prefill {row.pp_tps ?? "unreported"}{" "}
                      tok/s · Generation {row.tg_tps ?? "unreported"} tok/s ·{" "}
                      {row.quantization ?? "unreported"} · Molto{" "}
                      {row.omlx_version ?? "unreported"}
                    </span>
                    <Button
                      size="sm"
                      disabled={mutation.isPending}
                      onClick={() =>
                        act(
                          `${path}/settings/optimal`,
                          "Apply this published benchmark snapshot? Omitted profile fields reset to defaults.",
                          { benchmark_id: row.benchmark_id }
                        )
                      }
                    >
                      Apply snapshot
                    </Button>
                  </div>
                ))}
                {!optimal.data.found && (
                  <p className="text-sm text-muted-foreground">
                    No matching published snapshots.
                  </p>
                )}
              </div>
            )}
          </div>
          <FieldGroup>
            <Field>
              <FieldLabel htmlFor="settings-recipe">Settings recipe</FieldLabel>
              <Input
                id="settings-recipe"
                value={recipe}
                onChange={(event) => setRecipe(event.target.value)}
                placeholder="Paste a Molto recipe code"
              />
              <Button
                className="w-fit"
                disabled={!recipe.trim() || mutation.isPending}
                onClick={() =>
                  act(
                    `${path}/settings/recipe`,
                    "Apply this settings recipe? It may change engine settings and trigger a reload.",
                    { recipe: recipe.trim() }
                  )
                }
              >
                Apply recipe
              </Button>
            </Field>
            <Field>
              <FieldLabel htmlFor="optimal-benchmark">
                Benchmark result ID
              </FieldLabel>
              <Input
                id="optimal-benchmark"
                value={benchmark}
                onChange={(event) => setBenchmark(event.target.value)}
                placeholder="Published benchmark ID"
              />
              <Button
                className="w-fit"
                disabled={!benchmark.trim() || mutation.isPending}
                onClick={() =>
                  act(
                    `${path}/settings/optimal`,
                    "Apply the optimal configuration from this published benchmark result? It will be fetched from omlx.ai.",
                    { benchmark_id: benchmark.trim() }
                  )
                }
              >
                Apply optimal snapshot
              </Button>
            </Field>
          </FieldGroup>
          <Button
            variant="destructive"
            disabled={mutation.isPending}
            onClick={() =>
              act(
                `${path}/settings/reset`,
                "Reset every model override to defaults? Profiles and templates remain saved."
              )
            }
          >
            Reset all model settings
          </Button>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Recommended presets</CardTitle>
          <CardDescription>
            Refresh explicitly to fetch the current published presets.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Button
            variant="outline"
            disabled={mutation.isPending}
            onClick={() => act("presets/refresh")}
          >
            Refresh presets
          </Button>
          <QueryState query={presets}>
            {presets.data && (
              <PresetList
                values={refreshedPresets ?? presets.data.presets}
                disabled={mutation.isPending}
                apply={(preset) =>
                  act(
                    `${path}/settings`,
                    `Apply ${preset.display_name || preset.name}? Only the listed preset settings change.`,
                    preset.settings,
                    "PATCH"
                  )
                }
              />
            )}
          </QueryState>
        </CardContent>
      </Card>
    </div>
  )
}
function PresetList({
  values,
  disabled,
  apply,
}: {
  values: Template[]
  disabled: boolean
  apply: (preset: Template) => void
}) {
  return (
    <div className="space-y-3">
      {values.map((preset) => (
        <div key={preset.name} className="rounded-md border p-3">
          <p className="font-medium">{preset.display_name || preset.name}</p>
          <p className="text-sm text-muted-foreground">{preset.description}</p>
          <dl className="my-3 grid gap-2 sm:grid-cols-2">
            {Object.entries(preset.settings).map(([key, value]) => (
              <div key={key}>
                <dt className="text-xs text-muted-foreground">
                  {key.replaceAll("_", " ")}
                </dt>
                <dd className="text-sm">{JSON.stringify(value)}</dd>
              </div>
            ))}
          </dl>
          <Button disabled={disabled} onClick={() => apply(preset)}>
            Apply preset
          </Button>
        </div>
      ))}
    </div>
  )
}
function TemplateEditor({
  template,
  current,
  options,
  done,
}: {
  template?: Template
  current: Record<string, unknown>
  options: import("./model-options").ModelOptions
  done: () => void
}) {
  const [name, setName] = useState(template?.name ?? "")
  const [display, setDisplay] = useState(template?.display_name ?? "")
  const [description, setDescription] = useState(template?.description ?? "")
  const [values] = useState(
    () =>
      template?.settings ??
      Object.fromEntries(
        options.fields
          .filter((field) => field.template)
          .map((field) => [field.key, current[field.key] ?? null])
      )
  )
  return (
    <div className="space-y-4 rounded-md border p-4">
      <FieldGroup>
        <Field>
          <FieldLabel htmlFor="template-name">Template name</FieldLabel>
          <Input
            id="template-name"
            value={name}
            onChange={(event) => setName(event.target.value)}
          />
        </Field>
        <Field>
          <FieldLabel htmlFor="template-display">
            Template display name
          </FieldLabel>
          <Input
            id="template-display"
            value={display}
            onChange={(event) => setDisplay(event.target.value)}
          />
        </Field>
        <Field>
          <FieldLabel htmlFor="template-description">
            Template description
          </FieldLabel>
          <Input
            id="template-description"
            value={description}
            onChange={(event) => setDescription(event.target.value)}
          />
        </Field>
      </FieldGroup>
      <SettingsForm
        values={values}
        fields={optionFields(options, "template")}
        alwaysSave
        submitLabel="Save template"
        save={(patch) => {
          if (!name.trim()) throw new Error("Enter a template name.")
          return managementRequest(
            template
              ? `templates/${encodeURIComponent(template.name)}`
              : "templates",
            {
              method: template ? "PUT" : "POST",
              body: {
                ...(template
                  ? { new_name: name.trim() }
                  : { name: name.trim() }),
                display_name: display,
                description: description || null,
                settings: { ...values, ...patch },
              },
            }
          )
        }}
        describe={() => "Template saved."}
        saved={async () => done()}
      />
      <Button
        variant="ghost"
        onClick={() => {
          if (
            window.confirm(
              "Close this editor and discard unsaved template changes?"
            )
          )
            done()
        }}
      >
        Close template editor
      </Button>
    </div>
  )
}
