import { Link, createFileRoute } from "@tanstack/react-router"
import { useQuery } from "@tanstack/react-query"
import { z } from "zod"
import {
  modelsQuery,
  modelSettingsQuery,
  stateQuery,
  managementKey,
  useManagement,
} from "@/features/management/queries"
import { modelState, settingsMessage } from "@/features/management/api"
import { ModelActions } from "@/features/models/actions"
import { SettingsForm } from "@/features/settings/form"
import {
  modelOptionsQuery,
  optionFields,
} from "@/features/models/model-options"
import { WorkspaceModelPanel } from "@/features/workspace/model-panel"
import { ModelMtplxImport } from "@/features/models/settings-mtplx"
import { ModelHelpers } from "@/features/models/helpers"
import { managementRequest } from "@/features/management/request"
import type { SettingsResult } from "@/features/management/api"
import { ProfilesPanel } from "@/features/profiles/panel"
import { PageTitle, QueryState } from "@/components/page-state"
import { bytes } from "@/lib/format"
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"

export const Route = createFileRoute("/models_/$modelId")({
  validateSearch: z.object({
    tab: z
      .enum(["summary", "settings", "profiles", "workspace"])
      .catch("summary")
      .default("summary"),
  }),
  component: ModelPage,
})
function ModelPage() {
  const { modelId } = Route.useParams()
  const { tab } = Route.useSearch()
  const navigate = Route.useNavigate()
  const { api, queryClient } = useManagement()
  const inventory = useQuery(modelsQuery(api))
  const query = useQuery(modelSettingsQuery(api, modelId))
  const state = useQuery(stateQuery(api))
  const model = inventory.data?.models.find((item) => item.id === modelId)
  const options = useQuery(modelOptionsQuery(modelId))
  return (
    <QueryState query={inventory}>
      {model ? (
        <>
          <div>
            <Button
              variant="ghost"
              size="sm"
              render={<Link to="/models" />}
              nativeButton={false}
            >
              Back to models
            </Button>
          </div>
          <PageTitle
            title={model.settings.model_alias || model.id}
            description={model.id}
          >
            <Badge variant={model.load_failed ? "destructive" : "secondary"}>
              {modelState(model)}
            </Badge>
          </PageTitle>
          <ModelActions
            model={model}
            disabled={inventory.isError || state.data?.preparation_active}
          />
          {model.load_failure_message && (
            <Alert variant="destructive">
              <AlertTitle>Model failed to load</AlertTitle>
              <AlertDescription>{model.load_failure_message}</AlertDescription>
            </Alert>
          )}
          <Tabs
            value={tab}
            onValueChange={(next) =>
              void navigate({
                search: { tab: next as typeof tab },
                replace: true,
              })
            }
          >
            <TabsList>
              <TabsTrigger value="summary">Summary</TabsTrigger>
              <TabsTrigger value="settings">Settings</TabsTrigger>
              <TabsTrigger value="profiles">Profiles</TabsTrigger>
              <TabsTrigger value="workspace">Workspace</TabsTrigger>
            </TabsList>
            <TabsContent value="summary">
              <Card>
                <CardHeader>
                  <CardTitle>Checkpoint and engine</CardTitle>
                </CardHeader>
                <CardContent>
                  <dl className="grid gap-5 sm:grid-cols-2">
                    {[
                      ["Model ID", model.id],
                      ["Location", model.model_path ?? "Not reported"],
                      ["Task", model.model_type],
                      ["Engine", model.engine_type],
                      [
                        "Architecture",
                        model.config_model_type ?? "Not reported",
                      ],
                      [
                        "Estimated weights",
                        typeof model.estimated_size === "number"
                          ? bytes(model.estimated_size)
                          : "Not reported",
                      ],
                      [
                        "Resident model estimate",
                        model.loaded &&
                        typeof model.resident_estimated_size === "number"
                          ? bytes(model.resident_estimated_size)
                          : "Not reported",
                      ],
                      [
                        "Checkpoint context length",
                        model.model_context_length?.toLocaleString() ??
                          "Not reported",
                      ],
                    ].map(([label, value]) => (
                      <div key={label}>
                        <dt className="text-sm text-muted-foreground">
                          {label}
                        </dt>
                        <dd className="font-medium break-all">{value}</dd>
                      </div>
                    ))}
                  </dl>
                </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="settings" keepMounted>
              <ModelMtplxImport
                modelId={modelId}
                loaded={model.loaded}
                busy={
                  model.is_loading ||
                  model.is_unloading ||
                  !!state.data?.preparation_active
                }
              />
              <QueryState query={options}>
                <QueryState query={query}>
                  {query.data && (
                    <Card>
                      <CardHeader>
                        <CardTitle>Model configuration</CardTitle>
                        <CardDescription>
                          Model overrides are saved by Molto. Engine changes may
                          wait for active requests to finish.
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        {options.data?.active_profile && (
                          <Alert className="mb-4">
                            <AlertTitle>
                              Active profile: {options.data.active_profile}
                            </AlertTitle>
                            <AlertDescription>
                              {options.data.profile_drift
                                ? "Current settings differ from the saved profile. Reapply the profile to restore it, or update the profile to keep these changes."
                                : "Current settings match the saved profile."}
                            </AlertDescription>
                          </Alert>
                        )}
                        <SettingsForm
                          values={query.data.settings}
                          fields={
                            options.data ? optionFields(options.data) : []
                          }
                          save={(patch) =>
                            managementRequest<SettingsResult>(
                              `models/${encodeURIComponent(modelId)}/settings`,
                              { method: "PATCH", body: patch }
                            )
                          }
                          describe={settingsMessage}
                          saved={() =>
                            queryClient.invalidateQueries({
                              queryKey: managementKey,
                            })
                          }
                        />
                      </CardContent>
                    </Card>
                  )}
                </QueryState>
              </QueryState>
              <ModelHelpers modelId={modelId} />
            </TabsContent>
            <TabsContent value="workspace">
              <WorkspaceModelPanel modelId={modelId} />
            </TabsContent>
            <TabsContent value="profiles" keepMounted>
              <ProfilesPanel modelId={modelId} />
            </TabsContent>
          </Tabs>
        </>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>Model not found</CardTitle>
            <CardDescription>
              This ID is absent from the current inventory. Rescan the
              configured directories if you added it recently.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Link to="/models" className="underline">
              Return to models
            </Link>
          </CardContent>
        </Card>
      )}
    </QueryState>
  )
}
