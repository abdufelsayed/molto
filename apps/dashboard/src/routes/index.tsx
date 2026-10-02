import { useEffect, useRef } from "react"
import {
  usePreference,
  usePreferencesReady,
} from "@/features/preferences/provider"
import { overviewScopeSchema } from "@/features/preferences/app-schema"
import { Link, createFileRoute, useRouterState } from "@tanstack/react-router"
import { useQuery } from "@tanstack/react-query"
import { z } from "zod"
import {
  stateQuery,
  statsQuery,
  modelsQuery,
  useManagement,
} from "@/features/management/queries"
import { ModelTable } from "@/features/models/table"
import { PageTitle, QueryState } from "@/components/page-state"
import { bytes, count, percentage } from "@/lib/format"
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
  SelectGroup,
} from "@/components/ui/select"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"

export const Route = createFileRoute("/")({
  validateSearch: z.object({
    scope: z.enum(["session", "alltime"]).catch("session").optional(),
  }),
  component: Overview,
})
function Overview() {
  const { api } = useManagement()
  const { scope = "session" } = Route.useSearch()
  const navigate = Route.useNavigate()
  const [rememberedScope, setRememberedScope] = usePreference("overview.scope")
  const preferencesReady = usePreferencesReady()
  const hydratedScope = useRef(false)
  const restoringScope = useRef<string | undefined>(undefined)
  const pathname = useRouterState({
    select: (state) => state.location.pathname,
  })
  const rawSearch = useRouterState({
    select: (state) => state.location.search,
  })
  useEffect(() => {
    if (!preferencesReady || pathname !== "/") return
    const rawScope = overviewScopeSchema.safeParse(rawSearch.scope)
    if ((rawScope.success ? rawScope.data : "session") !== scope) return
    if (restoringScope.current !== undefined) {
      if (scope !== restoringScope.current) return
      restoringScope.current = undefined
    }
    if (!hydratedScope.current) {
      hydratedScope.current = true
      const explicit = overviewScopeSchema.safeParse(rawSearch.scope)
      const restored = explicit.success ? explicit.data : rememberedScope
      if (restored !== scope) {
        restoringScope.current = restored
        void navigate({ search: { scope: restored }, replace: true })
        return
      }
    }
    if (rememberedScope !== scope) setRememberedScope(scope)
  }, [
    preferencesReady,
    pathname,
    scope,
    rawSearch,
    rememberedScope,
    navigate,
    setRememberedScope,
  ])
  const state = useQuery(stateQuery(api))
  const stats = useQuery(statsQuery(api, scope))
  const models = useQuery(modelsQuery(api))
  const current = state.data
  return (
    <>
      <PageTitle
        title="Overview"
        description="Current model residency and serving statistics from Molto."
      />
      <QueryState query={state}>
        {current && (
          <>
            {current.preparation_active && (
              <Alert>
                <AlertTitle>Preparation is active</AlertTitle>
                <AlertDescription>
                  Molto has reserved engine admission for a local preparation
                  job. Loading and rescanning may be unavailable until it
                  finishes.
                </AlertDescription>
              </Alert>
            )}
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              {[
                [
                  "Loaded models",
                  count(current.loaded_count),
                  `${current.model_count} discovered`,
                ],
                [
                  "Model memory",
                  bytes(current.current_model_memory),
                  "Engine-pool accounting",
                ],
                [
                  "Memory ceiling",
                  bytes(current.final_ceiling),
                  "Current pool admission ceiling",
                ],
                [
                  "Default model",
                  current.default_model ?? "None",
                  "Used when the API request omits a model",
                ],
              ].map(([label, value, hint]) => (
                <Card key={label}>
                  <CardHeader>
                    <CardDescription>{label}</CardDescription>
                    <CardTitle className="text-2xl break-all">
                      {value}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="text-xs text-muted-foreground">
                    {hint}
                  </CardContent>
                </Card>
              ))}
            </div>
            <Card>
              <CardHeader>
                <CardTitle>Model memory</CardTitle>
                <CardDescription>
                  Reported resident models against the current memory ceiling.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-3">
                <div className="flex justify-between text-sm">
                  <span>
                    {bytes(current.current_model_memory)} of{" "}
                    {bytes(current.final_ceiling)}
                  </span>
                  <span>
                    {percentage(
                      current.current_model_memory,
                      current.final_ceiling
                    )}
                    %
                  </span>
                </div>
                <Progress
                  value={Math.min(
                    100,
                    percentage(
                      current.current_model_memory,
                      current.final_ceiling
                    )
                  )}
                />
              </CardContent>
              <CardFooter className="text-xs text-muted-foreground">
                This reports Molto model residency. It does not measure total
                process memory or transient execution peaks.
              </CardFooter>
            </Card>
          </>
        )}
      </QueryState>
      {current && (
        <>
          <PageTitle
            title="Serving statistics"
            description={
              scope === "session"
                ? "Counters for the current Molto server session."
                : "Persisted totals collected by Molto."
            }
          >
            <Select
              items={[
                { value: "session", label: "This session" },
                { value: "alltime", label: "All time" },
              ]}
              value={scope}
              onValueChange={(value) =>
                void navigate({
                  search: { scope: value as typeof scope },
                  replace: true,
                })
              }
            >
              <SelectTrigger aria-label="Statistics scope" className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  <SelectItem value="session">This session</SelectItem>
                  <SelectItem value="alltime">All time</SelectItem>
                </SelectGroup>
              </SelectContent>
            </Select>
          </PageTitle>
          <QueryState query={stats}>
            {stats.data && (
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                {[
                  ["Requests", count(stats.data.total_requests)],
                  ["Served tokens", count(stats.data.total_tokens_served)],
                  ["Cached tokens", count(stats.data.total_cached_tokens)],
                  [
                    "Cache efficiency",
                    `${stats.data.cache_efficiency.toFixed(1)}%`,
                  ],
                  [
                    "Prefill speed",
                    `${stats.data.avg_prefill_tps.toFixed(1)} tok/s`,
                  ],
                  [
                    "Generation speed",
                    `${stats.data.avg_generation_tps.toFixed(1)} tok/s`,
                  ],
                ].map(([label, value]) => (
                  <Card key={label}>
                    <CardHeader>
                      <CardDescription>{label}</CardDescription>
                      <CardTitle className="text-2xl">{value}</CardTitle>
                    </CardHeader>
                  </Card>
                ))}
              </div>
            )}
          </QueryState>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle>Models</CardTitle>
              <Button
                variant="outline"
                size="sm"
                render={<Link to="/models" />}
                nativeButton={false}
              >
                View library
              </Button>
            </CardHeader>
            <CardContent>
              <QueryState query={models}>
                {models.data?.models.length ? (
                  <ModelTable models={models.data.models} />
                ) : (
                  <p className="text-sm text-muted-foreground">
                    No models discovered. Add a checkpoint to a configured
                    directory and rescan from Models.
                  </p>
                )}
              </QueryState>
            </CardContent>
          </Card>
        </>
      )}
    </>
  )
}
