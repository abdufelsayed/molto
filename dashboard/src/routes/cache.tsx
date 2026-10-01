import { createFileRoute } from "@tanstack/react-router"
import { useQuery, useMutation } from "@tanstack/react-query"
import { useState } from "react"
import { toast } from "sonner"
import {
  cacheQuery,
  stateQuery,
  managementKey,
  useManagement,
} from "@/features/management/queries"
import { errorMessage } from "@/features/management/api"
import { PageTitle, QueryState } from "@/components/page-state"
import { CacheProbe } from "@/features/monitoring/cache-probe"
import { bytes, count } from "@/lib/format"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogCancel,
} from "@/components/ui/alert-dialog"
import { Table, TableBody, TableRow, TableCell } from "@/components/ui/table"

export const Route = createFileRoute("/cache")({ component: CachePage })
function metrics(
  value: Record<string, unknown>,
  prefix = ""
): { key: string; value: string }[] {
  return Object.entries(value).flatMap(([key, child]) => {
    const path = prefix
      ? `${prefix} / ${key.replaceAll("_", " ")}`
      : key.replaceAll("_", " ")
    if (child && typeof child === "object" && !Array.isArray(child))
      return metrics(child as Record<string, unknown>, path)
    if (typeof child === "number")
      return [
        {
          key: path,
          value: key.endsWith("_bytes")
            ? bytes(child)
            : Number.isInteger(child)
              ? count(child)
              : child.toFixed(2),
        },
      ]
    if (typeof child === "boolean")
      return [{ key: path, value: child ? "Enabled" : "Disabled" }]
    if (typeof child === "string") return [{ key: path, value: child }]
    return []
  })
}
function CachePage() {
  const { api, queryClient } = useManagement()
  const query = useQuery(cacheQuery(api))
  const state = useQuery(stateQuery(api))
  const [kind, setKind] = useState<"hot" | "ssd" | null>(null)
  const clear = useMutation({
    mutationFn: (tier: "hot" | "ssd") => api.clearCache(tier),
    onSuccess: (result) => {
      toast.success(
        `${result.kind === "hot" ? "Hot" : "SSD"} cache cleared. ${count(result.total_cleared)} entries or files reclaimed.`
      )
      setKind(null)
    },
    onSettled: () => queryClient.invalidateQueries({ queryKey: managementKey }),
  })
  return (
    <>
      <PageTitle
        title="Cache"
        description="Inspect the cache metrics reported by each loaded engine."
      />
      <QueryState query={query}>
        {query.data && (
          <>
            <div className="grid gap-4 sm:grid-cols-2">
              {(["hot", "ssd"] as const).map((tier) => (
                <Card key={tier}>
                  <CardHeader>
                    <CardTitle>
                      {tier === "hot" ? "Hot cache" : "SSD cache"}
                    </CardTitle>
                    <CardDescription>
                      {tier === "hot"
                        ? "Reuse held in memory by loaded engines."
                        : (query.data.ssd_cache_dir ??
                          "No SSD cache directory reported.")}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Button
                      variant="outline"
                      disabled={
                        query.isError ||
                        clear.isPending ||
                        state.data?.preparation_active
                      }
                      onClick={() => {
                        clear.reset()
                        setKind(tier)
                      }}
                    >
                      Clear {tier === "hot" ? "hot" : "SSD"} cache
                    </Button>
                  </CardContent>
                </Card>
              ))}
            </div>
            <div className="grid gap-4">
              {query.data.models.map((model) => (
                <Card key={model.model_id}>
                  <CardHeader>
                    <CardTitle className="break-all">
                      {model.model_id}
                    </CardTitle>
                    <CardDescription>
                      Metrics reported by this engine. Available fields differ
                      between engine types.
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {model.stats ? (
                      <Table>
                        <TableBody>
                          {metrics(model.stats).map((metric) => (
                            <TableRow key={metric.key}>
                              <TableCell className="text-muted-foreground">
                                {metric.key}
                              </TableCell>
                              <TableCell className="text-right tabular-nums">
                                {metric.value}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    ) : (
                      <p className="text-sm text-muted-foreground">
                        This engine does not report cache metrics.
                      </p>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
            {!query.data.models.length && (
              <p className="py-8 text-center text-muted-foreground">
                No loaded engines reporting cache metrics.
              </p>
            )}
          </>
        )}
      </QueryState>
      <CacheProbe
        models={query.data?.models.map((model) => model.model_id) ?? []}
      />
      <AlertDialog
        open={kind !== null}
        onOpenChange={(open) => {
          if (!open && !clear.isPending) setKind(null)
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              Clear {kind === "hot" ? "hot" : "SSD"} cache?
            </AlertDialogTitle>
            <AlertDialogDescription>
              {kind === "hot"
                ? "This clears supported hot caches across loaded engines while keeping models loaded."
                : "This deletes known saved SSD cache files, including files belonging to unloaded models."}{" "}
              Later requests may recompute prefixes. Active requests prevent
              clearing.
            </AlertDialogDescription>
          </AlertDialogHeader>
          {clear.isError && (
            <Alert variant="destructive">
              <AlertTitle>Cache could not be cleared</AlertTitle>
              <AlertDescription>{errorMessage(clear.error)}</AlertDescription>
            </Alert>
          )}
          <AlertDialogFooter>
            <AlertDialogCancel disabled={clear.isPending}>
              Cancel
            </AlertDialogCancel>
            <Button
              variant="destructive"
              disabled={clear.isPending}
              onClick={() => {
                if (kind) clear.mutate(kind)
              }}
            >
              {clear.isPending ? "Clearing…" : "Confirm clear"}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}
