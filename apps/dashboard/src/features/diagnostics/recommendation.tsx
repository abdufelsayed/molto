import { useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { managementRequest } from "@/features/management/request"
import { errorMessage } from "@/features/management/api"
import { Button } from "@/components/ui/button"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog"
import type { Run } from "./types"
const mapping: Record<string, string> = {
  dflash_enabled: "dflash_enabled",
  specprefill_enabled: "specprefill_enabled",
  mtp_enabled: "mtp_enabled",
  vlm_mtp_enabled: "vlm_mtp_enabled",
  enabled: "qwen35_ane_prefill_enabled",
  mlp_fraction: "qwen35_ane_prefill_fraction",
  shared_fraction: "qwen35_ane_prefill_shared_fraction",
  sequence_length: "qwen35_ane_prefill_sequence_length",
  gdn_enabled: "qwen35_ane_prefill_gdn",
  gdn_fraction: "qwen35_ane_prefill_gdn_fraction",
  cpu_enabled: "qwen35_ane_prefill_cpu_enabled",
  cpu_fraction: "qwen35_ane_prefill_cpu_fraction",
  cpu_down_fraction: "qwen35_ane_prefill_cpu_down_fraction",
  cpu_gdn_fraction: "qwen35_ane_prefill_cpu_gdn_fraction",
  cpu_threads: "qwen35_ane_prefill_cpu_threads",
  cpu_shared_resource: "qwen35_ane_prefill_cpu_shared_resource",
  fused_down: "qwen35_ane_prefill_fused_down",
  tail_padding_min_tokens: "qwen35_ane_prefill_tail_padding_min_tokens",
}
export function recommendedPatch(
  run: Run,
  results: unknown
): Record<string, unknown> {
  if (run.status !== "completed" || run.request.external) return {}
  if (run.kind === "context" && Array.isArray(results)) {
    const first: unknown = results[0]
    if (
      first &&
      typeof first === "object" &&
      "applied_tokens" in first &&
      typeof first.applied_tokens === "number" &&
      first.applied_tokens > 0
    )
      return { max_context_window: first.applied_tokens }
  }
  if (
    run.kind === "ane" &&
    run.recommendation &&
    typeof run.recommendation === "object"
  ) {
    const recommendation = run.recommendation as Record<string, unknown>
    if (
      recommendation.dflash_enabled !== false ||
      recommendation.specprefill_enabled !== false ||
      (recommendation.backend === "k2" &&
        (recommendation.mtp_enabled !== false ||
          recommendation.vlm_mtp_enabled !== false))
    )
      return {}
    return Object.fromEntries(
      Object.entries(recommendation)
        .filter(
          ([name, value]) =>
            name in mapping && value !== null && value !== undefined
        )
        .map(([name, value]) => [mapping[name], value])
    )
  }
  return {}
}
type ApplyResult = {
  requires_reload?: boolean
  reload_error?: string | null
  reload_deferred?: boolean
  auto_reloaded?: boolean
  auto_unloaded?: boolean
}
export function ApplyRecommendation({
  run,
  results,
}: {
  run: Run
  results: unknown
}) {
  const [open, setOpen] = useState(false)
  const client = useQueryClient()
  const patch = recommendedPatch(run, results)
  const model =
    typeof run.request.model_id === "string" ? run.request.model_id : undefined
  const apply = useMutation({
    mutationFn: () =>
      managementRequest<ApplyResult>(
        `models/${encodeURIComponent(model ?? "")}/settings`,
        { method: "PATCH", body: patch }
      ),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["molto"] })
    },
  })
  if (!model || !Object.keys(patch).length) {
    if (run.kind === "ane" && run.status === "completed" && run.recommendation)
      return (
        <p className="text-sm text-muted-foreground">
          This saved recommendation does not record all tested prerequisite
          settings. Run ANE tuning again before applying a recommendation.
        </p>
      )
    return null
  }
  return (
    <>
      <Button
        variant="outline"
        onClick={() => {
          apply.reset()
          setOpen(true)
        }}
      >
        Review recommendation
      </Button>
      <Dialog
        open={open}
        onOpenChange={(next) => {
          if (!apply.isPending) setOpen(next)
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Apply recommended settings?</DialogTitle>
            <DialogDescription>
              Save this patch to {model}. Existing model settings validation and
              safe reload rules apply. Active requests drain before unloading; a
              warmed engine may need loading again.
            </DialogDescription>
          </DialogHeader>
          <pre className="max-h-72 overflow-auto rounded bg-muted p-3 text-xs">
            {JSON.stringify(patch, null, 2)}
          </pre>
          {apply.isError && (
            <Alert variant="destructive">
              <AlertTitle>Recommendation could not be applied</AlertTitle>
              <AlertDescription>{errorMessage(apply.error)}</AlertDescription>
            </Alert>
          )}
          {apply.data && (
            <Alert
              variant={apply.data.reload_error ? "destructive" : "default"}
            >
              <AlertTitle>Settings saved</AlertTitle>
              <AlertDescription>
                {apply.data.reload_error
                  ? `Reload failed: ${apply.data.reload_error}`
                  : apply.data.reload_deferred
                    ? "Reload deferred until active requests finish. Load the model after it unloads."
                    : apply.data.auto_reloaded
                      ? "Model reloaded with the new settings."
                      : apply.data.auto_unloaded
                        ? "Model unloaded. The next request loads its new settings."
                        : apply.data.requires_reload
                          ? "The model requires reloading to use these settings."
                          : "The saved settings are ready for the next model load."}
              </AlertDescription>
            </Alert>
          )}
          <DialogFooter>
            <Button
              variant="outline"
              disabled={apply.isPending}
              onClick={() => setOpen(false)}
            >
              {apply.isSuccess ? "Close" : "Cancel"}
            </Button>
            <Button
              disabled={apply.isPending || apply.isSuccess}
              onClick={() => apply.mutate()}
            >
              {apply.isPending ? "Applying…" : "Apply recommendation"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
