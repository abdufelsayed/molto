import { useMutation, useQuery } from "@tanstack/react-query"
import { useManagement, managementKey } from "@/features/management/queries"
import { managementRequest } from "@/features/management/request"
import { errorMessage } from "@/features/management/api"
import { modelOptionsQuery } from "./model-options"
import type { ModelOptions } from "./model-options"
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
import { Spinner } from "@/components/ui/spinner"
type ImportResult = {
  status: "ok"
  model_id: string
  merge_mode: string
  mtp_tensors: number
  options: ModelOptions
}
export function ModelMtplxImport({
  modelId,
  loaded,
  busy,
}: {
  modelId: string
  loaded: boolean
  busy: boolean
}) {
  const { queryClient } = useManagement()
  const query = modelOptionsQuery(modelId)
  const options = useQuery(query)
  const mutation = useMutation({
    mutationFn: () =>
      managementRequest<ImportResult>(
        `models/${encodeURIComponent(modelId)}/import-mtplx`,
        { method: "POST" }
      ),
    onSuccess: async (result) => {
      queryClient.setQueryData(query.queryKey, result.options)
      await queryClient.invalidateQueries({ queryKey: managementKey })
    },
  })
  const capability = options.data?.capabilities?.mtplx_import
  const reason = options.data?.mtplx_import_reason
  const mtpReason = options.data?.fields.find(
    (field) => field.key === "mtp_enabled"
  )?.unsupported_reason
  if (
    !options.data?.mtplx_sidecar_detected &&
    capability !== true &&
    !mutation.data
  )
    return null
  return (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle>MTPLX sidecar import</CardTitle>
        <CardDescription>
          {reason ||
            mtpReason ||
            "Import detected MTPLX sidecar weights into this checkpoint so native multi-token prediction can use them."}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          This changes checkpoint files on disk. The model must be unloaded and
          idle; importing does not enable MTP or load the model.
        </p>
        {(loaded || busy) && (
          <Alert>
            <AlertTitle>
              {busy ? "Model operation in progress" : "Unload the model first"}
            </AlertTitle>
            <AlertDescription>
              {busy
                ? "Wait until loading, unloading, or preparation finishes before importing."
                : "Use Unload model above, wait for it to finish, then import the sidecar."}
            </AlertDescription>
          </Alert>
        )}
        <Button
          variant="outline"
          disabled={capability !== true || loaded || busy || mutation.isPending}
          onClick={() => {
            if (
              window.confirm(
                "Import MTPLX sidecar weights? This modifies this model's checkpoint files on disk. The model must be unloaded and idle. Review the refreshed MTP capabilities after import before enabling MTP."
              )
            )
              mutation.mutate()
          }}
        >
          {mutation.isPending && <Spinner />}
          {mutation.isPending
            ? "Importing MTPLX sidecar"
            : "Import MTPLX sidecar"}
        </Button>
        {mutation.isError && (
          <Alert variant="destructive">
            <AlertTitle>Sidecar import was not confirmed</AlertTitle>
            <AlertDescription>
              {errorMessage(mutation.error)} Refresh the model to check its
              current compatibility before retrying.
            </AlertDescription>
          </Alert>
        )}
        {mutation.data && (
          <Alert>
            <AlertTitle>MTPLX sidecar imported</AlertTitle>
            <AlertDescription>
              Imported {mutation.data.mtp_tensors} MTP tensors using{" "}
              {mutation.data.merge_mode}. Model capabilities have been
              refreshed. Review MTP settings before loading the model.
            </AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  )
}
