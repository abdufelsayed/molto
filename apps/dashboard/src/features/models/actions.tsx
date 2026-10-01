import { useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { toast } from "sonner"
import type { Model } from "@/features/management/api"
import { errorMessage } from "@/features/management/api"
import { managementKey, useManagement } from "@/features/management/queries"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
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

export function ModelActions({
  model,
  disabled = false,
}: {
  model: Model
  disabled?: boolean
}) {
  const { api, queryClient } = useManagement()
  const [confirm, setConfirm] = useState(false)
  const mutation = useMutation({
    mutationFn: (operation: "load" | "unload") =>
      operation === "load" ? api.load(model.id) : api.unload(model.id),
    onSuccess: (result) => {
      toast.success(
        result.status === "unloading"
          ? "Unload requested; waiting for active requests to finish."
          : (result.message ?? "Model state updated.")
      )
      setConfirm(false)
    },
    onSettled: () => queryClient.invalidateQueries({ queryKey: managementKey }),
  })
  const busy =
    disabled || model.is_loading || model.is_unloading || mutation.isPending
  return (
    <div className="flex flex-col gap-3">
      <div className="flex gap-2">
        {!model.loaded ? (
          <Button
            size="sm"
            disabled={busy}
            onClick={() => mutation.mutate("load")}
          >
            {mutation.isPending && <Spinner />}Load model
          </Button>
        ) : (
          <Button
            size="sm"
            variant="outline"
            disabled={busy}
            onClick={() => setConfirm(true)}
          >
            Unload model
          </Button>
        )}
        {(model.is_loading || model.is_unloading) && (
          <span
            role="status"
            className="flex items-center gap-2 text-sm text-muted-foreground"
          >
            <Spinner />
            {model.is_loading ? "Loading" : "Unloading"}
          </span>
        )}
      </div>
      {mutation.isError && (
        <Alert variant="destructive">
          <AlertTitle>Model operation needs attention</AlertTitle>
          <AlertDescription>{errorMessage(mutation.error)}</AlertDescription>
        </Alert>
      )}
      <AlertDialog
        open={confirm}
        onOpenChange={(open) => {
          if (!mutation.isPending) setConfirm(open)
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Unload {model.id}?</AlertDialogTitle>
            <AlertDialogDescription>
              Molto will release the model after active work finishes. A later
              inference request can load it again.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={mutation.isPending}>
              Cancel
            </AlertDialogCancel>
            <Button
              variant="destructive"
              disabled={mutation.isPending}
              onClick={() => mutation.mutate("unload")}
            >
              {mutation.isPending && <Spinner />}Confirm unload
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
