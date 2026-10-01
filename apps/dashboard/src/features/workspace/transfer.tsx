import { useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { managementRequest } from "@/features/management/request"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Field, FieldLabel } from "@/components/ui/field"
import { Failure, Notes } from "./primitives"
import type { ImportPreview } from "./types"
import { configurationDiff, type ConfigurationBundle } from "./transfer-diff"
export function TransferPanel() {
  const client = useQueryClient()
  const [bundle, setBundle] = useState<unknown>(null)
  const [fileError, setFileError] = useState<Error | null>(null)
  const preview = useMutation({
    mutationFn: async (value: unknown) => {
      const result = await managementRequest<{
        dry_run: boolean
        plan: {
          changes?: NonNullable<ImportPreview["changes"]>
          blockers: { model_id: string | null; code: string; reason: string }[]
          can_apply: boolean
          affected_model_ids?: string[]
          matched_models: string[]
          missing_models: string[]
          settings_updates: number
          profile_updates: number
          collection_updates: number
        }
      }>("workspace/import", {
        method: "POST",
        body: { bundle: value, dry_run: true },
      })
      const current =
        await managementRequest<ConfigurationBundle>("workspace/export")
      return {
        valid: true,
        errors: [],
        blockers: result.plan.blockers.map(
          (blocker) =>
            `${blocker.model_id ? `${blocker.model_id}: ` : ""}${blocker.reason}`
        ),
        can_apply: result.plan.can_apply,
        affected_model_ids: result.plan.affected_model_ids,
        changes:
          result.plan.changes ??
          configurationDiff(current, value as ConfigurationBundle),
        plan: result.plan,
      } as ImportPreview
    },
  })
  const apply = useMutation({
    mutationFn: () =>
      managementRequest<unknown>("workspace/import", {
        method: "POST",
        body: {
          bundle,
          dry_run: false,
        },
      }),
    onSuccess: () => {
      setBundle(null)
      preview.reset()
      void client.invalidateQueries({ queryKey: ["molto"] })
    },
  })
  const exporting = useMutation({
    mutationFn: () => managementRequest<unknown>("workspace/export"),
    onSuccess: (data) => {
      const url = URL.createObjectURL(
        new Blob([JSON.stringify(data, null, 2)], { type: "application/json" })
      )
      const a = document.createElement("a")
      a.href = url
      a.download = "molto-workspace.json"
      a.click()
      URL.revokeObjectURL(url)
    },
  })
  async function read(file?: File) {
    setBundle(null)
    setFileError(null)
    preview.reset()
    apply.reset()
    if (!file) return
    try {
      const parsed: unknown = JSON.parse(await file.text())
      setBundle(parsed)
      preview.mutate(parsed)
    } catch (error) {
      setFileError(
        error instanceof Error
          ? error
          : new Error("Cannot read configuration file.")
      )
    }
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle>Configuration transfer</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          Export settings and collections. Import reviews configuration changes
          before applying them.
        </p>
        <Button
          variant="outline"
          disabled={exporting.isPending}
          onClick={() => exporting.mutate()}
        >
          Download configuration
        </Button>
        <Field>
          <FieldLabel htmlFor="workspace-import">
            Import configuration file
          </FieldLabel>
          <Input
            id="workspace-import"
            type="file"
            accept="application/json,.json"
            disabled={preview.isPending || apply.isPending}
            onChange={(e) => void read(e.target.files?.[0])}
          />
        </Field>
        <Failure error={fileError} />
        <Failure error={preview.error} />
        <Failure error={apply.error} />
        <Failure error={exporting.error} />
        {bundle !== null && (
          <Button
            variant="outline"
            disabled={preview.isPending || apply.isPending}
            onClick={() => {
              apply.reset()
              preview.mutate(bundle)
            }}
          >
            Refresh preview
          </Button>
        )}
        {preview.isPending && <p role="status">Validating configuration…</p>}
        {preview.data && (
          <div className="space-y-3">
            <Notes items={preview.data.errors} />
            {!!preview.data.affected_model_ids?.length && (
              <p className="text-sm">
                Affected models: {preview.data.affected_model_ids.join(", ")}
              </p>
            )}
            {!!preview.data.blockers?.length && (
              <div
                role="alert"
                className="space-y-2 rounded-lg border border-destructive p-3"
              >
                <p className="text-sm font-medium">Import is blocked</p>
                <Notes items={preview.data.blockers} />
                <p className="text-sm">
                  Unload affected models and wait for active work to finish,
                  then refresh this preview. The uploaded configuration stays
                  staged.
                </p>
              </div>
            )}
            {preview.data.changes?.length ? (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead>
                    <tr>
                      <th>Model / field</th>
                      <th>Current</th>
                      <th>Imported</th>
                    </tr>
                  </thead>
                  <tbody>
                    {preview.data.changes.map((change, i) => (
                      <tr key={i}>
                        <td className="p-2">
                          {change.model_id} {change.field}
                        </td>
                        <td className="p-2 break-all">
                          {JSON.stringify(change.before) ?? "Unset"}
                        </td>
                        <td className="p-2 break-all">
                          {JSON.stringify(change.after) ?? "Unset"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-sm">No field changes reported.</p>
            )}
            <Button
              disabled={
                apply.isPending ||
                apply.isError ||
                preview.isPending ||
                preview.isError ||
                !!preview.data.blockers?.length ||
                preview.data.can_apply !== true ||
                !bundle ||
                preview.data.valid !== true ||
                !!preview.data.errors?.length
              }
              onClick={() => apply.mutate()}
            >
              Apply reviewed configuration
            </Button>
          </div>
        )}
        {apply.isSuccess && <p role="status">Configuration applied.</p>}
      </CardContent>
    </Card>
  )
}
