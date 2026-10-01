import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { managementQuery } from "@/features/management/request"
import { QueryState } from "@/components/page-state"
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
import {
  Table,
  TableHeader,
  TableBody,
  TableHead,
  TableRow,
  TableCell,
} from "@/components/ui/table"
import { labelFor, resourceDraft } from "./types"

type Tier = "safe" | "balanced" | "aggressive" | "custom"
type Preview = {
  reserve_bytes: number
  free_bytes: number
  inactive_bytes: number
  other_apps_bytes: number
  static_bytes: number
  dynamic_bytes: number
  metal_cap_bytes: number
  ceiling_bytes: number
  binding: string
}
type Resources = {
  hardware: {
    physical_memory_bytes: number | null
    available_memory_bytes: number | null
    metal_cap_bytes: number | null
  }
  tier_previews: Partial<Record<Tier, Preview>>
  saved: {
    tier: Tier
    custom_ceiling_gb: number
    guard_enabled: boolean
    preview: Preview | null
  }
  runtime: {
    available: boolean
    tier: Tier | null
    custom_ceiling_gb: number | null
    guard_enabled: boolean | null
    ceiling_bytes: number | null
    breakdown: {
      static: number
      dynamic: number
      metal_cap: number
      hard_limit: number
    } | null
    wired_limit_request_bytes: number | null
  }
  draft: {
    tier: Tier
    custom_ceiling_gb: number
    guard_enabled: boolean
    preview: Preview | null
  }
  wired_limit: {
    limited: boolean | null
    recommended_bytes: number | null
    recommended_mib: number | null
    command: string | null
    copy_only: true
  }
  warnings: string[]
}
export function resourceBytes(value: number | null | undefined) {
  return value == null ? "Unavailable" : `${(value / 2 ** 30).toFixed(1)} GiB`
}
export function ResourcePanel({ memory }: { memory: Record<string, unknown> }) {
  const draft = resourceDraft(memory)
  const query = useQuery({
    ...managementQuery<Resources>(
      ["server", "resources", draft.path],
      draft.path
    ),
    enabled: draft.valid,
  })
  const [copyState, setCopyState] = useState("")
  const guard =
    typeof memory.prefill_memory_guard === "boolean"
      ? memory.prefill_memory_guard
      : query.data?.draft.guard_enabled
  if (!draft.valid)
    return (
      <Alert variant="destructive">
        <AlertTitle>Custom memory ceiling needs a value</AlertTitle>
        <AlertDescription>
          Enter a positive custom ceiling in GiB to preview the effective limit.
          Nothing has been saved.
        </AlertDescription>
      </Alert>
    )
  return (
    <QueryState query={query}>
      {query.data && (
        <Card>
          <CardHeader>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <CardTitle>Memory capacity & guard</CardTitle>
              <Badge variant="outline">Read-only preview</Badge>
            </div>
            <CardDescription>
              Compare your saved settings, running process, and current draft.
              Metal limits and other apps can reduce the usable ceiling.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <dl className="grid gap-4 sm:grid-cols-3">
              {[
                {
                  label: "Physical memory",
                  value: query.data.hardware.physical_memory_bytes,
                },
                {
                  label: "Available memory",
                  value: query.data.hardware.available_memory_bytes,
                },
                {
                  label: "Metal memory cap",
                  value: query.data.hardware.metal_cap_bytes,
                },
              ].map((item) => (
                <div key={item.label} className="rounded-lg bg-muted/40 p-3">
                  <dt className="text-xs text-muted-foreground">
                    {item.label}
                  </dt>
                  <dd className="mt-1 text-lg font-semibold">
                    {resourceBytes(item.value)}
                  </dd>
                </div>
              ))}
            </dl>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Configuration</TableHead>
                  <TableHead>Guard</TableHead>
                  <TableHead>Tier</TableHead>
                  <TableHead>Effective ceiling</TableHead>
                  <TableHead>Limited by</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                <TableRow>
                  <TableCell>Saved on disk</TableCell>
                  <TableCell>
                    {query.data.saved.guard_enabled ? "Enabled" : "Disabled"}
                  </TableCell>
                  <TableCell>{labelFor(query.data.saved.tier)}</TableCell>
                  <TableCell>
                    {query.data.saved.guard_enabled
                      ? resourceBytes(query.data.saved.preview?.ceiling_bytes)
                      : "Disabled"}
                  </TableCell>
                  <TableCell>
                    {query.data.saved.preview
                      ? labelFor(query.data.saved.preview.binding)
                      : "Unavailable"}
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Running process</TableCell>
                  <TableCell>
                    {query.data.runtime.guard_enabled == null
                      ? "Unavailable"
                      : query.data.runtime.guard_enabled
                        ? "Enabled"
                        : "Disabled"}
                  </TableCell>
                  <TableCell>
                    {query.data.runtime.tier
                      ? labelFor(query.data.runtime.tier)
                      : "Unavailable"}
                  </TableCell>
                  <TableCell>
                    {query.data.runtime.guard_enabled === false
                      ? "Disabled"
                      : resourceBytes(query.data.runtime.ceiling_bytes)}
                  </TableCell>
                  <TableCell>
                    {query.data.runtime.available
                      ? "Active enforcer"
                      : "Enforcer unavailable"}
                  </TableCell>
                </TableRow>
                <TableRow className="bg-muted/30">
                  <TableCell>Current draft</TableCell>
                  <TableCell>{guard ? "Enabled" : "Disabled"}</TableCell>
                  <TableCell>{labelFor(query.data.draft.tier)}</TableCell>
                  <TableCell>
                    {guard
                      ? resourceBytes(query.data.draft.preview?.ceiling_bytes)
                      : "Disabled"}
                  </TableCell>
                  <TableCell>
                    {query.data.draft.preview
                      ? labelFor(query.data.draft.preview.binding)
                      : "Unavailable"}
                  </TableCell>
                </TableRow>
              </TableBody>
            </Table>
            <p className="text-xs text-muted-foreground">
              Changing a field updates this preview only. Save your memory
              settings and restart oMLX to apply them. Available memory can
              change as other apps allocate memory.
            </p>
            {!guard && (
              <Alert>
                <AlertTitle>Memory guard is disabled in this draft</AlertTitle>
                <AlertDescription>
                  The preview does not enforce a process ceiling while the guard
                  is off.
                </AlertDescription>
              </Alert>
            )}
            {query.data.warnings.length > 0 && (
              <Alert>
                <AlertTitle>Capacity considerations</AlertTitle>
                <AlertDescription>
                  <ul className="list-inside list-disc space-y-1">
                    {query.data.warnings.map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                </AlertDescription>
              </Alert>
            )}
            <details className="rounded-lg border p-3">
              <summary className="cursor-pointer text-sm font-medium">
                Compare memory tiers
              </summary>
              <p className="mt-3 text-xs text-muted-foreground">
                The static limit reserves memory for macOS. The dynamic limit
                uses available memory or your custom ceiling. The effective
                ceiling uses the lowest applicable limit.
              </p>
              <Table className="mt-3">
                <TableHeader>
                  <TableRow>
                    <TableHead>Tier</TableHead>
                    <TableHead>Reserve</TableHead>
                    <TableHead>Static limit</TableHead>
                    <TableHead>Dynamic limit</TableHead>
                    <TableHead>Effective ceiling</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {Object.entries(query.data.tier_previews).map(
                    ([tier, preview]) =>
                      preview && (
                        <TableRow key={tier}>
                          <TableCell>{labelFor(tier)}</TableCell>
                          <TableCell>
                            {resourceBytes(preview.reserve_bytes)}
                          </TableCell>
                          <TableCell>
                            {resourceBytes(preview.static_bytes)}
                          </TableCell>
                          <TableCell>
                            {resourceBytes(preview.dynamic_bytes)}
                          </TableCell>
                          <TableCell>
                            {resourceBytes(preview.ceiling_bytes)}
                          </TableCell>
                        </TableRow>
                      )
                  )}
                </TableBody>
              </Table>
            </details>
            {query.data.wired_limit.limited === true && (
              <div className="space-y-3 rounded-lg border p-4">
                <div>
                  <p className="text-sm font-medium">
                    macOS Metal limit constrains this process
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    The running process requests{" "}
                    {resourceBytes(
                      query.data.runtime.wired_limit_request_bytes
                    )}
                    , above the current Metal cap.{" "}
                    {query.data.wired_limit.recommended_bytes != null
                      ? `The server recommends ${resourceBytes(query.data.wired_limit.recommended_bytes)}, with space reserved for macOS.`
                      : "No safe increase is currently available."}
                  </p>
                </div>
                {query.data.wired_limit.command && (
                  <>
                    <pre className="overflow-x-auto rounded-md bg-muted p-3 text-xs break-all whitespace-pre-wrap">
                      {query.data.wired_limit.command}
                    </pre>
                    <div className="flex flex-wrap items-center gap-3">
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          const command = query.data?.wired_limit.command
                          if (command)
                            void navigator.clipboard.writeText(command).then(
                              () => setCopyState("Command copied"),
                              () =>
                                setCopyState(
                                  "Clipboard unavailable. Select the command to copy it."
                                )
                            )
                        }}
                      >
                        Copy OS command
                      </Button>
                      <span className="text-xs text-muted-foreground">
                        Copy only. Review and run it yourself in Terminal.
                      </span>
                    </div>
                    {copyState && (
                      <p
                        role="status"
                        className="text-xs text-muted-foreground"
                      >
                        {copyState}
                      </p>
                    )}
                  </>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </QueryState>
  )
}
