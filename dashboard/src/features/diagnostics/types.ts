export type Kind = "throughput" | "accuracy" | "context" | "ane"
export type Run = {
  id: string
  kind: Kind
  status: string
  created_at: string | number
  updated_at: string | number
  request: { kind?: Kind; options?: Record<string, unknown> } & Record<
    string,
    unknown
  >
  progress?: Record<string, unknown> | number | null
  error?: string | null
  recommendation?: unknown
  results?: unknown
}
export type Capabilities = {
  kinds: { kind: Kind; targets: string[]; request_schema?: unknown }[]
  models: {
    model_id: string
    label?: string
    kinds?: Kind[]
    disabled_reason?: string | null
    ane_disabled_reason?: string | null
  }[]
  suites: { id: string; label?: string; description?: string }[]
}
export const labels: Record<Kind, string> = {
  throughput: "Throughput",
  accuracy: "Accuracy",
  context: "Context capacity",
  ane: "ANE tuning",
}
export const active = (run: Run) =>
  ["pending", "queued", "running", "cancelling", "cancel_requested"].includes(
    run.status
  )
export function resultRows(
  value: unknown,
  prefix = ""
): { metric: string; value: string; number?: number }[] {
  if (value === null || value === undefined) return []
  if (typeof value === "object")
    return Object.entries(value).flatMap(([key, child]) =>
      resultRows(child, prefix ? `${prefix} / ${key}` : key)
    )
  return [
    {
      metric: prefix.replaceAll("_", " "),
      value: typeof value === "string" ? value : JSON.stringify(value),
      number:
        typeof value === "number" && Number.isFinite(value) ? value : undefined,
    },
  ]
}
