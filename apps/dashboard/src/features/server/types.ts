export type Sections = Record<string, Record<string, unknown>>
export type ServerField = {
  key: string
  section: string
  label: string
  description: string
  type: string
  nullable?: boolean
  default: unknown
  choices?: (string | number | { value: string; label: string })[] | null
  minimum?: number | null
  maximum?: number | null
  secret?: boolean
  restart_required?: boolean
}
export type ServerSettings = {
  base_path: string
  sections: Sections
  fields: ServerField[]
  effective_model_dirs?: string[]
}
export type SettingsResult = {
  changed: string[]
  live_applied: string[]
  restart_required: string[]
}
export type SubKey = {
  id: string
  key: string
  name: string
  created_at: string
}
export type AuthKeys = {
  main_key: string
  sub_keys: SubKey[]
  [key: string]: unknown
}
export function labelFor(value: string) {
  return value
    .replaceAll("_", " ")
    .replace(/^./, (first) => first.toUpperCase())
}
export function editedPatch(edits: Sections, values: Sections): Sections {
  const patch: Sections = {}
  for (const [section, fields] of Object.entries(edits)) {
    for (const [key, value] of Object.entries(fields)) {
      if (JSON.stringify(value) !== JSON.stringify(values[section]?.[key])) {
        patch[section] ??= {}
        patch[section][key] = value
      }
    }
  }
  return patch
}
export function draftValue(
  field: ServerField,
  edits: Sections,
  values: Sections
) {
  return Object.hasOwn(edits[field.section] ?? {}, field.key)
    ? edits[field.section]?.[field.key]
    : values[field.section]?.[field.key]
}

export function resourceDraft(memory: Record<string, unknown>) {
  const rawTier = memory.memory_guard_tier
  const tier =
    typeof rawTier === "string" &&
    ["safe", "balanced", "aggressive", "custom"].includes(rawTier)
      ? (rawTier as "safe" | "balanced" | "aggressive" | "custom")
      : undefined
  const custom =
    typeof memory.memory_guard_custom_ceiling_gb === "number"
      ? memory.memory_guard_custom_ceiling_gb
      : undefined
  const valid =
    tier !== "custom" ||
    (custom !== undefined && Number.isFinite(custom) && custom > 0)
  const params = new URLSearchParams()
  if (typeof memory.prefill_memory_guard === "boolean")
    params.set("guard_enabled", String(memory.prefill_memory_guard))
  if (tier) params.set("tier", tier)
  if (tier === "custom" && valid && custom !== undefined)
    params.set("custom_ceiling_gb", String(custom))
  return {
    tier,
    custom,
    valid,
    path: `server/resources${params.size ? `?${params.toString()}` : ""}`,
  }
}
