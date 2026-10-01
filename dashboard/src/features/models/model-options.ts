import type { SettingField } from "@/features/settings/form"
import { managementQuery } from "@/features/management/request"
export type ModelOptionField = {
  key: string
  label: string
  type: string
  default: unknown
  minimum?: number | null
  maximum?: number | null
  choices?: unknown[] | null
  profile: boolean
  template: boolean
  description: string
  group?: string
  advanced?: boolean
  supported?: boolean
  unsupported_reason?: string | null
  reload_required?: boolean
}
export type ModelOptions = {
  fields: ModelOptionField[]
  defaults: Record<string, unknown>
  profile_fields: string[]
  template_fields: string[]
  family?: string
  capabilities?: Record<string, boolean>
  active_profile?: string | null
  mtplx_sidecar_detected?: boolean
  mtplx_import_reason?: string | null
  profile_drift?: boolean
}
export const modelOptionsQuery = (id?: string) =>
  managementQuery<ModelOptions>(
    ["model-options", id],
    id ? `models/${encodeURIComponent(id)}/options` : "model-options"
  )
export function optionFields(
  options: ModelOptions,
  eligibility?: "profile" | "template"
): SettingField[] {
  return options.fields
    .filter((field) => !eligibility || field[eligibility])
    .map((field) => ({
      key: field.key,
      label:
        (
          {
            model_alias: "API alias",
            ttl_seconds: "Idle timeout in seconds",
            is_pinned: "Keep model pinned",
            is_hidden: "Hide from inference model list",
            is_default: "Use as default model",
            top_p: "Top P",
            top_k: "Top K",
            min_p: "Min P",
          } as Record<string, string>
        )[field.key] ?? field.label,
      kind:
        field.type === "boolean"
          ? "boolean"
          : field.type === "object" || field.type === "dict"
            ? "object"
            : field.type === "array" || field.type === "list"
              ? "list"
              : field.choices?.length
                ? "select"
                : field.key === "guided_grammar" || field.key === "description"
                  ? "multiline"
                  : field.type === "string"
                    ? "text"
                    : undefined,
      min: field.minimum ?? undefined,
      max: field.maximum ?? undefined,
      step: field.type === "integer" ? 1 : "any",
      options: field.choices?.map((value) => ({
        value: String(value),
        label: String(value),
        typedValue: value,
      })),
      hint: [field.description, field.unsupported_reason]
        .filter(Boolean)
        .join(" "),
      disabled: field.supported === false,
      group:
        field.group && !["general", "advanced"].includes(field.group)
          ? field.group
          : field.key.includes("thinking") || field.key === "reasoning_parser"
            ? "Reasoning"
            : /ane|dflash|mtp|offload|specprefill|turboquant|index_cache|oq_a8|deepseek|qwen4/.test(
                  field.key
                )
              ? "Acceleration and family settings"
              : "Common settings",
      advanced: field.advanced,
      reloadRequired: field.reload_required,
      defaultValue: field.default,
    }))
}
