import { useId, useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { errorMessage } from "@/features/management/api"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import {
  Field,
  FieldGroup,
  FieldLabel,
  FieldDescription,
} from "@/components/ui/field"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
import { Spinner } from "@/components/ui/spinner"
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
  SelectGroup,
} from "@/components/ui/select"

export type SettingField = {
  key: string
  label: string
  kind?: "text" | "boolean" | "select" | "object" | "list" | "multiline"
  group?: string
  advanced?: boolean
  disabled?: boolean
  reloadRequired?: boolean
  defaultValue?: unknown
  min?: number
  max?: number
  step?: number | "any"
  options?: { value: string; label: string; typedValue?: unknown }[]
  hint?: string
}
export const samplingFields: SettingField[] = [
  { key: "max_context_window", label: "Context limit", min: 1, step: 1 },
  { key: "max_tokens", label: "Maximum output tokens", min: 1, step: 1 },
  { key: "temperature", label: "Temperature", min: 0, step: "any" },
  { key: "top_p", label: "Top P", min: 0, max: 1, step: "any" },
  { key: "top_k", label: "Top K", min: 0, step: 1 },
  {
    key: "repetition_penalty",
    label: "Repetition penalty",
    min: 0.01,
    step: "any",
  },
]
export function SettingsForm<T>({
  values,
  fields,
  save,
  describe,
  saved,
  alwaysSave = false,
  submitLabel = "Save changes",
}: {
  values: Record<string, unknown>
  fields: SettingField[]
  save: (patch: Record<string, unknown>) => Promise<T>
  describe: (result: T) => string
  saved: () => Promise<unknown>
  alwaysSave?: boolean
  submitLabel?: string
}) {
  const formId = useId()
  const [edits, setEdits] = useState<Record<string, unknown>>({})
  const [notice, setNotice] = useState("")
  const dirty = Object.keys(edits).length > 0 || alwaysSave
  const mutation = useMutation({
    mutationFn: () => save(edits),
    onSuccess: async (result) => {
      setNotice(describe(result))
      setEdits({})
      await saved()
    },
  })
  const change = (key: string, value: unknown) => {
    setEdits((current) => ({ ...current, [key]: value }))
    setNotice("")
  }
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault()
        if (dirty) mutation.mutate()
      }}
      className="flex flex-col gap-6"
    >
      <FieldGroup>
        {Array.from(
          new Set(
            fields.map(
              (field) =>
                `${field.group ?? "Settings"}${field.advanced ? " · Advanced" : ""}`
            )
          )
        ).map((group) => (
          <details
            key={group}
            open={!group.endsWith(" · Advanced")}
            className="space-y-5"
          >
            <summary className="cursor-pointer font-medium">{group}</summary>
            <div className="grid gap-5 sm:grid-cols-2">
              {fields
                .filter(
                  (field) =>
                    `${field.group ?? "Settings"}${field.advanced ? " · Advanced" : ""}` ===
                    group
                )
                .map((field) => {
                  const value =
                    field.key in edits ? edits[field.key] : values[field.key]
                  const id = `${formId}-setting-${field.key}`
                  return (
                    <Field
                      key={field.key}
                      className={
                        field.kind === "object" || field.kind === "multiline"
                          ? "sm:col-span-2"
                          : undefined
                      }
                    >
                      <div className="flex items-center justify-between gap-2">
                        <FieldLabel htmlFor={id}>{field.label}</FieldLabel>
                        <Button
                          type="button"
                          size="sm"
                          variant="ghost"
                          className="h-6 px-1 text-xs"
                          aria-label={`Reset ${field.label} to default`}
                          disabled={mutation.isPending}
                          onClick={() => change(field.key, null)}
                        >
                          Reset
                        </Button>
                      </div>
                      {field.kind === "object" || field.kind === "list" ? (
                        <StructuredValue
                          value={value ?? (field.kind === "list" ? [] : {})}
                          change={(next) => change(field.key, next)}
                          disabled={mutation.isPending || field.disabled}
                          label={field.label}
                        />
                      ) : field.kind === "multiline" ? (
                        <textarea
                          id={id}
                          className="min-h-28 w-full rounded-md border p-2 font-mono text-sm"
                          value={
                            typeof value === "string" ||
                            typeof value === "number" ||
                            typeof value === "boolean"
                              ? String(value)
                              : ""
                          }
                          disabled={mutation.isPending || field.disabled}
                          placeholder="Default"
                          onChange={(event) =>
                            change(field.key, event.target.value || null)
                          }
                        />
                      ) : field.kind === "boolean" ? (
                        <div className="flex items-center gap-2">
                          <Switch
                            id={id}
                            checked={value === true}
                            disabled={mutation.isPending || field.disabled}
                            onCheckedChange={(checked) =>
                              change(field.key, checked)
                            }
                          />
                          <span className="text-xs text-muted-foreground">
                            {value == null
                              ? "Default"
                              : value === true
                                ? "Enabled"
                                : "Disabled"}
                          </span>
                        </div>
                      ) : field.kind === "select" ? (
                        <Select
                          value={
                            typeof value === "string" ||
                            typeof value === "number" ||
                            typeof value === "boolean"
                              ? String(value)
                              : ""
                          }
                          onValueChange={(next) =>
                            change(
                              field.key,
                              field.options?.find(
                                (option) => option.value === next
                              )?.typedValue ?? next
                            )
                          }
                        >
                          <SelectTrigger
                            id={id}
                            disabled={mutation.isPending || field.disabled}
                          >
                            <SelectValue placeholder="Default" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectGroup>
                              {field.options?.map((option) => (
                                <SelectItem
                                  key={option.value}
                                  value={option.value}
                                >
                                  {option.label}
                                </SelectItem>
                              ))}
                            </SelectGroup>
                          </SelectContent>
                        </Select>
                      ) : (
                        <Input
                          id={id}
                          type={field.kind === "text" ? "text" : "number"}
                          min={field.min}
                          max={field.max}
                          step={field.step ?? "any"}
                          value={
                            typeof value === "number" ||
                            typeof value === "string"
                              ? value
                              : ""
                          }
                          placeholder="Default"
                          disabled={mutation.isPending || field.disabled}
                          onChange={(event) =>
                            change(
                              field.key,
                              event.target.value === ""
                                ? null
                                : field.kind === "text"
                                  ? event.target.value
                                  : Number(event.target.value)
                            )
                          }
                        />
                      )}
                      {field.hint && (
                        <FieldDescription>{field.hint}</FieldDescription>
                      )}
                      {field.reloadRequired && (
                        <FieldDescription>
                          Changing this field requires the model engine to
                          reload.
                        </FieldDescription>
                      )}
                      {value == null && (
                        <FieldDescription>
                          Server default
                          {field.defaultValue !== undefined
                            ? `: ${JSON.stringify(field.defaultValue)}`
                            : ""}
                        </FieldDescription>
                      )}
                    </Field>
                  )
                })}
            </div>
          </details>
        ))}
      </FieldGroup>
      {mutation.isError && (
        <Alert variant="destructive">
          <AlertTitle>Changes were not confirmed</AlertTitle>
          <AlertDescription>{errorMessage(mutation.error)}</AlertDescription>
        </Alert>
      )}
      {notice && (
        <Alert>
          <AlertTitle>Configuration update</AlertTitle>
          <AlertDescription>{notice}</AlertDescription>
        </Alert>
      )}
      <div className="flex flex-wrap items-center gap-3">
        <Button type="submit" disabled={!dirty || mutation.isPending}>
          {mutation.isPending && <Spinner />}
          {submitLabel}
        </Button>
        <Button
          type="button"
          variant="outline"
          disabled={!dirty || mutation.isPending}
          onClick={() => {
            setEdits({})
            mutation.reset()
            setNotice("")
          }}
        >
          Discard changes
        </Button>
        {dirty && (
          <span role="status" className="text-sm text-muted-foreground">
            Unsaved changes
          </span>
        )}
      </div>
      <p className="text-xs text-muted-foreground">
        Only edited fields are sent. Reset restores the server default for that
        field. Background refreshes preserve your edits.
      </p>
    </form>
  )
}

function StructuredValue({
  value,
  change,
  disabled,
  label,
}: {
  value: unknown
  change: (value: unknown) => void
  disabled?: boolean
  label: string
}) {
  const entries = Array.isArray(value)
    ? value.map((item, index) => [String(index), item] as const)
    : value && typeof value === "object"
      ? Object.entries(value)
      : []
  const list = Array.isArray(value)
  const update = (key: string, next: unknown) => {
    if (list) {
      const copy = [...value]
      copy[Number(key)] = next
      change(copy)
    } else change({ ...Object.fromEntries(entries), [key]: next })
  }
  return (
    <div className="space-y-3 rounded-md border p-3">
      {entries.map(([key, item]) => (
        <div key={key} className="flex flex-wrap items-start gap-2">
          {list ? (
            <span className="pt-2 text-sm">{Number(key) + 1}</span>
          ) : (
            <Input
              aria-label={`${label} key ${key}`}
              className="w-40"
              defaultValue={key}
              disabled={disabled}
              onBlur={(event) => {
                const next = event.target.value.trim()
                if (next === key) return
                if (!next || entries.some(([other]) => other === next)) {
                  event.target.value = key
                  return
                }
                change(
                  Object.fromEntries(
                    entries.map(([other, val]) => [
                      other === key ? next : other,
                      val,
                    ])
                  )
                )
              }}
            />
          )}
          <Select
            value={
              Array.isArray(item)
                ? "list"
                : item === null
                  ? "null"
                  : typeof item
            }
            onValueChange={(kind) =>
              update(
                key,
                kind === "boolean"
                  ? false
                  : kind === "number"
                    ? 0
                    : kind === "object"
                      ? {}
                      : kind === "list"
                        ? []
                        : kind === "null"
                          ? null
                          : ""
              )
            }
          >
            <SelectTrigger
              disabled={disabled}
              aria-label={`${label} ${key} value type`}
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                {["string", "number", "boolean", "object", "list", "null"].map(
                  (kind) => (
                    <SelectItem key={kind} value={kind}>
                      {kind}
                    </SelectItem>
                  )
                )}
              </SelectGroup>
            </SelectContent>
          </Select>
          {item !== null && typeof item === "object" ? (
            <div className="w-full">
              <StructuredValue
                value={item}
                change={(next) => update(key, next)}
                disabled={disabled}
                label={`${label} ${key}`}
              />
            </div>
          ) : typeof item === "boolean" ? (
            <Switch
              aria-label={`${label} ${key}`}
              checked={item}
              disabled={disabled}
              onCheckedChange={(next) => update(key, next)}
            />
          ) : item !== null ? (
            <Input
              aria-label={`${label} ${key} value`}
              className="flex-1"
              type={typeof item === "number" ? "number" : "text"}
              value={
                typeof item === "string" || typeof item === "number" ? item : ""
              }
              disabled={disabled}
              onChange={(event) =>
                update(
                  key,
                  typeof item === "number"
                    ? Number(event.target.value)
                    : event.target.value
                )
              }
            />
          ) : (
            <span>Null</span>
          )}
          <Button
            type="button"
            variant="ghost"
            disabled={disabled}
            onClick={() =>
              change(
                list
                  ? value.filter((_, index) => index !== Number(key))
                  : Object.fromEntries(
                      entries.filter(([other]) => other !== key)
                    )
              )
            }
          >
            Remove
          </Button>
        </div>
      ))}
      <Button
        type="button"
        variant="outline"
        size="sm"
        disabled={disabled}
        onClick={() => {
          if (list) change([...value, ""])
          else {
            let index = 1
            while (entries.some(([key]) => key === `key_${index}`)) index++
            change({ ...Object.fromEntries(entries), [`key_${index}`]: "" })
          }
        }}
      >
        Add {list ? "item" : "argument"}
      </Button>
    </div>
  )
}
