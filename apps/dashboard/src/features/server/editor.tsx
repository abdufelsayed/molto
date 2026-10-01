import { useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { managementRequest } from "@/features/management/request"
import { errorMessage } from "@/features/management/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Switch } from "@/components/ui/switch"
import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card"
import {
  Field,
  FieldLabel,
  FieldDescription,
  FieldGroup,
} from "@/components/ui/field"
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectGroup,
  SelectItem,
} from "@/components/ui/select"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
import { draftValue, editedPatch, labelFor } from "./types"
import { ResourcePanel } from "./resources"
import type {
  ServerField,
  ServerSettings,
  Sections,
  SettingsResult,
} from "./types"

export const settingsGroups = [
  {
    id: "network",
    title: "Network",
    description: "Listening address, access, and connection behavior.",
    sections: ["server", "network", "cors", "ssl"],
  },
  {
    id: "models",
    title: "Models & resources",
    description: "Model directories, memory budgets, and engine scheduling.",
    sections: [
      "model",
      "models",
      "memory",
      "resources",
      "scheduler",
      "process",
    ],
  },
  {
    id: "cache",
    title: "Cache",
    description: "Prefix reuse, disk storage, and cache limits.",
    sections: ["cache", "ssd_cache", "paged_cache"],
  },
  {
    id: "defaults",
    title: "Defaults",
    description: "Generation and inference defaults shared by your models.",
    sections: ["sampling", "generation", "embedding", "default", "defaults"],
  },
  {
    id: "integrations",
    title: "Integrations & tools",
    description: "External services, tool parsing, and web search.",
    sections: [
      "integrations",
      "integration",
      "tools",
      "tool",
      "web_search",
      "huggingface",
      "modelscope",
      "claude_code",
      "mcp",
    ],
  },
  {
    id: "logging",
    title: "Logging",
    description: "Request history, log files, and diagnostics.",
    sections: ["logging", "log", "usage", "monitoring"],
  },
] as const

export function groupFor(section: string) {
  return (
    settingsGroups.find((group) =>
      group.sections.some(
        (name) => section === name || section.startsWith(`${name}_`)
      )
    )?.id ?? "models"
  )
}

function display(value: unknown, secret = false) {
  if (secret && value) return "••••••••"
  if (value === null || value === undefined) return "Automatic"
  if (Array.isArray(value)) return value.join(", ") || "None"
  if (typeof value === "object") return "Configured"
  return typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
    ? String(value)
    : "Automatic"
}

function SettingControl({
  field,
  value,
  change,
  disabled,
}: {
  field: ServerField
  value: unknown
  change: (value: unknown) => void
  disabled: boolean
}) {
  const id = `server-${field.section}-${field.key}`
  const choices = field.choices?.map((choice) =>
    typeof choice === "object"
      ? choice
      : { value: String(choice), label: String(choice) }
  )
  if (field.type === "boolean" && field.nullable) {
    return (
      <Select
        items={[
          { value: "automatic", label: "Automatic" },
          { value: "enabled", label: "Enabled" },
          { value: "disabled", label: "Disabled" },
        ]}
        value={
          value == null ? "automatic" : value === true ? "enabled" : "disabled"
        }
        onValueChange={(next) =>
          change(next === "automatic" ? null : next === "enabled")
        }
      >
        <SelectTrigger id={id} className="w-full" disabled={disabled}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            <SelectItem value="automatic">Automatic</SelectItem>
            <SelectItem value="enabled">Enabled</SelectItem>
            <SelectItem value="disabled">Disabled</SelectItem>
          </SelectGroup>
        </SelectContent>
      </Select>
    )
  }
  if (field.type === "boolean" || typeof field.default === "boolean") {
    return (
      <div className="flex items-center gap-3">
        <Switch
          id={id}
          checked={value === true}
          onCheckedChange={change}
          disabled={disabled}
        />
        <span className="text-sm text-muted-foreground">
          {value === true ? "Enabled" : "Disabled"}
        </span>
      </div>
    )
  }
  if (choices?.length) {
    return (
      <Select
        items={[
          ...(field.nullable
            ? [{ value: "__automatic", label: "Automatic" }]
            : []),
          ...choices,
        ]}
        value={
          typeof value === "string" || typeof value === "number"
            ? String(value)
            : "__automatic"
        }
        onValueChange={(next) =>
          change(
            next === "__automatic"
              ? null
              : typeof field.default === "number"
                ? Number(next)
                : next
          )
        }
      >
        <SelectTrigger id={id} className="w-full" disabled={disabled}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            {field.nullable && (
              <SelectItem value="__automatic">Automatic</SelectItem>
            )}
            {choices.map((choice) => (
              <SelectItem key={choice.value} value={choice.value}>
                {choice.label}
              </SelectItem>
            ))}
          </SelectGroup>
        </SelectContent>
      </Select>
    )
  }
  if (
    field.type === "array" ||
    field.type === "list" ||
    Array.isArray(field.default) ||
    Array.isArray(value)
  ) {
    return (
      <>
        <Textarea
          id={id}
          value={Array.isArray(value) ? value.join("\n") : ""}
          placeholder="One entry per line"
          rows={3}
          disabled={disabled}
          onChange={(event) =>
            change(
              event.target.value
                .split("\n")
                .map((line) => line.trim())
                .filter(Boolean)
            )
          }
        />
        <FieldDescription>
          One entry per line. An empty list removes all entries.
        </FieldDescription>
      </>
    )
  }
  const numeric =
    ["number", "integer", "float", "int"].includes(field.type) ||
    typeof field.default === "number"
  return (
    <Input
      id={id}
      type={field.secret ? "password" : numeric ? "number" : "text"}
      autoComplete={field.secret ? "new-password" : "off"}
      min={field.minimum ?? undefined}
      max={field.maximum ?? undefined}
      step={field.type === "integer" || field.type === "int" ? 1 : "any"}
      disabled={disabled}
      value={
        typeof value === "string" || typeof value === "number" ? value : ""
      }
      placeholder="Automatic"
      onChange={(event) =>
        change(
          event.target.value === "" && numeric
            ? null
            : numeric
              ? Number(event.target.value)
              : event.target.value
        )
      }
    />
  )
}

export function ServerEditor({
  data,
  defaults,
  active,
}: {
  data: ServerSettings
  defaults?: ServerSettings
  active: string
}) {
  const queryClient = useQueryClient()
  const [edits, setEdits] = useState<Sections>({})
  const [search, setSearch] = useState("")
  const [notice, setNotice] = useState<SettingsResult | null>(null)
  const [preview, setPreview] = useState(false)
  const patch = editedPatch(edits, data.sections)
  const dirtyFields = Object.entries(patch).flatMap(([section, values]) =>
    Object.entries(values).map(([key, value]) => ({ section, key, value }))
  )
  const fields = data.fields.filter((field) =>
    search.trim()
      ? `${field.section} ${field.label} ${field.description} ${field.key}`
          .toLowerCase()
          .includes(search.toLowerCase().trim())
      : groupFor(field.section) === active
  )
  const sections = [...new Set(fields.map((field) => field.section))]
  const mutation = useMutation({
    mutationFn: () =>
      managementRequest<SettingsResult>("server/settings", {
        method: "PATCH",
        body: { sections: patch },
      }),
    onSuccess: async (result) => {
      setNotice(result)
      setEdits({})
      setPreview(false)
      await queryClient.invalidateQueries({ queryKey: ["molto"] })
    },
  })
  function change(field: ServerField, value: unknown) {
    setEdits((current) => ({
      ...current,
      [field.section]: { ...current[field.section], [field.key]: value },
    }))
    setNotice(null)
  }
  const group = settingsGroups.find((item) => item.id === active)
  return (
    <form
      className="flex flex-col gap-5"
      onSubmit={(event) => {
        event.preventDefault()
        if (dirtyFields.length && !mutation.isPending) mutation.mutate()
      }}
    >
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="text-lg font-semibold">
            {search ? "Search results" : group?.title}
          </h2>
          <p className="text-sm text-muted-foreground">{group?.description}</p>
        </div>
        <Input
          aria-label="Search all server settings"
          className="sm:max-w-72"
          placeholder="Search settings…"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
      </div>
      {active === "models" && (
        <ResourcePanel memory={{ ...data.sections.memory, ...edits.memory }} />
      )}
      {active === "models" && data.effective_model_dirs && (
        <div className="rounded-lg border bg-muted/30 p-4">
          <p className="text-sm font-medium">Effective model directories</p>
          <p className="mt-1 text-xs text-muted-foreground">
            These are the directories the running server scans. Startup
            arguments can override saved paths.
          </p>
          <ul className="mt-2 space-y-1 font-mono text-xs break-all">
            {data.effective_model_dirs.map((path) => (
              <li key={path}>{path}</li>
            ))}
          </ul>
        </div>
      )}
      {!fields.length && (
        <p className="rounded-lg border border-dashed p-8 text-center text-muted-foreground">
          {search
            ? "No settings match your search."
            : "This server exposes no settings in this section."}
        </p>
      )}
      {sections.map((section) => (
        <Card key={section}>
          <CardHeader>
            <div className="flex items-center justify-between gap-2">
              <CardTitle>{labelFor(section)}</CardTitle>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                disabled={mutation.isPending}
                onClick={() => {
                  fields
                    .filter((field) => field.section === section)
                    .forEach((field) =>
                      change(
                        field,
                        defaults?.sections[section]?.[field.key] ??
                          field.default
                      )
                    )
                  setPreview(true)
                }}
              >
                Preview defaults
              </Button>
            </div>
            <CardDescription>
              Saved in {data.base_path || "the server configuration"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <FieldGroup className="grid gap-6 md:grid-cols-2">
              {fields
                .filter((field) => field.section === section)
                .map((field) => (
                  <Field key={field.key}>
                    <div className="flex items-center justify-between gap-2">
                      <FieldLabel
                        htmlFor={`server-${field.section}-${field.key}`}
                      >
                        {field.label}
                      </FieldLabel>
                      {field.restart_required && (
                        <Badge variant="outline">Restart</Badge>
                      )}
                    </div>
                    <SettingControl
                      field={field}
                      value={draftValue(field, edits, data.sections)}
                      change={(value) => change(field, value)}
                      disabled={mutation.isPending}
                    />
                    <FieldDescription>{field.description}</FieldDescription>
                    <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
                      <span>
                        Default:{" "}
                        {display(
                          defaults?.sections[field.section]?.[field.key] ??
                            field.default,
                          field.secret
                        )}
                      </span>
                      <Button
                        type="button"
                        size="sm"
                        variant="ghost"
                        className="h-6 px-2 text-xs"
                        disabled={mutation.isPending}
                        onClick={() =>
                          change(
                            field,
                            defaults?.sections[field.section]?.[field.key] ??
                              field.default
                          )
                        }
                      >
                        Use default
                      </Button>
                    </div>
                  </Field>
                ))}
            </FieldGroup>
          </CardContent>
        </Card>
      ))}
      {mutation.isError && (
        <Alert variant="destructive">
          <AlertTitle>Changes were not saved</AlertTitle>
          <AlertDescription>{errorMessage(mutation.error)}</AlertDescription>
        </Alert>
      )}
      {notice && (
        <Alert>
          <AlertTitle>Settings saved</AlertTitle>
          <AlertDescription className="flex flex-col gap-1">
            <span>
              {notice.changed.length} fields changed.{" "}
              {notice.live_applied.length} applied to the running server.
            </span>
            {notice.restart_required.length > 0 && (
              <span>
                Restart required for: {notice.restart_required.join(", ")}.
              </span>
            )}
          </AlertDescription>
        </Alert>
      )}
      {preview && dirtyFields.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Review pending changes</CardTitle>
            <CardDescription>
              Nothing changes on the server until you save.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2 text-sm">
              {dirtyFields.map(({ section, key, value }) => {
                const field = data.fields.find(
                  (item) => item.section === section && item.key === key
                )
                return (
                  <li
                    key={`${section}.${key}`}
                    className="flex flex-wrap justify-between gap-2 border-b pb-2"
                  >
                    <span>{field?.label ?? key}</span>
                    <span className="text-muted-foreground">
                      {display(data.sections[section]?.[key], field?.secret)} →{" "}
                      {display(value, field?.secret)}
                    </span>
                  </li>
                )
              })}
            </ul>
          </CardContent>
        </Card>
      )}
      <div className="sticky bottom-3 z-10 flex flex-wrap items-center gap-2 rounded-xl border bg-background/95 p-3 shadow-sm backdrop-blur">
        <Button
          type="submit"
          disabled={!dirtyFields.length || mutation.isPending}
        >
          {mutation.isPending
            ? "Saving…"
            : `Save${dirtyFields.length ? ` ${dirtyFields.length} changes` : " changes"}`}
        </Button>
        <Button
          type="button"
          variant="outline"
          disabled={!dirtyFields.length || mutation.isPending}
          onClick={() => setPreview((current) => !current)}
        >
          Review changes
        </Button>
        <Button
          type="button"
          variant="ghost"
          disabled={!Object.keys(edits).length || mutation.isPending}
          onClick={() => {
            setEdits({})
            setPreview(false)
            mutation.reset()
          }}
        >
          Discard
        </Button>
        {dirtyFields.length > 0 && (
          <span role="status" className="text-xs text-muted-foreground">
            Draft preserved during refresh
          </span>
        )}
      </div>
    </form>
  )
}
