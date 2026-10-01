import { useId, useState, type ReactNode } from "react"
import { ChevronDown } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Field,
  FieldDescription,
  FieldGroup,
  FieldLabel,
} from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Switch } from "@/components/ui/switch"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { Slider } from "@/components/ui/slider"
import {
  defaults,
  engines,
  toolNames,
  type AgentSettings,
} from "../agent/types"

export type SettingsPanelProps = {
  settings: AgentSettings
  onChange: (settings: AgentSettings) => void
  models: string[]
  running: boolean
}

function Choice({
  label,
  value,
  values,
  onChange,
  description,
}: {
  label: string
  value: string
  values: readonly (string | { value: string; label: string })[]
  onChange: (value: string) => void
  description?: string
}) {
  const id = useId()
  const items = values.map((item) =>
    typeof item === "string" ? { value: item, label: item } : item
  )
  return (
    <Field className="gap-1.5">
      <FieldLabel htmlFor={id}>{label}</FieldLabel>
      <Select
        value={value}
        onValueChange={(next) => {
          if (next !== null) onChange(next)
        }}
        items={items}
      >
        <SelectTrigger id={id} className="w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {items.map((item) => (
            <SelectItem key={item.value} value={item.value}>
              {item.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {description && (
        <FieldDescription className="text-xs">{description}</FieldDescription>
      )}
    </Field>
  )
}

function NumberControl({
  label,
  value,
  onChange,
  min,
  max,
  step = "any",
  slider = false,
}: {
  label: string
  value: number | undefined
  onChange: (value: number | undefined) => void
  min?: number
  max?: number
  step?: number | "any"
  slider?: boolean
}) {
  const id = useId()
  return (
    <Field className="gap-1.5">
      <FieldLabel htmlFor={id}>{label}</FieldLabel>
      <Input
        id={id}
        type="number"
        value={value ?? ""}
        placeholder="Inherit"
        min={min}
        max={max}
        step={step}
        onChange={(event) =>
          onChange(
            event.target.value === "" ? undefined : Number(event.target.value)
          )
        }
      />
      {slider && value !== undefined && (
        <Slider
          aria-label={`${label} slider`}
          value={[value]}
          min={min}
          max={max}
          step={step === "any" ? 0.01 : step}
          onValueChange={(next) =>
            onChange(Array.isArray(next) ? next[0] : next)
          }
        />
      )}
    </Field>
  )
}

function TextControl({
  label,
  value,
  onChange,
  multiline = false,
  description,
  placeholder,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  multiline?: boolean
  description?: string
  placeholder?: string
}) {
  const id = useId()
  return (
    <Field className="gap-1.5">
      <FieldLabel htmlFor={id}>{label}</FieldLabel>
      {multiline ? (
        <Textarea
          id={id}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
          className="min-h-20 font-mono text-xs"
        />
      ) : (
        <Input
          id={id}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
        />
      )}
      {description && (
        <FieldDescription className="text-xs">{description}</FieldDescription>
      )}
    </Field>
  )
}

function Toggle({
  label,
  checked,
  onChange,
  description,
}: {
  label: string
  checked: boolean
  onChange: (value: boolean) => void
  description?: string
}) {
  const id = useId()
  return (
    <Field orientation="horizontal" className="justify-between gap-3">
      <div className="min-w-0">
        <FieldLabel htmlFor={id}>{label}</FieldLabel>
        {description && (
          <FieldDescription className="mt-1 text-xs">
            {description}
          </FieldDescription>
        )}
      </div>
      <Switch id={id} checked={checked} onCheckedChange={onChange} />
    </Field>
  )
}

function Section({
  title,
  children,
  open = false,
}: {
  title: string
  children: ReactNode
  open?: boolean
}) {
  return (
    <Collapsible defaultOpen={open} className="border-b pb-3">
      <CollapsibleTrigger className="group flex w-full items-center justify-between py-3 text-left text-sm font-medium">
        {title}
        <ChevronDown className="size-4 text-muted-foreground transition-transform group-data-open:rotate-180 motion-reduce:transition-none" />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <FieldGroup className="gap-4 pb-1">{children}</FieldGroup>
      </CollapsibleContent>
    </Collapsible>
  )
}

function Environment({
  env,
  onChange,
}: {
  env: Record<string, string>
  onChange: (env: Record<string, string>) => void
}) {
  const [draft, setDraft] = useState<string>()
  const [error, setError] = useState("")
  const id = useId()
  return (
    <Field className="gap-1.5" data-invalid={!!error}>
      <FieldLabel htmlFor={id}>Environment</FieldLabel>
      <Textarea
        id={id}
        aria-invalid={!!error}
        aria-describedby={`${id}-help`}
        className="min-h-20 font-mono text-xs"
        value={draft ?? JSON.stringify(env, null, 2)}
        onChange={(event) => {
          const next = event.target.value
          setDraft(next)
          try {
            const parsed: unknown = JSON.parse(next || "{}")
            if (
              !parsed ||
              typeof parsed !== "object" ||
              Array.isArray(parsed) ||
              Object.values(parsed).some((value) => typeof value !== "string")
            )
              throw new Error("Use a JSON object with string values.")
            onChange(parsed as Record<string, string>)
            setError("")
          } catch {
            setError(
              "Use a JSON object with string values. The last valid environment is kept."
            )
          }
        }}
        onBlur={() => {
          if (!error) setDraft(undefined)
        }}
      />
      <FieldDescription
        id={`${id}-help`}
        className={error ? "text-xs text-destructive" : "text-xs"}
      >
        {error || 'JSON object, for example {"LANG": "en_US.UTF-8"}.'}
      </FieldDescription>
    </Field>
  )
}

export function SettingsPanel({
  settings,
  onChange,
  models,
  running,
}: SettingsPanelProps) {
  const set = <K extends keyof AgentSettings>(
    key: K,
    value: AgentSettings[K]
  ) => onChange({ ...settings, [key]: value })
  const search = <K extends keyof AgentSettings["search"]>(
    key: K,
    value: AgentSettings["search"][K]
  ) => set("search", { ...settings.search, [key]: value })
  const number = (
    key: keyof AgentSettings,
    label: string,
    min?: number,
    max?: number,
    step: number | "any" = "any",
    slider = false
  ) => (
    <NumberControl
      key={key}
      label={label}
      value={settings[key] as number | undefined}
      min={min}
      max={max}
      step={step}
      slider={slider}
      onChange={(value) => onChange({ ...settings, [key]: value })}
    />
  )
  const triState = (key: "enable_thinking" | "specprefill", label: string) => (
    <Choice
      label={label}
      value={
        settings[key] === undefined ? "inherit" : settings[key] ? "on" : "off"
      }
      values={[
        { value: "inherit", label: "Inherit" },
        { value: "on", label: "On" },
        { value: "off", label: "Off" },
      ]}
      onChange={(value) =>
        set(key, value === "inherit" ? undefined : value === "on")
      }
    />
  )
  return (
    <div className="space-y-1 px-4 pb-4" aria-label="Session settings">
      <div className="flex items-center justify-between pt-4">
        <h2 className="text-sm font-semibold">Session settings</h2>
        <Button
          variant="ghost"
          size="sm"
          onClick={() =>
            onChange({
              ...defaults(),
              model: settings.model,
              system: settings.system,
            })
          }
        >
          Reset controls
        </Button>
      </div>
      <p className="pb-2 text-xs text-muted-foreground">
        {running
          ? "Changes apply to the next run. This run keeps its settings."
          : "Blank generation controls inherit the model defaults."}
      </p>
      <FieldGroup className="gap-4">
        <Choice
          label="Model"
          value={settings.model}
          values={[
            { value: "", label: "Choose a local model" },
            ...new Set([
              ...models,
              ...(settings.model ? [settings.model] : []),
            ]),
          ]}
          onChange={(value) => set("model", value)}
        />
        <div className="grid grid-cols-2 gap-3">
          {number("temperature", "Temperature", 0, 5, 0.01, true)}
          {number("top_p", "Top P", 0, 1, 0.01, true)}
          {number("max_tokens", "Output tokens", 1, undefined, 1)}
          {number("seed", "Seed", undefined, undefined, 1)}
        </div>
      </FieldGroup>
      <Section title="Tools" open>
        <div className="flex gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => set("tools", [...toolNames])}
          >
            Enable all
          </Button>
          <Button size="sm" variant="outline" onClick={() => set("tools", [])}>
            Disable all
          </Button>
        </div>
        {toolNames.map((tool) => (
          <Toggle
            key={tool}
            label={tool}
            checked={settings.tools.includes(tool)}
            onChange={(checked) =>
              set(
                "tools",
                checked
                  ? [...settings.tools, tool]
                  : settings.tools.filter((name) => name !== tool)
              )
            }
          />
        ))}
        <Choice
          label="Tool choice"
          value={settings.toolChoice}
          values={["auto", "none", "required", ...settings.tools]}
          onChange={(value) =>
            set("toolChoice", value as AgentSettings["toolChoice"])
          }
        />
        {!(["auto", "none", "required"] as string[]).includes(
          settings.toolChoice
        ) &&
          !settings.tools.includes(
            settings.toolChoice as (typeof toolNames)[number]
          ) && (
            <p role="alert" className="text-xs text-destructive">
              The chosen tool is disabled. Choose another tool or enable it.
            </p>
          )}
        <Choice
          label="Tool execution"
          value={settings.execution}
          values={[
            { value: "auto", label: "Automatic" },
            { value: "confirm", label: "Confirm each call" },
          ]}
          onChange={(value) =>
            set("execution", value as AgentSettings["execution"])
          }
        />
        <div className="grid grid-cols-2 gap-3">
          {number("maxSteps", "Maximum steps", 1, 100, 1)}
          {number("toolTimeout", "Tool timeout (ms)", 100, 300_000, 1)}
          {number("outputLimit", "Output limit (chars)", 100, 1_000_000, 1)}
        </div>
      </Section>
      <Section title="Sampling and output">
        <div className="grid grid-cols-2 gap-3">
          {number("top_k", "Top K", 0, undefined, 1)}
          {number("min_p", "Min P", 0, 1)}
          {number("presence_penalty", "Presence penalty", -2, 2)}
          {number("frequency_penalty", "Frequency penalty", -2, 2)}
          {number("repetition_penalty", "Repetition penalty", 0)}
          {number(
            "repetition_context_size",
            "Repetition context",
            1,
            undefined,
            1
          )}
          {number("xtc_probability", "XTC probability", 0, 1)}
          {number("xtc_threshold", "XTC threshold", 0, 1)}
        </div>
        <TextControl
          label="Stop sequences"
          value={settings.stop.join("\n")}
          multiline
          onChange={(value) => set("stop", value ? value.split("\n") : [])}
          description="One literal stop sequence per line."
        />
      </Section>
      <Section title="Thinking and model options">
        <FieldDescription className="text-xs">
          Support depends on the selected model and its chat template. Unset
          options inherit defaults.
        </FieldDescription>
        {triState("enable_thinking", "Thinking")}
        <TextControl
          label="Reasoning effort"
          value={settings.reasoning_effort?.toString() ?? ""}
          placeholder="Inherit"
          description="Model-specific label (such as low or high) or numeric value."
          onChange={(value) =>
            set(
              "reasoning_effort",
              !value
                ? undefined
                : Number.isFinite(Number(value))
                  ? Number(value)
                  : value
            )
          }
        />
        {number("thinking_budget", "Thinking budget (tokens)", 0, undefined, 1)}
        {triState("specprefill", "Speculative prefill")}
        <div className="grid grid-cols-2 gap-3">
          {number("specprefill_keep_pct", "Keep fraction", 0.1, 0.5)}
          {number(
            "specprefill_threshold",
            "Prefill threshold",
            1,
            undefined,
            1
          )}
        </div>
      </Section>
      <Section title="Sandbox">
        <TextControl
          label="Working directory"
          value={settings.cwd}
          onChange={(value) => set("cwd", value)}
          placeholder="/workspace"
        />
        <Environment
          env={settings.env}
          onChange={(value) => set("env", value)}
        />
        <Toggle
          label="Read-only filesystem"
          checked={settings.readOnly}
          onChange={(value) => set("readOnly", value)}
          description="Blocks writes, including writes through bash."
        />
        <Toggle
          label="Sandbox network"
          checked={settings.network}
          onChange={(value) => set("network", value)}
          description="Network access follows the enabled web tools."
        />
        <div className="grid grid-cols-2 gap-3">
          {number("maxFileBytes", "File limit (bytes)", 1024, 100_000_000, 1)}
          {number("maxCommands", "Command limit", 1, 1_000_000, 1)}
          {number("maxLoopIterations", "Loop limit", 1, 1_000_000, 1)}
        </div>
      </Section>
      <Section title="Web search and fetch">
        <Choice
          label="Search provider"
          value={settings.search.provider}
          values={[
            { value: "ddgs", label: "DDGS · automatic engines" },
            { value: "ddgs_custom", label: "DDGS · custom engines" },
            { value: "duckduckgo", label: "DuckDuckGo" },
            { value: "brave", label: "Brave API" },
            { value: "searxng", label: "SearXNG" },
          ]}
          onChange={(value) =>
            search("provider", value as AgentSettings["search"]["provider"])
          }
        />
        {settings.search.provider === "ddgs_custom" && (
          <FieldGroup className="gap-3">
            {engines.map((engine) => (
              <Toggle
                key={engine}
                label={engine}
                checked={settings.search.engines.includes(engine)}
                onChange={(checked) =>
                  search(
                    "engines",
                    checked
                      ? [...settings.search.engines, engine]
                      : settings.search.engines.filter(
                          (item) => item !== engine
                        )
                  )
                }
              />
            ))}
          </FieldGroup>
        )}
        <NumberControl
          label="Maximum results"
          value={settings.search.maxResults}
          min={1}
          max={10}
          step={1}
          onChange={(value) => {
            if (value !== undefined) search("maxResults", value)
          }}
        />
        <Choice
          label="Result content"
          value={settings.search.content}
          values={[
            { value: "snippet", label: "Snippets" },
            { value: "full", label: "Fetch full pages" },
          ]}
          onChange={(value) => search("content", value as "snippet" | "full")}
        />
        <NumberControl
          label="Page content limit (chars)"
          value={settings.search.maxChars}
          min={100}
          max={100_000}
          step={1}
          onChange={(value) => {
            if (value !== undefined) search("maxChars", value)
          }}
        />
        <Toggle
          label="Truncate long pages"
          checked={settings.search.truncate}
          onChange={(value) => search("truncate", value)}
        />
        <FieldDescription className="text-xs">
          Provider credentials stay on the dashboard server.
        </FieldDescription>
      </Section>
      <Section title="Advanced request JSON">
        <TextControl
          label="Chat template arguments"
          value={settings.template}
          onChange={(value) => set("template", value)}
          multiline
          placeholder="{}"
          description="JSON object; blank inherits the model template."
        />
        <TextControl
          label="Response format"
          value={settings.responseFormat}
          onChange={(value) => set("responseFormat", value)}
          multiline
          placeholder="{}"
        />
        <TextControl
          label="Structured output"
          value={settings.structuredOutput}
          onChange={(value) => set("structuredOutput", value)}
          multiline
          placeholder="{}"
        />
      </Section>
    </div>
  )
}
