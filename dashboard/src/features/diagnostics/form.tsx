import { useState } from "react"
import {
  Field,
  FieldGroup,
  FieldLabel,
  FieldDescription,
} from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectGroup,
  SelectItem,
} from "@/components/ui/select"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
import { labels, type Kind, type Capabilities } from "./types"
const prompts = [1024, 4096, 8192, 16384, 32768, 65536, 131072, 200000]
const corpora = [
  "code_python",
  "code_mixed",
  "novel_ko",
  "novel_en",
  "novel_ja",
]
export function Choice({
  id,
  label,
  value,
  choices,
  change,
}: {
  id: string
  label: string
  value: string
  choices: { value: string; label: string; disabled?: boolean }[]
  change: (value: string) => void
}) {
  return (
    <Field>
      <FieldLabel htmlFor={id}>{label}</FieldLabel>
      <Select
        value={value}
        onValueChange={(next) => {
          if (next) change(next)
        }}
      >
        <SelectTrigger id={id} className="w-full">
          <SelectValue>
            {choices.find((item) => item.value === value)?.label ??
              "Choose an option"}
          </SelectValue>
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            {choices.map((item) => (
              <SelectItem
                key={item.value}
                value={item.value}
                disabled={item.disabled}
              >
                {item.label}
              </SelectItem>
            ))}
          </SelectGroup>
        </SelectContent>
      </Select>
    </Field>
  )
}
export function DiagnosticForm({
  capabilities,
  pending,
  busy,
  error,
  submit,
}: {
  capabilities: Capabilities
  pending: boolean
  busy: boolean
  error?: string
  submit: (kind: Kind, options: Record<string, unknown>) => void
}) {
  const [kind, setKind] = useState<Kind>("throughput")
  const [target, setTarget] = useState("local")
  const [localModel, setLocalModel] = useState("")
  const [externalModel, setExternalModel] = useState("")
  const model = target === "external" ? externalModel : localModel
  const setModel = target === "external" ? setExternalModel : setLocalModel
  const [url, setUrl] = useState("")
  const [key, setKey] = useState("")
  const [values, setValues] = useState<Record<string, string | boolean>>({
    prompt_lengths: "1024,4096",
    batch_sizes: "",
    generation_length: "128",
    context_profile: "code_python",
    warmup_mode: "quick",
    batch_size: "1",
    samples: "100",
    max_tokens_override: "",
    sampling_profile: "deterministic",
    target_tokens: "131072",
    backend: "qwen",
    sequence_length: "2048",
    repeats: "2",
    allow_cpu: true,
    allow_cpu_gate: true,
    allow_cpu_down: true,
    allow_ane_gdn: true,
    allow_cpu_gdn: true,
    allow_cpu_shared_resource: true,
  })
  const [suites, setSuites] = useState<string[]>([])
  const [validation, setValidation] = useState("")
  const spec = capabilities.kinds.find((item) => item.kind === kind)
  const selected = capabilities.models.find((item) => item.model_id === model)
  const external = target === "external"
  const reason = !spec
    ? "This runner is not available."
    : !spec.targets.includes(target)
      ? "This diagnostic supports local models only."
      : !external && selected?.disabled_reason
        ? selected.disabled_reason
        : !external && selected?.kinds && !selected.kinds.includes(kind)
          ? kind === "ane"
            ? (selected.ane_disabled_reason ??
              "This model does not support ANE tuning.")
            : "This model is incompatible with the selected diagnostic."
          : undefined
  const set = (name: string, value: string | boolean) =>
    setValues((previous) => ({ ...previous, [name]: value }))
  const number = (name: string) => Number(values[name])
  const numbers = (name: string) =>
    String(values[name] ?? "")
      .split(",")
      .filter((item) => item.trim())
      .map(Number)
  function start(event: React.FormEvent) {
    event.preventDefault()
    setValidation("")
    if (!model.trim()) {
      setValidation("Choose a model.")
      return
    }
    const options: Record<string, unknown> = { model_id: model.trim() }
    if (external) {
      if (!/^https?:\/\//.test(url)) {
        setValidation("Enter an HTTP or HTTPS endpoint URL.")
        return
      }
      options.external = {
        base_url: url,
        model: model.trim(),
        api_key: key,
        ...(kind === "accuracy" && values.max_tokens_override
          ? { max_tokens_override: number("max_tokens_override") }
          : {}),
      }
    }
    if (kind === "throughput") {
      const lengths = numbers("prompt_lengths")
      const batches = numbers("batch_sizes")
      if (
        !lengths.length ||
        lengths.some((value) => !prompts.includes(value)) ||
        batches.some((value) => ![2, 4, 8].includes(value)) ||
        !Number.isInteger(number("generation_length")) ||
        number("generation_length") < 1 ||
        number("generation_length") > 65536
      ) {
        setValidation(
          "Use supported token lengths and batch sizes, and a positive output length."
        )
        return
      }
      Object.assign(options, {
        prompt_lengths: lengths,
        batch_sizes: batches,
        generation_length: number("generation_length"),
        context_profile: values.context_profile,
        warmup_mode: values.warmup_mode,
        align_prompt_to_ane: !!values.align_prompt_to_ane,
        force_lm_engine: !!values.force_lm_engine,
      })
    }
    if (kind === "accuracy") {
      if (
        !suites.length ||
        !Number.isInteger(number("samples")) ||
        number("samples") < 0
      ) {
        setValidation(
          "Choose at least one suite and a nonnegative sample count."
        )
        return
      }
      Object.assign(options, {
        benchmarks: Object.fromEntries(
          suites.map((suite) => [suite, number("samples")])
        ),
        batch_size: number("batch_size"),
        enable_thinking: !external && !!values.enable_thinking,
        sampling_profile: values.sampling_profile,
      })
    }
    if (kind === "context") options.target_tokens = number("target_tokens")
    if (kind === "ane") {
      if (
        number("sequence_length") < 1024 ||
        number("sequence_length") % 64 ||
        !Number.isInteger(number("repeats")) ||
        number("repeats") < 1 ||
        number("repeats") > 5
      ) {
        setValidation(
          "ANE sequence length must be a multiple of 64 at least 1024. Repeats must be 1 to 5."
        )
        return
      }
      Object.assign(options, {
        backend: values.backend,
        sequence_length: number("sequence_length"),
        repeats: number("repeats"),
      })
      for (const name of [
        "allow_cpu",
        "allow_cpu_gate",
        "allow_cpu_down",
        "allow_ane_gdn",
        "allow_cpu_gdn",
        "allow_cpu_shared_resource",
      ])
        options[name] = !!values[name]
    }
    submit(kind, options)
  }
  const input = (
    name: string,
    label: string,
    hint?: string,
    minimum?: number
  ) => (
    <Field key={name}>
      <FieldLabel htmlFor={`diagnostic-${name}`}>{label}</FieldLabel>
      <Input
        id={`diagnostic-${name}`}
        type={minimum === undefined ? "text" : "number"}
        min={minimum}
        step={1}
        value={String(values[name] ?? "")}
        onChange={(event) => set(name, event.target.value)}
      />
      {hint && <FieldDescription>{hint}</FieldDescription>}
    </Field>
  )
  const choice = (name: string, label: string, choices: string[]) => (
    <Choice
      id={`diagnostic-${name}`}
      label={label}
      value={String(values[name])}
      choices={choices.map((value) => ({
        value,
        label: value.replaceAll("_", " "),
      }))}
      change={(value) => set(name, value)}
    />
  )
  const toggle = (name: string, label: string) => (
    <Field key={name} orientation="horizontal">
      <FieldLabel htmlFor={`diagnostic-${name}`}>{label}</FieldLabel>
      <Switch
        id={`diagnostic-${name}`}
        checked={!!values[name]}
        onCheckedChange={(value) => set(name, value)}
      />
    </Field>
  )
  return (
    <form onSubmit={start} className="grid gap-6">
      <FieldGroup>
        <Choice
          id="diagnostic-kind"
          label="Diagnostic"
          value={kind}
          choices={Object.entries(labels).map(([value, label]) => ({
            value,
            label,
          }))}
          change={(value) => setKind(value as Kind)}
        />
        <Choice
          id="diagnostic-target"
          label="Target"
          value={target}
          choices={[
            { value: "local", label: "Local model" },
            {
              value: "external",
              label: "External OpenAI-compatible endpoint",
              disabled: !spec?.targets.includes("external"),
            },
          ]}
          change={(value) => {
            setTarget(value)
          }}
        />
        {external ? (
          <>
            <Field>
              <FieldLabel htmlFor="external-model">
                External model name
              </FieldLabel>
              <Input
                id="external-model"
                value={model}
                onChange={(event) => setModel(event.target.value)}
              />
            </Field>
            <Field>
              <FieldLabel htmlFor="external-url">Endpoint base URL</FieldLabel>
              <Input
                id="external-url"
                placeholder="https://provider.example/v1"
                value={url}
                onChange={(event) => setUrl(event.target.value)}
              />
            </Field>
            <Field>
              <FieldLabel htmlFor="external-key">Endpoint API key</FieldLabel>
              <Input
                id="external-key"
                type="password"
                autoComplete="off"
                value={key}
                onChange={(event) => setKey(event.target.value)}
              />
              <FieldDescription>
                Sent only when you start the run. Omitted from retained history.
              </FieldDescription>
            </Field>
          </>
        ) : (
          <Choice
            id="diagnostic-model"
            label="Model"
            value={model}
            choices={capabilities.models.map((item) => ({
              value: item.model_id,
              label: `${item.label ?? item.model_id}${item.disabled_reason || (kind === "ane" && item.ane_disabled_reason) ? ` (${item.disabled_reason || item.ane_disabled_reason})` : ""}`,
              disabled:
                !!item.disabled_reason ||
                (!!item.kinds && !item.kinds.includes(kind)),
            }))}
            change={setModel}
          />
        )}
        {!external && !capabilities.models.length && (
          <p className="text-sm text-muted-foreground">
            No local models are available. Add a language model in the library
            or choose an external endpoint for throughput or accuracy.
          </p>
        )}
      </FieldGroup>
      <FieldGroup className="grid gap-5 sm:grid-cols-2">
        {kind === "throughput" && (
          <>
            {input(
              "prompt_lengths",
              "Input token lengths",
              `Comma-separated: ${prompts.join(", ")}`
            )}
            {input("generation_length", "Output tokens", undefined, 1)}
            {input(
              "batch_sizes",
              "Concurrent batch sizes",
              "Comma-separated: 2, 4, 8. Leave empty for single requests."
            )}
            {choice("context_profile", "Prompt corpus", corpora)}
            {choice("warmup_mode", "Warmup", ["quick", "ane_2048"])}
            {toggle("align_prompt_to_ane", "Align prompts to ANE blocks")}
            {toggle("force_lm_engine", "Force language model engine")}
          </>
        )}
        {kind === "accuracy" && (
          <>
            <Field className="sm:col-span-2">
              <FieldLabel>Evaluation suites</FieldLabel>
              {capabilities.suites.map((suite) => (
                <label
                  key={suite.id}
                  className="flex items-start gap-2 text-sm"
                >
                  <input
                    type="checkbox"
                    checked={suites.includes(suite.id)}
                    onChange={(event) =>
                      setSuites((previous) =>
                        event.target.checked
                          ? [...previous, suite.id]
                          : previous.filter((id) => id !== suite.id)
                      )
                    }
                  />
                  <span>
                    {suite.label ?? suite.id}
                    {suite.description && (
                      <span className="block text-xs text-muted-foreground">
                        {suite.description}
                      </span>
                    )}
                  </span>
                </label>
              ))}
            </Field>
            {input(
              "samples",
              "Samples per suite",
              "0 evaluates the full dataset. Missing datasets may be downloaded by the runner.",
              0
            )}
            {choice("batch_size", "Evaluation batch size", [
              "1",
              "2",
              "4",
              "8",
              "16",
              "32",
            ])}
            {external &&
              input(
                "max_tokens_override",
                "External output token floor",
                "Optional. Raises each evaluation question’s output budget for reasoning models.",
                1
              )}
            {choice("sampling_profile", "Sampling", [
              "deterministic",
              "model_settings",
            ])}
            {!external && toggle("enable_thinking", "Enable thinking")}
          </>
        )}
        {kind === "context" &&
          choice("target_tokens", "Context search ceiling", [
            "16384",
            "32768",
            "65536",
            "131072",
            "262144",
            "524288",
          ])}
        {kind === "ane" && (
          <>
            {choice("backend", "Model family", ["qwen", "k2"])}
            {input(
              "sequence_length",
              "Sequence length",
              "Multiple of 64, at least 1024.",
              1024
            )}
            {input("repeats", "Repeats per candidate", "1 to 5", 1)}
            {toggle("allow_cpu", "Test CPU candidates")}
            {toggle("allow_cpu_gate", "Test CPU gate projection")}
            {toggle("allow_cpu_down", "Test CPU down projection")}
            {toggle("allow_ane_gdn", "Test ANE GDN")}
            {toggle("allow_cpu_gdn", "Test CPU GDN")}
            {toggle("allow_cpu_shared_resource", "Test shared CPU resources")}
          </>
        )}
      </FieldGroup>
      <Alert>
        <AlertTitle>Run impact</AlertTitle>
        <AlertDescription>
          {external
            ? "This sends evaluation requests to your endpoint and may incur provider charges. The run holds the shared preparation and diagnostics lock until cleanup finishes. External runs do not unload local models."
            : kind === "context" || kind === "ane"
              ? "This waits for running requests to drain, unloads all resident models, then loads the target and runs inference. Warmed engines will need loading again. Exclusive GPU admission stays held until work and cancellation cleanup finish."
              : "This waits for running requests to drain and unloads other resident models and the target before loading it for inference. Warmed engines will need loading again. Exclusive GPU admission stays held until work and cancellation cleanup finish."}
          {kind === "ane" &&
            " The runner tests temporary settings and returns recommendations."}{" "}
          Results stay on this server; no community upload occurs.
        </AlertDescription>
      </Alert>
      {(reason || validation || error) && (
        <Alert variant="destructive">
          <AlertTitle>Run cannot start</AlertTitle>
          <AlertDescription>{reason || validation || error}</AlertDescription>
        </Alert>
      )}
      <Button type="submit" disabled={pending || busy || !!reason || !model}>
        {pending
          ? "Starting…"
          : busy
            ? "Waiting for active diagnostic to stop"
            : `Run ${labels[kind].toLowerCase()}`}
      </Button>
    </form>
  )
}
