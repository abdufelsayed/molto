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
import { FieldGroup, Field, FieldLabel } from "@/components/ui/field"
import { Switch } from "@/components/ui/switch"
import { Button } from "@/components/ui/button"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
import { Choice, TextField, useAction, Failure } from "./controls"
import { DiffusionPreparation } from "./diffusion"
import type { LocalCatalog } from "./types"
import type { Operation } from "@/features/operations/page"
export function Toggle({
  label,
  value,
  onChange,
}: {
  label: string
  value: boolean
  onChange: (value: boolean) => void
}) {
  return (
    <Field orientation="horizontal">
      <FieldLabel>{label}</FieldLabel>
      <Switch aria-label={label} checked={value} onCheckedChange={onChange} />
    </Field>
  )
}
export function Prepare() {
  const query = useQuery(
    managementQuery<LocalCatalog>(
      ["prepare-models"],
      "acquisition/prepare/models",
      5000
    )
  )
  const [path, setPath] = useState("")
  const [output, setOutput] = useState("")
  const [level, setLevel] = useState("4")
  const [group, setGroup] = useState("64")
  const [dtype, setDtype] = useState("bfloat16")
  const [proxy, setProxy] = useState("")
  const [assistant, setAssistant] = useState("")
  const [cache, setCache] = useState("")
  const [preserve, setPreserve] = useState(false)
  const [enhanced, setEnhanced] = useState(false)
  const [automatic, setAutomatic] = useState(true)
  const [textOnly, setTextOnly] = useState(false)
  const [reuse, setReuse] = useState(true)
  const [strict, setStrict] = useState(false)
  const [samples, setSamples] = useState("128")
  const [sequence, setSequence] = useState("512")
  const estimate = useAction<{
    effective_bpw: number
    output_size_bytes: number
    output_size_formatted: string
  }>()
  const run = useAction<Operation>()
  const model = query.data?.models.find((item) => item.path === path)
  const common = {
    model_path: path,
    oq_level: Number(level),
    group_size: Number(group),
    preserve_mtp: preserve,
  }
  const invalidateEstimate = () => {
    estimate.reset()
    run.reset()
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle>Prepare a local checkpoint</CardTitle>
        <CardDescription>
          Choose a source already in server storage. Conversion creates a
          compatible copy; quantization creates a smaller copy. The source
          remains available.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-5">
        <QueryState query={query}>
          {query.data && (
            <>
              <Choice
                label="Source model"
                value={path}
                onChange={(next) => {
                  setPath(next)
                  setOutput(
                    query.data.models.find((item) => item.path === next)
                      ?.conversion.output_name ?? ""
                  )
                  invalidateEstimate()
                }}
                options={query.data.models.map((item) => ({
                  value: item.path,
                  label: `${item.name} · ${item.precision} · ${item.format}`,
                }))}
              />
              {!query.data.models.length && (
                <p className="text-sm text-muted-foreground">
                  No local checkpoints found. Download a source first.
                </p>
              )}
              {model && (
                <>
                  <p className="text-sm break-all text-muted-foreground">
                    {model.path}
                  </p>
                  {query.data.options.exclusive_busy && (
                    <Alert>
                      <AlertTitle>Preparation is busy</AlertTitle>
                      <AlertDescription>
                        Wait for the current GPU operation to finish before
                        starting another preparation.
                      </AlertDescription>
                    </Alert>
                  )}
                  <Card>
                    <CardHeader>
                      <CardTitle>1. Convert when needed</CardTitle>
                      <CardDescription>
                        {model.conversion.available
                          ? `Creates ${model.conversion.target_format ?? "a compatible checkpoint"} using ${model.conversion.adapter ?? "the model adapter"}.`
                          : (model.conversion.reason ??
                            "No conversion is needed for this checkpoint.")}
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="grid gap-4">
                      {model.conversion.available && (
                        <>
                          <TextField
                            label="Output directory name"
                            value={output}
                            onChange={setOutput}
                            description="A new directory inside the server's configured model storage."
                          />
                          <Button
                            disabled={
                              run.isPending ||
                              query.isError ||
                              query.data.options.exclusive_busy ||
                              !/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(output) ||
                              output.includes("..")
                            }
                            onClick={() =>
                              run.mutate({
                                path: "acquisition/prepare/convert",
                                body: { model_path: path, output_name: output },
                              })
                            }
                          >
                            Convert checkpoint
                          </Button>
                        </>
                      )}
                    </CardContent>
                  </Card>
                  {model.quantization.adapter === "mflux" ? (
                    <DiffusionPreparation
                      model={model}
                      blocked={
                        query.isError || query.data.options.exclusive_busy
                      }
                    />
                  ) : (
                    <Card>
                      <CardHeader>
                        <CardTitle>
                          2. Quantize a full precision source
                        </CardTitle>
                        <CardDescription>
                          {model.quantization.available
                            ? model.quantization.requires_conversion
                              ? "Convert this source first, then select the converted output."
                              : "Estimate the output before starting oQ quantization."
                            : (model.quantization.reason ??
                              "Quantization is unavailable.")}
                        </CardDescription>
                      </CardHeader>
                      {model.quantization.available &&
                        !model.quantization.requires_conversion && (
                          <CardContent className="grid gap-5">
                            <FieldGroup className="grid gap-4 sm:grid-cols-2">
                              <Choice
                                label="oQ level"
                                value={level}
                                onChange={(next) => {
                                  setLevel(next)
                                  invalidateEstimate()
                                }}
                                options={query.data.options.oq_levels.map(
                                  (value) => ({
                                    value: String(value),
                                    label: `oQ${value}`,
                                  })
                                )}
                              />
                              <Choice
                                label="Group size"
                                value={group}
                                onChange={(next) => {
                                  setGroup(next)
                                  invalidateEstimate()
                                }}
                                options={query.data.options.group_sizes.map(
                                  (value) => ({
                                    value: String(value),
                                    label: String(value),
                                  })
                                )}
                              />
                              <Choice
                                label="Output dtype"
                                value={dtype}
                                onChange={setDtype}
                                options={query.data.options.dtypes.map(
                                  (value) => ({ value, label: value })
                                )}
                              />
                              <Toggle
                                label="Preserve MTP weights"
                                value={preserve}
                                onChange={(next) => {
                                  setPreserve(next)
                                  invalidateEstimate()
                                }}
                              />
                            </FieldGroup>
                            <details className="grid gap-4">
                              <summary className="cursor-pointer text-sm font-medium">
                                Calibration and source options
                              </summary>
                              <FieldGroup className="mt-4">
                                <Toggle
                                  label="Choose proxy sensitivity automatically"
                                  value={automatic}
                                  onChange={setAutomatic}
                                />
                                <Choice
                                  label="Sensitivity source, optional"
                                  value={proxy}
                                  onChange={setProxy}
                                  options={[
                                    {
                                      value: "",
                                      label: "No explicit sensitivity source",
                                    },
                                    ...query.data.models.map((item) => ({
                                      value: item.path,
                                      label: item.name,
                                    })),
                                  ]}
                                />
                                <Choice
                                  label="MTP assistant source, optional"
                                  value={assistant}
                                  onChange={setAssistant}
                                  options={[
                                    { value: "", label: "No assistant source" },
                                    ...query.data.models.map((item) => ({
                                      value: item.path,
                                      label: item.name,
                                    })),
                                  ]}
                                />
                                <Toggle
                                  label="Quantize text components only"
                                  value={textOnly}
                                  onChange={setTextOnly}
                                />
                                <Toggle
                                  label="Enhanced calibration"
                                  value={enhanced}
                                  onChange={setEnhanced}
                                />
                                <TextField
                                  label="Calibration cache path, optional"
                                  value={cache}
                                  onChange={setCache}
                                  description="Must be inside configured model storage."
                                />
                                <Toggle
                                  label="Reuse calibration cache"
                                  value={reuse}
                                  onChange={setReuse}
                                />
                                <Toggle
                                  label="Require calibration cache match"
                                  value={strict}
                                  onChange={setStrict}
                                />
                                <TextField
                                  label="Calibration samples"
                                  value={samples}
                                  onChange={setSamples}
                                  type="number"
                                  min={1}
                                  max={4096}
                                />
                                <TextField
                                  label="Calibration sequence length"
                                  value={sequence}
                                  onChange={setSequence}
                                  type="number"
                                  min={1}
                                  max={32768}
                                />
                              </FieldGroup>
                            </details>
                            <Failure error={estimate.error} />
                            <Button
                              variant="outline"
                              disabled={
                                estimate.isPending ||
                                query.isError ||
                                run.isPending
                              }
                              onClick={() =>
                                estimate.mutate({
                                  path: "acquisition/prepare/estimate",
                                  body: common,
                                })
                              }
                            >
                              {estimate.isPending
                                ? "Estimating…"
                                : "Estimate output size"}
                            </Button>
                            {estimate.data && (
                              <Alert>
                                <AlertTitle>
                                  Estimated output{" "}
                                  {estimate.data.output_size_formatted}
                                </AlertTitle>
                                <AlertDescription>
                                  {estimate.data.effective_bpw.toFixed(2)}{" "}
                                  effective bits per weight. Calibration may
                                  change final size.
                                </AlertDescription>
                              </Alert>
                            )}
                            <Button
                              disabled={
                                run.isPending ||
                                query.isError ||
                                query.data.options.exclusive_busy ||
                                !estimate.data ||
                                !(
                                  Number(samples) >= 1 &&
                                  Number(samples) <= 4096 &&
                                  Number(sequence) >= 1 &&
                                  Number(sequence) <= 32768
                                )
                              }
                              onClick={() =>
                                run.mutate({
                                  path: "acquisition/prepare/quantize",
                                  body: {
                                    ...common,
                                    dtype,
                                    sensitivity_model_path: proxy,
                                    mtp_assistant_model_path: assistant,
                                    auto_proxy_sensitivity: automatic,
                                    text_only: textOnly,
                                    enhanced,
                                    imatrix_cache_path: cache,
                                    imatrix_reuse_cache: reuse,
                                    imatrix_strict: strict,
                                    imatrix_num_samples: Number(samples),
                                    imatrix_seq_length: Number(sequence),
                                  },
                                })
                              }
                            >
                              {run.isPending
                                ? "Starting…"
                                : "Start quantization"}
                            </Button>
                          </CardContent>
                        )}
                    </Card>
                  )}
                  <Failure error={run.error} />
                  {run.data && (
                    <Alert>
                      <AlertTitle>Preparation queued</AlertTitle>
                      <AlertDescription>
                        Follow operation {run.data.id} in Activity. Refresh the
                        source list when it finishes to select the new output.
                      </AlertDescription>
                    </Alert>
                  )}
                </>
              )}
            </>
          )}
        </QueryState>
      </CardContent>
    </Card>
  )
}
