import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import type { components } from "@omlx/contracts"
import { managementQuery } from "@/features/management/request"
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { FieldGroup, Field, FieldLabel } from "@/components/ui/field"
import { Textarea } from "@/components/ui/textarea"
import {
  Progress,
  ProgressLabel,
  ProgressValue,
} from "@/components/ui/progress"
import { QueryState } from "@/components/page-state"
import { Choice, TextField, Failure, useAction } from "./controls"
import type { LocalModel } from "./types"
type Task = components["schemas"]["DiffusionCalibrationTask"]
type Job = components["schemas"]["DiffusionJobView"]
const initialTask = (): Task => ({
  prompt: "",
  seed: 0,
  width: 256,
  height: 256,
})
export function DiffusionPreparation({
  model,
  blocked,
}: {
  model: LocalModel
  blocked: boolean
}) {
  const query = useQuery(
    managementQuery<components["schemas"]["DiffusionJobsResponse"]>(
      ["diffusion-jobs"],
      "diffusion/jobs",
      2000
    )
  )
  const [tasks, setTasks] = useState<Task[]>([initialTask()])
  const [rows, setRows] = useState("256")
  const [calibration, setCalibration] = useState("")
  const [bits, setBits] = useState("4")
  const [group, setGroup] = useState("64")
  const [budgetMode, setBudgetMode] = useState("ratio")
  const [budget, setBudget] = useState("1.10")
  const [protectedLayers, setProtectedLayers] = useState("")
  const action = useAction<Job>()
  const cancel = useAction<Job>()
  const matching =
    query.data?.jobs.filter((job) => job.model_id === model.id) ?? []
  const complete = matching.filter(
    (job) => job.kind === "calibration" && job.status === "completed"
  )
  const selectedCalibration = complete.find((job) => job.id === calibration)
  const updateTask = (index: number, patch: Partial<Task>) =>
    setTasks((current) =>
      current.map((task, position) =>
        position === index ? { ...task, ...patch } : task
      )
    )
  const validTasks = tasks.every(
    (task) =>
      task.prompt.trim() &&
      task.prompt.length <= 32768 &&
      Number.isInteger(task.seed) &&
      task.seed >= 0 &&
      task.seed <= 4294967295 &&
      Number.isInteger(task.width) &&
      task.width >= 64 &&
      task.width <= 2048 &&
      Number.isInteger(task.height) &&
      task.height >= 64 &&
      task.height <= 2048 &&
      (task.steps == null ||
        (Number.isInteger(task.steps) &&
          task.steps >= 1 &&
          task.steps <= 1000)) &&
      (task.guidance == null || task.guidance >= 0)
  )
  return (
    <Card>
      <CardHeader>
        <CardTitle>2. Calibrate and quantize diffusion weights</CardTitle>
        <CardDescription>
          {model.quantization.available
            ? "Calibration runs the prompts below on the server and stores measured activations. Quantization uses a completed calibration for this source."
            : model.quantization.reason}
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-5">
        {model.quantization.available && (
          <>
            <TextField
              label="Maximum calibration rows"
              value={rows}
              onChange={setRows}
              type="number"
              min={1}
              max={4096}
            />
            {tasks.map((task, index) => (
              <Card key={index}>
                <CardHeader>
                  <CardTitle>Calibration prompt {index + 1}</CardTitle>
                </CardHeader>
                <CardContent>
                  <FieldGroup className="grid gap-4 sm:grid-cols-2">
                    <TextField
                      label={`Prompt ${index + 1}`}
                      value={task.prompt}
                      onChange={(prompt) => updateTask(index, { prompt })}
                    />
                    <TextField
                      label={`Negative prompt ${index + 1}`}
                      value={task.negative_prompt ?? ""}
                      onChange={(negative_prompt) =>
                        updateTask(index, {
                          negative_prompt: negative_prompt || null,
                        })
                      }
                    />
                    {(
                      ["seed", "width", "height", "steps", "guidance"] as const
                    ).map((name) => (
                      <TextField
                        key={name}
                        label={`${name} ${index + 1}`}
                        value={task[name] == null ? "" : String(task[name])}
                        type="number"
                        onChange={(value) =>
                          updateTask(index, {
                            [name]:
                              value === "" &&
                              (name === "steps" || name === "guidance")
                                ? null
                                : Number(value),
                          })
                        }
                      />
                    ))}
                    <Button
                      variant="outline"
                      disabled={tasks.length === 1 || action.isPending}
                      onClick={() =>
                        setTasks((current) =>
                          current.filter((_, position) => position !== index)
                        )
                      }
                    >
                      Remove prompt
                    </Button>
                  </FieldGroup>
                </CardContent>
              </Card>
            ))}
            <div className="flex flex-wrap gap-2">
              <Button
                variant="outline"
                disabled={tasks.length >= 32 || action.isPending}
                onClick={() =>
                  setTasks((current) => [...current, initialTask()])
                }
              >
                Add calibration prompt
              </Button>
              <Button
                disabled={
                  blocked ||
                  action.isPending ||
                  !validTasks ||
                  !Number.isInteger(Number(rows)) ||
                  Number(rows) < 1 ||
                  Number(rows) > 4096
                }
                onClick={() =>
                  action.mutate({
                    path: "diffusion/calibrations",
                    body: { model_id: model.id, tasks, max_rows: Number(rows) },
                  })
                }
              >
                Run calibration
              </Button>
            </div>
            <FieldGroup className="grid gap-4 sm:grid-cols-2">
              <Choice
                label="Completed calibration"
                value={calibration}
                onChange={setCalibration}
                options={complete.map((job) => ({
                  value: job.id,
                  label: `${job.id} · ${job.phase}`,
                }))}
              />
              <Choice
                label="Weight bits"
                value={bits}
                onChange={setBits}
                options={(model.quantization.bits ?? [3, 4, 5, 6, 8]).map(
                  (value) => ({ value: String(value), label: String(value) })
                )}
              />
              <Choice
                label="Diffusion group size"
                value={group}
                onChange={setGroup}
                options={[32, 64, 128].map((value) => ({
                  value: String(value),
                  label: String(value),
                }))}
              />
              <Choice
                label="Memory budget"
                value={budgetMode}
                onChange={(next) => {
                  setBudgetMode(next)
                  setBudget(next === "ratio" ? "1.10" : "1")
                }}
                options={[
                  { value: "ratio", label: "Ratio to estimated size" },
                  { value: "bytes", label: "Fixed GiB budget" },
                ]}
              />
              <TextField
                label={
                  budgetMode === "ratio" ? "Budget ratio" : "Budget in GiB"
                }
                value={budget}
                onChange={setBudget}
                type="number"
                min={budgetMode === "ratio" ? 1 : 0.001}
              />
              <Field>
                <FieldLabel htmlFor="protected-diffusion-layers">
                  Protected layer patterns
                </FieldLabel>
                <Textarea
                  id="protected-diffusion-layers"
                  value={protectedLayers}
                  onChange={(event) => setProtectedLayers(event.target.value)}
                  placeholder="One pattern per line, optional"
                />
              </Field>
            </FieldGroup>
            {!complete.length && (
              <p className="text-sm text-muted-foreground">
                Complete a calibration for this model before quantizing.
              </p>
            )}
            <Button
              disabled={
                blocked ||
                action.isPending ||
                query.isError ||
                !selectedCalibration ||
                Number(budget) < (budgetMode === "ratio" ? 1 : 0.001) ||
                !Number.isFinite(Number(budget))
              }
              onClick={() =>
                action.mutate({
                  path: "diffusion/quantizations",
                  body: {
                    model_id: model.id,
                    calibration_job_id: calibration,
                    bits: Number(bits),
                    group_size: Number(group),
                    ...(budgetMode === "ratio"
                      ? { budget_ratio: Number(budget) }
                      : {
                          budget_bytes: Math.round(Number(budget) * 1024 ** 3),
                        }),
                    protected: protectedLayers
                      .split("\n")
                      .map((value) => value.trim())
                      .filter(Boolean),
                  },
                })
              }
            >
              Start calibrated quantization
            </Button>
          </>
        )}
        <Failure error={action.error} />
        <Failure error={cancel.error} />
        {action.data && (
          <p role="status" className="text-sm">
            Job {action.data.id} queued. Progress appears below.
          </p>
        )}
        <QueryState query={query}>
          {matching.map((job) => (
            <div key={job.id} className="grid gap-2 rounded-lg border p-4">
              <p className="text-sm font-medium break-all">
                {job.kind} · {job.status} · {job.id}
              </p>
              <Progress value={job.progress * 100}>
                <ProgressLabel>{job.phase}</ProgressLabel>
                <ProgressValue />
              </Progress>
              <p className="text-sm text-muted-foreground">{job.detail}</p>
              {job.error && (
                <p role="alert" className="text-sm text-destructive">
                  {job.error}
                </p>
              )}
              {["queued", "waiting", "running"].includes(job.status) && (
                <Button
                  variant="outline"
                  disabled={cancel.isPending || query.isError}
                  onClick={() =>
                    cancel.mutate({
                      path: `diffusion/jobs/${encodeURIComponent(job.id)}/cancel`,
                    })
                  }
                >
                  Cancel diffusion job
                </Button>
              )}
            </div>
          ))}
        </QueryState>
      </CardContent>
    </Card>
  )
}
