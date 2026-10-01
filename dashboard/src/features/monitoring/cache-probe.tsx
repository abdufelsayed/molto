import { useMutation } from "@tanstack/react-query"
import { useState } from "react"
import { managementRequest } from "@/features/management/request"
import { errorMessage } from "@/features/management/api"
import { Choice } from "./components"
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Table, TableBody, TableCell, TableRow } from "@/components/ui/table"
type ProbeResult = {
  model_loaded: boolean
  reason?: string
  total_tokens?: number
  block_size?: number
  total_blocks?: number
  blocks_ssd_hot?: number
  blocks_ssd_disk?: number
  blocks_cold?: number
  ssd_hit_tokens?: number
  cold_tokens?: number
}
export function CacheProbe({ models }: { models: string[] }) {
  const [model, setModel] = useState("")
  const [system, setSystem] = useState("")
  const [prompt, setPrompt] = useState("")
  const [budget, setBudget] = useState("")
  const probe = useMutation({
    mutationFn: () =>
      managementRequest<ProbeResult>("monitoring/cache/probe", {
        method: "POST",
        body: {
          model_id: model,
          messages: [
            ...(system ? [{ role: "system", content: system }] : []),
            { role: "user", content: prompt },
          ],
          ...(budget ? { thinking_budget: Number(budget) } : {}),
        },
      }),
  })
  return (
    <Card>
      <CardHeader>
        <CardTitle>Prefix cache probe</CardTitle>
        <CardDescription>
          Tokenize a draft request and inspect reusable cache blocks. This does
          not generate a response or load a model.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form
          onSubmit={(event) => {
            event.preventDefault()
            probe.mutate()
          }}
        >
          <FieldGroup>
            <Choice
              label="Model to probe"
              value={model}
              onChange={setModel}
              options={[
                { value: "", label: "Choose a loaded model" },
                ...models.map((id) => ({ value: id, label: id })),
              ]}
            />
            <Field>
              <FieldLabel htmlFor="probe-system">
                System prefix, optional
              </FieldLabel>
              <Textarea
                id="probe-system"
                value={system}
                onChange={(event) => setSystem(event.target.value)}
              />
            </Field>
            <Field>
              <FieldLabel htmlFor="probe-prompt">User message</FieldLabel>
              <Textarea
                id="probe-prompt"
                required
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
              />
            </Field>
            <Field>
              <FieldLabel htmlFor="probe-budget">
                Thinking budget, optional
              </FieldLabel>
              <Input
                id="probe-budget"
                type="number"
                min="0"
                max="131072"
                step="1"
                value={budget}
                onChange={(event) => setBudget(event.target.value)}
              />
            </Field>
            <Button
              className="w-fit"
              type="submit"
              disabled={!model || !prompt.trim() || probe.isPending}
            >
              {probe.isPending ? "Probing…" : "Run cache probe"}
            </Button>
          </FieldGroup>
        </form>
        {probe.isError && (
          <p role="alert" className="mt-4 text-destructive">
            {errorMessage(probe.error)}
          </p>
        )}
        {probe.data && (
          <div className="mt-4" role="status">
            {!probe.data.model_loaded ? (
              <p>
                {probe.data.reason ??
                  "The model is not loaded. Load it from Models before probing."}
              </p>
            ) : (
              <>
                <p className="mb-2 text-sm">
                  Result from the last explicit probe. Editing the draft does
                  not update this result.
                </p>
                <Table>
                  <TableBody>
                    {Object.entries(probe.data)
                      .filter(([key]) => key !== "model_loaded")
                      .map(([key, value]) => (
                        <TableRow key={key}>
                          <TableCell>{key.replaceAll("_", " ")}</TableCell>
                          <TableCell className="text-right">
                            {String(value)}
                          </TableCell>
                        </TableRow>
                      ))}
                  </TableBody>
                </Table>
              </>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
