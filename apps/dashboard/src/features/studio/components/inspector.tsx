import { useId, useState } from "react"
import { Check, Copy } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Field, FieldLabel } from "@/components/ui/field"
import { Textarea } from "@/components/ui/textarea"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import {
  diffFiles,
  type FileChange,
  type Run,
  type RunEvent,
  type VirtualFile,
} from "../agent/types"

export type InspectorProps = {
  run?: Run
  selectedEvent?: string
  onSelectEvent: (id: string) => void
  onNote: (note: string) => void
}

function json(value: unknown): string {
  if (value === undefined) return "Not recorded"
  try {
    return JSON.stringify(value, null, 2) ?? "Not recorded"
  } catch {
    return "Unable to serialize this value."
  }
}

function Raw({ value, label }: { value: unknown; label: string }) {
  const [copied, setCopied] = useState(false)
  const [error, setError] = useState("")
  const text = json(value)
  return (
    <div className="min-w-0 rounded-md border">
      <div className="flex items-center justify-between border-b px-3 py-1.5">
        <span className="text-xs font-medium">{label}</span>
        <Button
          variant="ghost"
          size="sm"
          aria-label={`Copy ${label}`}
          onClick={() => {
            void navigator.clipboard
              .writeText(text)
              .then(() => {
                setCopied(true)
                setError("")
                setTimeout(() => setCopied(false), 1500)
              })
              .catch(() =>
                setError("Copy failed. Select and copy the text below.")
              )
          }}
        >
          {copied ? (
            <Check className="size-3.5" />
          ) : (
            <Copy className="size-3.5" />
          )}
        </Button>
      </div>
      {error && (
        <p role="alert" className="px-3 text-xs text-destructive">
          {error}
        </p>
      )}
      <pre
        aria-label={label}
        className="max-h-96 overflow-auto p-3 font-mono text-xs leading-relaxed break-all whitespace-pre-wrap"
      >
        {text}
      </pre>
    </div>
  )
}

function elapsed(seconds: number | undefined): string {
  if (seconds === undefined) return "—"
  return seconds < 1
    ? `${Math.round(seconds * 1000)} ms`
    : `${seconds.toFixed(2)} s`
}

function EventDetails({ event }: { event: RunEvent }) {
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <Badge variant="outline">Step {event.step}</Badge>
        <Badge
          variant={
            event.state === "error" || event.type === "error"
              ? "destructive"
              : "secondary"
          }
        >
          {event.state ?? event.type}
        </Badge>
        {event.replayed && (
          <Badge variant="outline">Recorded result replayed</Badge>
        )}
        {event.duration !== undefined && (
          <span className="text-muted-foreground">
            {elapsed(event.duration)}
          </span>
        )}
        <span className="text-muted-foreground">
          {new Date(event.at).toLocaleTimeString()}
        </span>
      </div>
      {event.callId && (
        <p className="font-mono text-xs break-all text-muted-foreground">
          Call: {event.callId}
        </p>
      )}
      {event.text !== undefined && (
        <Raw
          label={event.type === "reasoning" ? "Model reasoning" : "Event text"}
          value={event.text}
        />
      )}
      {event.input !== undefined && (
        <Raw label="Tool input" value={event.input} />
      )}
      {event.output !== undefined && (
        <Raw
          label={
            event.state === "error"
              ? "Recorded tool error"
              : "Full recorded tool result"
          }
          value={event.output}
        />
      )}
      {event.modelOutput !== undefined && (
        <Raw
          label="Model-visible tool result (limited)"
          value={event.modelOutput}
        />
      )}
      {event.changes?.length ? <Changes changes={event.changes} /> : null}
    </div>
  )
}

function content(file: VirtualFile | undefined): string {
  if (!file) return "(absent)"
  if (file.type === "directory") return "(directory)"
  if (file.type === "symlink") return `→ ${file.target ?? ""}`
  if (!file.content) return "(empty file)"
  try {
    const bytes = Uint8Array.from(atob(file.content), (char) =>
      char.charCodeAt(0)
    )
    if (bytes.length > 100_000)
      return `(file is ${bytes.length.toLocaleString()} bytes; preview limited to 100,000 bytes)\n${new TextDecoder().decode(bytes.slice(0, 100_000))}`
    if (bytes.includes(0))
      return `(binary file; ${bytes.length.toLocaleString()} bytes)`
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes)
  } catch {
    return "(binary or unavailable file preview)"
  }
}

function Changes({ changes }: { changes: FileChange[] }) {
  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold">File changes · {changes.length}</h3>
      {changes.map((change) => (
        <Collapsible key={change.path} className="rounded-md border">
          <CollapsibleTrigger className="flex w-full items-center justify-between gap-2 p-2 text-left text-xs">
            <span className="min-w-0 font-mono break-all">{change.path}</span>
            <Badge variant="outline">
              {!change.before ? "Added" : !change.after ? "Removed" : "Changed"}
            </Badge>
          </CollapsibleTrigger>
          <CollapsibleContent className="space-y-2 px-2 pb-2">
            <Raw label="Before" value={content(change.before)} />
            <Raw label="After" value={content(change.after)} />
          </CollapsibleContent>
        </Collapsible>
      ))}
    </div>
  )
}

function usageSummary(usage: unknown): [string, string][] {
  if (!usage || typeof usage !== "object") return []
  const record = usage as Record<string, unknown>
  return [
    ["Input tokens", record.inputTokens ?? record.prompt_tokens],
    ["Output tokens", record.outputTokens ?? record.completion_tokens],
    ["Total tokens", record.totalTokens ?? record.total_tokens],
  ].flatMap(([label, value]) =>
    typeof value === "number"
      ? [[String(label), value.toLocaleString()] as [string, string]]
      : []
  )
}

export function Inspector({
  run,
  selectedEvent,
  onSelectEvent,
  onNote,
}: InspectorProps) {
  const noteId = useId()
  if (!run)
    return (
      <div className="flex min-h-40 items-center justify-center p-6 text-center text-sm text-muted-foreground">
        Run an experiment or select a previous response to inspect its settings,
        requests, tools, and files.
      </div>
    )
  const event = run.events.find((entry) => entry.id === selectedEvent)
  const changes = diffFiles(run.before, run.after)
  return (
    <div className="min-w-0 p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <h2 className="text-sm font-semibold">Run inspector</h2>
        <Badge variant={run.status === "error" ? "destructive" : "secondary"}>
          {run.status}
        </Badge>
        <Badge variant="outline">
          {run.mode === "replay" ? "Recorded tools" : "Live tools"}
        </Badge>
        <span className="text-xs text-muted-foreground">
          {new Date(run.created).toLocaleString()}
        </span>
      </div>
      {run.error && (
        <p role="alert" className="mb-3 text-sm text-destructive">
          {run.error}
        </p>
      )}
      <Tabs defaultValue="overview">
        <TabsList className="w-full">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="trace">Trace</TabsTrigger>
          <TabsTrigger value="request">Request</TabsTrigger>
          <TabsTrigger value="files">Files</TabsTrigger>
        </TabsList>
        <TabsContent value="overview" className="space-y-4 pt-3">
          <div className="space-y-1 text-xs">
            <p>
              <span className="text-muted-foreground">Model </span>
              <span className="font-mono break-all">{run.settings.model}</span>
            </p>
            <p className="text-muted-foreground">
              {run.steps.length} model calls ·{" "}
              {run.events.filter((entry) => entry.type === "tool").length} tool
              calls · {changes.length} changed files
            </p>
            {run.replayOf && (
              <p className="break-all text-muted-foreground">
                Replaying results from {run.replayOf}
              </p>
            )}
          </div>
          <Field className="gap-1.5">
            <FieldLabel htmlFor={noteId}>Research note</FieldLabel>
            <Textarea
              id={noteId}
              value={run.note}
              onChange={(change) => onNote(change.target.value)}
              placeholder="What did this run show?"
              className="min-h-20"
            />
          </Field>
          {run.steps.map((step, index) => (
            <div key={index} className="space-y-2 rounded-md border p-3">
              <div className="flex items-center justify-between gap-2 text-xs">
                <h3 className="font-semibold">Model call {index + 1}</h3>
                <span className="text-muted-foreground">
                  {step.finishReason ?? "In progress"}
                </span>
              </div>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                {[
                  ["Duration", elapsed(step.duration)],
                  ["First token", elapsed(step.firstToken)],
                  ...usageSummary(step.usage),
                ].map(([label, value]) => (
                  <div key={label} className="flex justify-between gap-2">
                    <dt className="text-muted-foreground">{label}</dt>
                    <dd className="font-mono">{value}</dd>
                  </div>
                ))}
              </dl>
              {(step.usage !== undefined || step.metadata !== undefined) && (
                <Collapsible>
                  <CollapsibleTrigger className="text-xs text-muted-foreground underline underline-offset-4">
                    Full usage and timing
                  </CollapsibleTrigger>
                  <CollapsibleContent className="space-y-2 pt-2">
                    <Raw label="Usage" value={step.usage} />
                    <Raw label="Provider metadata" value={step.metadata} />
                  </CollapsibleContent>
                </Collapsible>
              )}
            </div>
          ))}
          {run.sources.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-xs font-semibold">
                Sources · {run.sources.length}
              </h3>
              {run.sources.map((source) => (
                <a
                  key={source.id}
                  href={source.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block rounded-md border p-3 text-xs hover:bg-muted/50"
                >
                  <span className="font-medium">
                    {source.title || source.url}
                  </span>
                  <span className="mt-1 block truncate text-muted-foreground">
                    {source.url}
                  </span>
                  {source.snippet && (
                    <span className="mt-1 block text-muted-foreground">
                      {source.snippet}
                    </span>
                  )}
                </a>
              ))}
            </div>
          )}
          <Collapsible>
            <CollapsibleTrigger className="text-xs font-medium underline underline-offset-4">
              Recorded settings for this run
            </CollapsibleTrigger>
            <CollapsibleContent className="space-y-3 pt-2">
              <p className="text-xs text-muted-foreground">
                Requested controls and inherited configuration were saved at run
                start. These records do not claim every runtime setting was
                resolved.
              </p>
              <Raw label="Requested session controls" value={run.settings} />
              <Raw
                label="Inherited configuration at run start"
                value={run.environment}
              />
            </CollapsibleContent>
          </Collapsible>
        </TabsContent>
        <TabsContent value="trace" className="space-y-3 pt-3">
          <p className="text-xs text-muted-foreground">
            Model output and tool activity in recorded order. Select an event to
            inspect it.
          </p>
          {run.events.length === 0 ? (
            <p className="text-xs text-muted-foreground">
              No events recorded yet.
            </p>
          ) : (
            <div
              className="max-h-52 space-y-1 overflow-auto"
              aria-label="Run events"
            >
              {run.events.map((entry, index) => (
                <Button
                  key={entry.id}
                  variant={entry.id === selectedEvent ? "secondary" : "ghost"}
                  size="sm"
                  className="h-auto w-full justify-start gap-2 py-2 text-left"
                  onClick={() => onSelectEvent(entry.id)}
                  aria-pressed={entry.id === selectedEvent}
                >
                  <span className="font-mono text-xs text-muted-foreground">
                    {index + 1}
                  </span>
                  <span className="truncate text-xs">
                    {entry.tool ?? entry.type}
                  </span>
                  <span className="ml-auto text-xs text-muted-foreground">
                    {entry.replayed ? "replayed" : (entry.state ?? "")}
                  </span>
                </Button>
              ))}
            </div>
          )}
          {event && <EventDetails event={event} />}
        </TabsContent>
        <TabsContent value="request" className="space-y-3 pt-3">
          <Raw label="Input conversation" value={run.input} />
          {run.steps.map((step, index) => (
            <Collapsible
              key={index}
              defaultOpen={index === 0}
              className="rounded-md border"
            >
              <CollapsibleTrigger className="w-full p-3 text-left text-xs font-semibold">
                Model call {index + 1} · request and response
              </CollapsibleTrigger>
              <CollapsibleContent className="space-y-3 px-3 pb-3">
                <Raw label={`Request ${index + 1}`} value={step.request} />
                <Raw label={`Response ${index + 1}`} value={step.response} />
              </CollapsibleContent>
            </Collapsible>
          ))}
          <Raw label="Result conversation" value={run.messages} />
        </TabsContent>
        <TabsContent value="files" className="space-y-3 pt-3">
          <p className="text-xs text-muted-foreground">
            This run started with {run.before.length} filesystem entries and
            ended with {run.after.length}. Historical snapshots belong to this
            run.
          </p>
          {changes.length ? (
            <Changes changes={changes} />
          ) : (
            <p className="text-xs text-muted-foreground">No file changes.</p>
          )}
          <Collapsible>
            <CollapsibleTrigger className="text-xs font-medium underline underline-offset-4">
              Snapshot manifest
            </CollapsibleTrigger>
            <CollapsibleContent className="space-y-3 pt-3">
              <Raw
                label="Before manifest"
                value={run.before.map(({ content: bytes, ...file }) => ({
                  ...file,
                  base64Length: bytes?.length,
                }))}
              />
              <Raw
                label="After manifest"
                value={run.after.map(({ content: bytes, ...file }) => ({
                  ...file,
                  base64Length: bytes?.length,
                }))}
              />
            </CollapsibleContent>
          </Collapsible>
        </TabsContent>
      </Tabs>
    </div>
  )
}
