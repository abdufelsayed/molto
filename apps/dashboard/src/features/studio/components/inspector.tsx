import { useId, useState } from "react"
import { formatJson, JsonView } from "./json-view"
import { Check, ChevronRight, Copy } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Field, FieldLabel } from "@/components/ui/field"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "cn"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
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
  tab: "trace" | "request" | "response" | "usage"
  run?: Run
  selectedEvent?: string
  onSelectEvent: (id: string) => void
  onNote: (note: string) => void
}

function Raw({ value, label }: { value: unknown; label: string }) {
  const [copied, setCopied] = useState(false)
  const [error, setError] = useState("")
  const text = formatJson(value).text
  return (
    <div className="min-w-0">
      <div className="flex items-center justify-between gap-2 py-1.5">
        <span className="text-xs font-medium">{label}</span>
        <Tooltip>
          <TooltipTrigger
            render={
              <Button
                variant="ghost"
                size="icon-xs"
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
              />
            }
          >
            {copied ? <Check /> : <Copy />}
          </TooltipTrigger>
          <TooltipContent>{copied ? "Copied" : `Copy ${label}`}</TooltipContent>
        </Tooltip>
      </div>
      {error && (
        <p role="alert" className="px-3 text-xs text-destructive">
          {error}
        </p>
      )}
      <JsonView value={value} ariaLabel={label} />
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
    <div className="flex flex-col gap-3">
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
    <div className="flex flex-col gap-2">
      <h3 className="text-xs font-semibold">File changes · {changes.length}</h3>
      {changes.map((change) => (
        <Collapsible key={change.path} className="border-b border-border/50">
          <CollapsibleTrigger className="group flex w-full items-center justify-between gap-2 py-2 text-left text-xs hover:bg-muted/40">
            <span className="min-w-0 font-mono break-all">{change.path}</span>
            <Badge variant="outline">
              {!change.before ? "Added" : !change.after ? "Removed" : "Changed"}
            </Badge>
          </CollapsibleTrigger>
          <CollapsibleContent className="flex flex-col gap-2 pb-2">
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

function Disclosure({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <Collapsible className="border-b border-border/50">
      <CollapsibleTrigger className="group flex w-full items-center gap-2 py-2 text-left text-xs font-medium hover:bg-muted/40">
        <ChevronRight className="size-3.5 text-muted-foreground group-data-open:rotate-90" />
        {label}
      </CollapsibleTrigger>
      <CollapsibleContent className="flex flex-col gap-3 pb-3">
        {children}
      </CollapsibleContent>
    </Collapsible>
  )
}

export function Inspector({
  tab,
  run,
  selectedEvent,
  onSelectEvent,
  onNote,
}: InspectorProps) {
  const noteId = useId()
  if (!run)
    return (
      <p className="p-4 text-sm text-muted-foreground">
        Select a response to inspect its run.
      </p>
    )
  const event = run.events.find((entry) => entry.id === selectedEvent)
  const changes = diffFiles(run.before, run.after)
  return (
    <div className="flex min-w-0 flex-col gap-4 p-4">
      {run.error && (
        <p role="alert" className="text-sm text-destructive">
          {run.error}
        </p>
      )}
      {tab === "trace" && (
        <div className="grid gap-4 md:grid-cols-[minmax(180px,1fr)_minmax(0,2fr)]">
          <div
            aria-label="Run events"
            className="flex min-w-0 flex-col gap-0.5"
          >
            {run.events.length === 0 && (
              <p className="text-xs text-muted-foreground">
                No events recorded yet.
              </p>
            )}
            {run.events.map((entry, index) => (
              <Button
                key={entry.id}
                variant="ghost"
                size="sm"
                className={cn(
                  "h-auto w-full justify-start gap-2 py-2 text-left",
                  entry.id === selectedEvent && "bg-muted"
                )}
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
                  {entry.replayed ? "replayed" : entry.state}
                </span>
              </Button>
            ))}
          </div>
          <div className="min-w-0">
            {event ? (
              <EventDetails event={event} />
            ) : (
              <p className="py-2 text-xs text-muted-foreground">
                Select an event to inspect its details.
              </p>
            )}
          </div>
        </div>
      )}
      {tab === "request" && (
        <>
          <Raw label="Input conversation" value={run.input} />
          {run.steps.map((step, index) => (
            <Disclosure key={index} label={`Model call ${index + 1}`}>
              <Raw label={`Request ${index + 1}`} value={step.request} />
            </Disclosure>
          ))}
        </>
      )}
      {tab === "response" && (
        <>
          <Raw label="Result conversation" value={run.messages} />
          {run.steps.map((step, index) => (
            <Disclosure key={index} label={`Model call ${index + 1}`}>
              <Raw label={`Response ${index + 1}`} value={step.response} />
            </Disclosure>
          ))}
        </>
      )}
      {tab === "usage" && (
        <>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
            <span className="font-mono break-all text-foreground">
              {run.settings.model}
            </span>
            <span>{run.status}</span>
            <span>
              {run.mode === "replay" ? "Recorded tools" : "Live tools"}
            </span>
            <span>{new Date(run.created).toLocaleString()}</span>
            <span>
              {run.steps.length} model calls ·{" "}
              {run.events.filter((entry) => entry.type === "tool").length} tool
              calls
            </span>
            {run.replayOf && (
              <span className="break-all">
                Replaying results from {run.replayOf}
              </span>
            )}
          </div>
          {run.steps.map((step, index) => (
            <div
              key={index}
              className="flex flex-col gap-2 border-b border-border/50 pb-3"
            >
              <div className="flex justify-between gap-2 text-xs">
                <h3 className="font-medium">Model call {index + 1}</h3>
                <span className="text-muted-foreground">
                  {step.finishReason ?? "In progress"}
                </span>
              </div>
              <dl className="grid gap-x-6 gap-y-1 text-xs sm:grid-cols-2">
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
                <Disclosure label="Full usage and timing">
                  <Raw label="Usage" value={step.usage} />
                  <Raw label="Provider metadata" value={step.metadata} />
                </Disclosure>
              )}
            </div>
          ))}
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
          <Disclosure label="Recorded settings for this run">
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
          </Disclosure>
          <Disclosure label={`File changes · ${changes.length}`}>
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
            <Disclosure label="Snapshot manifest">
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
            </Disclosure>
          </Disclosure>
          {run.sources.length > 0 && (
            <Disclosure label={`Sources · ${run.sources.length}`}>
              {run.sources.map((source) => (
                <a
                  key={source.id}
                  href={source.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block py-2 text-xs hover:bg-muted/40"
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
            </Disclosure>
          )}
        </>
      )}
    </div>
  )
}
