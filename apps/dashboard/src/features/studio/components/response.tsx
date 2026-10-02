import { useEffect, useState } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import {
  Check,
  CircleAlert,
  LoaderCircle,
  PanelRightOpen,
  ChevronDown,
  Copy,
  FileCode,
  Search,
} from "lucide-react"
import { IconAction } from "./icon-action"
import { formatJson, JsonView } from "./json-view"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card"
import type { Run, RunEvent, Source } from "../agent/types"

function CodeBlock({
  code,
  language,
  save,
}: {
  code: string
  language: string
  save: (code: string, language: string) => void
}) {
  const [html, setHtml] = useState("")
  const [copied, setCopied] = useState(false)
  useEffect(() => {
    let live = true
    const timer = setTimeout(() => {
      void import("shiki/bundle/web")
        .then(({ codeToHtml }) =>
          codeToHtml(code, {
            lang: language || "text",
            themes: { light: "github-light", dark: "github-dark" },
          })
        )
        .then((value) => {
          if (live) setHtml(value)
        })
        .catch(() => {
          if (live) setHtml("")
        })
    }, 150)
    return () => {
      live = false
      clearTimeout(timer)
    }
  }, [code, language])
  return (
    <div className="group/code my-3 overflow-hidden rounded-md bg-muted/30">
      <div className="flex items-center justify-between px-3 py-1 text-xs text-muted-foreground">
        <span>{language || "text"}</span>
        <div className="flex gap-1">
          <IconAction
            label="Copy code"
            onClick={() => {
              void navigator.clipboard.writeText(code).then(() => {
                setCopied(true)
                setTimeout(() => setCopied(false), 1500)
              })
            }}
          >
            {copied ? <Check /> : <Copy />}
          </IconAction>
          <IconAction
            label="Save code to filesystem"
            onClick={() => save(code, language)}
          >
            <FileCode />
          </IconAction>
        </div>
      </div>
      {html ? (
        <div
          className="studio-code overflow-auto p-3 text-xs"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      ) : (
        <pre className="overflow-auto p-3 text-xs">
          <code>{code}</code>
        </pre>
      )}
    </div>
  )
}
type CitationNode = {
  type: string
  value?: string
  url?: string
  children?: CitationNode[]
}
function citationPlugin(sources: Source[]) {
  return () => (tree: CitationNode) => {
    const visit = (node: CitationNode) => {
      if (["code", "inlineCode", "link", "linkReference"].includes(node.type))
        return
      if (!node.children) return
      node.children = node.children.flatMap((child) => {
        if (child.type !== "text" || !child.value) {
          visit(child)
          return [child]
        }
        const parts: CitationNode[] = []
        let offset = 0
        for (const match of child.value.matchAll(/\[(\d+)\]/g)) {
          const source = sources.find((item) => item.id === match[1])
          if (!source) continue
          if (match.index > offset)
            parts.push({
              type: "text",
              value: child.value.slice(offset, match.index),
            })
          parts.push({
            type: "link",
            url: source.url,
            children: [{ type: "text", value: match[0] }],
          })
          offset = match.index + match[0].length
        }
        if (!offset) return [child]
        if (offset < child.value.length)
          parts.push({ type: "text", value: child.value.slice(offset) })
        return parts
      })
    }
    visit(tree)
  }
}
export function Markdown({
  text,
  sources = [],
  save,
}: {
  text: string
  sources?: Source[]
  save: (code: string, language: string) => void
}) {
  return (
    <div className="studio-markdown min-w-0 text-sm leading-7">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, citationPlugin(sources)]}
        components={{
          pre: ({ children }) => <>{children}</>,
          code: ({ className, children }) => {
            const value =
              typeof children === "string"
                ? children
                : Array.isArray(children)
                  ? children
                      .filter((child) => typeof child === "string")
                      .join("")
                  : ""
            return className || value.includes("\n") ? (
              <CodeBlock
                code={value.replace(/\n$/, "")}
                language={className?.replace("language-", "") ?? ""}
                save={save}
              />
            ) : (
              <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">
                {children}
              </code>
            )
          },
          img: ({ alt }) => (
            <span className="text-xs text-muted-foreground">
              [Image: {alt ?? "external image"}]
            </span>
          ),
          a: ({ href, children }) => {
            const source = sources.find((s) => s.url === href)
            const link = (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary underline underline-offset-4"
              >
                {children}
              </a>
            )
            return source ? (
              <HoverCard>
                <HoverCardTrigger render={link} />
                <HoverCardContent>
                  <p className="text-sm font-medium">{source.title}</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {source.snippet ?? source.url}
                  </p>
                </HoverCardContent>
              </HoverCard>
            ) : (
              link
            )
          },
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  )
}
function eventSummary(event: RunEvent): string {
  if (event.type === "reasoning") return "Model reasoning"
  if (event.type === "text") return "Intermediate response"
  if (event.type !== "tool") return event.text ?? event.type
  if (event.input && typeof event.input === "object") {
    const input = event.input as Record<string, unknown>
    for (const key of ["command", "query", "path", "url", "question"]) {
      if (typeof input[key] === "string")
        return input[key].replace(/\s+/g, " ").trim()
    }
  }
  return event.tool ?? "Tool call"
}

const disclosureClassName =
  "group/disclosure inline-flex min-w-0 items-center gap-1.5 py-1 text-left text-xs text-muted-foreground transition-colors hover:text-foreground focus-visible:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
const chevronClassName =
  "size-3 shrink-0 -rotate-90 transition-transform group-aria-expanded/disclosure:rotate-0 motion-reduce:transition-none"
const detailActions =
  "[@media(hover:hover)]:opacity-0 group-hover/detail:opacity-100 group-focus-within/detail:opacity-100"

function ResultDisclosure({
  label,
  value,
  defaultOpen = true,
}: {
  label: string
  value: unknown
  defaultOpen?: boolean
}) {
  const [copied, setCopied] = useState(false)
  const [error, setError] = useState("")
  if (value === undefined) return null
  return (
    <Collapsible defaultOpen={defaultOpen}>
      <div className="group/detail flex items-center gap-1">
        <CollapsibleTrigger className={disclosureClassName}>
          <ChevronDown className={chevronClassName} />
          {label}
        </CollapsibleTrigger>
        <IconAction
          label={copied ? `Copied ${label}` : `Copy ${label}`}
          className={detailActions}
          onClick={() => {
            void navigator.clipboard
              .writeText(formatJson(value).text)
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
          {copied ? <Check /> : <Copy />}
        </IconAction>
      </div>
      <CollapsibleContent className="pl-4">
        {error && (
          <p role="alert" className="text-xs text-destructive">
            {error}
          </p>
        )}
        <JsonView value={value} ariaLabel={label} />
      </CollapsibleContent>
    </Collapsible>
  )
}

function Activity({
  event,
  active,
  selected,
  inspect,
  save,
  sources,
}: {
  event: RunEvent
  active: boolean
  selected: boolean
  inspect: () => void
  save: (code: string, language: string) => void
  sources: Source[]
}) {
  const [manual, setManual] = useState<boolean>()
  const open = manual ?? active
  const failed =
    event.type === "error" ||
    event.state === "error" ||
    event.state === "denied"
  const state = event.replayed
    ? "replayed"
    : (event.state ?? (active ? "streaming" : ""))
  return (
    <Collapsible open={open} onOpenChange={setManual} className="min-w-0">
      <div
        className="group/activity flex min-w-0 items-center gap-1 text-muted-foreground hover:text-foreground data-[selected=true]:text-foreground"
        data-slot="activity-step"
        data-selected={selected}
      >
        <CollapsibleTrigger
          className={`${disclosureClassName} flex-1 group-hover/activity:text-foreground group-data-[selected=true]/activity:text-foreground`}
        >
          <ChevronDown className={chevronClassName} />
          <span className="min-w-0 flex-1 truncate text-xs">
            {event.type === "tool" && (
              <span className="mr-2 font-medium">{event.tool}</span>
            )}
            {eventSummary(event)}
          </span>
          {failed ? (
            <CircleAlert className="size-3 shrink-0 text-destructive" />
          ) : (
            active && (
              <LoaderCircle className="size-3 shrink-0 animate-spin motion-reduce:animate-none" />
            )
          )}
          {state && <span className="shrink-0 text-xs">{state}</span>}
          {event.duration !== undefined && (
            <span className="shrink-0 text-xs">
              {event.duration.toFixed(2)}s
            </span>
          )}
        </CollapsibleTrigger>
        <IconAction
          label={`Inspect ${event.tool ?? event.type}`}
          onClick={inspect}
          className="group-focus-within/activity:opacity-100 group-hover/activity:opacity-100 hover:bg-transparent dark:hover:bg-transparent [@media(hover:hover)]:opacity-0"
        >
          <PanelRightOpen />
        </IconAction>
      </div>
      <CollapsibleContent className="flex min-w-0 flex-col gap-1 pb-2 pl-4">
        {event.type === "reasoning" || event.type === "text" ? (
          <Markdown text={event.text ?? ""} save={save} sources={sources} />
        ) : (
          <>
            {event.text && event.type !== "tool" && (
              <p
                className={
                  failed
                    ? "text-xs text-destructive"
                    : "text-xs text-muted-foreground"
                }
              >
                {event.text}
              </p>
            )}
            <ResultDisclosure
              label="Arguments"
              value={
                event.input === undefined && event.type === "tool"
                  ? event.text || undefined
                  : event.input
              }
            />
            <ResultDisclosure
              label={failed ? "Error" : "Result"}
              value={event.output}
            />
            <ResultDisclosure
              label="Model-visible result (limited)"
              value={event.modelOutput}
              defaultOpen={false}
            />
            {!!event.changes?.length && (
              <ResultDisclosure
                label={`File changes (${event.changes.length})`}
                value={event.changes}
                defaultOpen={false}
              />
            )}
          </>
        )}
      </CollapsibleContent>
    </Collapsible>
  )
}

export function ResponseContent({
  run,
  selectedEvent,
  inspect,
  save,
}: {
  run: Run
  selectedEvent?: string
  inspect: (event?: string) => void
  save: (code: string, language: string) => void
}) {
  const [chainState, setChainState] = useState<Record<string, boolean>>({})
  const running = run.status === "running"
  const chainOpen = chainState[run.id] ?? running
  const technical = (event: RunEvent) =>
    event.type === "tool" || event.type === "reasoning"
  let lastTechnical = -1
  for (let index = 0; index < run.events.length; index++)
    if (technical(run.events[index]!)) lastTechnical = index
  const chain = run.events.slice(0, lastTechnical + 1)
  const answer = run.events.slice(lastTechnical + 1)
  const tools = chain.filter((event) => event.type === "tool")
  const reasoning = chain.some((event) => event.type === "reasoning")
  const failed = tools.filter(
    (event) => event.state === "error" || event.state === "denied"
  ).length
  return (
    <div className="flex min-w-0 flex-col gap-3">
      {chain.length > 0 && (
        <Collapsible
          open={chainOpen}
          onOpenChange={(open) =>
            setChainState((previous) => ({ ...previous, [run.id]: open }))
          }
        >
          <CollapsibleTrigger className={disclosureClassName}>
            <ChevronDown className={chevronClassName} />
            <span>Activity</span>
            {running ? (
              <LoaderCircle className="size-3 animate-spin motion-reduce:animate-none" />
            ) : null}
            <span className="text-xs">
              {[
                reasoning ? "Reasoning" : "",
                tools.length
                  ? `${tools.length} tool ${tools.length === 1 ? "call" : "calls"}`
                  : "",
                failed ? `${failed} failed` : "",
              ]
                .filter(Boolean)
                .join(" · ")}
            </span>
          </CollapsibleTrigger>
          <CollapsibleContent className="flex min-w-0 flex-col gap-0.5 pl-4">
            {chain.map((event, index) => (
              <Activity
                key={`${run.id}:${event.id}`}
                event={event}
                active={
                  running &&
                  (event.state === "running" ||
                    event.state === "waiting" ||
                    (event.type === "reasoning" && index === chain.length - 1))
                }
                selected={selectedEvent === event.id}
                inspect={() => inspect(event.id)}
                save={save}
                sources={run.sources}
              />
            ))}
          </CollapsibleContent>
        </Collapsible>
      )}
      {answer.map((event) =>
        event.type === "text" ? (
          <Markdown
            key={event.id}
            text={event.text ?? ""}
            sources={run.sources}
            save={save}
          />
        ) : (
          <p
            key={event.id}
            className={
              event.type === "error"
                ? "text-xs text-destructive"
                : "text-xs text-muted-foreground"
            }
          >
            {event.text}
          </p>
        )
      )}
      {running && (
        <span
          role="status"
          className="w-fit shimmer text-xs text-muted-foreground"
        >
          Generating
        </span>
      )}
      {run.sources.length > 0 && (
        <Collapsible>
          <CollapsibleTrigger className={disclosureClassName}>
            <ChevronDown className={chevronClassName} />
            <Search className="size-3" />
            {run.sources.length} sources
          </CollapsibleTrigger>
          <CollapsibleContent className="flex flex-col gap-1 py-1">
            {run.sources.map((source) => (
              <a
                key={source.id}
                href={source.url}
                target="_blank"
                rel="noopener noreferrer"
                className="py-1 pl-4 text-xs text-muted-foreground transition-colors hover:text-foreground focus-visible:text-foreground"
              >
                <span className="mr-2 text-muted-foreground">
                  [{source.id}]
                </span>
                {source.title}
                <span className="mt-0.5 block truncate text-muted-foreground">
                  {source.url}
                </span>
              </a>
            ))}
          </CollapsibleContent>
        </Collapsible>
      )}
    </div>
  )
}
