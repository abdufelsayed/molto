import { useEffect, useState } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import {
  Check,
  Brain,
  CircleAlert,
  FileText,
  Globe,
  ListTree,
  LoaderCircle,
  MessageCircle,
  PanelRightOpen,
  ChevronDown,
  Copy,
  FileCode,
  Search,
  Terminal,
  Wrench,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { IconAction } from "./icon-action"
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
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
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

function ResultDisclosure({ label, value }: { label: string; value: unknown }) {
  if (value === undefined) return null
  const text =
    typeof value === "string" ? value : JSON.stringify(value, null, 2)
  return (
    <Collapsible>
      <CollapsibleTrigger
        render={<Button variant="ghost" size="sm" className="justify-start" />}
      >
        <ChevronDown data-icon="inline-start" />
        {label}
      </CollapsibleTrigger>
      <CollapsibleContent>
        <pre className="max-h-80 overflow-auto py-2 pl-3 font-mono text-xs leading-relaxed break-all whitespace-pre-wrap">
          {text}
        </pre>
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
  const Icon =
    event.type === "reasoning"
      ? Brain
      : event.type === "text"
        ? MessageCircle
        : event.tool === "bash"
          ? Terminal
          : event.tool === "web_search"
            ? Search
            : event.tool === "fetch_url"
              ? Globe
              : ["read", "write", "edit"].includes(event.tool ?? "")
                ? FileText
                : event.tool === "ask_question"
                  ? MessageCircle
                  : Wrench
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
        className="group flex min-w-0 items-center gap-1 rounded-md hover:bg-muted/50 data-[selected=true]:bg-muted/50"
        data-selected={selected}
      >
        <CollapsibleTrigger
          render={
            <Button
              variant="ghost"
              size="sm"
              className="h-auto min-w-0 flex-1 justify-start py-2 text-left"
            />
          }
        >
          <Icon data-icon="inline-start" />
          <span className="min-w-0 flex-1 truncate text-xs">
            {event.type === "tool" && (
              <span className="mr-2 font-medium">{event.tool}</span>
            )}
            {eventSummary(event)}
          </span>
          {failed ? (
            <CircleAlert data-icon="inline-end" />
          ) : (
            active && (
              <LoaderCircle
                data-icon="inline-end"
                className="animate-spin motion-reduce:animate-none"
              />
            )
          )}
          {state && (
            <span className="text-xs text-muted-foreground">{state}</span>
          )}
          {event.duration !== undefined && (
            <span className="text-xs text-muted-foreground">
              {event.duration.toFixed(2)}s
            </span>
          )}
          <ChevronDown data-icon="inline-end" />
        </CollapsibleTrigger>
        <Tooltip>
          <TooltipTrigger
            render={
              <Button
                variant="ghost"
                size="icon-xs"
                aria-label={`Inspect ${event.tool ?? event.type}`}
                onClick={inspect}
              />
            }
          >
            <PanelRightOpen />
          </TooltipTrigger>
          <TooltipContent>Inspect {event.tool ?? event.type}</TooltipContent>
        </Tooltip>
      </div>
      <CollapsibleContent className="flex min-w-0 flex-col gap-1 py-1 pl-5">
        {event.type === "reasoning" || event.type === "text" ? (
          <Markdown text={event.text ?? ""} save={save} sources={sources} />
        ) : (
          <>
            {event.text && (
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
            <ResultDisclosure label="Arguments" value={event.input} />
            <ResultDisclosure
              label={failed ? "Error" : "Result"}
              value={event.output}
            />
            <ResultDisclosure
              label="Model-visible result (limited)"
              value={event.modelOutput}
            />
            {!!event.changes?.length && (
              <p className="px-2 py-1 text-xs text-muted-foreground">
                {event.changes.map((change) => change.path).join(", ")}
              </p>
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
          <CollapsibleTrigger
            render={
              <Button variant="ghost" size="sm" className="justify-start" />
            }
          >
            {running ? (
              <LoaderCircle
                data-icon="inline-start"
                className="animate-spin motion-reduce:animate-none"
              />
            ) : (
              <ListTree data-icon="inline-start" />
            )}
            <span>Activity</span>
            <span className="text-xs text-muted-foreground">
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
            <ChevronDown data-icon="inline-end" />
          </CollapsibleTrigger>
          <CollapsibleContent className="flex min-w-0 flex-col gap-1 py-1">
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
          <CollapsibleTrigger render={<Button variant="ghost" size="sm" />}>
            <Search data-icon="inline-start" />
            {run.sources.length} sources
            <ChevronDown data-icon="inline-end" />
          </CollapsibleTrigger>
          <CollapsibleContent className="flex flex-col gap-1 py-1">
            {run.sources.map((source) => (
              <a
                key={source.id}
                href={source.url}
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-md px-2 py-1.5 text-xs hover:bg-muted/50"
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
