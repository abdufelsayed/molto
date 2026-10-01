import { useEffect, useState } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import {
  Check,
  ChevronDown,
  Copy,
  FileCode,
  Search,
  Terminal,
  Wrench,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Bubble, BubbleContent } from "@/components/ui/bubble"
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
    <div className="my-2 overflow-hidden rounded-lg border bg-muted/30">
      <div className="flex items-center justify-between border-b px-3 py-1 text-xs text-muted-foreground">
        <span>{language || "text"}</span>
        <div className="flex gap-1">
          <Button
            size="icon-xs"
            variant="ghost"
            aria-label="Copy code"
            onClick={() => {
              void navigator.clipboard.writeText(code).then(() => {
                setCopied(true)
                setTimeout(() => setCopied(false), 1500)
              })
            }}
          >
            {copied ? <Check /> : <Copy />}
          </Button>
          <Button
            size="icon-xs"
            variant="ghost"
            aria-label="Save code to filesystem"
            onClick={() => save(code, language)}
          >
            <FileCode />
          </Button>
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
function Activity({
  event,
  selected,
  inspect,
}: {
  event: RunEvent
  selected: boolean
  inspect: () => void
}) {
  const Icon =
    event.tool === "bash"
      ? Terminal
      : event.tool === "web_search"
        ? Search
        : Wrench
  return (
    <Button
      variant={selected ? "secondary" : "outline"}
      className="h-auto w-full justify-start gap-2 px-3 py-2 text-left"
      onClick={inspect}
    >
      <Icon className="size-3.5 shrink-0" />
      <span className="flex-1 truncate font-mono text-xs">
        {event.tool}
        {event.input &&
        typeof event.input === "object" &&
        "command" in event.input
          ? ` · ${String(event.input.command)}`
          : ""}
      </span>
      <Badge
        variant={event.state === "error" ? "destructive" : "outline"}
        className="text-[10px]"
      >
        {event.replayed ? "replayed" : event.state}
      </Badge>
      {event.duration !== undefined && (
        <span className="text-xs text-muted-foreground">
          {event.duration.toFixed(2)}s
        </span>
      )}
    </Button>
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
  return (
    <div className="flex flex-col gap-3">
      {run.events.map((event) => {
        if (event.type === "text")
          return (
            <Markdown
              key={event.id}
              text={event.text ?? ""}
              sources={run.sources}
              save={save}
            />
          )
        if (event.type === "reasoning")
          return (
            <Collapsible
              key={event.id}
              defaultOpen={run.status === "running"}
              className="rounded-lg border bg-muted/20"
            >
              <CollapsibleTrigger
                render={
                  <Button
                    variant="ghost"
                    size="sm"
                    className="w-full justify-start"
                  />
                }
              >
                <ChevronDown />
                <span>Model reasoning</span>
                <span className="ml-auto text-xs text-muted-foreground">
                  Step {event.step}
                </span>
              </CollapsibleTrigger>
              <CollapsibleContent className="px-3 pb-3 text-muted-foreground">
                <Markdown text={event.text ?? ""} save={save} />
              </CollapsibleContent>
            </Collapsible>
          )
        if (event.type === "tool")
          return (
            <Activity
              key={event.id}
              event={event}
              selected={selectedEvent === event.id}
              inspect={() => inspect(event.id)}
            />
          )
        return (
          <Bubble
            key={event.id}
            variant={event.type === "error" ? "destructive" : "muted"}
          >
            <BubbleContent className="text-xs">{event.text}</BubbleContent>
          </Bubble>
        )
      })}
      {run.status === "running" && (
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
            <Search />
            {run.sources.length} sources
            <ChevronDown />
          </CollapsibleTrigger>
          <CollapsibleContent className="flex flex-col gap-2 py-2">
            {run.sources.map((source) => (
              <a
                key={source.id}
                href={source.url}
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-lg border p-3 text-xs hover:bg-muted"
              >
                <span className="mr-2 text-muted-foreground">
                  [{source.id}]
                </span>
                {source.title}
                <span className="mt-1 block truncate text-muted-foreground">
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
