import { z } from "zod"
import type { ModelMessage } from "ai"

export const toolNames = [
  "read",
  "write",
  "edit",
  "bash",
  "web_search",
  "fetch_url",
  "ask_question",
] as const
export type ToolName = (typeof toolNames)[number]
export const engines = [
  "brave",
  "duckduckgo",
  "grokipedia",
  "mojeek",
  "wikipedia",
  "yahoo",
  "yandex",
] as const
export const searchSchema = z.object({
  provider: z
    .enum(["ddgs", "ddgs_custom", "duckduckgo", "brave", "searxng"])
    .default("ddgs"),
  engines: z.array(z.enum(engines)).default([...engines]),
  maxResults: z.number().int().min(1).max(10).default(3),
  content: z.enum(["snippet", "full"]).default("snippet"),
  maxChars: z.number().int().min(100).max(100_000).default(20_000),
  truncate: z.boolean().default(true),
})
export const settingsSchema = z.object({
  model: z.string().max(512).default(""),
  system: z.string().max(200_000).default(""),
  tools: z.array(z.enum(toolNames)).default([...toolNames]),
  temperature: z.number().min(0).max(5).optional(),
  top_p: z.number().min(0).max(1).optional(),
  top_k: z.number().int().min(0).optional(),
  min_p: z.number().min(0).max(1).optional(),
  max_tokens: z.number().int().positive().optional(),
  repetition_penalty: z.number().positive().optional(),
  repetition_context_size: z.number().int().positive().optional(),
  presence_penalty: z.number().min(-2).max(2).optional(),
  frequency_penalty: z.number().min(-2).max(2).optional(),
  xtc_probability: z.number().min(0).max(1).optional(),
  xtc_threshold: z.number().min(0).max(1).optional(),
  seed: z.number().int().optional(),
  stop: z.array(z.string()).default([]),
  enable_thinking: z.boolean().optional(),
  reasoning_effort: z.union([z.string(), z.number()]).optional(),
  thinking_budget: z.number().int().min(0).optional(),
  specprefill: z.boolean().optional(),
  specprefill_keep_pct: z.number().min(0.1).max(0.5).optional(),
  specprefill_threshold: z.number().int().positive().optional(),
  template: z.string().default(""),
  responseFormat: z.string().default(""),
  structuredOutput: z.string().default(""),
  toolChoice: z
    .enum(["auto", "none", "required", ...toolNames])
    .default("auto"),
  execution: z.enum(["auto", "confirm"]).default("auto"),
  maxSteps: z.number().int().min(1).max(100).default(10),
  toolTimeout: z.number().int().min(100).max(300_000).default(30_000),
  outputLimit: z.number().int().min(100).max(1_000_000).default(20_000),
  cwd: z.string().startsWith("/").default("/workspace"),
  env: z.record(z.string(), z.string()).default({}),
  readOnly: z.boolean().default(false),
  network: z.boolean().default(false),
  maxFileBytes: z.number().int().min(1024).max(100_000_000).default(20_000_000),
  maxCommands: z.number().int().min(1).max(1_000_000).default(10_000),
  maxLoopIterations: z.number().int().min(1).max(1_000_000).default(10_000),
  search: searchSchema.default(() => searchSchema.parse({})),
})
export type AgentSettings = z.infer<typeof settingsSchema>
export const defaults = (): AgentSettings => settingsSchema.parse({})
export const fileSchema = z.object({
  path: z.string().startsWith("/").max(4096),
  type: z.enum(["file", "directory", "symlink"]),
  content: z.string().optional(), // file bytes encoded as base64
  target: z.string().optional(),
  mode: z.number().int().optional(),
})
export type VirtualFile = z.infer<typeof fileSchema>
export type Snapshot = VirtualFile[]
export type Source = {
  id: string
  title: string
  url: string
  snippet?: string
}
export type FileChange = {
  path: string
  before?: VirtualFile
  after?: VirtualFile
}
export type RunEvent = {
  id: string
  at: number
  step: number
  type: "text" | "reasoning" | "tool" | "error" | "status"
  text?: string
  callId?: string
  tool?: string
  input?: unknown
  output?: unknown
  modelOutput?: unknown
  state?: "running" | "waiting" | "done" | "error" | "denied"
  duration?: number
  replayed?: boolean
  changes?: FileChange[]
  sources?: Source[]
}
export type ModelStep = {
  request: unknown
  response?: unknown
  usage?: unknown
  metadata?: unknown
  finishReason?: string
  duration?: number
  firstToken?: number
}
export type Run = {
  id: string
  created: number
  settings: AgentSettings
  input: ModelMessage[]
  messages: ModelMessage[]
  events: RunEvent[]
  steps: ModelStep[]
  sources: Source[]
  before: Snapshot
  after: Snapshot
  status: "running" | "complete" | "stopped" | "error" | "limit" | "interrupted"
  mode: "live" | "replay"
  replayOf?: string
  environment?: unknown
  prompt?: { text: string; attachments?: string[] }
  error?: string
  note: string
}
export type Turn = {
  id: string
  text: string
  runIds: string[]
  selected?: string
  attachments?: string[]
}
export type StudioSession = {
  version: 1
  id: string
  title: string
  updated: number
  settings: AgentSettings
  turns: Turn[]
  runs: Run[]
  files: Snapshot
  draft: string
  branches?: Array<{
    id: string
    title: string
    turns: Turn[]
    files: Snapshot
  }>
}
export function closePendingTools(
  messages: ModelMessage[],
  reason = "Run ended before this tool call completed."
) {
  const pending = new Map<string, string>()
  for (const message of messages) {
    if (message.role === "assistant" && Array.isArray(message.content)) {
      for (const part of message.content)
        if (part.type === "tool-call")
          pending.set(part.toolCallId, part.toolName)
    } else if (message.role === "tool") {
      for (const part of message.content)
        if (part.type === "tool-result") pending.delete(part.toolCallId)
    }
  }
  const output = { ok: false, error: reason }
  const completed: ModelMessage[] = [...messages]
  for (const [toolCallId, toolName] of pending)
    completed.push({
      role: "tool",
      content: [
        {
          type: "tool-result",
          toolCallId,
          toolName,
          output: { type: "json", value: output },
        },
      ],
    })
  return { messages: completed, unresolved: [...pending.keys()], output }
}
export function newSession(settings = defaults()): StudioSession {
  return {
    version: 1,
    id: crypto.randomUUID(),
    title: "Untitled experiment",
    updated: Date.now(),
    settings,
    turns: [],
    runs: [],
    files: [],
    draft: "",
  }
}
export function diffFiles(before: Snapshot, after: Snapshot): FileChange[] {
  const a = new Map(before.map((f) => [f.path, f]))
  const b = new Map(after.map((f) => [f.path, f]))
  return [...new Set([...a.keys(), ...b.keys()])]
    .sort()
    .flatMap((path) =>
      JSON.stringify(a.get(path)) === JSON.stringify(b.get(path))
        ? []
        : [{ path, before: a.get(path), after: b.get(path) }]
    )
}
export function modelBody(settings: AgentSettings): Record<string, unknown> {
  const body: Record<string, unknown> = {
    include_mcp_tools: false,
    sampling_override: true,
  }
  for (const key of [
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "max_tokens",
    "repetition_penalty",
    "repetition_context_size",
    "presence_penalty",
    "frequency_penalty",
    "xtc_probability",
    "xtc_threshold",
    "seed",
    "enable_thinking",
    "reasoning_effort",
    "thinking_budget",
    "specprefill",
    "specprefill_keep_pct",
    "specprefill_threshold",
  ] as const)
    if (settings[key] !== undefined) body[key] = settings[key]
  if (settings.stop.length) body.stop = settings.stop
  for (const [key, value] of [
    ["chat_template_kwargs", settings.template],
    ["response_format", settings.responseFormat],
    ["structured_outputs", settings.structuredOutput],
  ]) {
    if (value?.trim()) {
      const parsed: unknown = JSON.parse(value)
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed))
        throw new Error(`${key} must be a JSON object.`)
      body[key!] = parsed
    }
  }
  return body
}
