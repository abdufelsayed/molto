import { streamText } from "ai"
import type { ModelMessage } from "ai"
import { createOpenAICompatible } from "@ai-sdk/openai-compatible"
import {
  closePendingTools,
  diffFiles,
  modelBody,
  settingsSchema,
} from "./types"
import type {
  AgentSettings,
  Run,
  RunEvent,
  Snapshot,
  Source,
  ToolName,
} from "./types"
import { enabledTools, toolDefinitions } from "./tools"
import { SandboxClient } from "./worker-client"

export type Interaction = { event: RunEvent; kind: "confirm" | "question" }
export type RunOptions = {
  settings: AgentSettings
  messages: ModelMessage[]
  files: Snapshot
  environment?: unknown
  signal: AbortSignal
  replay?: Run
  onUpdate: (run: Run) => void
  interact: (interaction: Interaction, signal: AbortSignal) => Promise<string>
  // Composition owns the transport; the agent knows only an OpenAI API and tools.
  modelUrl: string
  web: (
    name: "web_search" | "fetch_url",
    input: unknown,
    settings: AgentSettings,
    signal: AbortSignal
  ) => Promise<unknown>
}
function key(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(key).join(",")}]`
  if (value && typeof value === "object")
    return `{${Object.entries(value)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([k, v]) => `${JSON.stringify(k)}:${key(v)}`)
      .join(",")}}`
  return JSON.stringify(value) ?? "null"
}
function sourcesFrom(result: unknown): Array<Omit<Source, "id">> {
  if (!result || typeof result !== "object") return []
  const data = result as {
    results?: unknown[]
    url?: unknown
    content?: unknown
  }
  const rows =
    data.results ??
    (typeof data.url === "string" && typeof data.content === "string"
      ? [{ url: data.url, title: data.url }]
      : [])
  return rows.flatMap((row) => {
    if (!row || typeof row !== "object") return []
    const source = row as { url?: unknown; title?: unknown; snippet?: unknown }
    if (typeof source.url !== "string" || !/^https?:\/\//.test(source.url))
      return []
    return [
      {
        url: source.url,
        title: typeof source.title === "string" ? source.title : source.url,
        snippet:
          typeof source.snippet === "string" ? source.snippet : undefined,
      },
    ]
  })
}
export function boundedOutput(output: unknown, limit: number): unknown {
  const text = JSON.stringify(output) ?? "null"
  if (text.length <= limit) return output
  const wrapper = {
    truncated: true,
    originalCharacters: text.length,
    content: "",
  }
  let low = 0
  let high = Math.max(0, limit - JSON.stringify(wrapper).length)
  while (low < high) {
    const mid = Math.ceil((low + high) / 2)
    wrapper.content = text.slice(0, mid)
    if (JSON.stringify(wrapper).length <= limit) low = mid
    else high = mid - 1
  }
  wrapper.content = text.slice(0, low)
  return wrapper
}
export async function runAgent(options: RunOptions): Promise<Run> {
  const settings = settingsSchema.parse(options.settings)
  if (!settings.model) throw new Error("Select a local model.")
  const overrides = modelBody(settings)
  if (
    !settings.tools.includes(settings.toolChoice as ToolName) &&
    !["auto", "none", "required"].includes(settings.toolChoice)
  )
    throw new Error("The selected tool choice is disabled.")
  if (!settings.tools.length && settings.toolChoice === "required")
    throw new Error("Required tool choice needs an enabled tool.")
  const run: Run = {
    id: crypto.randomUUID(),
    created: Date.now(),
    settings: structuredClone(settings),
    input: structuredClone(options.messages),
    messages: structuredClone(options.messages),
    events: [],
    steps: [],
    sources: [],
    before: structuredClone(options.files),
    after: structuredClone(options.files),
    status: "running",
    mode: options.replay ? "replay" : "live",
    replayOf: options.replay?.id,
    environment: options.environment,
    note: "",
  }
  // Snapshots and tool payloads are immutable. Copy changing containers without
  // cloning megabytes of attached files for each streamed token.
  const update = () =>
    options.onUpdate({
      ...run,
      events: run.events.map((event) => ({ ...event })),
      steps: run.steps.map((step) => ({ ...step })),
      messages: [...run.messages],
      sources: [...run.sources],
    })
  const network = async (url: string, networkSignal: AbortSignal) => {
    const data = (await options.web(
      "fetch_url",
      { url },
      settings,
      AbortSignal.any([options.signal, networkSignal])
    )) as { content?: string; url?: string; ok?: boolean; error?: string }
    if (!data.content || data.ok === false)
      throw new Error(data.error ?? "URL fetch failed.")
    // Browser curl uses this same broker; record provenance and network activity.
    const event: RunEvent = {
      id: crypto.randomUUID(),
      at: Date.now(),
      step: run.steps.length,
      type: "tool",
      tool: "fetch_url",
      input: { url },
      output: addSources(data),
      state: "done",
    }
    run.events.push(event)
    update()
    return { content: data.content, url: data.url ?? url }
  }
  const sandbox = new SandboxClient(settings, network)
  const addSources = (output: unknown): unknown => {
    for (const source of sourcesFrom(output))
      if (!run.sources.some((s) => s.url === source.url))
        run.sources.push({ id: String(run.sources.length + 1), ...source })
    if (!output || typeof output !== "object") return output
    const citation = (url: unknown) => {
      const source = run.sources.find((s) => s.url === url)
      return source ? { source_id: source.id, citation: `[${source.id}]` } : {}
    }
    const data = output as Record<string, unknown>
    return {
      ...data,
      ...citation(data.url),
      ...(Array.isArray(data.results)
        ? {
            results: data.results.map((row: unknown) =>
              row && typeof row === "object"
                ? { ...row, ...citation((row as { url?: string }).url) }
                : row
            ),
          }
        : {}),
    }
  }
  const replayCalls =
    options.replay?.events.filter((e) => e.type === "tool" && e.callId) ?? []
  let replayIndex = 0
  try {
    update()
    await sandbox.restore(options.files, options.signal)
    run.before = structuredClone(sandbox.files)
    run.after = structuredClone(sandbox.files)
    for (let step = 0; step < settings.maxSteps; step++) {
      options.signal.throwIfAborted()
      const started = performance.now()
      const chunks: unknown[] = []
      const record = {
        request: undefined as unknown,
        response: { chunks } as unknown,
        usage: undefined as unknown,
        metadata: undefined as unknown,
        finishReason: undefined as string | undefined,
        duration: undefined as number | undefined,
        firstToken: undefined as number | undefined,
      }
      run.steps.push(record)
      const provider = createOpenAICompatible({
        name: "molto",
        baseURL: options.modelUrl,
        includeUsage: true,
        transformRequestBody: (body) => {
          const choice = settings.tools.length
            ? ["auto", "none", "required"].includes(settings.toolChoice)
              ? settings.toolChoice
              : { type: "function", function: { name: settings.toolChoice } }
            : "none"
          const request = { ...body, ...overrides, tool_choice: choice }
          record.request = structuredClone(request)
          update()
          return request
        },
        metadataExtractor: {
          extractMetadata: async () => undefined,
          createStreamExtractor: () => ({
            processChunk: (chunk) => {
              chunks.push(chunk)
            },
            buildMetadata: () => undefined,
          }),
        },
      })
      const stream = streamText({
        model: provider.chatModel(settings.model),
        instructions: settings.system.trim() ? settings.system : undefined,
        messages: run.messages,
        tools: enabledTools(settings),
        abortSignal: options.signal,
        maxRetries: 0,
      })
      const calls: Array<{
        id: string
        name: string
        input: unknown
        invalid?: boolean
        error?: unknown
      }> = []
      try {
        for await (const part of stream.stream) {
          options.signal.throwIfAborted()
          if (
            ["text-delta", "reasoning-delta", "tool-input-start"].includes(
              part.type
            ) &&
            record.firstToken === undefined
          )
            record.firstToken = (performance.now() - started) / 1000
          if (part.type === "text-delta" || part.type === "reasoning-delta") {
            const type = part.type === "text-delta" ? "text" : "reasoning"
            const previous = run.events.at(-1)
            if (previous?.type === type && previous.step === step + 1)
              previous.text = (previous.text ?? "") + part.text
            else
              run.events.push({
                id: crypto.randomUUID(),
                at: Date.now(),
                step: step + 1,
                type,
                text: part.text,
              })
            update()
          } else if (part.type === "tool-input-start") {
            run.events.push({
              id: crypto.randomUUID(),
              at: Date.now(),
              step: step + 1,
              type: "tool",
              callId: part.id,
              tool: part.toolName,
              text: "",
              state: "running",
            })
            update()
          } else if (part.type === "tool-input-delta") {
            const event = run.events.find((e) => e.callId === part.id)
            if (event) event.text = (event.text ?? "") + part.delta
            update()
          } else if (part.type === "tool-call") {
            calls.push({
              id: part.toolCallId,
              name: part.toolName,
              input: part.input,
              invalid: part.invalid,
              error: part.error,
            })
            let event = run.events.find((e) => e.callId === part.toolCallId)
            if (!event) {
              event = {
                id: crypto.randomUUID(),
                at: Date.now(),
                step: step + 1,
                type: "tool",
                callId: part.toolCallId,
                tool: part.toolName,
                state: "running",
              }
              run.events.push(event)
            }
            event.input = part.input
            update()
          } else if (part.type === "error") throw part.error
          else if (part.type === "finish-step") {
            record.usage = part.usage
            record.metadata = part.providerMetadata
            record.finishReason = part.finishReason
          }
        }
        record.response = { ...(await stream.finalStep).response, chunks }
      } finally {
        record.duration = (performance.now() - started) / 1000
      }
      run.messages.push(...(await stream.responseMessages))
      update()
      if (!calls.length) {
        run.status = "complete"
        break
      }
      for (const call of calls) {
        options.signal.throwIfAborted()
        const event = run.events.find((e) => e.callId === call.id)!
        const before = structuredClone(sandbox.files)
        const previousSources = new Set(run.sources.map((s) => s.url))
        const toolStarted = performance.now()
        let output: unknown
        try {
          if (!settings.tools.includes(call.name as ToolName))
            throw new Error("The model called a disabled or unknown tool.")
          if (call.invalid)
            throw new Error(`Invalid tool arguments: ${String(call.error)}`)
          const schema = toolDefinitions[call.name as ToolName].inputSchema
          // SDK already validates; parse again at the execution boundary.
          const input = (
            schema as { parse: (input: unknown) => unknown }
          ).parse(call.input) as Record<string, unknown>
          if (options.replay) {
            const recorded = replayCalls[replayIndex++]
            if (
              !recorded ||
              recorded.tool !== call.name ||
              key(recorded.input) !== key(call.input) ||
              recorded.output === undefined
            )
              throw new Error(
                "Replay diverged: no matching recorded result for this tool call. No live tool was executed."
              )
            output = structuredClone(recorded.output)
            for (const source of recorded.sources ?? [])
              if (!run.sources.some((s) => s.url === source.url))
                run.sources.push({
                  ...source,
                  id: String(run.sources.length + 1),
                })
            const files = new Map(sandbox.files.map((f) => [f.path, f]))
            for (const change of recorded.changes ?? []) {
              if (change.after) files.set(change.path, change.after)
              else files.delete(change.path)
            }
            await sandbox.restore([...files.values()], options.signal)
            event.replayed = true
          } else {
            if (
              settings.execution === "confirm" &&
              call.name !== "ask_question"
            ) {
              event.state = "waiting"
              update()
              const answer = await options.interact(
                { event, kind: "confirm" },
                options.signal
              )
              if (answer !== "allow") {
                output = { ok: false, error: "The user denied this tool call." }
                event.state = "denied"
              }
            }
            if (output === undefined) {
              const signal = AbortSignal.any([
                options.signal,
                AbortSignal.timeout(settings.toolTimeout),
              ])
              if (call.name === "ask_question") {
                event.state = "waiting"
                update()
                output = {
                  answer: await options.interact(
                    { event, kind: "question" },
                    options.signal
                  ),
                }
              } else if (
                call.name === "web_search" ||
                call.name === "fetch_url"
              ) {
                output = await options.web(call.name, input, settings, signal)
                if (typeof input.savePath === "string") {
                  const data = output as { content?: string; ok?: boolean }
                  if (data.ok !== false)
                    await sandbox.execute(
                      "write",
                      {
                        path: input.savePath,
                        content:
                          data.content ?? JSON.stringify(output, null, 2),
                      },
                      signal
                    )
                }
              } else
                output = await sandbox.execute(
                  call.name as ToolName,
                  input,
                  signal
                )
            }
          }
          if (event.state !== "denied") event.state = "done"
          if (
            event.state !== "denied" &&
            output &&
            typeof output === "object" &&
            "ok" in output &&
            output.ok === false
          )
            event.state = "error"
          if (
            event.state !== "denied" &&
            output &&
            typeof output === "object" &&
            "exitCode" in output &&
            output.exitCode !== 0
          )
            event.state = "error"
          output = addSources(output)
        } catch (error) {
          options.signal.throwIfAborted()
          output = {
            ok: false,
            error: error instanceof Error ? error.message : String(error),
          }
          event.state = "error"
          // Replay mismatch invalidates the experiment instead of switching to live.
          if (options.replay || !sandbox.available) {
            event.output = output
            event.changes = diffFiles(before, sandbox.files)
            throw error
          }
        }
        event.output = output
        event.modelOutput = boundedOutput(output, settings.outputLimit)
        event.sources = run.sources.filter((s) => !previousSources.has(s.url))
        event.duration = (performance.now() - toolStarted) / 1000
        event.changes = diffFiles(before, sandbox.files)
        run.after = structuredClone(sandbox.files)
        run.messages.push({
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: call.id,
              toolName: call.name,
              output: {
                type: "json",
                value: JSON.parse(JSON.stringify(event.modelOutput)),
              },
            },
          ],
        })
        update()
      }
      if (step + 1 === settings.maxSteps) {
        run.status = "limit"
        run.events.push({
          id: crypto.randomUUID(),
          at: Date.now(),
          step: step + 1,
          type: "status",
          text: `Stopped at the ${settings.maxSteps}-step limit.`,
        })
      }
    }
  } catch (error) {
    run.status = options.signal.aborted ? "stopped" : "error"
    run.error = error instanceof Error ? error.message : String(error)
    for (const event of run.events)
      if (event.state === "running" || event.state === "waiting")
        event.state = "error"
    run.events.push({
      id: crypto.randomUUID(),
      at: Date.now(),
      step: run.steps.length,
      type: "error",
      text: options.signal.aborted
        ? "Run stopped. The last completed filesystem snapshot was retained."
        : run.error,
    })
  } finally {
    // A stopped tool turn must still be a valid conversation for the next run.
    const repaired = closePendingTools(run.messages)
    run.messages = repaired.messages
    for (const id of repaired.unresolved) {
      const event = run.events.filter((entry) => entry.callId === id).at(-1)
      if (event) {
        event.state = "error"
        event.output ??= repaired.output
      }
    }
    run.after = structuredClone(
      sandbox.files.length ? sandbox.files : options.files
    )
    sandbox.close()
    update()
  }
  return run
}
