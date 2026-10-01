import { test } from "node:test"
import assert from "node:assert/strict"
import { registerHooks } from "node:module"
import { readFileSync } from "node:fs"
import { transpileModule, ModuleKind, ScriptTarget } from "typescript"
import type {
  AgentSettings,
  Run,
  Snapshot,
} from "../../src/features/studio/agent/types"
const hooks = registerHooks({
  resolve(specifier, context, nextResolve) {
    try {
      return nextResolve(specifier, context)
    } catch (error) {
      if (specifier.startsWith(".") && !specifier.endsWith(".ts"))
        return nextResolve(`${specifier}.ts`, context)
      throw error
    }
  },
  load(url, context, nextLoad) {
    if (url.endsWith(".ts") && !url.includes("/node_modules/"))
      return {
        format: "module",
        shortCircuit: true,
        source: transpileModule(readFileSync(new URL(url), "utf8"), {
          compilerOptions: {
            module: ModuleKind.ESNext,
            target: ScriptTarget.ES2022,
          },
        }).outputText,
      }
    return nextLoad(url, context)
  },
})
const { VirtualSandbox, encodeBytes } =
  await import("../../src/features/studio/agent/sandbox.ts")
const { SandboxClient } =
  await import("../../src/features/studio/agent/worker-client.ts")
const { defaults, toolNames } =
  await import("../../src/features/studio/agent/types.ts")
const { runAgent, boundedOutput } =
  await import("../../src/features/studio/agent/run.ts")
hooks.deregister()
class MockWorker {
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: ((event: ErrorEvent) => void) | null = null
  sandbox = new VirtualSandbox()
  terminated = false
  static stall = false
  postMessage(message: {
    id: string
    action: string
    files?: Snapshot
    settings: AgentSettings
    name: (typeof toolNames)[number]
    args: Record<string, unknown>
  }) {
    if (MockWorker.stall && message.action === "execute") return
    void (async () => {
      try {
        const data =
          message.action === "restore"
            ? await this.sandbox.restore(message.files ?? [])
            : await this.sandbox.execute(
                message.name,
                message.args,
                message.settings
              )
        const files = await this.sandbox.snapshot()
        if (!this.terminated)
          this.onmessage?.({
            data: { id: message.id, data, files },
          } as MessageEvent)
      } catch (error) {
        if (!this.terminated)
          this.onmessage?.({
            data: { id: message.id, error: String(error) },
          } as MessageEvent)
      }
    })()
  }
  terminate() {
    this.terminated = true
  }
}
void test("sandbox file tools honor cwd, exact edits and session environment", async () => {
  const sandbox = new VirtualSandbox()
  await sandbox.restore([])
  const settings = {
    ...defaults(),
    cwd: "/workspace/experiment",
    env: { LABEL: "sample" },
  }
  await sandbox.execute(
    "write",
    { path: "a.txt", content: "first\nsecond\n" },
    settings
  )
  await sandbox.execute(
    "edit",
    { path: "a.txt", oldText: "second", newText: "third" },
    settings
  )
  assert.deepEqual(
    await sandbox.execute(
      "read",
      { path: "a.txt", offset: 2, limit: 1 },
      settings
    ),
    {
      path: "/workspace/experiment/a.txt",
      content: "third",
      totalLines: 3,
      truncated: false,
    }
  )
  const result = await sandbox.execute(
    "bash",
    { command: 'printf "%s:%s" "$PWD" "$LABEL"; cat a.txt' },
    settings
  )
  assert.equal(result.stdout, "/workspace/experiment:samplefirst\nthird\n")
  await assert.rejects(
    sandbox.execute(
      "edit",
      { path: "a.txt", oldText: "missing", newText: "x" },
      settings
    ),
    /not found/
  )
})
void test("read-only blocks shell redirects; curl requires both network and fetch tool", async () => {
  const sandbox = new VirtualSandbox()
  await sandbox.restore([])
  const settings = { ...defaults(), readOnly: true }
  await assert.rejects(
    sandbox.execute("write", { path: "x", content: "x" }, settings),
    /read-only/
  )
  const before = await sandbox.snapshot()
  await assert.rejects(
    sandbox.execute(
      "bash",
      { command: "echo changed > /workspace/x" },
      settings
    ),
    /read-only/
  )
  assert.deepEqual(
    (await sandbox.snapshot()).filter((file) =>
      file.path.startsWith("/workspace")
    ),
    before.filter((file) => file.path.startsWith("/workspace"))
  )
  let calls = 0
  const network = async (url: string) => {
    calls++
    return { content: "public", url }
  }
  for (const config of [
    { network: false },
    { network: true, tools: ["bash"] as AgentSettings["tools"] },
  ]) {
    const result = await sandbox.execute(
      "bash",
      { command: "curl https://example.com" },
      { ...defaults(), ...config },
      network
    )
    assert.notEqual(result.exitCode, 0)
  }
  assert.equal(calls, 0)
  const allowed = await sandbox.execute(
    "bash",
    { command: "curl https://example.com" },
    { ...defaults(), network: true },
    network
  )
  assert.equal(allowed.stdout, "public")
  assert.equal(calls, 1)
})
void test("binary bytes and symlinks survive snapshots", async () => {
  const sandbox = new VirtualSandbox()
  await sandbox.restore([
    {
      path: "/workspace/binary",
      type: "file",
      content: encodeBytes(new Uint8Array([0, 255, 128, 10])),
      mode: 0o640,
    },
    { path: "/workspace/link", type: "symlink", target: "binary" },
  ])
  const snapshot = await sandbox.snapshot()
  const clone = new VirtualSandbox()
  await clone.restore(snapshot)
  assert.deepEqual(await clone.snapshot(), snapshot)
  assert.deepEqual(
    await clone.fs.readFileBuffer("/workspace/link"),
    new Uint8Array([0, 255, 128, 10])
  )
})
void test("terminated worker retains completed snapshot and rejects future calls", async () => {
  const original = globalThis.Worker
  globalThis.Worker = MockWorker as unknown as typeof Worker
  const client = new SandboxClient(defaults(), async (url) => ({
    content: "",
    url,
  }))
  try {
    await client.restore([])
    await client.execute("write", { path: "kept", content: "completed" })
    const completed = structuredClone(client.files)
    MockWorker.stall = true
    const controller = new AbortController()
    const pending = client.execute(
      "bash",
      { command: "sleep 100" },
      controller.signal
    )
    controller.abort()
    await assert.rejects(pending, /stopped/)
    assert.equal(client.available, false)
    assert.deepEqual(client.files, completed)
    await assert.rejects(client.execute("read", { path: "kept" }), /terminated/)
  } finally {
    MockWorker.stall = false
    client.close()
    globalThis.Worker = original
  }
})
function completion(calls?: Array<{ name: string; args: unknown }>) {
  const chunks = calls
    ? [
        {
          choices: [
            {
              index: 0,
              delta: {
                role: "assistant",
                tool_calls: calls.map((call, index) => ({
                  index,
                  id: `call-${index}`,
                  type: "function",
                  function: {
                    name: call.name,
                    arguments: JSON.stringify(call.args),
                  },
                })),
              },
              finish_reason: null,
            },
          ],
        },
        { choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] },
      ]
    : [
        {
          choices: [
            {
              index: 0,
              delta: { role: "assistant", content: "Finished." },
              finish_reason: null,
            },
          ],
        },
        { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
      ]
  return new Response(
    chunks
      .map(
        (chunk) =>
          `data: ${JSON.stringify({ id: "test", object: "chat.completion.chunk", created: 1, model: "local", ...chunk })}\n\n`
      )
      .join("") + "data: [DONE]\n\n",
    { headers: { "content-type": "text/event-stream" } }
  )
}
async function withAgent(
  calls: Array<{ name: string; args: unknown }> | undefined,
  options: { replay?: Run; abort?: boolean; partial?: "error" | "abort" } = {}
) {
  const originalFetch = globalThis.fetch
  const originalWorker = globalThis.Worker
  const requests: Record<string, unknown>[] = []
  let liveCalls = 0
  globalThis.Worker = MockWorker as unknown as typeof Worker
  globalThis.fetch = async (_input, init) => {
    if (typeof init?.body !== "string")
      throw new Error("Expected JSON request body.")
    requests.push(JSON.parse(init.body) as Record<string, unknown>)
    if (options.partial) {
      const chunk = {
        id: "partial",
        object: "chat.completion.chunk",
        created: 1,
        model: "local",
        choices: [
          {
            index: 0,
            delta: { role: "assistant", content: "Partial answer" },
            finish_reason: null,
          },
        ],
      }
      return new Response(
        new ReadableStream<Uint8Array>({
          start(stream) {
            stream.enqueue(
              new TextEncoder().encode(`data: ${JSON.stringify(chunk)}\n\n`)
            )
            setTimeout(
              () => stream.error(new Error("Synthetic stream interrupted")),
              10
            )
          },
        }),
        { headers: { "content-type": "text/event-stream" } }
      )
    }
    return completion(requests.length === 1 ? calls : undefined)
  }
  const controller = new AbortController()
  try {
    const run = await runAgent({
      settings: { ...defaults(), model: "local" },
      messages: [{ role: "user", content: "Try this" }],
      files: [],
      signal: controller.signal,
      replay: options.replay,
      modelUrl: "http://local/v1",
      onUpdate: (run) => {
        if (
          options.partial === "abort" &&
          run.events.some((event) => event.type === "text")
        )
          controller.abort()
      },
      interact: async () => "answer",
      web: async (name, input) => {
        liveCalls++
        if (options.abort) {
          controller.abort()
          controller.signal.throwIfAborted()
        }
        return name === "web_search"
          ? {
              ok: true,
              results: [
                {
                  title: "First",
                  url: "https://first.example",
                  snippet: "one",
                },
              ],
            }
          : { ok: true, url: (input as { url: string }).url, content: "two" }
      },
    })
    return { run, requests, liveCalls }
  } finally {
    globalThis.fetch = originalFetch
    globalThis.Worker = originalWorker
  }
}
void test("default request omits application system and inherited generation settings", async () => {
  const { run, requests } = await withAgent(undefined)
  assert.equal(run.status, "complete")
  const request = requests[0]!
  assert.deepEqual(request.messages, [{ role: "user", content: "Try this" }])
  assert.deepEqual(
    (request.tools as Array<{ function: { name: string } }>)
      .map((tool) => tool.function.name)
      .sort(),
    [...toolNames].sort()
  )
  for (const setting of [
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "max_tokens",
    "enable_thinking",
  ])
    assert.equal(setting in request, false, setting)
})
void test("source identifiers stay stable; replay never calls live tools and rejects divergence", async () => {
  const calls = [
    { name: "web_search", args: { query: "sample" } },
    { name: "fetch_url", args: { url: "https://second.example" } },
  ]
  const live = await withAgent(calls)
  assert.equal(live.run.status, "complete")
  assert.deepEqual(
    live.run.sources.map((source) => source.id),
    ["1", "2"]
  )
  const outputs = live.run.events
    .filter((event) => event.type === "tool")
    .map((event) => event.output) as Array<{
    source_id?: string
    results?: Array<{ source_id: string }>
  }>
  assert.equal(outputs[0]?.results?.[0]?.source_id, "1")
  assert.equal(outputs[1]?.source_id, "2")
  const replay = await withAgent(calls, { replay: live.run })
  assert.equal(replay.run.status, "complete")
  assert.equal(replay.liveCalls, 0)
  assert.ok(
    replay.run.events
      .filter((event) => event.type === "tool")
      .every((event) => event.replayed)
  )
  const divergent = await withAgent(
    [{ name: "web_search", args: { query: "different" } }],
    { replay: live.run }
  )
  assert.equal(divergent.run.status, "error")
  assert.match(divergent.run.error!, /Replay diverged/)
  assert.equal(divergent.liveCalls, 0)
})
void test("stopped tool runs keep matching results for next context", async () => {
  const { run } = await withAgent(
    [{ name: "web_search", args: { query: "sample" } }],
    { abort: true }
  )
  assert.equal(run.status, "stopped")
  const calls = run.messages.flatMap((message) =>
    message.role === "assistant" && Array.isArray(message.content)
      ? message.content
          .filter((part) => part.type === "tool-call")
          .map((part) => part.toolCallId)
      : []
  )
  const results = run.messages.flatMap((message) =>
    message.role === "tool"
      ? message.content
          .filter((part) => part.type === "tool-result")
          .map((part) => part.toolCallId)
      : []
  )
  assert.ok(calls.length > 0)
  assert.deepEqual(results, calls)
})
void test("model output stays bounded with heavily escaped JSON", () => {
  const output = boundedOutput({ text: '\\"\n'.repeat(20_000) }, 100)
  assert.ok(JSON.stringify(output).length <= 100)
  assert.equal((output as { truncated: boolean }).truncated, true)
})

void test("failed and stopped partial streams retain raw chunks and elapsed duration", async () => {
  for (const mode of ["error", "abort"] as const) {
    const { run } = await withAgent(undefined, { partial: mode })
    assert.equal(run.status, mode === "abort" ? "stopped" : "error")
    assert.equal(
      run.events.find((event) => event.type === "text")?.text,
      "Partial answer"
    )
    const record = run.steps[0]!
    assert.ok(Number.isFinite(record.duration))
    assert.ok(record.duration! >= 0)
    assert.ok((record.response as { chunks: unknown[] }).chunks.length > 0)
    assert.match(JSON.stringify(record.response), /Partial answer/)
  }
})
