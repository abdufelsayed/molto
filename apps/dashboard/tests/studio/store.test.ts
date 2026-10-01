import { test } from "node:test"
import assert from "node:assert/strict"
import { registerHooks } from "node:module"
import { readFileSync } from "node:fs"
import { transpileModule, ModuleKind, ScriptTarget } from "typescript"
import type { Run } from "../../src/features/studio/agent/types"

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
const { importSession } = await import("../../src/features/studio/store.ts")
const { newSession } = await import("../../src/features/studio/agent/types.ts")
hooks.deregister()

void test("import repairs interrupted tool context and activity while preserving the exported record", () => {
  const session = newSession()
  const run: Run = {
    id: "run",
    created: 1,
    settings: session.settings,
    input: [{ role: "user", content: "write" }],
    messages: [
      {
        role: "assistant",
        content: [
          {
            type: "tool-call",
            toolCallId: "pending",
            toolName: "write",
            input: { path: "x", content: "data" },
          },
        ],
      },
    ],
    events: [
      {
        id: "pending",
        at: 1,
        step: 1,
        type: "tool",
        state: "waiting",
        callId: "pending",
        tool: "write",
      },
      {
        id: "complete",
        at: 1,
        step: 1,
        type: "tool",
        state: "done",
        output: { preserved: true },
      },
    ],
    steps: [{ request: {} }],
    sources: [],
    before: [],
    after: [{ path: "/workspace/kept", type: "file", content: "eA==" }],
    status: "running",
    mode: "live",
    note: "Keep this note",
    prompt: { text: "Original prompt", attachments: ["/workspace/kept"] },
  }
  session.runs.push(run)
  session.files = run.after
  const original = structuredClone(session)
  const imported = importSession(JSON.stringify(session))
  assert.notEqual(imported.id, session.id)
  assert.equal(imported.title, `${session.title} (imported)`)
  const recovered = imported.runs[0]!
  assert.equal(recovered.status, "interrupted")
  assert.deepEqual(recovered.messages.at(-1), {
    role: "tool",
    content: [
      {
        type: "tool-result",
        toolCallId: "pending",
        toolName: "write",
        output: { type: "json", value: { ok: false, error: recovered.error } },
      },
    ],
  })
  assert.equal(recovered.events[0]?.state, "error")
  assert.deepEqual(recovered.events[1], run.events[1])
  assert.equal(recovered.events.at(-1)?.type, "error")
  assert.equal(recovered.events.at(-1)?.text, recovered.error)
  assert.equal(recovered.note, run.note)
  assert.deepEqual(recovered.prompt, run.prompt)
  assert.deepEqual(recovered.after, run.after)
  assert.deepEqual(imported.files, session.files)
  assert.deepEqual(session, original)
  // A second import must not duplicate recovery results or error events.
  const again = importSession(JSON.stringify(imported)).runs[0]!
  assert.deepEqual(again.messages, recovered.messages)
  assert.deepEqual(again.events, recovered.events)
})
