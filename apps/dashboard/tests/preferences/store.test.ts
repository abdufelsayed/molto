import { test } from "node:test"
import assert from "node:assert/strict"
import { registerHooks } from "node:module"
import { readFileSync } from "node:fs"
import { transpileModule, ModuleKind, ScriptTarget } from "typescript"

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
const { createPreferenceStore, decodePreferences, storageKey } =
  await import("../../src/features/preferences/store.ts")
hooks.deregister()
const saved = (values: Record<string, unknown>, version = 1) =>
  JSON.stringify({ version, values })

void test("hydration restores saved preferences without writing defaults and isolates roots", () => {
  let writes = 0
  const preferences = createPreferenceStore()
  preferences.set("app.sidebar", true)
  assert.equal(preferences.store.get().ready, false)
  preferences.hydrate({
    getItem: () => saved({ "app.sidebar": false, "studio.active": "older" }),
    setItem: () => {
      writes++
    },
  })
  assert.equal(preferences.get("app.sidebar"), false)
  assert.equal(preferences.get("studio.active"), "older")
  assert.equal(writes, 0)
  const other = createPreferenceStore()
  assert.equal(other.get("app.sidebar"), true)
  assert.equal(other.store.get().ready, false)
})

void test("malformed and future storage fall back; invalid and unknown entries do not erase valid entries", () => {
  assert.deepEqual(decodePreferences("{"), {})
  assert.deepEqual(decodePreferences(saved({ "app.sidebar": false }, 2)), {})
  assert.deepEqual(
    decodePreferences(
      saved({
        "app.sidebar": false,
        "studio.settingsTab": "gone",
        "studio.observability": { expanded: true, size: -1 },
        credentials: "must not restore",
      })
    ),
    { "app.sidebar": false }
  )
})

void test("functional updates are synchronous and scoped choices survive a new store", () => {
  const entries = new Map<string, string>()
  const storage = {
    getItem: (key: string) => entries.get(key) ?? null,
    setItem: (key: string, value: string) => {
      entries.set(key, value)
    },
  }
  const preferences = createPreferenceStore()
  preferences.hydrate(storage)
  preferences.set("studio.attachments", ["/workspace/a"], "first")
  preferences.set(
    "studio.attachments",
    (paths) => [...paths, "/workspace/b"],
    "first"
  )
  preferences.set("studio.disclosure", false, "first:run:chain")
  preferences.set("studio.path", "/workspace/other", "second")
  const restored = createPreferenceStore()
  restored.hydrate(storage)
  assert.deepEqual(restored.get("studio.attachments", "first"), [
    "/workspace/a",
    "/workspace/b",
  ])
  assert.equal(restored.get("studio.disclosure", "first:run:chain"), false)
  restored.removeScope("first")
  assert.equal(restored.get("studio.disclosure", "first:run:chain"), undefined)
  assert.deepEqual(restored.get("studio.attachments", "first"), [])
  assert.equal(restored.get("studio.path", "second"), "/workspace/other")
  assert.ok(entries.has(storageKey))
})

void test("denied reads and quota failures leave in-memory updates usable", () => {
  const preferences = createPreferenceStore()
  preferences.hydrate({
    getItem: () => {
      throw new Error("denied")
    },
    setItem: () => {
      throw new Error("quota")
    },
  })
  preferences.set("app.sidebar", false)
  preferences.set("studio.inspectorTab", "usage")
  assert.equal(preferences.store.get().ready, true)
  assert.equal(preferences.get("app.sidebar"), false)
  assert.equal(preferences.get("studio.inspectorTab"), "usage")
})

void test("scoped preferences are bounded and invalid layouts cannot be restored", () => {
  const values = Object.fromEntries(
    Array.from({ length: 250 }, (_, index) => [
      `studio.run:session-${index}`,
      `run-${index}`,
    ])
  )
  const decoded = decodePreferences(
    saved({
      ...values,
      "studio.layout:wrong": { a: 60, b: 60 },
      "studio.layout:right": { "studio-main": 65, "studio-settings": 35 },
    })
  )
  assert.equal(
    Object.keys(decoded).filter((key) => key.startsWith("studio.run:")).length,
    200
  )
  assert.equal(decoded["studio.run:session-0"], undefined)
  assert.equal(decoded["studio.run:session-249"], "run-249")
  assert.equal(decoded["studio.layout:wrong"], undefined)
  assert.deepEqual(decoded["studio.layout:right"], {
    "studio-main": 65,
    "studio-settings": 35,
  })
})

void test("large valid preferences stay within the hydration budget and roundtrip", () => {
  let raw: string | null = null
  const storage = {
    getItem: () => raw,
    setItem: (_key: string, value: string) => {
      raw = value
    },
  }
  const preferences = createPreferenceStore()
  preferences.hydrate(storage)
  const paths = Array.from(
    { length: 100 },
    (_, index) => `/${index}/${"x".repeat(4000)}`
  )
  for (let index = 0; index < 5; index++)
    preferences.set("studio.attachments", paths, `session-${index}`)
  preferences.set("studio.inspectorTab", "usage")
  const serialized = storage.getItem()
  assert.ok(serialized && serialized.length <= 1_000_000)
  const restored = createPreferenceStore()
  restored.hydrate(storage)
  assert.equal(restored.get("studio.inspectorTab"), "usage")
  assert.deepEqual(restored.get("studio.attachments", "session-4"), paths)
  assert.deepEqual(restored.store.get().values, preferences.store.get().values)
})
