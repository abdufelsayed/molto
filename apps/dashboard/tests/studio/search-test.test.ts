import { test } from "node:test"
import assert from "node:assert/strict"
import { registerHooks } from "node:module"
import { readFileSync } from "node:fs"

// Match the bundler's extensionless TS/JSON imports when running server units in Node.
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
    if (url.endsWith(".json"))
      return {
        format: "module",
        source: `export default ${readFileSync(new URL(url), "utf8")}`,
        shortCircuit: true,
      }
    return nextLoad(url, context)
  },
})
const { searchTest } = await import("../../src/server/studio/search-test.ts")
const { createSession } = await import("../../src/server/connection.server.ts")
hooks.deregister()

void test("search connectivity test authorizes origin and session before reading credentials", async () => {
  assert.equal(
    (
      await searchTest(
        new Request("http://dashboard/api/test", {
          method: "POST",
          headers: { origin: "http://evil" },
          body: "{}",
        })
      )
    ).status,
    403
  )
  assert.equal(
    (
      await searchTest(
        new Request("http://dashboard/api/test", {
          method: "POST",
          headers: { origin: "http://dashboard" },
          body: "{}",
        })
      )
    ).status,
    401
  )
})
void test("pending search settings override saved values without a persistence request", async () => {
  const connected = createSession(
    new Request("http://dashboard"),
    "server-secret"
  )
  const cookie = connected.headers.get("set-cookie")!.split(";")[0]!
  const request = (body: unknown) =>
    new Request("http://dashboard/api/test", {
      method: "POST",
      headers: {
        origin: "http://dashboard",
        cookie,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    })
  const originalFetch = globalThis.fetch
  const calls: { url: string; method: string; authorization: string | null }[] =
    []
  globalThis.fetch = async (input, init) => {
    calls.push({
      url:
        typeof input === "string"
          ? input
          : input instanceof URL
            ? input.href
            : input.url,
      method: init?.method ?? "GET",
      authorization: new Headers(init?.headers).get("Authorization"),
    })
    return Response.json({
      sections: {
        integrations: {
          web_search_provider: "brave",
          web_search_brave_api_key: "stored-key",
        },
      },
    })
  }
  try {
    const response = await searchTest(
      request({ provider: "brave", brave_api_key: "" })
    )
    const body = (await response.json()) as {
      ok: boolean
      error: { message: string }
    }
    assert.equal(response.status, 200)
    assert.equal(body.ok, false)
    assert.match(body.error.message, /key is not configured/)
    assert.equal(calls.length, 1)
    assert.equal(calls[0]?.method, "GET")
    assert.equal(calls[0]?.authorization, "Bearer server-secret")
    assert.equal((await searchTest(request({ unknown: "value" }))).status, 422)
    assert.equal(calls.length, 1)
  } finally {
    globalThis.fetch = originalFetch
  }
})
