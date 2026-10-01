import { test } from "node:test"
import assert from "node:assert/strict"
import { createServer } from "node:http"
import {
  publicAddress,
  validateUrl,
  markdown,
  parseEngineHtml,
  guardedText,
  executeWeb,
} from "../../src/server/studio/web.ts"
import { searchSchema } from "../../src/features/studio/agent/types.ts"

void test("public network classification blocks special and mapped addresses", () => {
  for (const ip of [
    "127.0.0.1",
    "10.0.0.1",
    "169.254.169.254",
    "192.168.1.1",
    "::1",
    "::ffff:127.0.0.1",
    "fc00::1",
    "0.0.0.0",
  ])
    assert.equal(publicAddress(ip), false, ip)
  assert.equal(publicAddress("8.8.8.8"), true)
  assert.throws(() => validateUrl("file:///etc/passwd"))
  assert.throws(() => validateUrl("http://user:secret@example.com"))
})
void test("search parser unwraps engine redirects and preserves provenance", () => {
  const results = parseEngineHtml(
    "duckduckgo",
    '<div class="result"><h2><a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fpaper">Paper</a></h2><a class="result__snippet">Useful result</a></div>',
    "https://html.duckduckgo.com/html/"
  )
  assert.deepEqual(results, [
    {
      title: "Paper",
      url: "https://example.com/paper",
      snippet: "Useful result",
      engine: "duckduckgo",
    },
  ])
})
void test("readable markdown removes active content and resolves relative links", () => {
  const value = markdown(
    '<html><body><script>secret()</script><main><h1>Title</h1><a href="/paper">Paper</a><a href="javascript:bad()">Unsafe</a></main></body></html>',
    "https://example.com"
  )
  assert.match(value, /# Title/)
  assert.match(value, /https:\/\/example.com\/paper/)
  assert.doesNotMatch(value, /secret|javascript:/)
})
void test("guard refuses localhost before making a request", async () => {
  await assert.rejects(
    guardedText("http://127.0.0.1:9", AbortSignal.timeout(1000)),
    /non-public/
  )
})
void test("configured private search service is bounded; redirects cannot escape origin exception", async () => {
  const server = createServer((req, res) => {
    if (req.url?.startsWith("/search")) {
      res.setHeader("content-type", "application/json")
      res.end(
        JSON.stringify({
          results: [
            {
              title: "Local search",
              url: "https://example.com",
              content: "result",
            },
          ],
        })
      )
    } else if (req.url === "/redirect") {
      res.writeHead(302, { location: "http://127.0.0.1:9/private" })
      res.end()
    } else if (req.url === "/large") {
      res.setHeader("content-length", 3 * 1024 * 1024)
      res.end("large")
    } else {
      res.end("ok")
    }
  })
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve))
  const address = server.address()
  assert.ok(address && typeof address === "object")
  const origin = `http://127.0.0.1:${address.port}`
  try {
    const result = await executeWeb(
      "web_search",
      { query: "paper" },
      searchSchema.parse({ provider: "searxng" }),
      { searxngUrl: origin }
    )
    assert.equal(result.ok, true)
    await assert.rejects(
      guardedText(`${origin}/redirect`, AbortSignal.timeout(2000), {
        privateOrigin: origin,
      }),
      /non-public/
    )
    await assert.rejects(
      guardedText(`${origin}/large`, AbortSignal.timeout(2000), {
        privateOrigin: origin,
      }),
      /2 MiB/
    )
    const missing = await executeWeb(
      "web_search",
      { query: "paper" },
      searchSchema.parse({ provider: "brave" })
    )
    assert.equal(missing.ok, false)
    assert.match(String(missing.error), /key is not configured/)
    const empty = await executeWeb(
      "web_search",
      { query: "paper" },
      searchSchema.parse({ provider: "ddgs_custom", engines: [] })
    )
    assert.equal(empty.ok, false)
  } finally {
    await new Promise<void>((resolve, reject) =>
      server.close((error) => (error ? reject(error) : resolve()))
    )
  }
})
