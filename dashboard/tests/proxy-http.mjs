import assert from "node:assert/strict"
import { createServer } from "node:http"
import { readFile } from "node:fs/promises"
import { once } from "node:events"
import { test } from "node:test"
import ts from "typescript"
import { H3, toNodeHandler } from "nitro/h3"

const source = await readFile(
  new URL("../src/server/inference-proxy.server.ts", import.meta.url),
  "utf8"
)
const compiled = ts.transpileModule(
  source.replaceAll(
    '"nitro/h3"',
    JSON.stringify(import.meta.resolve("nitro/h3"))
  ),
  {
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.ES2022,
    },
  }
).outputText
const { inferenceProxy, inferencePathAllowed } = await import(
  `data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`
)

async function fixture(upstreamHandler, run) {
  const upstream = createServer(upstreamHandler)
  upstream.listen(0, "127.0.0.1")
  await once(upstream, "listening")
  const previous = process.env.OMLX_API_URL
  process.env.OMLX_API_URL = `http://127.0.0.1:${upstream.address().port}`
  const app = new H3()
  app.use(inferenceProxy)
  app.all("/**", () => new Response("dashboard"))
  const publicServer = createServer(toNodeHandler(app))
  publicServer.listen(0, "127.0.0.1")
  await once(publicServer, "listening")
  try {
    await run(`http://127.0.0.1:${publicServer.address().port}`)
  } finally {
    if (previous === undefined) delete process.env.OMLX_API_URL
    else process.env.OMLX_API_URL = previous
    publicServer.closeAllConnections()
    upstream.closeAllConnections()
    await Promise.all([
      new Promise((resolve) => publicServer.close(resolve)),
      new Promise((resolve) => upstream.close(resolve)),
    ])
  }
}

await test("inference inventory rejects unsupported methods, unsafe paths, and management", () => {
  assert.equal(inferencePathAllowed("/v1/models/org/model/load", "POST"), true)
  for (const [path, method] of [
    ["/v1/models", "POST"],
    ["/management/v1/state", "GET"],
    ["//elsewhere/v1/models", "GET"],
    ["/v1/models/a/%2e%2e/load", "POST"],
    ["/v1/models/a%5Cb/load", "POST"],
    ["/v1/models/a%252fb/load", "POST"],
    ["/v1/not-a-route", "GET"],
  ])
    assert.equal(inferencePathAllowed(path, method), false)
})

await test("multipart request, binary response, credentials, and errors survive native proxy", async () => {
  const bytes = Buffer.from([0, 255, 1, 128])
  const body = Buffer.concat([
    Buffer.from(
      '--boundary\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'
    ),
    bytes,
    Buffer.from("\r\n--boundary--\r\n"),
  ])
  await fixture(
    async (req, res) => {
      assert.equal(req.headers.authorization, "Bearer client-key")
      assert.equal(req.headers["x-api-key"], "sub-key")
      assert.equal(req.headers["x-forwarded-for"], "127.0.0.1")
      assert.equal(req.headers.forwarded, undefined)
      assert.equal(req.headers.cookie, undefined)
      assert.equal(
        req.headers["content-type"],
        "multipart/form-data; boundary=boundary"
      )
      const chunks = []
      for await (const chunk of req) chunks.push(chunk)
      assert.deepEqual(Buffer.concat(chunks), body)
      res.writeHead(422, { "content-type": "application/octet-stream" })
      res.end(bytes)
    },
    async (origin) => {
      const response = await fetch(`${origin}/v1/audio/transcriptions`, {
        method: "POST",
        body,
        headers: {
          authorization: "Bearer client-key",
          "x-api-key": "sub-key",
          "content-type": "multipart/form-data; boundary=boundary",
          "x-forwarded-for": "forged",
          forwarded: "for=forged",
          cookie: "omlx_dashboard_session=secret",
        },
      })
      assert.equal(response.status, 422)
      assert.equal(
        response.headers.get("content-type"),
        "application/octet-stream"
      )
      assert.deepEqual(Buffer.from(await response.arrayBuffer()), bytes)
    }
  )
})

await test("SSE first chunk arrives before generation finishes and client cancellation closes upstream", async () => {
  let finish
  let closed
  const release = new Promise((resolve) => {
    finish = resolve
  })
  const cancelled = new Promise((resolve) => {
    closed = resolve
  })
  await fixture(
    (req, res) => {
      res.on("close", closed)
      res.writeHead(200, { "content-type": "text/event-stream" })
      res.write("data: first\n\n")
      void release.then(() => res.end("data: last\n\n"))
    },
    async (origin) => {
      const controller = new AbortController()
      const response = await fetch(`${origin}/v1/chat/completions`, {
        method: "POST",
        body: "{}",
        signal: controller.signal,
      })
      const reader = response.body.getReader()
      assert.equal(
        new TextDecoder().decode((await reader.read()).value),
        "data: first\n\n"
      )
      controller.abort()
      await Promise.race([
        cancelled,
        new Promise((_, reject) =>
          setTimeout(
            () => reject(new Error("Upstream was not cancelled")),
            2000
          )
        ),
      ])
      finish()
    }
  )
})

await test("unknown routes and methods never reach upstream; redirects are not followed", async () => {
  let calls = 0
  await fixture(
    (req, res) => {
      calls++
      res.writeHead(307, { location: "/management/v1/state" })
      res.end()
    },
    async (origin) => {
      for (const [path, method] of [
        ["/v1/models", "POST"],
        ["/management/v1/state", "GET"],
        ["/admin/api/state", "GET"],
        ["/api/cluster/devices", "GET"],
      ]) {
        assert.equal((await fetch(`${origin}${path}`, { method })).status, 404)
      }
      assert.equal(calls, 0)
      assert.equal((await fetch(`${origin}/v1/models`)).status, 502)
      assert.equal(calls, 1)
      assert.equal(
        await (await fetch(`${origin}/settings`)).text(),
        "dashboard"
      )
    }
  )
})

await test("remote cluster protocol has an explicit method inventory and forwards actual peer identity", async () => {
  const allowed = [
    ["GET", "/api/cluster/node_id"],
    ["GET", "/api/cluster/pair/status/node-123"],
    ["POST", "/api/cluster/pair/request"],
    ["POST", "/api/cluster/pair/request/cancel"],
    ["GET", "/cluster/join/bootstrap.py"],
    ["POST", "/cluster/join/claim"],
    ["GET", "/cluster/join/source"],
    ["POST", "/cluster/join/complete"],
    ["GET", "/api/cluster/models/org/model/manifest"],
  ]
  let calls = 0
  await fixture(
    (req, res) => {
      calls++
      assert.equal(req.headers["x-forwarded-for"], "127.0.0.1")
      assert.equal(req.headers["x-forwarded-proto"], "http")
      assert.notEqual(req.headers["x-forwarded-host"], "attacker.example")
      assert.equal(req.headers.forwarded, undefined)
      assert.equal(req.headers["x-real-ip"], undefined)
      assert.equal(req.headers.cookie, undefined)
      assert.equal(req.headers.authorization, "Bearer peer-key")
      assert.equal(req.headers["x-api-key"], "peer-key")
      res.end(req.url)
    },
    async (origin) => {
      for (const [method, path] of allowed) {
        assert.equal(inferencePathAllowed(path, method), true)
        const response = await fetch(`${origin}${path}?token=join-token`, {
          method,
          headers: {
            authorization: "Bearer peer-key",
            "x-api-key": "peer-key",
            "x-forwarded-for": "203.0.113.1",
            "x-forwarded-host": "attacker.example",
            "x-forwarded-proto": "https",
            forwarded: "for=attacker",
            "x-real-ip": "203.0.113.2",
            cookie: "omlx_dashboard_session=admin-secret",
          },
        })
        assert.equal(response.status, 200)
        assert.equal(await response.text(), `${path}?token=join-token`)
        const wrongMethod = method === "GET" ? "POST" : "GET"
        assert.equal(inferencePathAllowed(path, wrongMethod), false)
        assert.equal(
          (await fetch(`${origin}${path}`, { method: wrongMethod })).status,
          404
        )
      }
      for (const path of [
        "/api/cluster/pair/approve",
        "/api/cluster/devices",
        "/admin/api/cluster/deployments",
        "/cluster/join/not-a-route",
        "/api/cluster/models/a/%2e%2e/manifest",
      ]) {
        assert.equal(inferencePathAllowed(path, "GET"), false)
      }
      assert.equal(calls, allowed.length)
    }
  )
})
