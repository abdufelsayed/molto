import assert from "node:assert/strict"
import { createServer } from "node:http"
import { once } from "node:events"
import { readFile } from "node:fs/promises"
import { test } from "node:test"
import ts from "typescript"
import nodeAdapter from "crossws/adapters/node"
import WebSocket from "crossws/websocket"

let helperSource = await readFile(
  new URL("../src/server/inference-proxy.server.ts", import.meta.url),
  "utf8"
)
const schema = await readFile(
  new URL("../src/lib/openapi.json", import.meta.url),
  "utf8"
)
const targetSource = (
  await readFile(
    new URL("../src/server/management-target.server.ts", import.meta.url),
    "utf8"
  )
)
  .replace(
    'import openapi from "@/lib/openapi.json"',
    `const openapi = ${schema}`
  )
  .replace("export function operationTarget", "function operationTarget")
helperSource = helperSource.replace(
  'import { operationTarget } from "./management-target.server"',
  () => targetSource
)

const compile = (source) =>
  `data:text/javascript;base64,${Buffer.from(ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } }).outputText).toString("base64")}`
const helper = compile(
  helperSource.replaceAll(
    '"nitro/h3"',
    JSON.stringify(import.meta.resolve("nitro/h3"))
  )
)
const routeSource = await readFile(
  new URL(
    "../server/routes/v1/audio/transcriptions/realtime.ts",
    import.meta.url
  ),
  "utf8"
)
const route = await import(
  compile(
    routeSource
      .replace(
        '"../../../../../src/server/inference-proxy.server"',
        JSON.stringify(helper)
      )
      .replace('"crossws"', JSON.stringify(import.meta.resolve("crossws")))
      .replace(
        '"crossws/websocket"',
        JSON.stringify(import.meta.resolve("crossws/websocket"))
      )
      .replace('"nitro"', JSON.stringify(import.meta.resolve("nitro")))
  )
)

await test("native realtime proxy preserves early auth frame, binary audio, and bidirectional close", async () => {
  let upstreamClosed
  const closed = new Promise((resolve) => {
    upstreamClosed = resolve
  })
  const seen = []
  const upstream = createServer()
  const upstreamAdapter = nodeAdapter({
    hooks: {
      message(peer, message) {
        if (message.text() === "upstream-close") {
          peer.close(1008, "fixture rejection")
          return
        }
        const bytes = message.uint8Array()
        seen.push(
          message.text().startsWith("{") ? message.text() : Array.from(bytes)
        )
        peer.send(message.text().startsWith("{") ? message.text() : bytes)
      },
      close: upstreamClosed,
    },
  })
  upstream.on("upgrade", (...args) => {
    void upstreamAdapter.handleUpgrade(...args)
  })
  upstream.listen(0, "127.0.0.1")
  await once(upstream, "listening")
  const previous = process.env.OMLX_API_URL
  process.env.OMLX_API_URL = `http://127.0.0.1:${upstream.address().port}`
  const publicServer = createServer()
  const publicAdapter = nodeAdapter({ hooks: route.realtimeHooks })
  publicServer.on("upgrade", (...args) => {
    void publicAdapter.handleUpgrade(...args)
  })
  publicServer.listen(0, "127.0.0.1")
  await once(publicServer, "listening")
  const client = new WebSocket(
    `ws://127.0.0.1:${publicServer.address().port}/v1/audio/transcriptions/realtime`
  )
  const echoes = []
  let echoed
  const replies = new Promise((resolve) => {
    echoed = resolve
  })
  client.addEventListener("message", (event) => {
    echoes.push(event.data)
    if (echoes.length === 2) echoed()
  })
  try {
    await new Promise((resolve, reject) => {
      client.addEventListener("open", resolve, { once: true })
      client.addEventListener("error", reject, { once: true })
    })
    const auth = JSON.stringify({
      type: "start",
      model: "fixture",
      api_key: "client-secret",
    })
    client.send(auth)
    client.send(new Uint8Array([0, 255, 1, 128]))
    await Promise.race([
      replies,
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error("No realtime replies")), 2000)
      ),
    ])
    assert.deepEqual(seen, [auth, [0, 255, 1, 128]])
    assert.equal(echoes[0], auth)
    client.close(1000, "finished")
    await Promise.race([
      closed,
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error("Upstream socket not closed")), 2000)
      ),
    ])
    const second = new WebSocket(
      `ws://127.0.0.1:${publicServer.address().port}/v1/audio/transcriptions/realtime`
    )
    const secondClosed = new Promise((resolve) =>
      second.addEventListener("close", resolve, { once: true })
    )
    await new Promise((resolve, reject) => {
      second.addEventListener("open", resolve, { once: true })
      second.addEventListener("error", reject, { once: true })
    })
    second.send("upstream-close")
    const closeEvent = await Promise.race([
      secondClosed,
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error("Client socket not closed")), 2000)
      ),
    ])
    assert.equal(closeEvent.code, 1008)
    assert.equal(closeEvent.reason, "fixture rejection")
  } finally {
    client.close()
    if (previous === undefined) delete process.env.OMLX_API_URL
    else process.env.OMLX_API_URL = previous
    publicServer.closeAllConnections()
    upstream.closeAllConnections()
    await Promise.all([
      new Promise((resolve) => publicServer.close(resolve)),
      new Promise((resolve) => upstream.close(resolve)),
    ])
  }
})
