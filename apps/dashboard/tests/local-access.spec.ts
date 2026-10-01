import { readFile } from "node:fs/promises"
import ts from "typescript"
import { expect, test } from "@playwright/test"

type LocalModule = {
  localSessionHandler: (event: {
    req: Request & { ip?: string }
  }) => Promise<Response>
}
type Gateway = {
  session: (req: Request) => { key: string } | undefined
  disconnect: (req: Request) => Response
}
async function modules() {
  const compile = (source: string) =>
    `data:text/javascript;base64,${Buffer.from(ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } }).outputText).toString("base64")}`
  const schema = await readFile(
    "../../packages/contracts/generated/openapi.json",
    "utf8"
  )
  const targetUrl = compile(
    (await readFile("src/server/management-target.server.ts", "utf8")).replace(
      'import openapi from "@molto/contracts/openapi.json"',
      () => `const openapi = ${schema}`
    )
  )
  const gatewayUrl = compile(
    (await readFile("src/server/connection.server.ts", "utf8")).replace(
      '"./management-target.server"',
      () => JSON.stringify(targetUrl)
    )
  )
  const guardUrl = compile(
    await readFile("server/utils/local-access.ts", "utf8")
  )
  const source = (await readFile("server/routes/api/local-session.ts", "utf8"))
    .replace(
      'import { defineHandler } from "nitro"',
      "const defineHandler = (handler:unknown) => handler"
    )
    .replace('"../../utils/local-access"', JSON.stringify(guardUrl))
    .replace(
      '"../../../src/server/connection.server"',
      JSON.stringify(gatewayUrl)
    )
  return {
    local: (await import(compile(source))) as LocalModule,
    gateway: (await import(gatewayUrl)) as Gateway,
  }
}
const origin = "http://127.0.0.1:3000"
function req(
  headers: HeadersInit = { origin },
  ip: string | undefined = "127.0.0.1",
  url = `${origin}/api/local-session`,
  method = "POST",
  body: unknown = { automatic: true }
): Request & { ip?: string } {
  return Object.assign(
    new Request(url, {
      method,
      headers,
      ...(method === "POST" ? { body: JSON.stringify(body) } : {}),
    }),
    { ip }
  )
}

test("local session checks peer, binding, origin and forwarding headers before accessing a credential", async () => {
  const { local } = await modules()
  const originalFetch = globalThis.fetch
  const previous = {
    HOST: process.env.HOST,
    NITRO_HOST: process.env.NITRO_HOST,
    MOLTO_LOCAL_ACCESS_TOKEN: process.env.MOLTO_LOCAL_ACCESS_TOKEN,
  }
  let calls = 0
  process.env.NITRO_HOST = "127.0.0.1"
  process.env.MOLTO_LOCAL_ACCESS_TOKEN = "test-capability"
  globalThis.fetch = () => {
    calls++
    return Promise.resolve(Response.json({ key: "main-secret" }))
  }
  try {
    const forbidden = [
      req({ origin: "https://attacker.example" }),
      req({}),
      req({ origin, "sec-fetch-site": "cross-site" }),
      req({ origin }, "203.0.113.7"),
      req(
        { origin: "http://attacker.example:3000" },
        "127.0.0.1",
        "http://attacker.example:3000/api/local-session"
      ),
      ...[
        "forwarded",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-proto",
        "x-real-ip",
      ].map((name) => req({ origin, [name]: "127.0.0.1" })),
    ]
    for (const request of forbidden)
      expect((await local.localSessionHandler({ req: request })).status).toBe(
        403
      )
    const missingPeer = req()
    delete missingPeer.ip
    expect((await local.localSessionHandler({ req: missingPeer })).status).toBe(
      403
    )
    process.env.NITRO_HOST = "0.0.0.0"
    expect((await local.localSessionHandler({ req: req() })).status).toBe(403)
    process.env.NITRO_HOST = "127.0.0.1"
    delete process.env.MOLTO_LOCAL_ACCESS_TOKEN
    expect((await local.localSessionHandler({ req: req() })).status).toBe(403)
    expect(calls).toBe(0)
  } finally {
    globalThis.fetch = originalFetch
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key]
      else process.env[key] = value
    }
  }
})

test("local bootstrap exposes only an opaque session and honors explicit disconnect", async () => {
  const { local, gateway } = await modules()
  const originalFetch = globalThis.fetch
  const previous = {
    NITRO_HOST: process.env.NITRO_HOST,
    MOLTO_LOCAL_ACCESS_TOKEN: process.env.MOLTO_LOCAL_ACCESS_TOKEN,
  }
  process.env.NITRO_HOST = "127.0.0.1"
  process.env.MOLTO_LOCAL_ACCESS_TOKEN = "test-capability"
  let calls = 0
  globalThis.fetch = (url, options) => {
    calls++
    const address =
      typeof url === "string" ? url : url instanceof URL ? url.href : url.url
    expect(address).toMatch(/\/_internal\/dashboard-key$/)
    expect(options?.method).toBe("POST")
    const headers = new Headers(options?.headers)
    expect(headers.get("x-molto-local-access")).toBe("test-capability")
    expect(headers.get("cookie")).toBeNull()
    expect(headers.get("authorization")).toBeNull()
    return Promise.resolve(Response.json({ key: "main-secret" }))
  }
  try {
    const status = await local.localSessionHandler({
      req: req({}, "127.0.0.1", undefined, "GET"),
    })
    expect(await status.json()).toEqual({ available: true })
    expect(status.headers.get("set-cookie")).toBeNull()
    expect(calls).toBe(0)
    const opened = await local.localSessionHandler({ req: req() })
    expect(opened.status).toBe(200)
    expect(opened.headers.get("cache-control")).toBe("no-store")
    const cookie = opened.headers.get("set-cookie")!
    expect(cookie).toMatch(/HttpOnly; SameSite=Strict/)
    expect(cookie).not.toContain("main-secret")
    expect(await opened.text()).not.toContain("main-secret")
    expect(gateway.session(req({ origin, cookie }))?.key).toBe("main-secret")
    const closed = gateway.disconnect(req({ origin, cookie }))
    const disconnected = closed.headers.get("set-cookie")!
    const automatic = await local.localSessionHandler({
      req: req({ origin, cookie: disconnected }),
    })
    expect(automatic.status).toBe(409)
    expect(calls).toBe(1)
    const manual = await local.localSessionHandler({
      req: req({ origin, cookie: disconnected }, undefined, undefined, "POST", {
        automatic: false,
      }),
    })
    expect(manual.status).toBe(200)
    globalThis.fetch = () =>
      Promise.resolve(Response.json({ detail: "no key" }, { status: 409 }))
    const fresh = await local.localSessionHandler({ req: req() })
    expect(fresh.status).toBe(409)
    expect(fresh.headers.get("set-cookie")).toBeNull()
  } finally {
    globalThis.fetch = originalFetch
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key]
      else process.env[key] = value
    }
  }
})
