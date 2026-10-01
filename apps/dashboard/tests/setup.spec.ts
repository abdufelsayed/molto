import { readFile } from "node:fs/promises"
import ts from "typescript"
import { expect, test } from "@playwright/test"
import type { Page } from "@playwright/test"

type SetupModule = {
  setupTransportAllowed: (
    request: { ip?: string; headers: Headers },
    bindHost?: string
  ) => boolean
  setupHandler: (event: { req: Request & { ip?: string } }) => Promise<Response>
}
async function setupModule() {
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
  const source = (await readFile("server/routes/api/setup.ts", "utf8"))
    .replace(
      'import { defineHandler } from "nitro"',
      "const defineHandler = (handler:unknown) => handler"
    )
    .replace(
      '"../../../src/server/connection.server"',
      JSON.stringify(gatewayUrl)
    )
  const setup = (await import(compile(source))) as SetupModule
  return { setup }
}
function setupRequest(
  method = "POST",
  body: unknown = { key: "first-main-key", confirmation: "first-main-key" },
  headers: HeadersInit = { origin: "http://127.0.0.1:3000" },
  ip = "127.0.0.1"
) {
  return Object.assign(
    new Request("http://127.0.0.1:3000/api/setup", {
      method,
      headers,
      ...(method === "POST" ? { body: JSON.stringify(body) } : {}),
    }),
    { ip }
  )
}

test("setup transport guards reject public binding, nonlocal peers, and forged forwarding headers", async () => {
  const { setup } = await setupModule()
  expect(
    setup.setupTransportAllowed(
      { ip: "127.0.0.1", headers: new Headers() },
      "127.0.0.1"
    )
  ).toBe(true)
  expect(
    setup.setupTransportAllowed(
      { ip: "::ffff:127.0.0.1", headers: new Headers() },
      "::1"
    )
  ).toBe(true)
  for (const [ip, host] of [
    ["203.0.113.7", "127.0.0.1"],
    ["127.0.0.1", "0.0.0.0"],
    ["127.0.0.1", "::"],
    [undefined, "127.0.0.1"],
    ["127.0.0.1", ""],
  ])
    expect(
      setup.setupTransportAllowed({ ip, headers: new Headers() }, host)
    ).toBe(false)
  for (const header of [
    "forwarded",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "x-real-ip",
  ])
    expect(
      setup.setupTransportAllowed(
        { ip: "127.0.0.1", headers: new Headers({ [header]: "127.0.0.1" }) },
        "127.0.0.1"
      )
    ).toBe(false)
})

test("setup BFF rejects cross-origin and invalid confirmations before calling upstream", async () => {
  const { setup } = await setupModule()
  const originalFetch = globalThis.fetch
  const previousHost = process.env.HOST
  const previousNitroHost = process.env.NITRO_HOST
  let calls = 0
  process.env.HOST = "127.0.0.1"
  delete process.env.NITRO_HOST
  globalThis.fetch = () => {
    calls++
    return Promise.resolve(Response.json({ configured: true }))
  }
  try {
    expect(
      (
        await setup.setupHandler({
          req: setupRequest("POST", undefined, {
            origin: "https://attacker.example",
          }),
        })
      ).status
    ).toBe(403)
    expect(
      (
        await setup.setupHandler({
          req: setupRequest("POST", {
            key: "first-key",
            confirmation: "different",
          }),
        })
      ).status
    ).toBe(422)
    expect(
      (
        await setup.setupHandler({
          req: setupRequest("POST", {
            key: "with spaces",
            confirmation: "with spaces",
          }),
        })
      ).status
    ).toBe(422)
    expect(
      (
        await setup.setupHandler({
          req: setupRequest("POST", undefined, {
            origin: "http://127.0.0.1:3000",
            "sec-fetch-site": "cross-site",
          }),
        })
      ).status
    ).toBe(403)
    expect(
      (
        await setup.setupHandler({
          req: setupRequest("POST", undefined, undefined, "203.0.113.8"),
        })
      ).status
    ).toBe(403)
    const remoteStatus = await setup.setupHandler({
      req: setupRequest("GET", undefined, undefined, "203.0.113.8"),
    })
    expect(await remoteStatus.json()).toMatchObject({
      setup_required: false,
      allowed: false,
    })
    expect(
      (
        await setup.setupHandler({
          req: setupRequest("POST", undefined, {
            origin: "http://127.0.0.1:3000",
            "x-forwarded-for": "127.0.0.1",
          }),
        })
      ).status
    ).toBe(403)
    const rebound = Object.assign(
      new Request("http://attacker.example:3000/api/setup", {
        method: "POST",
        headers: { origin: "http://attacker.example:3000" },
        body: JSON.stringify({
          key: "first-main-key",
          confirmation: "first-main-key",
        }),
      }),
      { ip: "127.0.0.1" }
    )
    expect((await setup.setupHandler({ req: rebound })).status).toBe(403)
    process.env.HOST = "0.0.0.0"
    expect((await setup.setupHandler({ req: setupRequest() })).status).toBe(403)
    expect(calls).toBe(0)
  } finally {
    globalThis.fetch = originalFetch
    if (previousHost === undefined) delete process.env.HOST
    else process.env.HOST = previousHost
    if (previousNitroHost === undefined) delete process.env.NITRO_HOST
    else process.env.NITRO_HOST = previousNitroHost
  }
})

test("setup BFF filters safe status and defers session creation to the canonical connection route", async () => {
  const { setup } = await setupModule()
  const originalFetch = globalThis.fetch
  const previousHost = process.env.HOST
  const previousNitroHost = process.env.NITRO_HOST
  process.env.HOST = "127.0.0.1"
  delete process.env.NITRO_HOST
  const calls: { path: string; headers: Headers; body: unknown }[] = []
  globalThis.fetch = (url, options) => {
    const path = new URL(
      typeof url === "string" ? url : url instanceof URL ? url.href : url.url
    ).pathname
    calls.push({
      path,
      headers: new Headers(options?.headers),
      body:
        typeof options?.body === "string"
          ? (JSON.parse(options.body) as unknown)
          : null,
    })
    if (path.endsWith("/setup"))
      return Promise.resolve(
        Response.json(
          options?.method === "POST"
            ? { configured: true, main_key: "not-to-browser" }
            : {
                setup_required: true,
                allowed: true,
                reason: null,
                main_key: "not-to-browser",
              }
        )
      )
    return Promise.resolve(Response.json({ status: "running" }))
  }
  try {
    const status = await setup.setupHandler({ req: setupRequest("GET") })
    expect(await status.json()).toEqual({
      setup_required: true,
      allowed: true,
      reason: null,
    })
    const response = await setup.setupHandler({ req: setupRequest() })
    expect(response.status).toBe(200)
    expect(response.headers.get("cache-control")).toBe("no-store")
    expect(response.headers.get("set-cookie")).toBeNull()
    expect(await response.json()).toEqual({ configured: true })
    expect(calls.map((call) => call.path)).toEqual([
      "/management/v1/setup",
      "/management/v1/setup",
    ])
    const mutation = calls.find((call) => call.body !== null)!
    expect(mutation.path).toBe("/management/v1/setup")
    expect(mutation.headers.get("authorization")).toBeNull()
    expect(mutation.headers.get("forwarded")).toBeNull()
    expect(mutation.headers.get("x-forwarded-for")).toBeNull()
    expect(mutation.body).toEqual({
      key: "first-main-key",
      confirmation: "first-main-key",
    })
  } finally {
    globalThis.fetch = originalFetch
    if (previousHost === undefined) delete process.env.HOST
    else process.env.HOST = previousHost
    if (previousNitroHost === undefined) delete process.env.NITRO_HOST
    else process.env.NITRO_HOST = previousNitroHost
  }
})

async function mockSetup(
  page: Page,
  setupAvailable = true,
  failSetup = false,
  failConnection = false
) {
  const submitted: unknown[] = []
  let configured = false
  let connected = false
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path === "/api/setup") {
      if (route.request().method() === "POST") {
        submitted.push(route.request().postDataJSON())
        if (failSetup) {
          await route.fulfill({
            status: 500,
            json: {
              detail: "Could not persist the main key. No key was configured.",
            },
          })
          return
        }
        configured = true
        await route.fulfill({
          json: { configured: true },
        })
      } else
        await route.fulfill({
          json: {
            setup_required: setupAvailable && !configured,
            allowed: setupAvailable && !configured,
            reason: null,
          },
        })
    } else if (path === "/api/connection") {
      if (route.request().method() === "POST") {
        if (failConnection) {
          await route.fulfill({
            status: 503,
            json: { detail: "Fixture connection unavailable." },
          })
          return
        }
        connected = true
      }
      await route.fulfill({
        json: { connected, server: "http://127.0.0.1:8000" },
      })
    } else
      await route.fulfill({
        status: 401,
        json: { detail: "Connect with the main key." },
      })
  })
  return submitted
}

test("first-run dialog generates, reveals, copies, confirms, and creates the main key without browser storage", async ({
  page,
}) => {
  const submitted = await mockSetup(page)
  await page.context().grantPermissions(["clipboard-read", "clipboard-write"])
  await page.goto("/settings")
  await page
    .getByRole("button", { name: "API access", exact: true })
    .first()
    .click()
  await expect(
    page.getByRole("dialog", { name: "Set up Molto access" })
  ).toBeVisible()
  await page
    .getByLabel("New main API key", { exact: true })
    .fill("first-browser-key")
  await page
    .getByLabel("Confirm main API key", { exact: true })
    .fill("different-key")
  await expect(
    page.getByRole("button", {
      name: "Create main key and connect",
      exact: true,
    })
  ).toBeDisabled()
  await page
    .getByRole("button", { name: "Generate secure key", exact: true })
    .click()
  const key = await page
    .getByLabel("New main API key", { exact: true })
    .inputValue()
  expect(key).toMatch(/^molto_[0-9a-f]{64}$/)
  await expect(
    page.getByLabel("New main API key", { exact: true })
  ).toHaveAttribute("type", "password")
  await page
    .getByRole("button", { name: "Reveal new main key", exact: true })
    .click()
  await expect(
    page.getByLabel("New main API key", { exact: true })
  ).toHaveAttribute("type", "text")
  await page.getByRole("button", { name: "Copy key", exact: true }).click()
  await expect(
    page.getByText("Key copied. Save it in your password manager.")
  ).toBeVisible()
  const connectionRequest = page.waitForRequest(
    (request) =>
      new URL(request.url()).pathname === "/api/connection" &&
      request.method() === "POST"
  )
  await page
    .getByRole("button", { name: "Create main key and connect", exact: true })
    .click()
  expect((await connectionRequest).postDataJSON()).toEqual({ key })
  await expect(page.getByRole("dialog")).toHaveCount(0)
  expect(submitted).toEqual([{ key, confirmation: key }])
  expect(
    await page.evaluate(() =>
      JSON.stringify({
        local: Object.fromEntries(
          Object.keys(localStorage).map((key) => [
            key,
            localStorage.getItem(key),
          ])
        ),
        session: Object.fromEntries(
          Object.keys(sessionStorage).map((key) => [
            key,
            sessionStorage.getItem(key),
          ])
        ),
        cookies: document.cookie,
      })
    )
  ).not.toContain(key)
})

test("configured servers retain the existing main-key connection flow", async ({
  page,
}) => {
  await mockSetup(page, false)
  await page.goto("/settings")
  await page
    .getByRole("button", { name: "API access", exact: true })
    .first()
    .click()
  await expect(
    page.getByRole("dialog", { name: "Connect to Molto" })
  ).toBeVisible()
  await expect(page.getByLabel("API key", { exact: true })).toBeVisible()
  await expect(
    page.getByRole("button", { name: "Generate secure key", exact: true })
  ).toHaveCount(0)
})

test("setup failure preserves the key draft and allows an explicit retry", async ({
  page,
}) => {
  const submitted = await mockSetup(page, true, true)
  await page.goto("/settings")
  await page
    .getByRole("button", { name: "API access", exact: true })
    .first()
    .click()
  await expect(
    page.getByRole("dialog", { name: "Set up Molto access" })
  ).toBeVisible()
  await page
    .getByLabel("New main API key", { exact: true })
    .fill("first-browser-key")
  await page
    .getByLabel("Confirm main API key", { exact: true })
    .fill("first-browser-key")
  await page
    .getByRole("button", { name: "Create main key and connect", exact: true })
    .click()
  await expect(
    page.getByText("Could not persist the main key. No key was configured.")
  ).toBeVisible()
  await expect(
    page.getByLabel("New main API key", { exact: true })
  ).toHaveValue("first-browser-key")
  await expect(
    page.getByRole("button", {
      name: "Create main key and connect",
      exact: true,
    })
  ).toBeEnabled()
  expect(submitted).toHaveLength(1)
})

test("local first-run setup persists the real fixture key and connects the normal dashboard session", async ({
  page,
  request,
  context,
}) => {
  await request.post("http://127.0.0.1:8765/__test__/reset?setup=true")
  const key = "fixture-first-run-main-key"
  try {
    expect(
      await (
        await request.get("http://127.0.0.1:8765/__test__/persisted-auth")
      ).json()
    ).toEqual({ api_key: null })
    await page.goto("/settings")
    await expect(
      page.getByText("Connection required", { exact: true }).first()
    ).toBeVisible()
    await page
      .getByRole("button", { name: "API access", exact: true })
      .first()
      .click()
    await expect(
      page.getByRole("dialog", { name: "Set up Molto access" })
    ).toBeVisible()
    await page.getByLabel("New main API key", { exact: true }).fill(key)
    await page.getByLabel("Confirm main API key", { exact: true }).fill(key)
    await page
      .getByRole("button", { name: "Create main key and connect", exact: true })
      .click()
    await expect(page.getByRole("dialog")).toHaveCount(0)
    await expect(
      page.getByText("Running server", { exact: true })
    ).toBeVisible()
    expect(
      await (
        await request.get("http://127.0.0.1:8765/__test__/persisted-auth")
      ).json()
    ).toEqual({ api_key: key })
    const cookie = (await context.cookies()).find(
      (value) => value.name === "molto_dashboard_session"
    )
    expect(cookie?.httpOnly).toBe(true)
    expect(cookie?.sameSite).toBe("Strict")
    expect(cookie?.value).not.toContain(key)
    expect(
      await page.evaluate(() =>
        JSON.stringify({
          local: Object.fromEntries(
            Object.keys(localStorage).map((name) => [
              name,
              localStorage.getItem(name),
            ])
          ),
          session: Object.fromEntries(
            Object.keys(sessionStorage).map((name) => [
              name,
              sessionStorage.getItem(name),
            ])
          ),
          cookies: document.cookie,
        })
      )
    ).not.toContain(key)
    await page.reload()
    await expect(
      page.getByText("Running server", { exact: true })
    ).toBeVisible()
    const repeated = await request.post("/api/setup", {
      headers: { origin: new URL(page.url()).origin },
      data: { key: "replacement-key", confirmation: "replacement-key" },
    })
    expect(repeated.status()).toBe(409)
    expect(
      await (
        await request.get("http://127.0.0.1:8765/__test__/persisted-auth")
      ).json()
    ).toEqual({ api_key: key })
  } finally {
    await request.post("http://127.0.0.1:8765/__test__/reset")
  }
})

test("real fixture setup persistence failure leaves first-run state and the draft intact", async ({
  page,
  request,
}) => {
  await request.post("http://127.0.0.1:8765/__test__/reset?setup=true")
  await request.post("http://127.0.0.1:8765/__test__/fail-save")
  try {
    await page.goto("/settings")
    await expect(
      page.getByText("Connection required", { exact: true }).first()
    ).toBeVisible()
    await page
      .getByRole("button", { name: "API access", exact: true })
      .first()
      .click()
    await expect(
      page.getByRole("dialog", { name: "Set up Molto access" })
    ).toBeVisible()
    await page
      .getByLabel("New main API key", { exact: true })
      .fill("fixture-failed-setup-key")
    await page
      .getByLabel("Confirm main API key", { exact: true })
      .fill("fixture-failed-setup-key")
    const result = page.waitForResponse(
      (response) =>
        new URL(response.url()).pathname === "/api/setup" &&
        response.request().method() === "POST"
    )
    await page
      .getByRole("button", { name: "Create main key and connect", exact: true })
      .click()
    expect((await result).status()).toBe(500)
    await expect(
      page.getByLabel("New main API key", { exact: true })
    ).toHaveValue("fixture-failed-setup-key")
    await expect(
      page.getByRole("button", {
        name: "Create main key and connect",
        exact: true,
      })
    ).toBeEnabled()
    expect(
      await (
        await request.get("http://127.0.0.1:8765/__test__/persisted-auth")
      ).json()
    ).toEqual({ api_key: null })
    expect(await (await request.get("/api/setup")).json()).toMatchObject({
      setup_required: true,
      allowed: true,
    })
    expect(
      (
        await request.get("http://127.0.0.1:8765/management/v1/state", {
          headers: { authorization: "Bearer fixture-failed-setup-key" },
        })
      ).status()
    ).toBe(401)
  } finally {
    await request.post("http://127.0.0.1:8765/__test__/reset")
  }
})

test("a created key with failed connection recovers through ordinary login without creating it again", async ({
  page,
}) => {
  const submitted = await mockSetup(page, true, false, true)
  await page.goto("/settings")
  await page
    .getByRole("button", { name: "API access", exact: true })
    .first()
    .click()
  await expect(
    page.getByRole("dialog", { name: "Set up Molto access" })
  ).toBeVisible()
  await page
    .getByLabel("New main API key", { exact: true })
    .fill("created-main-key")
  await page
    .getByLabel("Confirm main API key", { exact: true })
    .fill("created-main-key")
  await page
    .getByRole("button", { name: "Create main key and connect", exact: true })
    .click()
  await expect(
    page.getByText(
      /Your main key was created, but the dashboard connection failed/
    )
  ).toBeVisible()
  await expect(
    page.getByRole("dialog", { name: "Connect to Molto" })
  ).toBeVisible()
  await expect(page.getByLabel("API key", { exact: true })).toHaveValue(
    "created-main-key"
  )
  await page.getByRole("button", { name: "Connect", exact: true }).click()
  await expect(
    page.getByText("Fixture connection unavailable.", { exact: true })
  ).toBeVisible()
  expect(submitted).toEqual([
    { key: "created-main-key", confirmation: "created-main-key" },
  ])
})
