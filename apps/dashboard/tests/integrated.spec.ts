import { expect, test } from "@playwright/test"
import type { Page } from "@playwright/test"

const model = "mlx-community/test-model"
const modelPath = `/models/${encodeURIComponent(model)}`

async function connect(page: Page, path = "/settings") {
  await page.goto(path)
  await page.getByRole("button", { name: "API access" }).first().click()
  await page.getByLabel("API key", { exact: true }).fill("dashboard-test-key")
  await page.getByRole("button", { name: "Connect", exact: true }).click()
  await expect(
    page.getByRole("dialog", { name: "Connect to oMLX" })
  ).toHaveCount(0)
}

test.beforeEach(async ({ request }) => {
  expect(
    (await request.post("http://127.0.0.1:8765/__test__/reset")).ok()
  ).toBe(true)
})

test("real management pages load without browser errors", async ({ page }) => {
  const errors: string[] = []
  page.on("pageerror", (error) => errors.push(error.message))
  await connect(page)
  for (const path of [
    "/settings",
    "/add-model",
    "/activity",
    "/monitoring",
    "/logs",
    "/diagnostics",
    "/cluster",
    `${modelPath}?tab=settings`,
    `${modelPath}?tab=profiles`,
  ]) {
    await page.goto(path)
    await expect(page.getByRole("heading", { level: 1 }).first()).toBeVisible()
    await expect(
      page.getByText("Connection required", { exact: true })
    ).toHaveCount(0)
  }
  expect(errors).toEqual([])
})

test("subkey lifecycle and main rotation retain only the rotating session", async ({
  page,
  browser,
  baseURL,
}) => {
  await connect(page)
  const old = await browser.newContext({ baseURL })
  try {
    expect(
      (
        await old.request.post("/api/connection", {
          headers: { Origin: baseURL! },
          data: { key: "dashboard-test-key" },
        })
      ).ok()
    ).toBe(true)
    await page.getByRole("tab", { name: "API keys", exact: true }).click()
    await page.getByRole("button", { name: "Create key", exact: true }).click()
    await page.getByLabel("Name", { exact: true }).fill("Notebook")
    await page.getByRole("button", { name: "Save key", exact: true }).click()
    const row = page
      .locator("div.rounded-lg.border.p-4")
      .filter({ hasText: "Notebook" })
    await expect(row).toBeVisible()
    await row.getByRole("button", { name: "Reveal Notebook key" }).click()
    await row.getByRole("button", { name: "Edit", exact: true }).click()
    await page.getByLabel("Name", { exact: true }).fill("Laptop")
    await page.getByRole("button", { name: "Save key", exact: true }).click()
    await page
      .locator("div.rounded-lg.border.p-4")
      .filter({ hasText: "Laptop" })
      .getByRole("button", { name: "Revoke", exact: true })
      .click()
    await page.getByLabel("Type REVOKE to confirm").fill("REVOKE")
    await page.getByRole("button", { name: "Revoke key", exact: true }).click()
    await expect(
      page.locator("div.rounded-lg.border.p-4").filter({ hasText: "Laptop" })
    ).toHaveCount(0)
    await page
      .getByRole("button", { name: "Rotate main key", exact: true })
      .click()
    await page.getByLabel("New main key").fill("rotated-fixture-key")
    await page.getByLabel("Type ROTATE to confirm").fill("ROTATE")
    await page.getByRole("button", { name: "Rotate key", exact: true }).click()
    await expect(page.getByText(/Main key rotated/)).toBeVisible()
    await page.reload()
    await expect(
      page.getByRole("heading", { name: "Server settings", exact: true })
    ).toBeVisible()
    expect((await page.request.get("/api/omlx/server/settings")).ok()).toBe(
      true
    )
    expect((await old.request.get("/api/omlx/server/settings")).status()).toBe(
      401
    )
  } finally {
    await old.close()
  }
})

test("nested settings persist and failed persistence keeps drafts", async ({
  page,
  request,
}) => {
  await connect(page)
  await page.getByRole("tab", { name: "Defaults", exact: true }).click()
  await page.getByLabel("Temperature", { exact: true }).fill("0.31")
  const saved = page.waitForRequest(
    (value) =>
      value.method() === "PATCH" && value.url().endsWith("/server/settings")
  )
  await page
    .getByRole("button", { name: "Review changes", exact: true })
    .click()
  await page
    .getByRole("button", { name: "Save 1 changes", exact: true })
    .click()
  expect((await saved).postDataJSON()).toEqual({
    sections: { sampling: { temperature: 0.31 } },
  })
  await expect(page.getByText("Settings saved", { exact: true })).toBeVisible()
  await page.reload()
  await page.getByRole("tab", { name: "Defaults", exact: true }).click()
  await expect(page.getByLabel("Temperature", { exact: true })).toHaveValue(
    "0.31"
  )
  await request.post("http://127.0.0.1:8765/__test__/fail-save")
  await page.getByLabel("Temperature", { exact: true }).fill("0.42")
  await page
    .getByRole("button", { name: "Review changes", exact: true })
    .click()
  await page
    .getByRole("button", { name: "Save 1 changes", exact: true })
    .click()
  await expect(
    page.getByText("Settings could not be saved", { exact: true })
  ).toBeVisible()
  await expect(page.getByLabel("Temperature", { exact: true })).toHaveValue(
    "0.42"
  )
  const settings = await page.request.get("/api/omlx/server/settings")
  expect((await settings.json()).sections.sampling.temperature).toBe(0.31)
})

test("retained history and rotated log filters use backend queries", async ({
  page,
}) => {
  await connect(page, "/monitoring")
  await expect(
    page.getByText("Historical usage", { exact: true })
  ).toBeVisible()
  const usage = await page.request.get(
    `/api/omlx/monitoring/usage?range=today&model=${encodeURIComponent(model)}`
  )
  expect(usage.ok()).toBe(true)
  const record = await usage.json()
  expect(record.state).toBe("available")
  expect(record.totals.requests).toBe(1)
  const yesterday = page.waitForResponse((response) => {
    const url = new URL(response.url())
    return (
      url.pathname.endsWith("/monitoring/usage") &&
      url.searchParams.get("range") === "yesterday"
    )
  })
  await page.getByLabel("Date range", { exact: true }).click()
  await page.getByRole("option", { name: "Yesterday", exact: true }).click()
  expect((await (await yesterday).json()).models).toEqual(
    expect.arrayContaining([
      expect.objectContaining({ model_id: "local/embedding", requests: 1 }),
    ])
  )
  await expect(
    page.locator("summary").filter({ hasText: "local/embedding" })
  ).toBeVisible()
  await page.goto("/logs")
  await expect(page.getByLabel("Server log tail")).toContainText(
    "fixture ready"
  )
  await page.getByLabel("Level", { exact: true }).click()
  await page.getByRole("option", { name: "ERROR", exact: true }).click()
  await expect(page.getByLabel("Server log tail")).toContainText(
    "fixture failure"
  )
  await expect(page.getByLabel("Server log tail")).not.toContainText(
    "fixture ready"
  )
  await expect(page.getByLabel("Server log tail")).not.toContainText(
    "fixture info continuation"
  )
  await expect(page.getByLabel("Server log tail")).toContainText(
    "traceback continuation"
  )
  await page.getByLabel("Level", { exact: true }).click()
  await page.getByRole("option", { name: "All levels", exact: true }).click()
  await page.getByLabel("Log file", { exact: true }).click()
  await page.getByRole("option", { name: "server.log.1", exact: true }).click()
  await expect(page.getByLabel("Server log tail")).toContainText(
    "rotated fixture record"
  )
})

test("operation history and local discovery expose real persisted contracts", async ({
  page,
}) => {
  await connect(page, "/activity")
  const operations = await page.request.get("/api/omlx/operations")
  expect(operations.ok()).toBe(true)
  expect((await operations.json()).operations).toEqual(
    expect.arrayContaining([
      expect.objectContaining({ kind: "fixture_verify", status: "succeeded" }),
    ])
  )
  await page.goto("/add-model")
  await page.getByRole("button", { name: "2. Prepare local model" }).click()
  const catalog = await page.request.get("/api/omlx/acquisition/prepare/models")
  expect(catalog.ok()).toBe(true)
  expect((await catalog.json()).models.length).toBeGreaterThanOrEqual(4)
  expect((await page.request.get("/api/omlx/templates")).ok()).toBe(true)
  expect(
    (
      await page.request.get(
        `/api/omlx/models/${encodeURIComponent(model)}/profiles`
      )
    ).ok()
  ).toBe(true)
})

test("templates and profiles preserve explicit field resets through the gateway", async ({
  page,
  baseURL,
}) => {
  await connect(page, `${modelPath}?tab=settings`)
  const headers = { Origin: baseURL! }
  const target = `/api/omlx/models/${encodeURIComponent(model)}`
  expect(
    (
      await page.request.post("/api/omlx/templates", {
        headers,
        data: {
          name: "reset-template",
          settings: { temperature: null, top_p: 0.8 },
        },
      })
    ).ok()
  ).toBe(true)
  expect(
    (
      await page.request.post(`${target}/templates/reset-template/apply`, {
        headers,
      })
    ).ok()
  ).toBe(true)
  await page.reload()
  await expect(page.getByLabel("Temperature", { exact: true })).toHaveValue("")
  await expect(page.getByLabel("Top P", { exact: true })).toHaveValue("0.8")
  expect(
    (
      await page.request.post(`${target}/profiles`, {
        headers,
        data: {
          name: "reset-profile",
          settings: { top_p: null, temperature: 0.25 },
        },
      })
    ).ok()
  ).toBe(true)
  expect(
    (
      await page.request.post(`${target}/profiles/reset-profile/apply`, {
        headers,
      })
    ).ok()
  ).toBe(true)
  await page.reload()
  await expect(page.getByLabel("Temperature", { exact: true })).toHaveValue(
    "0.25"
  )
  await expect(page.getByLabel("Top P", { exact: true })).toHaveValue("")
})

test("gateway rejects unsupported methods and encoded unsafe paths", async ({
  page,
  baseURL,
}) => {
  await connect(page)
  for (const path of [
    "/api/omlx/server/restart",
    "/api/omlx/models/local%2F..%2Fescape/settings",
    "/api/omlx/cluster/arbitrary-command",
  ]) {
    const response = await page.request.get(path)
    expect(response.status(), path).toBe(404)
    expect(await response.json(), path).toEqual({
      detail: "Unknown management operation.",
    })
  }
  // The HTTP adapter may normalize encoded backslashes to model separators
  // before the handler receives the URL. Either form remains a model lookup.
  const normalized = await page.request.get(
    "/api/omlx/models/local%5Cescape/settings"
  )
  expect(normalized.status()).toBe(404)
  expect((await normalized.json()).detail).toMatch(
    /Unknown management operation|Model not found/
  )
  const rejected = await page.request.post("/api/omlx/server/settings", {
    headers: { Origin: baseURL! },
    data: { sections: { sampling: { temperature: 0.99 } } },
  })
  expect(rejected.status()).toBe(404)
  expect(await rejected.json()).toEqual({
    detail: "Unknown management operation.",
  })
  const settings = await page.request.get("/api/omlx/server/settings")
  expect((await settings.json()).sections.sampling.temperature).not.toBe(0.99)
})
