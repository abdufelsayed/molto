import { spawn } from "node:child_process"
import type { ChildProcess } from "node:child_process"
import { expect, test } from "@playwright/test"

const local = "http://127.0.0.1:3008"
const publicBind = "http://127.0.0.1:3009"
const children: ChildProcess[] = []

test.beforeAll(async () => {
  for (const [host, port] of [
    ["127.0.0.1", "3008"],
    ["0.0.0.0", "3009"],
  ]) {
    const child = spawn(process.execPath, [".output/server/index.mjs"], {
      env: {
        ...process.env,
        HOST: host,
        NITRO_HOST: host,
        PORT: port,
        NITRO_PORT: port,
        MOLTO_API_URL: "http://127.0.0.1:8765",
        MOLTO_LOCAL_ACCESS_TOKEN: "browser-fixture-capability",
      },
      stdio: "ignore",
    })
    children.push(child)
    await expect
      .poll(async () => {
        if (child.exitCode !== null)
          throw new Error("Local dashboard exited during startup")
        return fetch(`http://127.0.0.1:${port}/api/connection`).then(
          (r) => r.status,
          () => 0
        )
      })
      .toBe(200)
  }
})
test.afterAll(async () => {
  await Promise.all(
    children.map(
      (child) =>
        new Promise<void>((resolve) => {
          if (child.exitCode !== null) {
            resolve()
            return
          }
          child.once("exit", () => resolve())
          child.kill("SIGTERM")
        })
    )
  )
})
test.beforeEach(async ({ request }) => {
  await request.post("http://127.0.0.1:8765/__test__/reset")
})
test.afterEach(async ({ request }) => {
  await request.post("http://127.0.0.1:8765/__test__/reset")
})

test("bundled local dashboard opens key management and creates inference keys without a login key", async ({
  page,
  context,
}) => {
  const opened = page.waitForResponse(
    (r) =>
      r.url() === `${local}/api/local-session` &&
      r.request().method() === "POST"
  )
  await page.goto(`${local}/settings`)
  await page.getByRole("tab", { name: "API keys", exact: true }).click()
  await expect(
    page.getByRole("button", { name: "Create key", exact: true })
  ).toBeVisible()
  const response = await opened
  expect(response.status()).toBe(200)
  expect(await response.text()).not.toContain("dashboard-test-key")
  const cookie = (await context.cookies()).find(
    (c) => c.name === "molto_dashboard_session"
  )
  expect(cookie?.httpOnly).toBe(true)
  expect(cookie?.sameSite).toBe("Strict")
  expect(cookie?.value).not.toContain("dashboard-test-key")
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
  ).not.toContain("dashboard-test-key")
  await page.getByRole("button", { name: "Create key", exact: true }).click()
  await page.getByLabel("Name", { exact: true }).fill("Local notebook")
  await page.getByRole("button", { name: "Save key", exact: true }).click()
  const row = page
    .locator("div.rounded-lg.border.p-4")
    .filter({ hasText: "Local notebook" })
  await expect(row).toBeVisible()
  await page.reload()
  await page.getByRole("tab", { name: "API keys", exact: true }).click()
  await expect(row).toBeVisible()
  await context.clearCookies()
  await page.reload()
  await page.getByRole("tab", { name: "API keys", exact: true }).click()
  await expect(
    page.getByRole("button", { name: "Create key", exact: true })
  ).toBeVisible()
})

test("local disconnect survives reload and can reconnect without copying a key", async ({
  page,
}) => {
  await page.goto(`${local}/settings`)
  await page.getByRole("button", { name: "Disconnect", exact: true }).click()
  await expect(
    page.getByText("Connection required", { exact: true }).first()
  ).toBeVisible()
  await page.reload()
  await expect(
    page.getByText("Connection required", { exact: true }).first()
  ).toBeVisible()
  await page
    .getByRole("button", { name: "API access", exact: true })
    .first()
    .click()
  await page
    .getByRole("button", { name: "Connect locally", exact: true })
    .click()
  await expect(page.getByRole("dialog")).toHaveCount(0)
  await expect(
    page.getByRole("button", { name: "Disconnect", exact: true })
  ).toBeVisible()
})

test("a public bind still requires the main key", async ({ page }) => {
  await page.goto(`${publicBind}/settings`)
  await expect(
    page.getByText("Connection required", { exact: true }).first()
  ).toBeVisible()
  await page
    .getByRole("button", { name: "API access", exact: true })
    .first()
    .click()
  await expect(
    page.getByRole("button", { name: "Connect locally", exact: true })
  ).toHaveCount(0)
  await page.getByLabel("API key", { exact: true }).fill("dashboard-test-key")
  await page.getByRole("button", { name: "Connect", exact: true }).click()
  await expect(
    page.getByRole("button", { name: "Disconnect", exact: true })
  ).toBeVisible()
})

test("new local sessions use the main key rotated by another client", async ({
  page,
  request,
}) => {
  await page.goto(`${local}/settings`)
  await expect(
    page.getByRole("button", { name: "Disconnect", exact: true })
  ).toBeVisible()
  const result = await request.patch(
    "http://127.0.0.1:8765/management/v1/auth/main-key",
    {
      headers: { Authorization: "Bearer dashboard-test-key" },
      data: { key: "rotated-by-another-client" },
    }
  )
  expect(result.status()).toBe(200)
  await page.reload()
  await page.getByRole("tab", { name: "API keys", exact: true }).click()
  await expect(
    page.getByRole("button", { name: "Create key", exact: true })
  ).toBeVisible()
})
