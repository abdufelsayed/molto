import { test, expect } from "@playwright/test"
import type { Page } from "@playwright/test"

const model = "mlx-community/test-model"
const modelPath = `/models/${encodeURIComponent(model)}`
async function connect(page: Page) {
  await page.getByRole("button", { name: "API access" }).first().click()
  await page.getByLabel("API key", { exact: true }).fill("dashboard-test-key")
  await page.getByRole("button", { name: "Connect", exact: true }).click()
  await expect(
    page.getByRole("dialog", { name: "Connect to oMLX" })
  ).toHaveCount(0)
}
test.beforeEach(async ({ request }) => {
  await request.post("http://127.0.0.1:8765/__test__/reset")
})
test("main-key authentication, session persistence, overview and signout", async ({
  page,
  context,
}) => {
  const errors: string[] = []
  page.on("pageerror", (error) => errors.push(error.message))
  await page.goto("/")
  await expect(
    page.getByText("Connection required", { exact: true }).first()
  ).toBeVisible()
  await page.getByRole("button", { name: "API access" }).first().click()
  await page.getByLabel("API key", { exact: true }).fill("inference-sub-key")
  await page.getByRole("button", { name: "Connect", exact: true }).click()
  await expect(page.getByText(/oMLX rejected this key/)).toBeVisible()
  await page.getByLabel("API key", { exact: true }).fill("dashboard-test-key")
  await page.getByRole("button", { name: "Connect", exact: true }).click()
  await expect(
    page.getByRole("link", { name: model, exact: true })
  ).toBeVisible()
  await expect(page.getByText("45.0 tok/s", { exact: true })).toBeVisible()
  const cookie = (await context.cookies()).find(
    (value) => value.name === "omlx_dashboard_session"
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
  await page.reload()
  await expect(
    page.getByRole("link", { name: model, exact: true })
  ).toBeVisible()
  await page.getByRole("link", { name: "Settings", exact: true }).click()
  await page.getByRole("button", { name: "Disconnect", exact: true }).click()
  await expect(
    page.getByText("Connection required", { exact: true }).first()
  ).toBeVisible()
  expect(errors).toEqual([])
})
test("URL filters, model IDs containing slashes, load and real rescan", async ({
  page,
}) => {
  await page.goto("/models")
  await connect(page)
  await page.getByLabel("Search library").fill("unloaded")
  await expect(page).toHaveURL(/q=unloaded/)
  await page
    .getByRole("button", { name: "local/unloaded-model", exact: true })
    .click()
  await page
    .getByRole("link", { name: "Open model configuration", exact: true })
    .click()
  await expect(
    page.getByRole("heading", { name: "local/unloaded-model", exact: true })
  ).toBeVisible()
  await page.getByRole("button", { name: "Load model", exact: true }).click()
  await expect(
    page.getByRole("button", { name: "Unload model", exact: true })
  ).toBeVisible()
  await page
    .getByRole("button", { name: "Back to models", exact: true })
    .click()
  await page.getByRole("button", { name: "Rescan directories" }).click()
  await page.getByLabel("Search library").fill("new-checkpoint")
  await expect(
    page.getByRole("button", { name: "local/new-checkpoint", exact: true })
  ).toBeVisible()
})
test("deferred unload remains pending until backend finishes", async ({
  page,
}) => {
  await page.goto("/models/local%2Fbusy-model")
  await connect(page)
  await page.getByRole("button", { name: "Unload model", exact: true }).click()
  await page.getByRole("button", { name: "Confirm unload" }).click()
  await expect(page.getByText(/Unload requested; waiting/)).toBeVisible()
  await expect(
    page.getByRole("button", { name: "Load model", exact: true })
  ).toBeVisible()
})
test("model edits survive refresh, patch only edited fields and reset explicitly", async ({
  page,
}) => {
  await page.goto(`${modelPath}?tab=settings`)
  await connect(page)
  await page.getByLabel("Temperature", { exact: true }).fill("0.3")
  await page.getByRole("button", { name: "Refresh dashboard" }).click()
  await expect(page.getByLabel("Temperature", { exact: true })).toHaveValue(
    "0.3"
  )
  const request = page.waitForRequest(
    (value) => value.method() === "PATCH" && value.url().endsWith("/settings")
  )
  await page.getByRole("button", { name: "Save changes", exact: true }).click()
  expect((await request).postDataJSON()).toEqual({ temperature: 0.3 })
  await expect(page.getByText("Settings saved.", { exact: true })).toBeVisible()
  await page
    .getByRole("button", { name: "Reset Temperature to default" })
    .click()
  await page.getByRole("button", { name: "Save changes", exact: true }).click()
  await expect(page.getByLabel("Temperature", { exact: true })).toHaveValue("")
  await page.getByRole("switch", { name: "Keep model pinned" }).click()
  await page
    .getByRole("switch", { name: "Hide from inference model list" })
    .click()
  await page.getByRole("switch", { name: "Use as default model" }).click()
  await page.getByRole("button", { name: "Save changes", exact: true }).click()
  await expect(page.getByText("Settings saved.", { exact: true })).toBeVisible()
  await page.reload()
  await expect(
    page.getByRole("switch", { name: "Keep model pinned" })
  ).toBeChecked()
})
test("global settings report restart requirements", async ({ page }) => {
  await page.goto("/settings")
  await connect(page)
  await page
    .getByRole("tab", { name: "Models & resources", exact: true })
    .click()
  await page.getByLabel("Max concurrent requests", { exact: true }).fill("3")
  await page
    .getByRole("button", { name: "Review changes", exact: true })
    .click()
  await page
    .getByRole("button", { name: "Save 1 changes", exact: true })
    .click()
  await expect(
    page.getByText(/Restart required for: scheduler.max_concurrent_requests/)
  ).toBeVisible()
  await page.reload()
  await page
    .getByRole("tab", { name: "Models & resources", exact: true })
    .click()
  await expect(
    page.getByLabel("Max concurrent requests", { exact: true })
  ).toHaveValue("3")
})
test("profile create, rename, apply and delete through oMLX persistence", async ({
  page,
}) => {
  await page.goto(`${modelPath}?tab=profiles`)
  await connect(page)
  await page
    .getByRole("button", { name: "Create profile", exact: true })
    .click()
  await page.getByLabel("Profile name", { exact: true }).fill("focused")
  const profiles = page.getByRole("tabpanel", { name: "Profiles", exact: true })
  await profiles.getByLabel("Display name", { exact: true }).fill("Focused")
  await profiles.getByLabel("Temperature", { exact: true }).fill("0.2")
  await profiles
    .getByRole("button", { name: "Save changes", exact: true })
    .click()
  await expect(page.getByText("Focused", { exact: true })).toBeVisible()
  await page.getByRole("button", { name: "Edit profile", exact: true }).click()
  await page.getByLabel("Profile name", { exact: true }).fill("focused-v2")
  await profiles
    .getByRole("button", { name: "Save changes", exact: true })
    .click()
  page.once("dialog", (dialog) => dialog.accept())
  await page.getByRole("button", { name: "Apply profile", exact: true }).click()
  await expect(page.getByText("Profile applied", { exact: true })).toBeVisible()
  await page.getByRole("tab", { name: "Settings", exact: true }).click()
  await expect(page.getByLabel("Temperature", { exact: true })).toHaveValue(
    "0.2"
  )
  await page.getByRole("tab", { name: "Profiles", exact: true }).click()
  await page
    .getByRole("button", { name: "Delete profile", exact: true })
    .click()
  await page.getByRole("button", { name: "Confirm delete" }).click()
  await expect(page.getByText(/No saved profiles/)).toBeVisible()
})
test("cache busy error and successful clearing are separately reported", async ({
  page,
}) => {
  await page.goto("/cache")
  await connect(page)
  await expect(page.getByText("128 MiB", { exact: true }).first()).toBeVisible()
  await page
    .getByRole("button", { name: "Clear hot cache", exact: true })
    .click()
  await page.getByRole("button", { name: "Confirm clear" }).click()
  await expect(
    page.getByText(/Cannot clear cache while requests are active/)
  ).toBeVisible()
  await page.getByRole("button", { name: "Cancel", exact: true }).click()
  await page.goto("/models/local%2Fbusy-model")
  await page.getByRole("button", { name: "Unload model", exact: true }).click()
  await page.getByRole("button", { name: "Confirm unload" }).click()
  await expect(
    page.getByRole("button", { name: "Load model", exact: true })
  ).toBeVisible()
  await page.getByRole("link", { name: "Cache", exact: true }).click()
  await page
    .getByRole("button", { name: "Clear SSD cache", exact: true })
    .click()
  await page.getByRole("button", { name: "Confirm clear" }).click()
  await expect(page.getByText(/SSD cache cleared/)).toBeVisible()
})
test("stale data recovers, failed save keeps draft, mobile navigation and theme", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto(`${modelPath}?tab=settings`)
  await connect(page)
  await page.getByLabel("Temperature", { exact: true }).fill("0.4")
  await page.route("**/api/omlx/models/**/settings", (route) =>
    route.request().method() === "PATCH"
      ? route.fulfill({
          status: 503,
          json: { detail: "Settings storage unavailable" },
        })
      : route.continue()
  )
  await page.getByRole("button", { name: "Save changes", exact: true }).click()
  await expect(
    page.getByText("Settings storage unavailable", { exact: true })
  ).toBeVisible()
  await expect(page.getByLabel("Temperature", { exact: true })).toHaveValue(
    "0.4"
  )
  await page.unrouteAll()
  await page.getByRole("button", { name: "Toggle Sidebar" }).click()
  await page.getByRole("link", { name: "Overview", exact: true }).click()
  await expect(
    page.getByRole("heading", { name: "Overview", exact: true })
  ).toBeVisible()
  await page.route("**/api/omlx/state", (route) =>
    route.fulfill({ status: 503, json: { detail: "Server restarting" } })
  )
  await expect(
    page.getByText("Showing the last received data", { exact: true })
  ).toBeVisible()
  await page.unrouteAll()
  await expect(
    page.getByText("Showing the last received data", { exact: true })
  ).toHaveCount(0)
  await page.getByRole("button", { name: "Toggle color theme" }).click()
  await expect(page.locator("html")).toHaveClass(/dark/)
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth
    )
  ).toBe(true)
})
test("gateway rejects unauthenticated and cross-origin operations", async ({
  request,
  baseURL,
}) => {
  expect((await request.get("/api/omlx/state")).status()).toBe(401)
  expect(
    (
      await request.post("/api/connection", {
        headers: { Origin: "https://foreign.example" },
        data: { key: "dashboard-test-key" },
      })
    ).status()
  ).toBe(403)
  expect(
    (
      await request.post("/api/connection", {
        headers: { Origin: baseURL! },
        data: { key: "dashboard-test-key" },
      })
    ).status()
  ).toBe(200)
  expect((await request.get("/api/omlx/state")).status()).toBe(200)
  expect(
    (
      await request.post("/api/omlx/cache/hot/clear", {
        headers: { Origin: "https://foreign.example" },
      })
    ).status()
  ).toBe(403)
  expect((await request.get("/api/omlx/server/restart")).status()).toBe(404)
})
