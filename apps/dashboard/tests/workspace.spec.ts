import { test, expect } from "@playwright/test"
import type { Page } from "@playwright/test"

const model = "mlx-community/test-model"
const unloaded = "local/unloaded-model"

async function openLibrary(page: Page, search = "") {
  await page.goto(`/models${search}`)
  await page.getByRole("button", { name: "API access" }).first().click()
  await page.getByLabel("API key", { exact: true }).fill("dashboard-test-key")
  await page.getByRole("button", { name: "Connect", exact: true }).click()
  await expect(
    page.getByRole("dialog", { name: "Connect to Molto" })
  ).toHaveCount(0)
  await expect(
    page.getByRole("heading", { name: "Model library" })
  ).toBeVisible()
}

test.beforeEach(async ({ request }) => {
  await request.post("http://127.0.0.1:8765/__test__/reset")
})

test("library search and task, health and lifecycle filters survive reload", async ({
  page,
}) => {
  await openLibrary(page)
  await page.getByLabel("Search library").fill("unloaded")
  await expect(page).toHaveURL(/q=unloaded/)
  await page.getByRole("combobox", { name: "Filter by lifecycle" }).click()
  await page.getByRole("option", { name: "unloaded", exact: true }).click()
  await page.getByRole("combobox", { name: "Filter by task" }).click()
  await page
    .getByRole("option", { name: "text-generation", exact: true })
    .click()
  await page.getByRole("combobox", { name: "Filter by health" }).click()
  await page.getByRole("option", { name: "unverified", exact: true }).click()
  await page.reload()
  await expect(page.getByLabel("Search library")).toHaveValue("unloaded")
  await expect(
    page.getByRole("combobox", { name: "Filter by lifecycle" })
  ).toContainText("unloaded")
  await expect(
    page.getByRole("combobox", { name: "Filter by task" })
  ).toContainText("text-generation")
  await expect(
    page.getByRole("combobox", { name: "Filter by health" })
  ).toContainText("unverified")
  await expect(
    page.getByRole("button", { name: unloaded, exact: true })
  ).toBeVisible()
  await expect(
    page.getByRole("button", { name: model, exact: true })
  ).toHaveCount(0)
})

test("collection startup preload is explicit and retained when editing", async ({
  page,
}) => {
  await openLibrary(page)
  await page
    .getByRole("checkbox", { name: `Select ${unloaded}`, exact: true })
    .check()
  await page.getByRole("button", { name: "Estimate 1 selected models" }).click()
  await expect(
    page.getByText("Selection fits the reported memory budget.")
  ).toBeVisible()
  await page
    .getByRole("button", { name: "Create collection from 1 selected" })
    .click()
  const dialog = page.getByRole("dialog", { name: "Save collection" })
  await dialog.getByLabel("Name", { exact: true }).fill("Startup study")
  await expect(
    dialog.getByRole("checkbox", { name: "Preload models on server startup" })
  ).not.toBeChecked()
  await dialog
    .getByRole("checkbox", { name: "Preload models on server startup" })
    .check()
  await expect(dialog.getByText(/pins every member/)).toBeVisible()
  await dialog
    .getByRole("button", { name: "Save collection", exact: true })
    .click()
  await expect(dialog).toHaveCount(0)
  await expect(page.getByText("Startup study", { exact: true })).toBeVisible()
  await expect(page.getByText(/Preloads on startup/)).toBeVisible()
  await page.getByRole("button", { name: "Edit", exact: true }).click()
  await expect(
    dialog.getByRole("checkbox", { name: "Preload models on server startup" })
  ).toBeChecked()
  await dialog
    .getByLabel("Description", { exact: true })
    .fill("Keep the startup policy")
  await dialog
    .getByRole("button", { name: "Save collection", exact: true })
    .click()
  await expect(dialog).toHaveCount(0)
  const response = await page.request.get("/api/molto/workspace/collections")
  expect(response.ok()).toBe(true)
  const data = (await response.json()) as {
    collections: { name: string; description: string; preload: boolean }[]
  }
  expect(data.collections).toHaveLength(1)
  expect(data.collections[0]).toMatchObject({
    name: "Startup study",
    description: "Keep the startup policy",
    preload: true,
  })
  await page.getByRole("button", { name: "Edit", exact: true }).click()
  await dialog
    .getByRole("checkbox", { name: "Preload models on server startup" })
    .uncheck()
  await dialog
    .getByRole("button", { name: "Save collection", exact: true })
    .click()
  await expect(dialog).toHaveCount(0)
  await expect(page.getByText(/Loads only when requested/)).toBeVisible()
})

test("configuration file import shows the server field diff before explicit apply", async ({
  page,
  baseURL,
}) => {
  await openLibrary(page)
  const response = await page.request.get("/api/molto/workspace/export")
  expect(response.ok()).toBe(true)
  const bundle = (await response.json()) as {
    models: { id: string; settings: Record<string, unknown> }[]
  }
  const target = bundle.models.find((item) => item.id === model)
  expect(target).toBeDefined()
  target!.settings.temperature = 0.25
  let applied = 0
  page.on("request", (request) => {
    if (
      request.url().endsWith("/workspace/import") &&
      request.method() === "POST" &&
      (request.postDataJSON() as { dry_run: boolean }).dry_run === false
    )
      applied++
  })
  await page.getByLabel("Import configuration file").setInputFiles({
    name: "workspace.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(bundle)),
  })
  await expect(
    page.getByRole("cell", {
      name: `${model} settings.temperature`,
      exact: true,
    })
  ).toBeVisible()
  await expect(
    page.getByRole("cell", { name: "0.7", exact: true })
  ).toBeVisible()
  await expect(
    page.getByRole("cell", { name: "0.25", exact: true })
  ).toBeVisible()
  expect(applied).toBe(0)
  await expect(
    page.getByText("Import is blocked", { exact: true })
  ).toBeVisible()
  await expect(
    page.getByRole("button", { name: "Apply reviewed configuration" })
  ).toBeDisabled()
  const unloadedResponse = await page.request.post(
    `/api/molto/models/${encodeURIComponent(model)}/unload`,
    { headers: { Origin: baseURL! } }
  )
  expect(unloadedResponse.ok()).toBe(true)
  await page.getByRole("button", { name: "Refresh preview" }).click()
  await expect(
    page.getByText("Import is blocked", { exact: true })
  ).toHaveCount(0)
  await expect(
    page.getByRole("cell", {
      name: `${model} settings.temperature`,
      exact: true,
    })
  ).toBeVisible()
  expect(applied).toBe(0)
  await page
    .getByRole("button", { name: "Apply reviewed configuration" })
    .click()
  await expect(
    page.getByText("Configuration applied.", { exact: true })
  ).toBeVisible()
  expect(applied).toBe(1)
  const exported = await page.request.get("/api/molto/workspace/export")
  const result = (await exported.json()) as typeof bundle
  expect(
    result.models.find((item) => item.id === model)?.settings.temperature
  ).toBe(0.25)
})

test("deletion preview enforces blockers and a failed delete requires a new preview", async ({
  page,
}) => {
  await openLibrary(page)
  let deletes = 0
  page.on("request", (request) => {
    if (request.url().endsWith("/delete") && request.method() === "DELETE")
      deletes++
  })
  await page.getByRole("button", { name: model, exact: true }).click()
  await page.getByRole("button", { name: "Preview file deletion" }).click()
  const deletion = page.getByRole("dialog", { name: "Delete model files" })
  await expect(
    deletion.getByText("Model is loaded", { exact: true })
  ).toBeVisible()
  await deletion.getByLabel("Type the model identifier to confirm").fill(model)
  await expect(
    deletion.getByRole("button", { name: "Delete previewed files" })
  ).toBeDisabled()
  expect(deletes).toBe(0)
  await deletion.getByRole("button", { name: "Cancel", exact: true }).click()
  await page
    .getByRole("dialog", { name: model, exact: true })
    .getByRole("button", { name: "Close", exact: true })
    .click()
  await page.getByRole("button", { name: unloaded, exact: true }).click()
  await page.getByRole("button", { name: "Preview file deletion" }).click()
  await deletion
    .getByLabel("Type the model identifier to confirm")
    .fill(unloaded)
  await page.route("**/api/molto/workspace/models/**/delete", (route) =>
    route.fulfill({
      status: 503,
      json: { detail: "Fixture deletion unavailable" },
    })
  )
  await deletion.getByRole("button", { name: "Delete previewed files" }).click()
  await expect(
    deletion.getByText("Fixture deletion unavailable", { exact: true })
  ).toBeVisible()
  await expect(
    deletion.getByRole("button", { name: "Delete previewed files" })
  ).toBeDisabled()
  expect(deletes).toBe(1)
})
