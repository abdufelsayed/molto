import { test, expect } from "@playwright/test"
const model = "mlx-community/test-model"
const path = `/models/${encodeURIComponent(model)}?tab=settings`
test("metadata controls preserve typed drafts and send explicit resets", async ({
  page,
}) => {
  await page.route("**/api/molto/models/**/options", (route) =>
    route.fulfill({
      json: {
        fields: [
          {
            key: "temperature",
            label: "Temperature",
            type: "number",
            default: null,
            profile: true,
            template: true,
            description: "Sampling temperature.",
            group: "general",
            minimum: 0,
          },
          {
            key: "chat_template_kwargs",
            label: "Chat template arguments",
            type: "object",
            default: null,
            profile: true,
            template: true,
            description: "Arguments passed to the model chat template.",
            group: "general",
          },
          {
            key: "turboquant_kv_bits",
            label: "KV cache bits",
            type: "number",
            default: 4,
            profile: true,
            template: false,
            description: "Cache compression bit depth.",
            choices: [2, 4, 8],
            group: "advanced",
            advanced: true,
          },
          {
            key: "qwen35_ane_prefill_enabled",
            label: "ANE prefill",
            type: "boolean",
            default: false,
            profile: true,
            template: false,
            description: "Accelerate prompt processing.",
            supported: false,
            unsupported_reason: "Unavailable for this family",
            group: "advanced",
            advanced: true,
          },
        ],
        defaults: {},
        profile_fields: [],
        template_fields: [],
      },
    })
  )
  await page.route("**/api/molto/models/**/settings", (route) =>
    route.request().method() === "GET"
      ? route.fulfill({
          json: {
            model_id: model,
            settings: {
              temperature: 0.7,
              chat_template_kwargs: null,
              turboquant_kv_bits: 4,
            },
          },
        })
      : route.fulfill({
          json: {
            model_id: model,
            settings: {},
            auto_reloaded: false,
            auto_unloaded: false,
            reload_deferred: false,
            reload_error: null,
          },
        })
  )
  await page.route("**/api/molto/models/**/profiles", (route) =>
    route.fulfill({ json: { model_id: model, profiles: [] } })
  )
  await page.route("**/api/molto/templates", (route) =>
    route.fulfill({ json: { templates: [] } })
  )
  await page.goto(path)
  await page.getByRole("button", { name: "API access" }).first().click()
  await page.getByLabel("API key", { exact: true }).fill("dashboard-test-key")
  await page.getByRole("button", { name: "Connect", exact: true }).click()
  await page.getByLabel("Temperature", { exact: true }).fill("0.3")
  await page.getByRole("tab", { name: "Summary", exact: true }).click()
  await page.getByRole("tab", { name: "Settings", exact: true }).click()
  await expect(page.getByLabel("Temperature", { exact: true })).toHaveValue(
    "0.3"
  )
  await page.getByRole("button", { name: "Refresh dashboard" }).click()
  await expect(page.getByLabel("Temperature", { exact: true })).toHaveValue(
    "0.3"
  )
  await page.getByRole("button", { name: "Add argument", exact: true }).click()
  await page
    .getByLabel("Chat template arguments key key_1", { exact: true })
    .fill("mode")
  await page.getByLabel("Temperature", { exact: true }).click()
  await page
    .getByLabel("Chat template arguments mode value", { exact: true })
    .fill("fast")
  await page
    .getByText("Acceleration and family settings · Advanced", { exact: true })
    .click()
  await expect(
    page.getByRole("switch", { name: "ANE prefill", exact: true })
  ).toBeDisabled()
  await page.getByLabel("KV cache bits", { exact: true }).click()
  await page.getByRole("option", { name: "2", exact: true }).click()
  const request = page.waitForRequest(
    (request) =>
      request.method() === "PATCH" && request.url().endsWith("/settings")
  )
  await page.getByRole("button", { name: "Save changes", exact: true }).click()
  expect((await request).postDataJSON()).toEqual({
    temperature: 0.3,
    chat_template_kwargs: { mode: "fast" },
    turboquant_kv_bits: 2,
  })
  await page
    .getByRole("button", { name: "Reset Temperature to default", exact: true })
    .click()
  const reset = page.waitForRequest(
    (request) =>
      request.method() === "PATCH" && request.url().endsWith("/settings")
  )
  await page.getByRole("button", { name: "Save changes", exact: true }).click()
  expect((await reset).postDataJSON()).toEqual({ temperature: null })
  await page.getByRole("tab", { name: "Profiles", exact: true }).click()
  await page
    .getByRole("button", { name: "Create profile", exact: true })
    .click()
  await page.getByLabel("Profile name", { exact: true }).fill("unsaved-profile")
  const profilesPanel = page.getByRole("tabpanel", {
    name: "Profiles",
    exact: true,
  })
  await profilesPanel.getByLabel("Temperature", { exact: true }).fill("0.4")
  await page.getByRole("tab", { name: "Settings", exact: true }).click()
  await page
    .getByRole("button", {
      name: "Create template from current settings",
      exact: true,
    })
    .click()
  await page
    .getByLabel("Template name", { exact: true })
    .fill("unsaved-template")
  await page.getByRole("tab", { name: "Summary", exact: true }).click()
  await page.getByRole("tab", { name: "Profiles", exact: true }).click()
  await expect(page.getByLabel("Profile name", { exact: true })).toHaveValue(
    "unsaved-profile"
  )
  await expect(
    profilesPanel.getByLabel("Temperature", { exact: true })
  ).toHaveValue("0.4")
  await page.getByRole("tab", { name: "Settings", exact: true }).click()
  await expect(page.getByLabel("Template name", { exact: true })).toHaveValue(
    "unsaved-template"
  )
})

test("MTPLX import requires confirmation and refreshes compatibility readback", async ({
  page,
}) => {
  const before = {
    fields: [],
    defaults: {},
    profile_fields: [],
    template_fields: [],
    capabilities: { mtplx_import: true, mtp: false },
    mtplx_sidecar_detected: true,
    mtplx_import_reason:
      "MTPLX sidecar weights must be imported before MTP is available.",
  }
  const after = {
    ...before,
    capabilities: { mtplx_import: false, mtp: true },
    mtplx_import_reason: "MTPLX weights are already imported.",
  }
  await page.route("**/api/molto/models", async (route) => {
    const response = await route.fetch()
    if (response.status() !== 200) {
      await route.fulfill({ response })
      return
    }
    const inventory = (await response.json()) as {
      models: Array<{
        id: string
        loaded: boolean
        is_loading: boolean
        is_unloading: boolean
      }>
    }
    inventory.models = inventory.models.map((model) => ({
      ...model,
      loaded: false,
      is_loading: false,
      is_unloading: false,
    }))
    await route.fulfill({ response, json: inventory })
  })
  await page.route("**/api/molto/models/**/options", (route) =>
    route.fulfill({ json: imports > 0 ? after : before })
  )
  let imports = 0
  await page.route("**/api/molto/models/**/import-mtplx", (route) => {
    imports++
    return route.fulfill({
      json: {
        status: "ok",
        model_id: model,
        merge_mode: "sidecar",
        mtp_tensors: 12,
        options: after,
      },
    })
  })
  await page.goto(path)
  await page.getByRole("button", { name: "API access" }).first().click()
  await page.getByLabel("API key", { exact: true }).fill("dashboard-test-key")
  await page.getByRole("button", { name: "Connect", exact: true }).click()
  const button = page.getByRole("button", {
    name: "Import MTPLX sidecar",
    exact: true,
  })
  await expect(button).toBeEnabled()
  page.once("dialog", (dialog) => {
    expect(dialog.message()).toContain("checkpoint files on disk")
    void dialog.dismiss()
  })
  await button.click()
  expect(imports).toBe(0)
  page.once("dialog", (dialog) => void dialog.accept())
  await button.click()
  await expect(
    page.getByText("MTPLX sidecar imported", { exact: true })
  ).toBeVisible()
  await expect(
    page.getByText(/Imported 12 MTP tensors using sidecar/)
  ).toBeVisible()
  expect(imports).toBe(1)
  await expect(button).toBeDisabled()
  await expect(
    page.getByText("MTPLX weights are already imported.", { exact: true })
  ).toBeVisible()
})

test("ordinary models do not show MTPLX import controls", async ({ page }) => {
  await page.route("**/api/molto/models/**/options", (route) =>
    route.fulfill({
      json: {
        fields: [
          {
            key: "temperature",
            label: "Temperature",
            type: "number",
            default: null,
            profile: true,
            template: true,
            description: "Sampling temperature.",
          },
        ],
        defaults: {},
        profile_fields: [],
        template_fields: [],
        capabilities: { mtplx_import: false },
        mtplx_sidecar_detected: false,
        mtplx_import_reason: "No MTPLX sidecar detected.",
      },
    })
  )
  await page.goto(path)
  await page.getByRole("button", { name: "API access" }).first().click()
  await page.getByLabel("API key", { exact: true }).fill("dashboard-test-key")
  await page.getByRole("button", { name: "Connect", exact: true }).click()
  await expect(page.getByLabel("Temperature", { exact: true })).toBeVisible()
  await expect(
    page.getByText("MTPLX sidecar import", { exact: true })
  ).toHaveCount(0)
  await expect(
    page.getByRole("button", { name: "Import MTPLX sidecar", exact: true })
  ).toHaveCount(0)
})
