import { test, expect } from "@playwright/test"
import type { Operation } from "../src/features/operations/page"
const local = {
  id: "source",
  name: "Source checkpoint",
  path: "/models/source",
  model_type: "llm",
  format: "mlx",
  precision: "FP16",
  size: 1000,
  conversion: { available: false, reason: "Already in MLX format" },
  quantization: { available: true, adapter: "oQ", requires_conversion: false },
}
const catalog = {
  models: [local],
  options: {
    oq_levels: [3, 4, 8],
    group_sizes: [32, 64, 128],
    dtypes: ["bfloat16", "float16"],
    exclusive_busy: false,
  },
}
test("guided discovery preserves staged input and sends one explicit download", async ({
  page,
}) => {
  const downloads: unknown[] = []
  await page.route("**/api/molto/server/settings", (route) =>
    route.fulfill({
      json: { fields: [], sections: {}, effective_model_dirs: ["/models"] },
    })
  )
  await page.route("**/api/molto/acquisition/**", async (route) => {
    const url = new URL(route.request().url())
    if (url.pathname.endsWith("/models"))
      return route.fulfill({ json: catalog })
    if (url.pathname.endsWith("/search"))
      return route.fulfill({
        json: {
          models: [
            { repo_id: "owner/checkpoint", name: "Checkpoint", size: 1000 },
          ],
        },
      })
    if (url.pathname.endsWith("/info"))
      return route.fulfill({
        json: {
          repo_id: "owner/checkpoint",
          name: "Checkpoint",
          files: [{ name: "config.json", size: 10 }],
          model_card: "Checkpoint documentation",
        },
      })
    if (url.pathname.endsWith("/downloads")) {
      downloads.push(route.request().postDataJSON())
      return route.fulfill({
        status: 202,
        json: { id: "download-one", status: "queued" },
      })
    }
    return route.fulfill({
      status: 404,
      json: { detail: "Unexpected test request" },
    })
  })
  await page.route("**/api/molto/diffusion/jobs", (route) =>
    route.fulfill({ json: { jobs: [] } })
  )
  await page.goto("/add-model")
  await page
    .getByLabel("Search models", { exact: true })
    .fill("staged checkpoint")
  await page.getByRole("button", { name: "2. Prepare local model" }).click()
  await page.getByRole("button", { name: "1. Find and download" }).click()
  await expect(page.getByLabel("Search models", { exact: true })).toHaveValue(
    "staged checkpoint"
  )
  await page.getByRole("button", { name: "Search", exact: true }).click()
  await page.getByRole("button", { name: "Inspect checkpoint" }).click()
  await expect(
    page.getByText("Compatibility checked locally after download")
  ).toBeVisible()
  await page.getByLabel("Download token, optional").fill("ephemeral-token")
  await page
    .getByRole("button", { name: "Download checkpoint", exact: true })
    .click()
  await expect(page.getByText("Download queued", { exact: true })).toBeVisible()
  expect(downloads).toEqual([
    { repo_id: "owner/checkpoint", token: "ephemeral-token" },
  ])
  await expect(page.getByLabel("Download token, optional")).toHaveValue("")
})
test("activity exposes advertised actions and retries only after confirmation", async ({
  page,
}) => {
  const operation: Operation = {
    id: "failed-one",
    kind: "download_hf",
    model_id: "owner/checkpoint",
    status: "failed",
    stage: "download failed",
    progress: 30,
    error: "Connection lost",
    actions: { cancel: false, retry: true, delete: false },
  }
  const submissions: unknown[] = []
  await page.route("**/api/molto/operations**", async (route) => {
    if (route.request().method() === "POST") {
      submissions.push(route.request().postDataJSON())
      return route.fulfill({
        json: { ...operation, id: "retry-one", status: "queued" },
      })
    }
    return route.fulfill({ json: { operations: [operation] } })
  })
  await page.route("**/api/molto/diffusion/jobs", (route) =>
    route.fulfill({ json: { jobs: [] } })
  )
  await page.goto("/activity")
  await expect(
    page.getByRole("button", { name: "Cancel", exact: true })
  ).toHaveCount(0)
  await expect(page.getByRole("button", { name: "Remove record" })).toHaveCount(
    0
  )
  await page.getByRole("button", { name: "Retry", exact: true }).click()
  expect(submissions).toEqual([])
  await page.getByLabel("Download token, optional").fill("retry-token")
  await page.getByRole("button", { name: "Confirm", exact: true }).click()
  await expect(page.getByRole("dialog")).toHaveCount(0)
  expect(submissions).toEqual([{ token: "retry-token" }])
})
test("publication validates token, reviews visibility, and sends staged fields", async ({
  page,
}) => {
  const published: unknown[] = []
  await page.route("**/api/molto/server/settings", (route) =>
    route.fulfill({
      json: { fields: [], sections: {}, effective_model_dirs: ["/models"] },
    })
  )
  await page.route("**/api/molto/diffusion/jobs", (route) =>
    route.fulfill({ json: { jobs: [] } })
  )
  await page.route("**/api/molto/acquisition/**", async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path.endsWith("/models")) return route.fulfill({ json: catalog })
    if (path.endsWith("/validate"))
      return route.fulfill({
        json: {
          valid: true,
          can_write: true,
          username: "owner",
          orgs: [{ name: "team" }],
        },
      })
    if (path.endsWith("/start")) {
      published.push(route.request().postDataJSON())
      return route.fulfill({ json: { id: "publish-one", status: "queued" } })
    }
    return route.fulfill({
      status: 404,
      json: { detail: "Unexpected request" },
    })
  })
  await page.goto("/add-model")
  await page.getByRole("button", { name: "3. Publish, optional" }).click()
  await page.getByLabel("Local model to publish").click()
  await page
    .getByRole("option", { name: "Source checkpoint", exact: true })
    .click()
  await page.getByLabel("Hugging Face write token").fill("write-token")
  await page.getByRole("button", { name: "Validate token and source" }).click()
  await page
    .getByLabel("Repository name", { exact: true })
    .fill("prepared-copy")
  await page.getByRole("button", { name: "Review publication" }).click()
  await expect(page.getByRole("dialog")).toContainText("private")
  expect(published).toEqual([])
  await page.getByRole("button", { name: "Confirm publication" }).click()
  await expect(
    page.getByText("Publication queued", { exact: true })
  ).toBeVisible()
  expect(published).toEqual([
    {
      model_path: "/models/source",
      repo_id: "owner/prepared-copy",
      token: "write-token",
      private: true,
      auto_readme: true,
      redownload_notice: false,
      readme_source_path: "",
    },
  ])
  await expect(page.getByLabel("Hugging Face write token")).toHaveValue("")
})
test("quantization requires an estimate and submits source options", async ({
  page,
}) => {
  const quantized: unknown[] = []
  await page.route("**/api/molto/server/settings", (route) =>
    route.fulfill({
      json: { fields: [], sections: {}, effective_model_dirs: ["/models"] },
    })
  )
  await page.route("**/api/molto/diffusion/jobs", (route) =>
    route.fulfill({ json: { jobs: [] } })
  )
  await page.route("**/api/molto/acquisition/**", async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path.endsWith("/models")) return route.fulfill({ json: catalog })
    if (path.endsWith("/estimate"))
      return route.fulfill({
        json: {
          effective_bpw: 4.2,
          output_size_bytes: 100,
          output_size_formatted: "100 B",
        },
      })
    if (path.endsWith("/quantize")) {
      quantized.push(route.request().postDataJSON())
      return route.fulfill({ json: { id: "quantize-one", status: "queued" } })
    }
    return route.fulfill({
      status: 404,
      json: { detail: "Unexpected request" },
    })
  })
  await page.goto("/add-model")
  await page.getByRole("button", { name: "2. Prepare local model" }).click()
  await page.getByLabel("Source model", { exact: true }).click()
  await page.getByRole("option", { name: /Source checkpoint/ }).click()
  await expect(
    page.getByRole("button", { name: "Start quantization", exact: true })
  ).toBeDisabled()
  await page.getByRole("button", { name: "Estimate output size" }).click()
  await expect(page.getByText("Estimated output 100 B")).toBeVisible()
  await page
    .getByRole("button", { name: "Start quantization", exact: true })
    .click()
  await expect(
    page.getByText("Preparation queued", { exact: true })
  ).toBeVisible()
  expect(quantized).toEqual([
    {
      model_path: "/models/source",
      oq_level: 4,
      group_size: 64,
      preserve_mtp: false,
      dtype: "bfloat16",
      sensitivity_model_path: "",
      mtp_assistant_model_path: "",
      auto_proxy_sensitivity: true,
      text_only: false,
      enhanced: false,
      imatrix_cache_path: "",
      imatrix_reuse_cache: true,
      imatrix_strict: false,
      imatrix_num_samples: 128,
      imatrix_seq_length: 512,
    },
  ])
})
