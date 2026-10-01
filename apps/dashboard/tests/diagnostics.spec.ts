import { test, expect } from "@playwright/test"
import { recommendedPatch } from "../src/features/diagnostics/recommendation"
import { active, resultRows, type Run } from "../src/features/diagnostics/types"

test("native cancellation remains active and nested results retain measurements", () => {
  expect(active({ status: "cancelling" } as Run)).toBe(true)
  expect(active({ status: "cancelled" } as Run)).toBe(false)
  expect(resultRows([{ tokens_per_second: 42, status: "partial" }])).toEqual([
    { metric: "0 / tokens per second", value: "42", number: 42 },
    { metric: "0 / status", value: "partial", number: undefined },
  ])
})

test("guided run request, preserved workload and cooperative cancellation", async ({
  page,
}) => {
  const kinds = ["throughput", "accuracy", "context", "ane"].map((kind) => ({
    kind,
    targets:
      kind === "throughput" || kind === "accuracy"
        ? ["local", "external"]
        : ["local"],
  }))
  let run: Run | undefined
  let submitted: unknown
  await page.route("**/api/molto/diagnostics/**", async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path.endsWith("capabilities"))
      return route.fulfill({
        json: {
          kinds,
          models: [
            {
              model_id: "test-model",
              kinds: ["throughput", "accuracy", "context", "ane"],
            },
          ],
          suites: [{ id: "mmlu", label: "MMLU" }],
        },
      })
    if (route.request().method() === "POST" && path.endsWith("/runs")) {
      submitted = route.request().postDataJSON()
      run = {
        id: "test-run",
        kind: "throughput",
        status: "running",
        created_at: "2026-10-01T12:00:00Z",
        updated_at: "2026-10-01T12:00:00Z",
        request: submitted as Run["request"],
        progress: { message: "Measuring prefill" },
        results: [],
      }
      return route.fulfill({ status: 202, json: run })
    }
    if (path.endsWith("/cancel") && run) {
      run.status = "cancelling"
      return route.fulfill({ json: run })
    }
    if (path.endsWith("/results"))
      return route.fulfill({ json: { results: [] } })
    if (path.endsWith("/runs"))
      return route.fulfill({ json: { runs: run ? [run] : [] } })
    return route.fulfill({ json: run })
  })
  await page.goto("/diagnostics")
  await page.getByLabel("Model", { exact: true }).click()
  await page.getByRole("option", { name: "test-model" }).click()
  await expect(
    page.getByText(/unloads other resident models and the target/)
  ).toBeVisible()
  await expect(
    page.getByText(/Warmed engines will need loading again/)
  ).toBeVisible()
  await page.getByLabel("Output tokens", { exact: true }).fill("64")
  await page
    .getByRole("button", { name: "Run throughput", exact: true })
    .click()
  await expect(
    page.getByRole("button", { name: "Cancel run", exact: true })
  ).toBeVisible()
  expect(submitted).toMatchObject({
    kind: "throughput",
    options: {
      model_id: "test-model",
      prompt_lengths: [1024, 4096],
      generation_length: 64,
      batch_sizes: [],
      context_profile: "code_python",
      warmup_mode: "quick",
    },
  })
  await expect(page.getByLabel("Output tokens", { exact: true })).toHaveValue(
    "64"
  )
  await page.getByRole("button", { name: "Cancel run", exact: true }).click()
  await expect(
    page.getByText(
      "Cancellation requested. Admission remains held until native work stops."
    )
  ).toBeVisible()
  await expect(
    page.getByRole("button", { name: "Cancel run", exact: true })
  ).toBeDisabled()
})

test("guided accuracy, context and ANE workloads use runner fields", async ({
  page,
}) => {
  const submissions: { kind: string; options: Record<string, unknown> }[] = []
  const kinds = ["throughput", "accuracy", "context", "ane"].map((kind) => ({
    kind,
    targets: ["local"],
  }))
  await page.route("**/api/molto/diagnostics/**", async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path.endsWith("capabilities"))
      return route.fulfill({
        json: {
          kinds,
          models: [
            {
              model_id: "test-model",
              kinds: ["throughput", "accuracy", "context", "ane"],
            },
          ],
          suites: [{ id: "mmlu", label: "MMLU" }],
        },
      })
    if (route.request().method() === "POST") {
      const body = route.request().postDataJSON() as {
        kind: string
        options: Record<string, unknown>
      }
      submissions.push(body)
      return route.fulfill({
        status: 202,
        json: {
          id: "finished",
          ...body,
          status: "completed",
          request: body.options,
          results: [],
          created_at: "2026-10-01",
          updated_at: "2026-10-01",
        },
      })
    }
    if (path.endsWith("/runs")) return route.fulfill({ json: { runs: [] } })
    if (path.endsWith("/results"))
      return route.fulfill({ json: { results: [] } })
    return route.fulfill({
      json: {
        id: "finished",
        kind: "accuracy",
        status: "completed",
        request: {},
        results: [],
        created_at: "2026-10-01",
        updated_at: "2026-10-01",
      },
    })
  })
  await page.goto("/diagnostics")
  await page.getByLabel("Model", { exact: true }).click()
  await page.getByRole("option", { name: "test-model" }).click()
  for (const [label, button] of [
    ["Accuracy", "Run accuracy"],
    ["Context capacity", "Run context capacity"],
    ["ANE tuning", "Run ane tuning"],
  ]) {
    await page.getByLabel("Diagnostic", { exact: true }).click()
    await page.getByRole("option", { name: label, exact: true }).click()
    if (label === "Context capacity" || label === "ANE tuning")
      await expect(page.getByText(/unloads all resident models/)).toBeVisible()
    if (label === "Accuracy")
      await page.getByLabel("MMLU", { exact: true }).check()
    await page.getByRole("button", { name: button, exact: true }).click()
    await expect
      .poll(() => submissions.length)
      .toBe(label === "Accuracy" ? 1 : label === "Context capacity" ? 2 : 3)
  }
  expect(submissions[0]).toMatchObject({
    kind: "accuracy",
    options: {
      model_id: "test-model",
      benchmarks: { mmlu: 100 },
      batch_size: 1,
      sampling_profile: "deterministic",
      enable_thinking: false,
    },
  })
  expect(submissions[1]).toMatchObject({
    kind: "context",
    options: { model_id: "test-model", target_tokens: 131072 },
  })
  expect(submissions[2]).toMatchObject({
    kind: "ane",
    options: {
      model_id: "test-model",
      backend: "qwen",
      sequence_length: 2048,
      repeats: 2,
      allow_cpu: true,
      allow_ane_gdn: true,
    },
  })
})

test("external credentials are sent only with explicit run and local-only runners explain incompatibility", async ({
  page,
}) => {
  let body: unknown
  await page.route("**/api/molto/diagnostics/**", async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path.endsWith("capabilities"))
      return route.fulfill({
        json: {
          kinds: ["throughput", "accuracy", "context", "ane"].map((kind) => ({
            kind,
            targets: ["throughput", "accuracy"].includes(kind)
              ? ["local", "external"]
              : ["local"],
          })),
          models: [],
          suites: [],
        },
      })
    if (route.request().method() === "POST") {
      body = route.request().postDataJSON()
      return route.fulfill({
        status: 409,
        json: { detail: "A local job is still draining." },
      })
    }
    return route.fulfill({ json: { runs: [] } })
  })
  await page.goto("/diagnostics")
  await page.getByLabel("Target", { exact: true }).click()
  await page
    .getByRole("option", { name: "External OpenAI-compatible endpoint" })
    .click()
  await expect(
    page.getByText(/shared preparation and diagnostics lock/)
  ).toBeVisible()
  await expect(
    page.getByText(/External runs do not unload local models/)
  ).toBeVisible()
  await page.getByLabel("External model name").fill("remote-model")
  await page.getByLabel("Endpoint base URL").fill("https://provider.example/v1")
  await page.getByLabel("Endpoint API key").fill("test-secret")
  expect(body).toBeUndefined()
  await page
    .getByRole("button", { name: "Run throughput", exact: true })
    .click()
  await expect(page.getByText("A local job is still draining.")).toBeVisible()
  expect(body).toMatchObject({
    kind: "throughput",
    options: {
      model_id: "remote-model",
      external: {
        base_url: "https://provider.example/v1",
        model: "remote-model",
        api_key: "test-secret",
      },
    },
  })
  await expect(page.getByLabel("External model name")).toHaveValue(
    "remote-model"
  )
  await page.getByLabel("Diagnostic", { exact: true }).click()
  await page
    .getByRole("option", { name: "Context capacity", exact: true })
    .click()
  await expect(
    page.getByText("This diagnostic supports local models only.")
  ).toBeVisible()
  await expect(
    page.getByRole("button", { name: "Run context capacity", exact: true })
  ).toBeDisabled()
})

test("recommendation patches use measured context and actual ANE fields only", () => {
  const run = {
    id: "unit-context",
    created_at: "2026-10-01",
    updated_at: "2026-10-01",
    status: "completed",
    kind: "context",
    request: { model_id: "local-model" },
  } satisfies Run
  expect(
    recommendedPatch(run, [{ applied_tokens: 32768, target_tokens: 131072 }])
  ).toEqual({ max_context_window: 32768 })
  expect(
    recommendedPatch({ ...run, status: "error" }, [{ applied_tokens: 32768 }])
  ).toEqual({})
  expect(
    recommendedPatch(
      {
        ...run,
        kind: "ane",
        recommendation: {
          enabled: false,
          mlp_fraction: null,
          cpu_enabled: true,
          cpu_fraction: 0.25,
          backend: "k2",
          dflash_enabled: false,
          specprefill_enabled: false,
          mtp_enabled: false,
          vlm_mtp_enabled: false,
          unknown_setting: 42,
        },
      },
      []
    )
  ).toEqual({
    qwen35_ane_prefill_enabled: false,
    qwen35_ane_prefill_cpu_enabled: true,
    qwen35_ane_prefill_cpu_fraction: 0.25,
    dflash_enabled: false,
    specprefill_enabled: false,
    mtp_enabled: false,
    vlm_mtp_enabled: false,
  })
  expect(
    recommendedPatch(
      {
        ...run,
        kind: "ane",
        recommendation: {
          enabled: true,
          backend: "k2",
          dflash_enabled: false,
          specprefill_enabled: false,
        },
      },
      []
    )
  ).toEqual({})
  expect(
    recommendedPatch(
      { ...run, kind: "ane", recommendation: { enabled: true } },
      []
    )
  ).toEqual({})
})

test("recommendations preview before applying and report reload failures", async ({
  page,
}) => {
  let patch: unknown
  const run = {
    id: "context-completed",
    kind: "context",
    status: "completed",
    created_at: "2026-10-01",
    updated_at: "2026-10-01",
    request: { model_id: "local-model" },
    results: [{ applied_tokens: 32768 }],
  }
  await page.route("**/api/molto/**", async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path.endsWith("/models/local-model/settings")) {
      patch = route.request().postDataJSON()
      return route.fulfill({
        json: { requires_reload: true, reload_error: "Model reload failed" },
      })
    }
    if (path.endsWith("capabilities"))
      return route.fulfill({
        json: {
          kinds: [{ kind: "context", targets: ["local"] }],
          models: [],
          suites: [],
        },
      })
    if (path.endsWith("/diagnostics/runs"))
      return route.fulfill({ json: { runs: [run] } })
    if (path.endsWith("/results"))
      return route.fulfill({ json: { results: run.results } })
    if (path.endsWith("/context-completed")) return route.fulfill({ json: run })
    return route.continue()
  })
  await page.goto("/diagnostics")
  await page
    .getByRole("button", { name: /Context capacity context-completed/ })
    .click()
  await page
    .getByRole("button", { name: "Review recommendation", exact: true })
    .click()
  const dialog = page.getByRole("dialog", {
    name: "Apply recommended settings?",
  })
  await expect(dialog).toContainText('"max_context_window": 32768')
  expect(patch).toBeUndefined()
  await dialog
    .getByRole("button", { name: "Apply recommendation", exact: true })
    .click()
  await expect(
    dialog.getByText("Reload failed: Model reload failed")
  ).toBeVisible()
  expect(patch).toEqual({ max_context_window: 32768 })
  await expect(
    dialog.getByRole("button", { name: "Apply recommendation", exact: true })
  ).toBeDisabled()
})
