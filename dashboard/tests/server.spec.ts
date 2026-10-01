import { test, expect } from "@playwright/test"
import type { Page } from "@playwright/test"
import { editedPatch, resourceDraft } from "../src/features/server/types"

const fixture = {
  base_path: "/tmp/dashboard-test",
  sections: {
    sampling: { temperature: 0.7 },
    scheduler: { max_concurrent_requests: 2 },
    model: { model_dirs: ["/models"] },
    memory: {
      prefill_memory_guard: true,
      memory_guard_tier: "balanced",
      memory_guard_custom_ceiling_gb: 20,
    },
  },
  fields: [
    {
      key: "memory_guard_tier",
      section: "memory",
      label: "Memory guard tier",
      description: "Memory reserve policy",
      type: "string",
      default: "balanced",
      choices: [
        { value: "safe", label: "Safe" },
        { value: "balanced", label: "Balanced" },
        { value: "aggressive", label: "Aggressive" },
        { value: "custom", label: "Custom" },
      ],
      restart_required: true,
    },
    {
      key: "memory_guard_custom_ceiling_gb",
      section: "memory",
      label: "Custom memory ceiling",
      description: "Custom ceiling in GiB",
      type: "number",
      default: 20,
      minimum: 0,
      restart_required: true,
    },
    {
      key: "prefill_memory_guard",
      section: "memory",
      label: "Prefill memory guard",
      description: "Enforce memory admission",
      type: "boolean",
      default: true,
      restart_required: true,
    },
    {
      key: "temperature",
      section: "sampling",
      label: "Temperature",
      description: "Generation temperature",
      type: "number",
      default: 1,
      minimum: 0,
      restart_required: false,
    },
    {
      key: "max_concurrent_requests",
      section: "scheduler",
      label: "Max concurrent requests",
      description: "Scheduler concurrency",
      type: "integer",
      default: 1,
      minimum: 1,
      restart_required: true,
    },
    {
      key: "model_dirs",
      section: "model",
      label: "Model directories",
      description: "Search roots",
      type: "array",
      default: [],
      restart_required: true,
    },
  ],
  effective_model_dirs: ["/models"],
}
async function mock(page: Page) {
  const state = structuredClone(fixture)
  const patches: unknown[] = []
  const mainChanges: unknown[] = []
  const resourceRequests: { method: string; url: string }[] = []
  const keyActions: { method: string; body: unknown }[] = []
  let subKeys: { id: string; key: string; name: string; created_at: string }[] =
    []
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname
    let body: unknown = {}
    if (path === "/api/connection")
      body = { connected: true, server: "http://test-server:8000" }
    else if (path.endsWith("/server/settings")) {
      if (route.request().method() === "PATCH") {
        patches.push(route.request().postDataJSON())
        body = {
          changed: ["sampling.temperature"],
          live_applied: ["sampling.temperature"],
          restart_required: [],
        }
      } else body = state
    } else if (path.endsWith("/server/defaults"))
      body = {
        ...fixture,
        sections: {
          sampling: { temperature: 1 },
          scheduler: { max_concurrent_requests: 1 },
          model: { model_dirs: [] },
        },
      }
    else if (path.endsWith("/server/info"))
      body = {
        version: "0.test",
        base_path: "/tmp/test",
        host: "127.0.0.1",
        port: 8000,
        restart_supported: false,
      }
    else if (path.endsWith("/server/resources")) {
      const requestUrl = new URL(route.request().url())
      resourceRequests.push({
        method: route.request().method(),
        url: requestUrl.search,
      })
      const tier = requestUrl.searchParams.get("tier") ?? "balanced"
      const custom = Number(
        requestUrl.searchParams.get("custom_ceiling_gb") ?? 20
      )
      const gib = 2 ** 30
      const preview = {
        reserve_bytes: 8 * gib,
        free_bytes: 40 * gib,
        inactive_bytes: 0,
        other_apps_bytes: 8 * gib,
        static_bytes: 48 * gib,
        dynamic_bytes: 40 * gib,
        metal_cap_bytes: 30 * gib,
        ceiling_bytes: (tier === "custom" ? Math.min(custom, 30) : 30) * gib,
        binding: tier === "custom" ? "custom" : "metal_cap",
      }
      body = {
        hardware: {
          physical_memory_bytes: 64 * gib,
          available_memory_bytes: 40 * gib,
          metal_cap_bytes: 30 * gib,
        },
        tier_previews: {
          safe: preview,
          balanced: preview,
          aggressive: preview,
          custom: preview,
        },
        saved: {
          tier: "balanced",
          custom_ceiling_gb: 20,
          guard_enabled: true,
          preview: { ...preview, ceiling_bytes: 30 * gib },
        },
        runtime: {
          available: true,
          tier: "safe",
          custom_ceiling_gb: 0,
          guard_enabled: true,
          ceiling_bytes: 28 * gib,
          breakdown: null,
          wired_limit_request_bytes: 40 * gib,
        },
        draft: {
          tier,
          custom_ceiling_gb: custom,
          guard_enabled:
            requestUrl.searchParams.get("guard_enabled") !== "false",
          preview,
        },
        wired_limit: {
          limited: true,
          recommended_bytes: 50 * gib,
          recommended_mib: 51200,
          command: "sudo sysctl iogpu.wired_limit_mb=51200",
          copy_only: true,
        },
        warnings: ["The Metal cap is below the requested memory budget."],
      }
    } else if (path.endsWith("/server/integrations"))
      body = { integrations: [] }
    else if (path.endsWith("/auth/keys"))
      body = {
        main_key: "admin-secret",
        sub_keys: subKeys,
        policy: {
          skip_api_key_verification: false,
          allow_unauthenticated_inference: false,
        },
      }
    else if (
      path.endsWith("/auth/subkeys") ||
      path.includes("/auth/subkeys/")
    ) {
      const method = route.request().method()
      const payload = route.request().postDataJSON() as {
        name?: string
        key?: string
      } | null
      keyActions.push({ method, body: payload })
      if (method === "POST")
        subKeys = [
          {
            id: "opaque-1",
            name: payload?.name ?? "",
            key: payload?.key ?? "generated-secret",
            created_at: "2026-10-01T10:00:00Z",
          },
        ]
      else if (method === "PATCH")
        subKeys = subKeys.map((entry) => ({ ...entry, ...payload }))
      else if (method === "DELETE") subKeys = []
      body =
        method === "DELETE"
          ? { deleted: true, id: "opaque-1" }
          : { sub_key: subKeys[0] }
    } else if (path.endsWith("/auth/main-key")) {
      mainChanges.push(route.request().postDataJSON())
      body = { main_key: "new-admin-secret", live_applied: true }
    } else if (path.endsWith("/server/web-search/test"))
      body = {
        ok: false,
        error: {
          code: "invalid_arguments",
          message: "A Brave API key is required.",
        },
      }
    else if (path.endsWith("/state"))
      body = { models: [], engines: [], status: "running" }
    await route.fulfill({ json: body })
  })
  return { state, patches, mainChanges, keyActions, resourceRequests }
}

test("nested patch excludes unchanged fields and retains explicit null and empty lists", () => {
  expect(
    editedPatch(
      {
        sampling: { temperature: 0.7, max_tokens: null },
        model: { model_dirs: [] },
      },
      {
        sampling: { temperature: 0.7, max_tokens: 100 },
        model: { model_dirs: ["/models"] },
      }
    )
  ).toEqual({ sampling: { max_tokens: null }, model: { model_dirs: [] } })
})

test("server drafts survive section changes and refresh; save sends edited fields only", async ({
  page,
}) => {
  const { state, patches } = await mock(page)
  await page.goto("/settings")
  await expect(page.getByText("0.test", { exact: true })).toBeVisible()
  await page.getByRole("tab", { name: "Defaults", exact: true }).click()
  await page.getByLabel("Temperature", { exact: true }).fill("0.3")
  await page.getByRole("tab", { name: "Network", exact: true }).click()
  await page.getByRole("tab", { name: "Defaults", exact: true }).click()
  state.sections.sampling.temperature = 0.8
  await page.getByRole("button", { name: "Refresh dashboard" }).click()
  await expect(page.getByLabel("Temperature", { exact: true })).toHaveValue(
    "0.3"
  )
  await page
    .getByRole("button", { name: "Review changes", exact: true })
    .click()
  await expect(
    page.getByText("Nothing changes on the server until you save.")
  ).toBeVisible()
  await page
    .getByRole("button", { name: "Save 1 changes", exact: true })
    .click()
  await expect
    .poll(() => patches)
    .toEqual([{ sections: { sampling: { temperature: 0.3 } } }])
  await expect(page.getByText("Settings saved", { exact: true })).toBeVisible()
})

test("main key starts masked and rotation requires explicit confirmation", async ({
  page,
}) => {
  const { mainChanges } = await mock(page)
  await page.goto("/settings")
  await expect(page.getByText("0.test", { exact: true })).toBeVisible()
  await page.getByRole("tab", { name: "API keys", exact: true }).click()
  await expect(page.getByText("admin-secret", { exact: true })).toHaveCount(0)
  await page.getByRole("button", { name: "Reveal main key" }).click()
  await expect(page.getByText("admin-secret", { exact: true })).toBeVisible()
  await page
    .getByRole("button", { name: "Rotate main key", exact: true })
    .click()
  await page.getByLabel("New main key").fill("new-admin-secret")
  await expect(
    page.getByRole("button", { name: "Rotate key", exact: true })
  ).toBeDisabled()
  await page.getByLabel("Type ROTATE to confirm").fill("ROTATE")
  await page.getByRole("button", { name: "Rotate key", exact: true }).click()
  await expect.poll(() => mainChanges).toEqual([{ key: "new-admin-secret" }])
  await expect(page.getByText(/Main key rotated/)).toBeVisible()
})

test("web search failures render provider errors without a page crash", async ({
  page,
}) => {
  await mock(page)
  await page.goto("/settings")
  await expect(page.getByText("0.test", { exact: true })).toBeVisible()
  await page
    .getByRole("tab", { name: "Integrations & tools", exact: true })
    .click()
  await page.getByRole("button", { name: "Test search", exact: true }).click()
  await expect(page.getByText("A Brave API key is required.")).toBeVisible()
})

test("inference keys can be named, edited, and revoked with confirmation", async ({
  page,
}) => {
  const { keyActions } = await mock(page)
  await page.goto("/settings")
  await expect(page.getByText("0.test", { exact: true })).toBeVisible()
  await page.getByRole("tab", { name: "API keys", exact: true }).click()
  await page.getByRole("button", { name: "Create key", exact: true }).click()
  await page.getByLabel("Name", { exact: true }).fill("Notebook")
  await page.getByRole("button", { name: "Save key", exact: true }).click()
  await expect(page.getByText("Notebook", { exact: false })).toBeVisible()
  await expect
    .poll(() => keyActions)
    .toEqual([{ method: "POST", body: { name: "Notebook" } }])
  await page.getByRole("button", { name: "Edit", exact: true }).click()
  await page.getByLabel("Name", { exact: true }).fill("Laptop")
  await page.getByRole("button", { name: "Save key", exact: true }).click()
  await expect(page.getByText("Laptop", { exact: false })).toBeVisible()
  await page.getByRole("button", { name: "Revoke", exact: true }).click()
  await expect(
    page.getByRole("button", { name: "Revoke key", exact: true })
  ).toBeDisabled()
  await page.getByLabel("Type REVOKE to confirm").fill("REVOKE")
  await page.getByRole("button", { name: "Revoke key", exact: true }).click()
  await expect(
    page.getByText("No inference keys yet", { exact: true })
  ).toBeVisible()
  expect(keyActions.map((entry) => entry.method)).toEqual([
    "POST",
    "PATCH",
    "DELETE",
  ])
})

test("default preview and discard leave the server untouched", async ({
  page,
}) => {
  const { patches } = await mock(page)
  await page.goto("/settings")
  await expect(page.getByText("0.test", { exact: true })).toBeVisible()
  await page.getByRole("tab", { name: "Defaults", exact: true }).click()
  await page.getByRole("button", { name: "Use default", exact: true }).click()
  await expect(page.getByLabel("Temperature", { exact: true })).toHaveValue("1")
  await page.getByLabel("Temperature", { exact: true }).fill("0.1")
  await page
    .getByRole("button", { name: "Review changes", exact: true })
    .click()
  await expect(
    page.getByText("Nothing changes on the server until you save.")
  ).toBeVisible()
  expect(patches).toEqual([])
  await page.getByRole("button", { name: "Discard", exact: true }).click()
  await expect(page.getByLabel("Temperature", { exact: true })).toHaveValue(
    "0.7"
  )
  expect(patches).toEqual([])
})

test("resource draft requests are read-only and reject an invalid custom ceiling", () => {
  expect(
    resourceDraft({ prefill_memory_guard: true, memory_guard_tier: "balanced" })
      .path
  ).toBe("server/resources?guard_enabled=true&tier=balanced")
  expect(
    resourceDraft({
      prefill_memory_guard: false,
      memory_guard_tier: "balanced",
    }).path
  ).toBe("server/resources?guard_enabled=false&tier=balanced")
  expect(
    resourceDraft({
      memory_guard_tier: "custom",
      memory_guard_custom_ceiling_gb: 8,
    })
  ).toMatchObject({
    valid: true,
    path: "server/resources?tier=custom&custom_ceiling_gb=8",
  })
  expect(
    resourceDraft({
      memory_guard_tier: "custom",
      memory_guard_custom_ceiling_gb: 0,
    }).valid
  ).toBe(false)
  expect(
    resourceDraft({
      memory_guard_tier: "custom",
      memory_guard_custom_ceiling_gb: Infinity,
    }).valid
  ).toBe(false)
})

test("resource previews follow unsaved memory settings and OS commands are copy-only", async ({
  page,
}) => {
  const { patches, resourceRequests } = await mock(page)
  await page.context().grantPermissions(["clipboard-read", "clipboard-write"])
  await page.goto("/settings")
  await expect(page.getByText("0.test", { exact: true })).toBeVisible()
  await page
    .getByRole("tab", { name: "Models & resources", exact: true })
    .click()
  await expect(
    page.getByText("Memory capacity & guard", { exact: true })
  ).toBeVisible()
  await expect(page.getByText("64.0 GiB", { exact: true })).toBeVisible()
  await expect(
    page.getByText("The Metal cap is below the requested memory budget.")
  ).toBeVisible()
  await page.getByLabel("Memory guard tier", { exact: true }).click()
  await page.getByRole("option", { name: "Custom", exact: true }).click()
  await page.getByLabel("Custom memory ceiling", { exact: true }).fill("8")
  await expect(
    page.getByRole("row").filter({ hasText: "Current draft" })
  ).toContainText("8.0 GiB")
  await expect(
    page.getByRole("row").filter({ hasText: "Running process" })
  ).toContainText("28.0 GiB")
  await expect(
    page.getByRole("row").filter({ hasText: "Saved on disk" })
  ).toContainText("30.0 GiB")
  await page
    .getByRole("button", { name: "Copy OS command", exact: true })
    .click()
  await expect(page.getByText("Command copied", { exact: true })).toBeVisible()
  expect(patches).toEqual([])
  expect(resourceRequests.length).toBeGreaterThan(0)
  expect(resourceRequests.every((request) => request.method === "GET")).toBe(
    true
  )
  expect(
    resourceRequests.some((request) =>
      request.url.includes("custom_ceiling_gb=8")
    )
  ).toBe(true)
  await page
    .getByRole("switch", { name: "Prefill memory guard", exact: true })
    .click()
  await expect(
    page.getByRole("row").filter({ hasText: "Current draft" })
  ).toContainText("Disabled")
  await expect
    .poll(() =>
      resourceRequests.some((request) =>
        request.url.includes("guard_enabled=false")
      )
    )
    .toBe(true)
  await page.getByLabel("Custom memory ceiling", { exact: true }).fill("0")
  await expect(
    page.getByText("Custom memory ceiling needs a value", { exact: true })
  ).toBeVisible()
  expect(patches).toEqual([])
})
