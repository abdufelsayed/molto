import { test, expect, type Page } from "@playwright/test"

async function fixtures(
  page: Page,
  available: boolean,
  overrides: Record<string, unknown> = {}
) {
  const requests: { path: string; method: string; body: unknown }[] = []
  await page.route("**/api/molto/cluster/**", async (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname.split("/cluster/")[1] ?? ""
    requests.push({
      path,
      method: request.method(),
      body: request.postDataJSON(),
    })
    if (!available) {
      await route.fulfill({
        status: 404,
        json: { detail: "Distributed inference is disabled" },
      })
      return
    }
    const responses: Record<string, unknown> = {
      devices: {
        self: {
          node_id: "local",
          friendly_name: "Coordinator",
          caps: { ram_gb: 32 },
          addrs: [{ ip: "192.168.1.1", if_type: "ethernet" }],
        },
        paired: [
          {
            node_id: "peer",
            friendly_name: "Worker",
            state: "suspect",
            ssh_target: "worker@192.168.1.2",
            addrs: [{ ip: "192.168.1.2" }],
          },
        ],
        discovered: [],
      },
      deployments: { deployments: [] },
      runtime: { jobs: [], launchers: [] },
      "pair/join": { status: "idle" },
      "join-status": { join_keys: [], nodes: [] },
      "rdma-links": {
        helper: { available: false, reason: "RDMA helper is not installed" },
        links: [],
      },
      "devices/manual": { ok: true, verified: false },
      models: {
        models: [
          {
            id: "demo",
            model_path: "/models/demo",
            display_name: "Demo model",
            model_source: "127.0.0.1",
            locations: [],
          },
        ],
        errors: [],
      },
      "node-budgets": {
        nodes: [
          {
            node_id: "local",
            capacity_bytes: 32 * 1024 ** 3,
            role: "workstation",
          },
          { node_id: "peer", capacity_bytes: 32 * 1024 ** 3, role: "headless" },
        ],
      },
      autoconfigure: {
        ready_to_activate: false,
        ready_to_stage: false,
        preflight: ["Worker is offline"],
        plan: { placement_signature: "0123456789abcdef" },
        activation: {
          approved_placement: "0123456789abcdef",
          model_path: "/models/demo",
          backend: "ring",
          nodes: [
            {
              node_id: "local",
              capacity_bytes: 32 * 1024 ** 3,
              role: "workstation",
            },
            {
              node_id: "peer",
              capacity_bytes: 32 * 1024 ** 3,
              role: "headless",
            },
          ],
          hosts: [
            { node_id: "local", ssh: "127.0.0.1" },
            { node_id: "peer", ssh: "worker@192.168.1.2" },
          ],
        },
      },
    }
    await route.fulfill({
      json: overrides[path] ?? responses[path] ?? { ok: true },
    })
  })
  return requests
}
test("standalone servers explain unavailable cluster without issuing mutations", async ({
  page,
}) => {
  const requests = await fixtures(page, false)
  await page.goto("/cluster")
  await expect(
    page.getByText("Cluster unavailable on this server")
  ).toBeVisible()
  await expect(
    page.getByText(/Standalone servers do not expose cluster management/)
  ).toBeVisible()
  await expect(
    page.getByRole("button", { name: "Activate reviewed placement" })
  ).toHaveCount(0)
  expect(requests.every((request) => request.method === "GET")).toBe(true)
})
test("explicit inventory and preview preserve unavailable preflight gates", async ({
  page,
}) => {
  const requests = await fixtures(page, true)
  await page.goto("/cluster")
  await expect(
    page.getByText("RDMA helper is not installed", { exact: false })
  ).toBeVisible()
  await expect(page.getByText(/suspect/)).toBeVisible()
  await page.getByRole("button", { name: "Read model inventory" }).click()
  await expect(
    page.getByRole("combobox", { name: "Cluster model" })
  ).toBeEnabled()
  await page.getByRole("combobox", { name: "Cluster model" }).click()
  await page.getByRole("option", { name: "Demo model" }).click()
  await page.getByRole("button", { name: "Preview placement" }).click()
  await expect(
    page.getByText("Worker is offline", { exact: true })
  ).toBeVisible()
  await expect(
    page.getByRole("button", { name: "Activate reviewed placement" })
  ).toBeDisabled()
  await expect(
    page.getByRole("button", { name: "Stage approved placement" })
  ).toBeDisabled()
  const preview = requests.find((request) => request.path === "autoconfigure")
  expect(preview?.body).toMatchObject({
    measure_performance: false,
    strategy: "auto",
    target_context_tokens: 8192,
  })
  expect(
    requests.some(
      (request) => request.path === "deployments" && request.method === "POST"
    )
  ).toBe(false)
})

test("TP2 deployment membership preserves preview, staging and approval contracts", async ({
  page,
}) => {
  const nodes = [
    { node_id: "local", capacity_bytes: 32 * 1024 ** 3, role: "workstation" },
    { node_id: "peer", capacity_bytes: 32 * 1024 ** 3, role: "headless" },
    { node_id: "extra", capacity_bytes: 64 * 1024 ** 3, role: "headless" },
  ]
  const requests = await fixtures(page, true, {
    devices: {
      self: {
        node_id: "local",
        friendly_name: "Coordinator",
        addrs: [{ ip: "192.168.1.1" }],
      },
      paired: [
        {
          node_id: "peer",
          friendly_name: "Worker",
          ssh_target: "worker@192.168.1.2",
          addrs: [{ ip: "192.168.1.2" }],
        },
        {
          node_id: "extra",
          friendly_name: "Extra worker",
          ssh_target: "worker@192.168.1.3",
          addrs: [{ ip: "192.168.1.3" }],
        },
      ],
      discovered: [],
    },
    deployments: {
      deployments: [
        {
          deployment_id: "active",
          tensor_parallel_size: 2,
          model: "/models/demo",
          hosts: [
            { node_id: "local", ssh: "127.0.0.1", ips: ["10.0.0.1"], rdma: [] },
            {
              node_id: "peer",
              ssh: "worker@192.168.1.2",
              ips: ["10.0.0.2"],
              rdma: [],
            },
          ],
          path_map: { local: "/local/model", peer: "/worker/model" },
        },
      ],
    },
    "node-budgets": { nodes },
    replan: {
      mode: "preview",
      backend: "ring",
      plan: { placement_signature: "0123456789abcdef" },
      changes: { added: ["extra"] },
    },
    stage: { job_id: "membership-job", status: "completed" },
    "stage/membership-job": { job_id: "membership-job", status: "completed" },
  })
  await page.goto("/cluster")
  await page
    .getByRole("button", { name: "Add Extra worker", exact: true })
    .click()
  await page
    .getByRole("button", { name: "Preview replan", exact: true })
    .click()
  await expect(
    page.getByRole("button", { name: "Stage reviewed replan" })
  ).toBeEnabled()
  const preview = requests.find((request) => request.path === "replan")
  expect(preview?.body).toMatchObject({
    tensor_parallel_size: 2,
    nodes,
    hosts: [
      { node_id: "local", ips: ["10.0.0.1"] },
      { node_id: "peer", ips: ["10.0.0.2"] },
      { node_id: "extra", ips: ["192.168.1.3"] },
    ],
    path_map: { local: "/local/model", peer: "/worker/model" },
  })
  expect(preview?.body).not.toHaveProperty("approved_placement")
  await page.getByRole("button", { name: "Stage reviewed replan" }).click()
  await expect(
    page.getByRole("button", { name: "Approve replan and reload" })
  ).toBeEnabled()
  await page.getByRole("button", { name: "Approve replan and reload" }).click()
  await expect
    .poll(() => requests.filter((request) => request.path === "replan").length)
    .toBe(2)
  expect(
    requests.filter((request) => request.path === "replan")[1]?.body
  ).toMatchObject({
    ...(preview?.body as object),
    approved_placement: "0123456789abcdef",
  })
  expect(
    requests.find((request) => request.path === "stage")?.body
  ).toMatchObject({
    activation: {
      tensor_parallel_size: 2,
      ...(preview?.body as object),
      backend: "ring",
      approved_placement: "0123456789abcdef",
      preflight: true,
    },
  })
})
