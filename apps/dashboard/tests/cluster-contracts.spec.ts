import { test, expect } from "@playwright/test"
import {
  devicesSchema,
  proposalSchema,
  budgetsSchema,
  stageSchema,
  inventoryHosts,
} from "../src/features/cluster/contracts"

const signature = "0123456789abcdef0123456789abcdef"
function proposal() {
  return {
    ready_to_stage: true,
    ready_to_activate: false,
    plan: {
      placement_signature: signature,
      assignments: [{ node_id: "local", rank: 0 }],
    },
    activation: {
      approved_placement: signature,
      model_path: "/models/demo",
      backend: "ring",
      nodes: ["local", "peer"].map((node_id) => ({
        node_id,
        capacity_bytes: 1024,
        role: "headless",
        manual_memory_limit: true,
      })),
      hosts: [
        { node_id: "local", ssh: "127.0.0.1", ips: ["192.168.1.1"] },
        { node_id: "peer", ssh: "worker@192.168.1.2", ips: ["192.168.1.2"] },
      ],
      path_map: { peer: "/worker/models/demo" },
      preflight: true,
    },
  }
}
test("signed placement preserves exact server activation fields", () => {
  const wire = proposal()
  expect(proposalSchema.parse(wire).activation).toEqual(wire.activation)
  wire.activation.approved_placement = "abcdef0123456789abcdef0123456789"
  expect(proposalSchema.safeParse(wire).success).toBe(false)
})
test("unknown memory cannot become a fabricated planning budget", () => {
  expect(
    budgetsSchema.safeParse({
      nodes: [{ node_id: "peer", capacity_bytes: 0, role: "headless" }],
    }).success
  ).toBe(false)
  expect(
    budgetsSchema.safeParse({ nodes: [{ node_id: "peer", role: "headless" }] })
      .success
  ).toBe(false)
})
test("trusted inventory uses enrolled SSH targets and retains suspect liveness", () => {
  const devices = devicesSchema.parse({
    self: { node_id: "local", addrs: [{ ip: "192.168.1.1" }] },
    paired: [
      {
        node_id: "peer",
        state: "suspect",
        ssh_target: "worker@192.168.1.2",
        addrs: [{ ip: "192.168.1.2" }],
      },
    ],
    discovered: [],
  })
  expect(devices.paired[0]?.state).toBe("suspect")
  expect(inventoryHosts([devices.self!, ...devices.paired], "local")).toEqual([
    { node_id: "local", ssh: "127.0.0.1" },
    { node_id: "peer", ssh: "worker@192.168.1.2" },
  ])
  expect(devicesSchema.safeParse({ paired: [], discovered: [] }).success).toBe(
    false
  )
})
test("staging keeps per-node failure evidence and rejects missing job identity", () => {
  const job = {
    job_id: "job-1",
    status: "failed",
    nodes: [{ node_id: "peer", error: "Insufficient disk" }],
  }
  expect(stageSchema.parse(job)).toEqual(job)
  expect(stageSchema.safeParse({ status: "completed" }).success).toBe(false)
})

test("TP2 membership replans preserve tensor parallelism, fabric and paths", async () => {
  const { deploymentsSchema, membershipHosts, membershipReplanBody } =
    await import("../src/features/cluster/contracts")
  const deployment = deploymentsSchema.parse({
    deployments: [
      {
        deployment_id: "active",
        tensor_parallel_size: 2,
        model: "/models/demo",
        hosts: [
          {
            node_id: "local",
            ssh: "127.0.0.1",
            ips: ["10.0.0.1"],
            rdma: ["rdma0"],
          },
          {
            node_id: "old",
            ssh: "worker@10.0.0.2",
            ips: ["10.0.0.2"],
            rdma: ["rdma1"],
          },
        ],
        path_map: { local: "/models/local-demo", old: "/models/old-demo" },
      },
    ],
  }).deployments[0]!
  const devices = devicesSchema.parse({
    self: { node_id: "local", addrs: [{ ip: "192.168.1.1" }] },
    paired: [
      {
        node_id: "new",
        ssh_target: "worker@192.168.1.3",
        addrs: [{ ip: "192.168.1.3" }],
      },
    ],
    discovered: [],
  })
  const hosts = membershipHosts(
    deployment,
    [devices.self!, ...devices.paired],
    "local",
    ["local", "new"]
  )
  expect(hosts[0]).toEqual(deployment.hosts[0])
  expect(hosts[1]).toMatchObject({
    node_id: "new",
    ssh: "worker@192.168.1.3",
    ips: ["192.168.1.3"],
  })
  const nodes = [
    { node_id: "local", capacity_bytes: 4096, role: "workstation" },
    { node_id: "new", capacity_bytes: 8192, role: "headless" },
  ]
  const body = membershipReplanBody({
    deployment,
    nodes,
    hosts,
    inventory: [
      {
        id: "demo",
        model_path: "/models/demo",
        model_source: "worker@192.168.1.3",
        locations: [
          { node_id: "local", model_path: "/replacement/local" },
          {
            node_id: "new",
            model_path: "/new/full-model",
            ssh: "worker@192.168.1.3",
            python_executable: "/new/python",
          },
        ],
      },
    ],
    target_context_tokens: 16384,
    execution_profile: "interactive",
  })
  expect(body.tensor_parallel_size).toBe(2)
  const staged = proposalSchema.parse({
    plan: { placement_signature: signature },
    activation: {
      ...body,
      backend: "ring",
      approved_placement: signature,
      preflight: true,
    },
  }).activation
  expect(staged.tensor_parallel_size).toBe(2)
  expect(body.nodes).toEqual(nodes)
  expect(body.path_map).toEqual({
    local: "/models/local-demo",
    new: "/new/full-model",
  })
  expect(body.model_source_python).toBe("/new/python")
  expect(body).not.toHaveProperty("approved_placement")
})
