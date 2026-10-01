import { z } from "zod"
import {
  managementRequest,
  type RequestOptions,
} from "@/features/management/request"

const record = z.record(z.string(), z.unknown())
export const deviceSchema = z
  .object({
    node_id: z.string().min(1),
    friendly_name: z.string().optional(),
    state: z.string().optional(),
    ssh_target: z.string().optional(),
    addrs: z
      .array(z.object({ ip: z.string(), if_type: z.string().optional() }))
      .default([]),
    caps: record.default({}),
  })
  .passthrough()
export const devicesSchema = z
  .object({
    self: deviceSchema.nullable(),
    paired: z.array(deviceSchema),
    discovered: z.array(deviceSchema),
  })
  .passthrough()
export type Device = z.infer<typeof deviceSchema>
export const modelSchema = z
  .object({
    id: z.string(),
    model_path: z.string().min(1),
    display_name: z.string().optional(),
    model_source: z.string().optional(),
    source_node_id: z.string().optional(),
    python_executable: z.string().optional(),
    locations: z
      .array(
        z
          .object({
            node_id: z.string(),
            model_path: z.string(),
            ssh: z.string().optional(),
            python_executable: z.string().optional(),
          })
          .passthrough()
      )
      .default([]),
  })
  .passthrough()
export const inventorySchema = z
  .object({
    models: z.array(modelSchema),
    errors: z
      .array(
        z.object({ detail: z.string(), node_id: z.string() }).passthrough()
      )
      .default([]),
  })
  .passthrough()
export const planNodeSchema = z
  .object({
    node_id: z.string(),
    capacity_bytes: z.number().positive(),
    reserve_bytes: z.number().nonnegative().optional(),
    role: z.string(),
  })
  .passthrough()
export const budgetsSchema = z.object({ nodes: z.array(planNodeSchema) })
export const proposalSchema = z
  .object({
    ready_to_activate: z.boolean().optional(),
    ready_to_stage: z.boolean().optional(),
    plan: z.object({ placement_signature: z.string().min(16) }).passthrough(),
    activation: z
      .object({
        approved_placement: z.string().min(16),
        model_path: z.string(),
        nodes: z.array(planNodeSchema).min(2),
        hosts: z.array(record).min(2),
        backend: z.enum(["ring", "jaccl", "jaccl-ring"]),
      })
      .passthrough(),
  })
  .passthrough()
  .refine(
    (value) =>
      value.plan.placement_signature === value.activation.approved_placement,
    "The activation signature does not match the preview."
  )
export type Proposal = z.infer<typeof proposalSchema>
export const stageSchema = z
  .object({ job_id: z.string().min(1), status: z.string() })
  .passthrough()
export const deploymentsSchema = z
  .object({
    deployments: z.array(
      z
        .object({
          deployment_id: z.string(),
          model: z.string().optional(),
          model_path: z.string().optional(),
          tensor_parallel_size: z.number().int().min(1).max(64).default(1),
          hosts: z
            .array(
              z
                .object({
                  node_id: z.string(),
                  ssh: z.string(),
                  ips: z.array(z.string()).default([]),
                  rdma: z.array(z.unknown()).default([]),
                  python_executable: z.string().nullable().optional(),
                })
                .passthrough()
            )
            .default([]),
          path_map: z.record(z.string(), z.string()).default({}),
          assignments: z
            .array(
              z
                .object({ node_id: z.string(), role: z.string().optional() })
                .passthrough()
            )
            .default([]),
        })
        .passthrough()
    ),
  })
  .passthrough()
export const joinStatusSchema = z
  .object({
    join_keys: z.array(
      z
        .object({
          join_id: z.string(),
          status: z.string(),
          expires_at: z.number().optional(),
        })
        .passthrough()
    ),
    nodes: z.array(
      z.object({ node_id: z.string(), ssh: z.string() }).passthrough()
    ),
  })
  .passthrough()
export const rdmaSchema = z
  .object({
    helper: z.object({
      available: z.boolean(),
      reason: z.string().nullable().optional(),
    }),
    links: z.array(
      z
        .object({
          name: z.string(),
          verified: z.boolean(),
          reason: z.string().nullable().optional(),
          stale_reason: z.string().nullable().optional(),
          in_use_by: z.string().nullable().optional(),
          verifying: z.boolean().optional(),
        })
        .passthrough()
    ),
  })
  .passthrough()
export const evidenceSchema = record
export async function clusterRequest<S extends z.ZodType>(
  path: string,
  schema: S,
  options: RequestOptions = {}
): Promise<z.infer<S>> {
  return schema.parse(
    await managementRequest<unknown>(`cluster/${path}`, options)
  )
}
export function inventoryHosts(devices: Device[], selfId: string | undefined) {
  return devices
    .map((device) => ({
      node_id: device.node_id,
      ssh:
        device.node_id === selfId
          ? "127.0.0.1"
          : device.ssh_target ||
            device.addrs.find((address) => !address.ip.startsWith("fe80:"))
              ?.ip ||
            "",
    }))
    .filter((host) => host.ssh)
}

export const catalogueSchema = z
  .object({
    models: z.array(
      z
        .object({
          model_id: z.string(),
          fits: z.boolean(),
          reason: z.string().optional(),
          supports_pipeline: z.boolean().optional(),
          supports_tensor_parallel: z.boolean().optional(),
        })
        .passthrough()
    ),
  })
  .passthrough()

export type Deployment = z.infer<
  typeof deploymentsSchema
>["deployments"][number]
export function membershipHosts(
  deployment: Deployment,
  devices: Device[],
  selfId: string | undefined,
  ids: string[]
) {
  const live = inventoryHosts(devices, selfId)
  return ids.map((id) => {
    const prior = deployment.hosts.find((host) => host.node_id === id)
    if (prior) return prior
    const device = devices.find((node) => node.node_id === id)
    return {
      node_id: id,
      ssh: live.find((host) => host.node_id === id)?.ssh ?? "",
      ips: device?.addrs.map((address) => address.ip) ?? [],
      rdma: [],
    }
  })
}
export function membershipReplanBody({
  deployment,
  nodes,
  hosts,
  inventory,
  target_context_tokens,
  execution_profile,
}: {
  deployment: Deployment
  nodes: z.infer<typeof planNodeSchema>[]
  hosts: ReturnType<typeof membershipHosts>
  inventory: z.infer<typeof modelSchema>[]
  target_context_tokens: number
  execution_profile: string
}) {
  if (
    nodes.length < 2 ||
    hosts.length !== nodes.length ||
    hosts.some((host) => !host.ssh || !host.ips.length)
  )
    throw new Error(
      "Select at least two members with known SSH and collective addresses."
    )
  const modelPath = deployment.model ?? deployment.model_path
  if (!modelPath)
    throw new Error("The deployment does not report its model path.")
  const memberIds = new Set(nodes.map((node) => node.node_id))
  const model = inventory.find(
    (item) =>
      item.model_path === modelPath ||
      item.locations.some((location) => location.model_path === modelPath)
  )
  const paths = {
    ...Object.fromEntries(
      (model?.locations ?? []).map((location) => [
        location.node_id,
        location.model_path,
      ])
    ),
    ...deployment.path_map,
  }
  const source = model?.locations.find(
    (location) => location.ssh === model.model_source
  )
  return {
    deployment_id: deployment.deployment_id,
    model_path: modelPath,
    tensor_parallel_size: deployment.tensor_parallel_size,
    nodes,
    hosts,
    path_map: Object.fromEntries(
      Object.entries(paths).filter(([id]) => memberIds.has(id))
    ),
    ...(model?.model_source
      ? {
          model_source: model.model_source,
          model_source_python:
            source?.python_executable ?? model.python_executable,
        }
      : {}),
    target_context_tokens,
    execution_profile,
  }
}
export const replanPreviewSchema = z
  .object({
    mode: z.literal("preview"),
    backend: z.enum(["ring", "jaccl", "jaccl-ring"]),
    plan: z.object({ placement_signature: z.string().min(16) }).passthrough(),
  })
  .passthrough()
