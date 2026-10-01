import { managementQuery } from "@/features/management/request"
import type { Artifact, Registry, Storage } from "./types"
export type RegistryRecord = {
  id: string
  display_name: string
  path: string | null
  kind: string
  model_type: string
  capabilities: { tasks: string[] }
  storage: { estimated_bytes: number }
  health: { status: string; summary?: string }
  runtime: {
    loaded: boolean
    loading: boolean
    unloading: boolean
    pinned: boolean
    default: boolean
  }
  source: {
    revision: string | null
    cached_revisions: { revision: string; active: boolean }[]
  }
  update: {
    status: string
    local_revision?: string
    remote_revision?: string
    staged_revision?: string
  }
  lineage: {
    parents: { id: string; relation: string }[]
    children: { id: string; relation: string }[]
  }
  preparation_blocker?: string
}
export function registryQuery() {
  return {
    ...managementQuery<{ models: RegistryRecord[] }>(
      ["workspace", "registry"],
      "workspace/registry",
      5000
    ),
    select: (data: { models: RegistryRecord[] }): Registry => ({
      models: data.models.map((r): Artifact => ({
        id: r.id,
        name: r.display_name,
        path: r.path ?? "",
        model_type: r.model_type,
        task: r.capabilities.tasks[0] ?? r.model_type,
        health: r.health.status,
        exposed: r.kind === "virtual",
        size_bytes: r.storage.estimated_bytes,
        revision: r.source.revision,
        blockers: [r.health.summary, r.preparation_blocker].filter(
          (s): s is string => !!s
        ),
        record: r,
      })),
    }),
  }
}
export function storageQuery() {
  return {
    ...managementQuery<{
      roots: {
        id: string
        path: string
        disk_free_bytes: number
        disk_total_bytes: number
        physical_bytes: number
        exists: boolean
      }[]
    }>(["workspace", "storage"], "workspace/storage", 10000),
    select: (data: {
      roots: {
        id: string
        path: string
        disk_free_bytes: number
        disk_total_bytes: number
        physical_bytes: number
        exists: boolean
      }[]
    }): Storage => ({
      roots: data.roots.map((r) => ({
        ...r,
        free_bytes: r.disk_free_bytes,
        total_bytes: r.disk_total_bytes,
        used_bytes: r.physical_bytes,
      })),
    }),
  }
}
