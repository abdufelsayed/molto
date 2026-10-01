import type { RegistryRecord } from "./queries"
export type Artifact = {
  record: RegistryRecord
  id: string
  model_id?: string
  name?: string
  path: string
  root_id?: string
  task?: string
  model_type?: string
  health: string
  exposed?: boolean
  loaded?: boolean
  size_bytes?: number
  revision?: string | null
  source?: string | null
  blockers?: string[]
}
export type Registry = { models: Artifact[] }
export type Storage = {
  roots: {
    id: string
    path: string
    total_bytes?: number
    free_bytes?: number
    used_bytes?: number
    writable?: boolean
  }[]
}
export type Collection = {
  id: string
  name: string
  description?: string
  model_ids: string[]
  preload: boolean
}
export type Collections = { collections: Collection[] }
export type Plan = {
  additional_bytes: number
  current_bytes: number
  ceiling_bytes: number
  evictions: { id: string; bytes: number }[]
  steps: { action: string; model_id: string }[]
  fits?: boolean
  blockers?: string[]
  models?: { id: string; estimated_bytes?: number }[]
  warnings?: string[]
}
export type DeletePlan = {
  plan_token?: string
  model_id?: string
  path?: string
  physical_bytes?: number
  protected_reasons: string[]
  safe: boolean
}
export type ImportPreview = {
  affected_model_ids?: string[]
  can_apply?: boolean
  blockers?: string[]
  valid?: boolean
  errors?: string[]
  changes?: {
    model_id?: string | null
    field: string
    before?: unknown
    after?: unknown
  }[]
  preview_token?: string
}
export type LifecycleResult = {
  operation?: {
    id: string
    status: string
    stage: string
    progress: number
    error?: string
    result?: {
      status?: string
      summary?: string
      local_revision?: string
      remote_revision?: string
    }
  }
  status?: string
  message?: string
  revision?: string
  update_available?: boolean
  current_revision?: string
  latest_revision?: string
  revisions?: { revision: string; active?: boolean; path?: string }[]
  warnings?: string[]
  blockers?: string[]
}
