import { z } from "zod"
import type { components } from "@molto/contracts"

type Schema = components["schemas"]
export type State = Schema["StateResponse"]
export type Stats = Schema["StatsResponse"]
export type ModelSettings = Schema["ModelSettingsPatch"]
export type Settings = Schema["GlobalSettingsView"]
export type Profile = Schema["ProfileView"]
export type ProfileWrite = Schema["ProfileWrite"]
export type ProfileUpdate = Schema["ProfileUpdate"]
export type Cache = Schema["CacheResponse"]
export type SettingsResult = Schema["ModelSettingsUpdateResponse"]
const modelExtras = z.object({
  model_path: z.string().optional(),
  estimated_size: z.number().optional(),
  resident_estimated_size: z.number().optional(),
  actual_size: z.number().nullish(),
  config_model_type: z.string().nullish(),
  model_context_length: z.number().nullish(),
  load_failed: z.boolean().optional(),
  load_failure_message: z.string().nullish(),
  is_helper: z.boolean().optional(),
})
export type Model = Schema["ModelInventoryItem"] & z.infer<typeof modelExtras>

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number
  ) {
    super(message)
    this.name = "ApiError"
  }
}
export function errorMessage(error: Error) {
  if (error.name === "TimeoutError")
    return "The request timed out. Molto may still be working; check current state before retrying."
  if (error instanceof TypeError) return "Could not reach the dashboard server."
  return error.message
}
export function detail(data: unknown, status: number) {
  const error = z
    .object({
      detail: z.union([
        z.string(),
        z.array(
          z.object({
            msg: z.string(),
            loc: z.array(z.union([z.string(), z.number()])).optional(),
          })
        ),
        z.object({ message: z.string(), code: z.string().optional() }),
      ]),
    })
    .safeParse(data)
  if (!error.success) return `The server returned HTTP ${status}.`
  if (typeof error.data.detail === "string") return error.data.detail
  if (!Array.isArray(error.data.detail)) return error.data.detail.message
  return error.data.detail
    .map(
      (item) =>
        `${item.loc?.filter((part) => part !== "body").join(".") ?? ""}: ${item.msg}`
    )
    .join("; ")
}
export class ManagementClient {
  private async request<T>(
    path: string,
    signal?: AbortSignal,
    method = "GET",
    body?: unknown
  ): Promise<T> {
    const timeout = AbortSignal.timeout(method === "GET" ? 20_000 : 120_000)
    const response = await fetch(path, {
      method,
      credentials: "same-origin",
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      ...(method === "GET" || body === undefined
        ? {}
        : { body: JSON.stringify(body) }),
      cache: "no-store",
      signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
    })
    const data: unknown = await response.json().catch(() => null)
    if (!response.ok)
      throw new ApiError(detail(data, response.status), response.status)
    if (data === null)
      throw new Error("The server returned an invalid response.")
    return data as T
  }
  async connection(signal?: AbortSignal) {
    const status = await this.request<{
      connected: boolean
      server: string
      access?: "local" | "key"
      auto_connect?: boolean
    }>("/api/connection", signal)
    if (
      (!status.connected || status.access === "local") &&
      status.auto_connect !== false &&
      typeof window !== "undefined"
    ) {
      try {
        return await this.connectLocal(true, signal)
      } catch (error) {
        if (!(error instanceof ApiError) || ![403, 409].includes(error.status))
          throw error
      }
    }
    return status
  }
  localAccess(signal?: AbortSignal) {
    return this.request<{ available: boolean }>("/api/local-session", signal)
  }
  connectLocal(automatic = false, signal?: AbortSignal) {
    return this.request<{
      connected: boolean
      server: string
      access?: "local" | "key"
    }>("/api/local-session", signal, "POST", { automatic })
  }
  connect(key: string) {
    return this.request("/api/connection", undefined, "POST", { key })
  }
  disconnect() {
    return this.request("/api/connection", undefined, "DELETE")
  }
  state(signal?: AbortSignal) {
    return this.request<State>("/api/molto/state", signal)
  }
  async models(signal?: AbortSignal) {
    const data = await this.request<Schema["InventoryResponse"]>(
      "/api/molto/models",
      signal
    )
    return {
      ...data,
      models: data.models.map((item): Model => ({
        ...item,
        ...modelExtras.parse(item),
      })),
    }
  }
  stats(scope: "session" | "alltime", modelId = "", signal?: AbortSignal) {
    return this.request<Stats>(
      `/api/molto/stats?scope=${scope}&model_id=${encodeURIComponent(modelId)}`,
      signal
    )
  }
  settings(signal?: AbortSignal) {
    return this.request<Settings>("/api/molto/settings", signal)
  }
  updateSettings(patch: Schema["GlobalSettingsPatch"]) {
    return this.request<Schema["GlobalSettingsUpdateResponse"]>(
      "/api/molto/settings",
      undefined,
      "PATCH",
      patch
    )
  }
  modelSettings(id: string, signal?: AbortSignal) {
    return this.request<Schema["ModelSettingsResponse"]>(
      `${this.modelPath(id)}/settings`,
      signal
    )
  }
  updateModelSettings(id: string, patch: ModelSettings) {
    return this.request<SettingsResult>(
      `${this.modelPath(id)}/settings`,
      undefined,
      "PATCH",
      patch
    )
  }
  load(id: string) {
    return this.request<Schema["ModelOperationResponse"]>(
      `${this.modelPath(id)}/load`,
      undefined,
      "POST"
    )
  }
  unload(id: string) {
    return this.request<Schema["ModelOperationResponse"]>(
      `${this.modelPath(id)}/unload`,
      undefined,
      "POST"
    )
  }
  rescan() {
    return this.request<Schema["RefreshResponse"]>(
      "/api/molto/models/refresh",
      undefined,
      "POST"
    )
  }
  cache(signal?: AbortSignal) {
    return this.request<Cache>("/api/molto/cache", signal)
  }
  clearCache(kind: "hot" | "ssd") {
    return this.request<Schema["CacheClearResponse"]>(
      `/api/molto/cache/${kind}/clear`,
      undefined,
      "POST"
    )
  }
  profiles(id: string, signal?: AbortSignal) {
    return this.request<Schema["ProfilesResponse"]>(
      `${this.modelPath(id)}/profiles`,
      signal
    )
  }
  createProfile(id: string, profile: ProfileWrite) {
    return this.request<Schema["ProfileResponse"]>(
      `${this.modelPath(id)}/profiles`,
      undefined,
      "POST",
      profile
    )
  }
  updateProfile(id: string, name: string, profile: ProfileUpdate) {
    return this.request<Schema["ProfileResponse"]>(
      `${this.modelPath(id)}/profiles/${encodeURIComponent(name)}`,
      undefined,
      "PUT",
      profile
    )
  }
  deleteProfile(id: string, name: string) {
    return this.request(
      `${this.modelPath(id)}/profiles/${encodeURIComponent(name)}`,
      undefined,
      "DELETE"
    )
  }
  applyProfile(id: string, name: string) {
    return this.request<SettingsResult>(
      `${this.modelPath(id)}/profiles/${encodeURIComponent(name)}/apply`,
      undefined,
      "POST"
    )
  }
  private modelPath(id: string) {
    return `/api/molto/models/${encodeURIComponent(id)}`
  }
}
export function modelState(model: {
  is_loading: boolean
  is_unloading: boolean
  loaded: boolean
  load_failed?: boolean
}) {
  return model.is_unloading
    ? "Unloading"
    : model.is_loading
      ? "Loading"
      : model.load_failed
        ? "Failed"
        : model.loaded
          ? "Loaded"
          : "Unloaded"
}
export function settingsMessage(result: SettingsResult) {
  if (result.reload_error)
    return `Settings saved, but reloading failed: ${result.reload_error}`
  if (result.reload_deferred)
    return "Settings saved; waiting for active requests to finish before unloading. Load the model again once it unloads."
  if (result.auto_reloaded) return "Settings saved and model reloaded."
  if (result.auto_unloaded)
    return "Settings saved and model unloaded. Its next request will load the new configuration."
  return "Settings saved."
}
