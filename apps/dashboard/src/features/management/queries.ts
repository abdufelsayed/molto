import { queryOptions } from "@tanstack/react-query"
import { useRouter } from "@tanstack/react-router"
import { ApiError } from "./api"
import type { ManagementClient } from "./api"

export const managementKey = ["molto"] as const
function retry(attempt: number, error: Error) {
  return !(error instanceof ApiError && error.status < 500) && attempt < 1
}
const options = { retry, refetchIntervalInBackground: false }
function interval(
  query: { state: { error: Error | null } },
  milliseconds: number
) {
  return query.state.error instanceof ApiError &&
    query.state.error.status === 401
    ? false
    : milliseconds
}
export const connectionQuery = (api: ManagementClient) =>
  queryOptions({
    queryKey: [...managementKey, "connection"],
    queryFn: ({ signal }) => api.connection(signal),
    ...options,
  })
export const stateQuery = (api: ManagementClient) =>
  queryOptions({
    queryKey: [...managementKey, "state"],
    queryFn: ({ signal }) => api.state(signal),
    staleTime: 1_000,
    ...options,
    refetchInterval: (q) => interval(q, 2_000),
  })
export const modelsQuery = (api: ManagementClient) =>
  queryOptions({
    queryKey: [...managementKey, "models"],
    queryFn: ({ signal }) => api.models(signal),
    staleTime: 2_000,
    ...options,
    refetchInterval: (q) => interval(q, 3_000),
  })
export const statsQuery = (
  api: ManagementClient,
  scope: "session" | "alltime",
  id = ""
) =>
  queryOptions({
    queryKey: [...managementKey, "stats", scope, id],
    queryFn: ({ signal }) => api.stats(scope, id, signal),
    staleTime: 2_000,
    ...options,
    refetchInterval: (q) => interval(q, 5_000),
  })
export const settingsQuery = (api: ManagementClient) =>
  queryOptions({
    queryKey: [...managementKey, "settings"],
    queryFn: ({ signal }) => api.settings(signal),
    staleTime: 30_000,
    ...options,
  })
export const modelSettingsQuery = (api: ManagementClient, id: string) =>
  queryOptions({
    queryKey: [...managementKey, "model-settings", id],
    queryFn: ({ signal }) => api.modelSettings(id, signal),
    staleTime: 10_000,
    ...options,
  })
export const profilesQuery = (api: ManagementClient, id: string) =>
  queryOptions({
    queryKey: [...managementKey, "profiles", id],
    queryFn: ({ signal }) => api.profiles(id, signal),
    ...options,
  })
export const cacheQuery = (api: ManagementClient) =>
  queryOptions({
    queryKey: [...managementKey, "cache"],
    queryFn: ({ signal }) => api.cache(signal),
    ...options,
    refetchInterval: (q) => interval(q, 5_000),
  })
export function useManagement() {
  return useRouter().options.context
}
