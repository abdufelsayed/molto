import { queryOptions } from "@tanstack/react-query"
import { ApiError, detail } from "./api"

export type RequestOptions = {
  method?: "GET" | "POST" | "PATCH" | "PUT" | "DELETE"
  body?: unknown
  signal?: AbortSignal
}

export async function managementRequest<T>(
  path: string,
  options: RequestOptions = {}
): Promise<T> {
  const method = options.method ?? "GET"
  const timeout = AbortSignal.timeout(method === "GET" ? 30_000 : 180_000)
  const response = await fetch(`/api/omlx/${path.replace(/^\//, "")}`, {
    method,
    credentials: "same-origin",
    cache: "no-store",
    headers:
      options.body === undefined ? {} : { "Content-Type": "application/json" },
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    signal: options.signal
      ? AbortSignal.any([options.signal, timeout])
      : timeout,
  })
  const data: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    throw new ApiError(detail(data, response.status), response.status)
  }
  if (data === null) throw new Error("The server returned an invalid response.")
  return data as T
}

export function managementQuery<T>(
  key: readonly unknown[],
  path: string,
  poll?: number
) {
  return queryOptions({
    queryKey: ["omlx", ...key],
    queryFn: ({ signal }) => managementRequest<T>(path, { signal }),
    retry: (attempt, error) =>
      !(error instanceof ApiError && error.status < 500) && attempt < 1,
    refetchIntervalInBackground: false,
    refetchInterval: (query) =>
      query.state.error instanceof ApiError && query.state.error.status === 401
        ? false
        : (poll ?? false),
  })
}
