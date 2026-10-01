import { HTTPError, proxyRequest } from "nitro/h3"
import type { H3Event } from "nitro"
import { operationTarget } from "./management-target.server"

export const realtimePath = "/v1/audio/transcriptions/realtime"
const routes: Record<string, RegExp> = {
  GET: /^(?:\/health|\/api\/status|\/v1\/(?:models(?:\/status)?|responses\/[^/]+|images\/capabilities|audio\/voices|mcp\/(?:tools|servers)))$/,
  POST: /^\/v1\/(?:models\/.+\/(?:load|unload)|embeddings|rerank|completions|chat\/completions|messages(?:\/count_tokens)?|responses|images\/(?:generations|edits|operations)|audio\/(?:transcriptions|speech|process)|mcp\/execute|web\/(?:search|fetch))$/,
  DELETE: /^\/v1\/responses\/[^/]+$/,
}

const clusterRoutes: Record<string, RegExp> = {
  GET: /^(?:\/api\/cluster\/(?:node_id|pair\/status\/[^/]+|models\/.+\/manifest)|\/cluster\/join\/(?:bootstrap\.py|source))$/,
  POST: /^(?:\/api\/cluster\/pair\/request(?:\/cancel)?|\/cluster\/join\/(?:claim|complete))$/,
}

export function inferencePathAllowed(path: string, method: string) {
  if (!path.startsWith("/") || path.startsWith("//")) return false
  let decoded: string
  try {
    decoded = decodeURIComponent(path)
  } catch {
    return false
  }
  if (
    /[\\%?#]/.test(decoded) ||
    decoded
      .split("")
      .some(
        (character) =>
          character.charCodeAt(0) <= 32 || character.charCodeAt(0) === 127
      ) ||
    decoded
      .slice(1)
      .split("/")
      .some((part) => !part || part === "." || part === "..")
  )
    return false
  return (
    (routes[method]?.test(decoded) ?? false) ||
    (clusterRoutes[method]?.test(decoded) ?? false)
  )
}

export function inferenceBackendOrigin() {
  const target = new URL(process.env.OMLX_API_URL ?? "http://127.0.0.1:8000")
  if (
    !["http:", "https:"].includes(target.protocol) ||
    target.username ||
    target.password ||
    target.pathname !== "/" ||
    target.search ||
    target.hash
  )
    throw new Error(
      "OMLX_API_URL must be an HTTP(S) origin without credentials or a path."
    )
  return target.origin
}

export function inferenceProxy(event: H3Event) {
  const path = event.url.pathname
  const rawPath =
    event.req.runtime?.node?.req.url?.split("?")[0] || event.url.pathname
  const managementPrefix = "/api/management/v1/"
  let managementTarget: string | undefined
  const managementRequest =
    path.startsWith("/api/management") || rawPath.startsWith("/api/management")
  if (managementRequest) {
    if (rawPath !== path)
      throw new HTTPError({ status: 404, message: "Invalid management path." })
    if (event.req.headers.has("origin"))
      throw new HTTPError({
        status: 403,
        message: "Use dashboard sign-in for browser management requests.",
      })
    if (!/^Bearer [^\s]+$/i.test(event.req.headers.get("authorization") ?? ""))
      throw new HTTPError({
        status: 401,
        message: "An explicit Bearer main API key is required.",
      })
    managementTarget = path.startsWith(managementPrefix)
      ? operationTarget(event.req.method, path.slice(managementPrefix.length))
      : undefined
    if (!managementTarget)
      throw new HTTPError({
        status: 404,
        message: "Unknown management operation.",
      })
  }
  const reserved =
    path.startsWith("/v1/") ||
    path === "/health" ||
    path === "/api/status" ||
    path.startsWith("/management/") ||
    path.startsWith("/admin/") ||
    path.startsWith("/api/cluster") ||
    path.startsWith("/cluster/")
  if ((!reserved && !managementRequest) || path === realtimePath) return
  if (!managementTarget && !inferencePathAllowed(path, event.req.method))
    throw new HTTPError({
      status: 404,
      message: "Unknown inference operation.",
    })
  const filtered = [...event.req.headers.keys()].filter(
    (name) =>
      name === "cookie" ||
      name === "forwarded" ||
      name.startsWith("x-forwarded-") ||
      name === "x-real-ip"
  )
  const response = proxyRequest(
    event,
    `${inferenceBackendOrigin()}${managementTarget ?? path}${event.url.search}`,
    {
      filterHeaders: filtered,
      xfwd: true,
      fetchOptions: { redirect: "error" },
    }
  )
  return response.then((result) => {
    if (!managementTarget) return result
    const headers = new Headers(result.headers)
    headers.set("Cache-Control", "no-store")
    return new Response(result.body, {
      status: result.status,
      statusText: result.statusText,
      headers,
    })
  })
}
