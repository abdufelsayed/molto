import { HTTPError, proxyRequest } from "nitro/h3"
import type { H3Event } from "nitro"

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
  const reserved =
    path.startsWith("/v1/") ||
    path === "/health" ||
    path === "/api/status" ||
    path.startsWith("/management/") ||
    path.startsWith("/admin/") ||
    path.startsWith("/api/cluster") ||
    path.startsWith("/cluster/")
  if (!reserved || path === realtimePath) return
  if (!inferencePathAllowed(path, event.req.method))
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
  return proxyRequest(
    event,
    `${inferenceBackendOrigin()}${path}${event.url.search}`,
    {
      filterHeaders: filtered,
      xfwd: true,
      fetchOptions: { redirect: "error" },
    }
  )
}
