import { randomBytes } from "node:crypto"
import { operationTarget } from "./management-target.server"

const cookieName = "molto_dashboard_session"
const lifetime = 8 * 60 * 60 * 1000
type DashboardSession = {
  key: string
  expires: number
  access: "local" | "key"
}
// Nitro routes and TanStack's SSR service compile this module separately.
// They must use the same session store in the single dashboard process.
const sessionsKey = Symbol.for("molto.dashboard.sessions")
const shared = globalThis as typeof globalThis & {
  [sessionsKey]?: Map<string, DashboardSession>
}
const sessions = (shared[sessionsKey] ??= new Map<string, DashboardSession>())

export function backendUrl() {
  const url = new URL(process.env.MOLTO_API_URL ?? "http://127.0.0.1:8000")
  if (
    !["http:", "https:"].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash ||
    url.pathname !== "/"
  )
    throw new Error(
      "MOLTO_API_URL must be an HTTP(S) origin without credentials or a path."
    )
  return url.origin
}
export function json(data: unknown, status = 200, headers: HeadersInit = {}) {
  const responseHeaders = new Headers(headers)
  responseHeaders.set("Cache-Control", "no-store")
  return Response.json(data, {
    status,
    headers: responseHeaders,
  })
}
export function mutationAllowed(request: Request) {
  return (
    request.headers.get("sec-fetch-site") !== "cross-site" &&
    request.headers.get("origin") === new URL(request.url).origin
  )
}
function token(request: Request) {
  return request.headers
    .get("cookie")
    ?.split(/;\s*/)
    .find((part) => part.startsWith(`${cookieName}=`))
    ?.slice(cookieName.length + 1)
}
export function session(request: Request) {
  const id = token(request)
  const value = id ? sessions.get(id) : undefined
  if (value && value.expires > Date.now()) return value
  if (id) sessions.delete(id)
  return undefined
}
export function automaticConnectionAllowed(request: Request) {
  return token(request) !== "disconnected"
}
function cookie(request: Request, id: string, maxAge: number) {
  const secure = new URL(request.url).protocol === "https:" ? "; Secure" : ""
  return `${cookieName}=${id}; Path=/; HttpOnly; SameSite=Strict; Max-Age=${maxAge}${secure}`
}
export function createSession(
  request: Request,
  key: string,
  access: "local" | "key" = "key"
) {
  for (const [id, value] of sessions)
    if (value.expires <= Date.now()) sessions.delete(id)
  const previous = token(request)
  if (previous) sessions.delete(previous)
  if (sessions.size >= 128)
    return json(
      { detail: "Too many active dashboard sessions. Try again later." },
      503
    )
  const id = randomBytes(32).toString("hex")
  sessions.set(id, { key, access, expires: Date.now() + lifetime })
  return json({ connected: true, server: backendUrl(), access }, 200, {
    "Set-Cookie": cookie(request, id, lifetime / 1000),
  })
}
export async function connect(request: Request) {
  if (!mutationAllowed(request))
    return json({ detail: "Cross-origin requests are forbidden." }, 403)
  const payload: unknown = await request.json().catch(() => null)
  if (
    !payload ||
    typeof payload !== "object" ||
    !("key" in payload) ||
    typeof payload.key !== "string" ||
    payload.key.length > 4096
  )
    return json({ detail: "Enter your Molto main API key." }, 400)
  try {
    const response = await fetch(`${backendUrl()}/management/v1/state`, {
      headers: { Authorization: `Bearer ${payload.key}` },
      signal: AbortSignal.timeout(15_000),
      redirect: "error",
    })
    if (!response.ok)
      return json(
        {
          detail:
            response.status === 401
              ? "Molto rejected this key. Use the main key, not an inference subkey."
              : `Molto returned HTTP ${response.status}.`,
        },
        response.status
      )
    return createSession(request, payload.key)
  } catch {
    return json(
      { detail: "Could not reach Molto. Check the server and MOLTO_API_URL." },
      502
    )
  }
}
export function disconnect(request: Request) {
  if (!mutationAllowed(request))
    return json({ detail: "Cross-origin requests are forbidden." }, 403)
  const id = token(request)
  if (id) sessions.delete(id)
  return json({ connected: false }, 200, {
    "Set-Cookie": cookie(request, "disconnected", lifetime / 1000),
  })
}
export async function forward(request: Request, path: string) {
  if (request.method !== "GET" && !mutationAllowed(request))
    return json({ detail: "Cross-origin requests are forbidden." }, 403)
  const credential = session(request)
  if (!credential)
    return json({ detail: "Connect to Molto with the main API key." }, 401)
  const target = operationTarget(request.method, path)
  let parts: string[]
  try {
    parts = [
      ...decodeURIComponent(path).split("/"),
      ...decodeURIComponent(new URL(request.url).pathname).split("/"),
    ]
  } catch {
    return json({ detail: "Invalid management path." }, 400)
  }
  if (
    !target ||
    parts.some((part) => [".", ".."].includes(part)) ||
    parts.some((part) => part.includes("\\"))
  )
    return json({ detail: "Unknown management operation." }, 404)
  try {
    const source = new URL(request.url)
    const body = request.method === "GET" ? undefined : await request.text()
    const response = await fetch(`${backendUrl()}${target}${source.search}`, {
      method: request.method,
      headers: {
        Authorization: `Bearer ${credential.key}`,
        ...(request.method !== "GET"
          ? { "Content-Type": "application/json" }
          : {}),
      },
      body,
      signal: AbortSignal.timeout(request.method === "GET" ? 15_000 : 600_000),
      redirect: "error",
    })
    const data: unknown = await response
      .json()
      .catch(() => ({ detail: "Molto returned an invalid response." }))
    if (
      response.ok &&
      request.method === "PATCH" &&
      path === "auth/main-key" &&
      body
    ) {
      const payload: unknown = JSON.parse(body)
      if (
        payload &&
        typeof payload === "object" &&
        "key" in payload &&
        typeof payload.key === "string"
      ) {
        const previous = credential.key
        credential.key = payload.key
        for (const [id, value] of sessions)
          if (value !== credential && value.key === previous)
            sessions.delete(id)
      }
    }
    return json(data, response.status)
  } catch {
    return json(
      {
        detail:
          request.method === "GET"
            ? "Could not reach Molto."
            : "Lost contact with Molto. The operation may still be running; check current model state before retrying.",
      },
      502
    )
  }
}
