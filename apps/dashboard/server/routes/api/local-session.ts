import { defineHandler } from "nitro"
import type { H3Event } from "nitro"
import { localRequestAllowed } from "../../utils/local-access"
import {
  automaticConnectionAllowed,
  backendUrl,
  createSession,
  json,
  mutationAllowed,
  session,
} from "../../../src/server/connection.server"

export async function localSessionHandler(event: Pick<H3Event, "req">) {
  const request = event.req
  const capability = process.env.MOLTO_LOCAL_ACCESS_TOKEN
  const available = !!capability && localRequestAllowed(request)
  if (request.method === "GET") return json({ available })
  if (request.method !== "POST")
    return json({ detail: "Unsupported connection method." }, 405, {
      Allow: "GET, POST",
    })
  if (!available || !mutationAllowed(request))
    return json(
      {
        detail:
          "Local access requires a direct localhost connection to the bundled dashboard.",
      },
      403
    )
  const payload: unknown = await request.json().catch(() => null)
  if (
    !payload ||
    typeof payload !== "object" ||
    !("automatic" in payload) ||
    typeof payload.automatic !== "boolean"
  )
    return json({ detail: "Invalid local connection request." }, 400)
  if (payload.automatic && !automaticConnectionAllowed(request))
    return json(
      {
        detail:
          "This browser is disconnected. Choose Connect locally to reconnect.",
      },
      409
    )
  try {
    const response = await fetch(`${backendUrl()}/_internal/dashboard-key`, {
      method: "POST",
      headers: { "X-Molto-Local-Access": capability! },
      signal: AbortSignal.timeout(15_000),
      redirect: "error",
    })
    if (!response.ok)
      return json(
        {
          detail:
            response.status === 409
              ? "Create your first main key to set up Molto."
              : "Could not open local Molto access. Check the server and try again.",
        },
        response.status
      )
    const data: unknown = await response.json()
    if (
      !data ||
      typeof data !== "object" ||
      !("key" in data) ||
      typeof data.key !== "string" ||
      !data.key
    )
      return json(
        { detail: "Molto returned an invalid local access response." },
        502
      )
    const active = session(request)
    if (active) {
      active.key = data.key
      active.access = "local"
      return json({ connected: true, server: backendUrl(), access: "local" })
    }
    return createSession(request, data.key, "local")
  } catch {
    return json(
      { detail: "Could not reach Molto. Check the server and try again." },
      502
    )
  }
}

export default defineHandler(localSessionHandler)
