import { defineHandler } from "nitro"
import type { H3Event } from "nitro"
import {
  localRequestAllowed,
  localTransportAllowed as setupTransportAllowed,
} from "../../utils/local-access"

export { setupTransportAllowed }
import {
  backendUrl,
  json,
  mutationAllowed,
} from "../../../src/server/connection.server"

function setupError(data: unknown, status: number) {
  if (
    data &&
    typeof data === "object" &&
    "detail" in data &&
    typeof data.detail === "string"
  )
    return data.detail
  if (
    data &&
    typeof data === "object" &&
    "detail" in data &&
    data.detail &&
    typeof data.detail === "object" &&
    "message" in data.detail &&
    typeof data.detail.message === "string"
  )
    return data.detail.message
  if (status === 409)
    return "Molto already has a main key. Connect with that key."
  if (status === 403)
    return "Initial setup requires a directly connected local browser and loopback server binding."
  if (status === 422)
    return "Use matching keys with at least four printable ASCII characters and no spaces."
  return `Molto setup returned HTTP ${status}.`
}

export async function setupHandler(event: Pick<H3Event, "req">) {
  const request = event.req
  if (!["GET", "POST"].includes(request.method))
    return json({ detail: "Unsupported setup method." }, 405, {
      Allow: "GET, POST",
    })
  const allowed = localRequestAllowed(request)
  if (!allowed)
    return request.method === "GET"
      ? json({
          setup_required: false,
          allowed: false,
          reason:
            "Initial setup is available only through a directly connected local browser and a loopback dashboard binding.",
        })
      : json(
          {
            detail:
              "Initial setup requires a directly connected local browser and loopback dashboard binding.",
          },
          403
        )
  if (request.method === "POST" && !mutationAllowed(request))
    return json({ detail: "Cross-origin requests are forbidden." }, 403)
  let key: string | undefined
  if (request.method === "POST") {
    const payload: unknown = await request.json().catch(() => null)
    if (
      !payload ||
      typeof payload !== "object" ||
      !("key" in payload) ||
      !("confirmation" in payload) ||
      typeof payload.key !== "string" ||
      typeof payload.confirmation !== "string" ||
      !/^[!-~]{4,4096}$/.test(payload.key) ||
      payload.key !== payload.confirmation
    )
      return json(
        {
          detail:
            "Use matching keys with at least four printable ASCII characters and no spaces.",
        },
        422
      )
    key = payload.key
  }
  try {
    const response = await fetch(`${backendUrl()}/management/v1/setup`, {
      method: request.method,
      ...(key
        ? {
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ key, confirmation: key }),
          }
        : {}),
      signal: AbortSignal.timeout(15_000),
      redirect: "error",
    })
    const data: unknown = await response.json().catch(() => null)
    if (!response.ok)
      return json(
        { detail: setupError(data, response.status) },
        response.status
      )
    if (request.method === "GET") {
      if (
        !data ||
        typeof data !== "object" ||
        !("setup_required" in data) ||
        !("allowed" in data) ||
        typeof data.setup_required !== "boolean" ||
        typeof data.allowed !== "boolean"
      )
        return json({ detail: "Molto returned an invalid setup status." }, 502)
      return json({
        setup_required: data.setup_required,
        allowed: data.allowed,
        reason:
          "reason" in data && typeof data.reason === "string"
            ? data.reason
            : null,
      })
    }
    if (
      !data ||
      typeof data !== "object" ||
      !("configured" in data) ||
      data.configured !== true
    )
      return json(
        {
          detail:
            "Molto did not confirm initial setup. Check server status before retrying.",
        },
        502
      )
    return json({ configured: true })
  } catch {
    return json(
      {
        detail:
          request.method === "GET"
            ? "Could not check Molto initial setup."
            : "Lost contact with Molto during setup. Check setup status before retrying, or connect using your new key.",
      },
      502
    )
  }
}

export default defineHandler(setupHandler)
