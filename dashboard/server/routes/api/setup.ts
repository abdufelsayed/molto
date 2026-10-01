import { isIP } from "node:net"
import { defineHandler } from "nitro"
import type { H3Event } from "nitro"
import {
  backendUrl,
  json,
  mutationAllowed,
} from "../../../src/server/connection.server"

export function loopbackAddress(address: string | undefined) {
  if (!address) return false
  const value = address.toLowerCase()
  if (value === "::1" || value === "0:0:0:0:0:0:0:1") return true
  const mapped = value.startsWith("::ffff:") ? value.slice(7) : value
  return isIP(mapped) === 4 && mapped.split(".")[0] === "127"
}

export function setupTransportAllowed(
  request: Pick<H3Event["req"], "ip" | "headers">,
  bindHost = process.env.NITRO_HOST ?? process.env.HOST
) {
  // Setup is a direct-local operation. Never resolve a caller from forwarding headers.
  const forwarded = [...request.headers.keys()].some(
    (name) =>
      name === "forwarded" ||
      name === "x-real-ip" ||
      name.startsWith("x-forwarded-")
  )
  const localBind =
    bindHost === "localhost" ||
    loopbackAddress(bindHost?.replace(/^\[|\]$/g, ""))
  return localBind && loopbackAddress(request.ip) && !forwarded
}

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
    return "oMLX already has a main key. Connect with that key."
  if (status === 403)
    return "Initial setup requires a directly connected local browser and loopback server binding."
  if (status === 422)
    return "Use matching keys with at least four printable ASCII characters and no spaces."
  return `oMLX setup returned HTTP ${status}.`
}

export async function setupHandler(event: Pick<H3Event, "req">) {
  const request = event.req
  if (!["GET", "POST"].includes(request.method))
    return json({ detail: "Unsupported setup method." }, 405, {
      Allow: "GET, POST",
    })
  const urlHost = new URL(request.url).hostname.replace(/^\[|\]$/g, "")
  const localUrl = urlHost === "localhost" || loopbackAddress(urlHost)
  const allowed = setupTransportAllowed(request) && localUrl
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
        return json({ detail: "oMLX returned an invalid setup status." }, 502)
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
            "oMLX did not confirm initial setup. Check server status before retrying.",
        },
        502
      )
    return json({ configured: true })
  } catch {
    return json(
      {
        detail:
          request.method === "GET"
            ? "Could not check oMLX initial setup."
            : "Lost contact with oMLX during setup. Check setup status before retrying, or connect using your new key.",
      },
      502
    )
  }
}

export default defineHandler(setupHandler)
