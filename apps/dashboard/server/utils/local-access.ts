import { isIP } from "node:net"
import type { H3Event } from "nitro"

export function loopbackAddress(address: string | undefined) {
  if (!address) return false
  const value = address.toLowerCase()
  if (value === "::1" || value === "0:0:0:0:0:0:0:1") return true
  const mapped = value.startsWith("::ffff:") ? value.slice(7) : value
  return isIP(mapped) === 4 && mapped.split(".")[0] === "127"
}

export function localTransportAllowed(
  request: Pick<H3Event["req"], "ip" | "headers">,
  bindHost = process.env.NITRO_HOST ?? process.env.HOST
) {
  // Use the socket peer. Forwarding headers cannot establish local authority.
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

export function localRequestAllowed(request: H3Event["req"]) {
  const hostname = new URL(request.url).hostname.replace(/^\[|\]$/g, "")
  return (
    localTransportAllowed(request) &&
    (hostname === "localhost" || loopbackAddress(hostname))
  )
}
