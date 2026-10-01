import { createWebSocketProxy } from "crossws"
import NodeWebSocket from "crossws/websocket"
import { defineWebSocketHandler } from "nitro"
import {
  inferenceBackendOrigin,
  realtimePath,
} from "../../../../../src/server/inference-proxy.server"

export const realtimeHooks = createWebSocketProxy({
  target: (peer) => {
    const source = new URL(peer.request.url)
    return `${inferenceBackendOrigin().replace(/^http/, "ws")}${realtimePath}${source.search}`
  },
  WebSocket: NodeWebSocket,
  headers: (peer) => {
    const headers = new Headers()
    for (const name of ["authorization", "x-api-key", "origin"])
      if (peer.request.headers.has(name))
        headers.set(name, peer.request.headers.get(name)!)
    return headers
  },
  maxBufferSize: 1024 * 1024,
  connectTimeout: 10_000,
})

export default defineWebSocketHandler(realtimeHooks)
