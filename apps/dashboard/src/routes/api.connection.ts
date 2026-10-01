import { createFileRoute } from "@tanstack/react-router"
import {
  backendUrl,
  connect,
  disconnect,
  json,
  session,
} from "@/server/connection.server"

export const Route = createFileRoute("/api/connection")({
  server: {
    handlers: {
      GET: ({ request }) =>
        json({ connected: !!session(request), server: backendUrl() }),
      POST: ({ request }) => connect(request),
      DELETE: ({ request }) => disconnect(request),
    },
  },
})
