import { createFileRoute } from "@tanstack/react-router"
import {
  backendUrl,
  automaticConnectionAllowed,
  connect,
  disconnect,
  json,
  session,
} from "@/server/connection.server"

export const Route = createFileRoute("/api/connection")({
  server: {
    handlers: {
      GET: ({ request }) =>
        json({
          connected: !!session(request),
          server: backendUrl(),
          access: session(request)?.access,
          auto_connect: automaticConnectionAllowed(request),
        }),
      POST: ({ request }) => connect(request),
      DELETE: ({ request }) => disconnect(request),
    },
  },
})
