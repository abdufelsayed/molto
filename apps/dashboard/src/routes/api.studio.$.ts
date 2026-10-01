import { createFileRoute } from "@tanstack/react-router"
import { studioRequest } from "@/server/studio.server"

export const Route = createFileRoute("/api/studio/$")({
  server: {
    handlers: {
      GET: ({ request, params }) => studioRequest(request, params._splat ?? ""),
      POST: ({ request, params }) =>
        studioRequest(request, params._splat ?? ""),
    },
  },
})
