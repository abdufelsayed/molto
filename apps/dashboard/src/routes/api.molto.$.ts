import { createFileRoute } from "@tanstack/react-router"
import { forward } from "@/server/connection.server"
import { searchTest } from "@/server/studio/search-test"

export const Route = createFileRoute("/api/molto/$")({
  server: {
    handlers: {
      GET: ({ request, params }) => forward(request, params._splat ?? ""),
      POST: ({ request, params }) =>
        params._splat === "server/web-search/test"
          ? searchTest(request)
          : forward(request, params._splat ?? ""),
      PATCH: ({ request, params }) => forward(request, params._splat ?? ""),
      PUT: ({ request, params }) => forward(request, params._splat ?? ""),
      DELETE: ({ request, params }) => forward(request, params._splat ?? ""),
    },
  },
})
