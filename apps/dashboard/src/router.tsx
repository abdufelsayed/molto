import { setupRouterSsrQueryIntegration } from "@tanstack/react-router-ssr-query"
import { createRouter as createTanStackRouter } from "@tanstack/react-router"
import { QueryClient } from "@tanstack/react-query"
import { ManagementClient } from "@/features/management/api"
import { routeTree } from "./routeTree.gen"

export function getRouter() {
  const queryClient = new QueryClient()
  const api = new ManagementClient()
  const router = createTanStackRouter({
    routeTree,
    context: { queryClient, api },
    scrollRestoration: true,
    defaultPreload: "intent",
    defaultPreloadStaleTime: 0,
  })

  setupRouterSsrQueryIntegration({ router, queryClient })
  return router
}

declare module "@tanstack/react-router" {
  interface Register {
    router: ReturnType<typeof getRouter>
  }
}
