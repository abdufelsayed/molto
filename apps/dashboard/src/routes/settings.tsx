import { createFileRoute } from "@tanstack/react-router"
import { ServerSettingsPage } from "@/features/server/page"

export const Route = createFileRoute("/settings")({
  component: ServerSettingsPage,
})
