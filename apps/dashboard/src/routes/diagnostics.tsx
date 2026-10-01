import { createFileRoute } from "@tanstack/react-router"
import { DiagnosticsPage } from "@/features/diagnostics/page"
export const Route = createFileRoute("/diagnostics")({
  component: DiagnosticsPage,
})
