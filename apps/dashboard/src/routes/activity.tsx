import { createFileRoute } from "@tanstack/react-router"
import { ActivityPage } from "@/features/operations/page"
export const Route = createFileRoute("/activity")({ component: ActivityPage })
