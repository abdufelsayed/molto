import { createFileRoute } from "@tanstack/react-router"
import { AddModelPage } from "@/features/acquisition/page"
export const Route = createFileRoute("/add-model")({ component: AddModelPage })
