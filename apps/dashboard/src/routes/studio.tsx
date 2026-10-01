import { createFileRoute } from "@tanstack/react-router"
import { Studio } from "@/features/studio/studio"
export const Route = createFileRoute("/studio")({
  component: Studio,
  head: () => ({ meta: [{ title: "Studio · Molto" }] }),
})
