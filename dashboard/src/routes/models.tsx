import { createFileRoute } from "@tanstack/react-router"
import { z } from "zod"
import { LibraryPanel } from "@/features/workspace/library"
export const Route = createFileRoute("/models")({
  validateSearch: z.object({
    q: z.string().catch("").default(""),
    state: z
      .enum(["all", "loaded", "unloaded", "loading", "unloading", "failed"])
      .catch("all")
      .default("all"),
    task: z.string().catch("all").default("all"),
    health: z.string().catch("all").default("all"),
    type: z.string().catch("").default(""),
  }),
  component: LibraryPanel,
})
