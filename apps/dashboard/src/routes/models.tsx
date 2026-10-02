import { z } from "zod"
import { createFileRoute } from "@tanstack/react-router"
import { libraryFiltersSchema } from "@/features/preferences/app-schema"
import { LibraryPanel } from "@/features/workspace/library"
export const Route = createFileRoute("/models")({
  validateSearch: z.object({
    q: libraryFiltersSchema.shape.q.removeDefault().optional(),
    state: libraryFiltersSchema.shape.state.removeDefault().optional(),
    task: libraryFiltersSchema.shape.task.removeDefault().optional(),
    health: libraryFiltersSchema.shape.health.removeDefault().optional(),
    type: libraryFiltersSchema.shape.type.removeDefault().optional(),
  }),
  component: LibraryPanel,
})
