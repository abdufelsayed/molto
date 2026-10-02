import { z } from "zod"

const selection = z.string().max(4096).optional()
const layout = z
  .record(
    z
      .string()
      .refine((key) =>
        ["studio-sessions", "studio-main", "studio-settings"].includes(key)
      ),
    z.number().min(0).max(100)
  )
  .refine(
    (value) =>
      Object.hasOwn(value, "studio-main") &&
      Object.keys(value).length <= 3 &&
      Math.abs(
        Object.values(value).reduce((sum, size) => sum + size, 0) - 100
      ) < 0.1
  )

export const studioPreferences = {
  "studio.active": z.string().max(200).default(""),
  "studio.sessions": z.boolean().default(true),
  "studio.settings": z.boolean().default(true),
  "studio.settingsTab": z
    .enum(["prompt", "model", "tools", "sandbox"])
    .default("model"),
  "studio.inspectorTab": z
    .enum(["trace", "request", "response", "usage", "history"])
    .default("trace"),
  "studio.layout": layout.optional(),
  "studio.observability": z
    .object({
      expanded: z.boolean(),
      size: z.number().min(0).max(65),
    })
    .default({ expanded: false, size: 35 }),
  "studio.run": selection,
  "studio.event": selection,
  "studio.path": selection,
  "studio.attachments": z.array(z.string().max(4096)).max(100).default([]),
  "studio.disclosure": z.boolean().optional(),
} as const
