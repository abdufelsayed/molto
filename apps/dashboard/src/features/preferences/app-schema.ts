import { z } from "zod"

const choice = (values: readonly string[], fallback: string) =>
  z
    .string()
    .refine((value) => values.includes(value))
    .default(fallback)
const filterNumber = () => z.string().max(32).default("")
export const libraryFiltersSchema = z.object({
  q: z.string().max(2000).catch("").default(""),
  state: z
    .enum(["all", "loaded", "unloaded", "loading", "unloading", "failed"])
    .catch("all")
    .default("all"),
  task: z.string().max(512).catch("all").default("all"),
  health: z.string().max(512).catch("all").default("all"),
  type: z.string().max(512).catch("").default(""),
})
export const modelTabSchema = z.enum([
  "summary",
  "settings",
  "profiles",
  "workspace",
])
export const overviewScopeSchema = z.enum(["session", "alltime"])
export const appPreferences = {
  "app.sidebar": z.boolean().default(true),
  "logs.file": z.string().max(2048).default(""),
  "logs.level": choice(
    ["", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    ""
  ),
  "logs.lines": choice(["100", "500", "2000", "10000"], "500"),
  "monitoring.period": choice(
    ["today", "yesterday", "7d", "30d", "90d", "month"],
    "7d"
  ),
  "monitoring.model": z.string().max(512).default(""),
  "activity.filter": choice(["all", "active", "failed"], "all"),
  "acquisition.step": choice(["discover", "prepare", "publish"], "discover"),
  "discover.provider": choice(["hf", "ms"], "hf"),
  "discover.search": z.string().max(2000).default(""),
  "discover.sort": choice(
    ["trending", "downloads", "created", "updated", "likes"],
    "trending"
  ),
  "discover.mlx": choice(["true", "false"], "true"),
  "discover.minimumSize": filterNumber(),
  "discover.maximumSize": filterNumber(),
  "discover.minimumParams": filterNumber(),
  "discover.maximumParams": filterNumber(),
  "discover.sizeSort": choice(["off", "ascending", "descending"], "off"),
  "discover.budget": z.string().max(32).default("16"),
  "server.tab": choice(
    [
      "connection",
      "keys",
      "network",
      "models",
      "cache",
      "defaults",
      "integrations",
      "logging",
    ],
    "connection"
  ),
  "server.search": z.string().max(2000).default(""),
  "models.filters": libraryFiltersSchema.default(() =>
    libraryFiltersSchema.parse({})
  ),
  "models.tabs": z.record(z.string().max(512), modelTabSchema).default({}),
  "overview.scope": overviewScopeSchema.default("session"),
} as const
