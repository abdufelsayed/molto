import { openDB } from "idb"
import { z } from "zod"
import { closePendingTools, fileSchema, settingsSchema } from "./agent/types"
import type { StudioSession } from "./agent/types"

const database = () =>
  openDB("molto-studio", 1, {
    upgrade(db) {
      db.createObjectStore("sessions", { keyPath: "id" })
    },
  })
// The dashboard owns storage. The agent is given messages, settings and snapshots.
export async function loadSessions(): Promise<StudioSession[]> {
  const db = await database()
  const sessions = (await db.getAll("sessions")) as StudioSession[]
  db.close()
  return sessions.sort((a, b) => b.updated - a.updated).map(recoverSession)
}
function recoverSession(session: StudioSession): StudioSession {
  return {
    ...session,
    settings: settingsSchema.parse(session.settings),
    runs: session.runs.map((run) => {
      if (run.status !== "running") return run
      const reason =
        "This run was interrupted before it finished. Its last saved snapshot is available."
      return {
        ...run,
        status: "interrupted" as const,
        messages: closePendingTools(run.messages, reason).messages,
        events: [
          ...run.events.map((event) =>
            event.state === "running" || event.state === "waiting"
              ? {
                  ...event,
                  state: "error" as const,
                  output: event.output ?? { ok: false, error: reason },
                }
              : event
          ),
          {
            id: crypto.randomUUID(),
            at: Date.now(),
            step: run.steps.length,
            type: "error" as const,
            text: reason,
          },
        ],
        error: reason,
      }
    }),
  }
}
let pendingWrites: Promise<void> = Promise.resolve()
function enqueue(operation: () => Promise<void>) {
  const next = pendingWrites.catch(() => undefined).then(operation)
  pendingWrites = next
  return next
}
export function saveSession(session: StudioSession) {
  const snapshot = structuredClone(session)
  return enqueue(async () => {
    const db = await database()
    try {
      await db.put("sessions", snapshot)
    } finally {
      db.close()
    }
  })
}
export function removeSession(id: string) {
  // Wait for queued checkpoints so a late save cannot resurrect a deleted session.
  return enqueue(async () => {
    const db = await database()
    try {
      await db.delete("sessions", id)
    } finally {
      db.close()
    }
  })
}
export function download(
  name: string,
  content: string | Uint8Array,
  type = "application/json"
) {
  const blob = new Blob(
    [typeof content === "string" ? content : new Uint8Array(content)],
    { type }
  )
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement("a")
  anchor.href = url
  anchor.download = name
  anchor.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
const message = z
  .object({
    role: z.enum(["user", "assistant", "tool", "system"]),
    content: z.union([z.string(), z.array(z.record(z.string(), z.unknown()))]),
  })
  .passthrough()
const event = z
  .object({
    id: z.string(),
    at: z.number(),
    step: z.number(),
    type: z.enum(["text", "reasoning", "tool", "error", "status"]),
    text: z.string().optional(),
    callId: z.string().optional(),
    tool: z.string().optional(),
    state: z.enum(["running", "waiting", "done", "error", "denied"]).optional(),
  })
  .passthrough()
const importedSchema = z.object({
  version: z.literal(1),
  id: z.string(),
  title: z.string().max(1000),
  updated: z.number(),
  settings: settingsSchema,
  turns: z
    .array(
      z.object({
        id: z.string(),
        text: z.string(),
        runIds: z.array(z.string()),
        selected: z.string().optional(),
        attachments: z.array(z.string()).optional(),
      })
    )
    .max(10_000),
  files: z.array(fileSchema).max(100_000),
  draft: z.string().default(""),
  branches: z
    .array(
      z.object({
        id: z.string(),
        title: z.string(),
        turns: z.array(
          z.object({
            id: z.string(),
            text: z.string(),
            runIds: z.array(z.string()),
            selected: z.string().optional(),
            attachments: z.array(z.string()).optional(),
          })
        ),
        files: z.array(fileSchema),
      })
    )
    .optional(),
  runs: z
    .array(
      z.object({
        id: z.string(),
        created: z.number(),
        settings: settingsSchema,
        input: z.array(message),
        messages: z.array(message),
        events: z.array(event),
        steps: z.array(z.object({ request: z.unknown() }).passthrough()),
        sources: z.array(
          z.object({
            id: z.string(),
            title: z.string(),
            url: z
              .string()
              .url()
              .refine((value) => /^https?:\/\//.test(value)),
            snippet: z.string().optional(),
          })
        ),
        before: z.array(fileSchema),
        after: z.array(fileSchema),
        status: z.enum([
          "running",
          "complete",
          "stopped",
          "error",
          "limit",
          "interrupted",
        ]),
        mode: z.enum(["live", "replay"]),
        replayOf: z.string().optional(),
        environment: z.unknown().optional(),
        error: z.string().optional(),
        note: z.string(),
        prompt: z
          .object({
            text: z.string(),
            attachments: z.array(z.string()).optional(),
          })
          .optional(),
      })
    )
    .max(10_000),
})
export function importSession(text: string): StudioSession {
  if (text.length > 100_000_000)
    throw new Error("Experiment imports are limited to 100 MB.")
  const parsed = importedSchema.parse(JSON.parse(text))
  return recoverSession({
    ...parsed,
    id: crypto.randomUUID(),
    title: `${parsed.title} (imported)`,
    updated: Date.now(),
  } as StudioSession)
}
