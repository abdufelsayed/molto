import { tool } from "ai"
import { z } from "zod"
import type { AgentSettings } from "./types"

export const toolDefinitions = {
  read: tool({
    description:
      "Read a UTF-8 file in the virtual filesystem. Paths are relative to the working directory unless absolute.",
    inputSchema: z.object({
      path: z.string(),
      offset: z
        .number()
        .int()
        .min(1)
        .optional()
        .describe("First line, 1-based"),
      limit: z.number().int().positive().optional().describe("Maximum lines"),
    }),
  }),
  write: tool({
    description:
      "Create or replace a UTF-8 file in the virtual filesystem, creating parent directories.",
    inputSchema: z.object({ path: z.string(), content: z.string() }),
  }),
  edit: tool({
    description:
      "Replace an exact, uniquely occurring string in a UTF-8 file. Fails if the match is missing or ambiguous.",
    inputSchema: z.object({
      path: z.string(),
      oldText: z.string().min(1),
      newText: z.string(),
    }),
  }),
  bash: tool({
    description:
      "Execute bash in an isolated browser virtual filesystem. Standard text tools including rg, find, jq and sed are available. No host filesystem, Python, Node, or package installation. Each call starts at the session working directory with session environment variables.",
    inputSchema: z.object({ command: z.string() }),
  }),
  web_search: tool({
    description:
      "Search the web using the session's configured provider. Returns source URLs, titles and snippets or content. Optionally save the JSON result in the virtual filesystem.",
    inputSchema: z.object({
      query: z.string().min(1).max(300),
      savePath: z.string().optional(),
    }),
  }),
  fetch_url: tool({
    description:
      "Fetch a public HTTP(S) URL and extract readable content. Optionally save content in the virtual filesystem.",
    inputSchema: z.object({
      url: z.string().url().max(2048),
      savePath: z.string().optional(),
    }),
  }),
  ask_question: tool({
    description:
      "Ask the user a question and wait for their answer. Suggested options are optional; the user can always enter free text.",
    inputSchema: z.object({
      question: z.string().min(1),
      options: z.array(z.string()).max(10).optional(),
    }),
  }),
}
export function enabledTools(settings: AgentSettings) {
  return Object.fromEntries(
    settings.tools.map((name) => [name, toolDefinitions[name]])
  )
}
