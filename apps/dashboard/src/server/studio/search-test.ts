import { z } from "zod"
import {
  backendUrl,
  json,
  mutationAllowed,
  session,
} from "../connection.server"
import { engines, searchSchema } from "../../features/studio/agent/types"
import { executeWeb } from "./web"

const pendingSchema = z
  .object({
    provider: searchSchema.shape.provider.optional(),
    brave_api_key: z.string().max(4096).optional(),
    searxng_url: z.string().max(2048).optional(),
    ddgs_backends: z.string().max(512).optional(),
    max_results: z.number().int().min(1).max(10).optional(),
  })
  .strict()
const integrationsSchema = z.object({
  web_search_provider: searchSchema.shape.provider,
  web_search_brave_api_key: z.string().default(""),
  web_search_searxng_url: z.string().default(""),
  web_search_ddgs_backends: z.string().default(""),
  web_search_max_results: z.number().int().min(1).max(10).default(3),
})
export async function searchTest(request: Request): Promise<Response> {
  if (!mutationAllowed(request))
    return json({ detail: "Cross-origin requests are forbidden." }, 403)
  const credential = session(request)
  if (!credential) return json({ detail: "Connect to Molto first." }, 401)
  const pending = pendingSchema.safeParse(
    await request.json().catch(() => null)
  )
  if (!pending.success)
    return json({ detail: "Invalid web-search test settings." }, 422)
  try {
    const response = await fetch(
      `${backendUrl()}/management/v1/server/settings`,
      {
        headers: { Authorization: `Bearer ${credential.key}` },
        redirect: "error",
        signal: AbortSignal.any([request.signal, AbortSignal.timeout(15_000)]),
      }
    )
    if (!response.ok)
      return json(
        { detail: "Could not read saved search settings." },
        response.status
      )
    const payload = z
      .object({ sections: z.object({ integrations: integrationsSchema }) })
      .parse(await response.json())
    const saved = payload.sections.integrations
    const names = (pending.data.ddgs_backends ?? saved.web_search_ddgs_backends)
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean)
    if (names.some((name) => !(engines as readonly string[]).includes(name)))
      return json({ detail: "Unknown search engine." }, 422)
    const settings = searchSchema.parse({
      provider: pending.data.provider ?? saved.web_search_provider,
      engines: names,
      maxResults: pending.data.max_results ?? saved.web_search_max_results,
    })
    const result = await executeWeb(
      "web_search",
      { query: "molto web search connectivity check" },
      settings,
      {
        braveApiKey:
          pending.data.brave_api_key ?? saved.web_search_brave_api_key,
        searxngUrl: pending.data.searxng_url ?? saved.web_search_searxng_url,
      },
      request.signal
    )
    return json({
      ...result,
      ...(!result.ok
        ? {
            error: {
              code: "request_failed",
              message:
                typeof result.error === "string"
                  ? result.error
                  : "Search failed",
            },
          }
        : {}),
    })
  } catch {
    return json({ detail: "Could not test web search." }, 502)
  }
}
