import { z } from "zod"
import { backendUrl, session, mutationAllowed, json } from "./connection.server"
import { searchSchema } from "@/features/studio/agent/types"
import { executeWeb } from "./studio/web"

export async function studioRequest(request: Request, path: string) {
  const credential = session(request)
  if (!credential)
    return json({ detail: "Connect to Molto to use the studio." }, 401)
  if (request.method !== "GET" && !mutationAllowed(request))
    return json({ detail: "Cross-origin requests are forbidden." }, 403)
  const headers = {
    Authorization: `Bearer ${credential.key}`,
    "Content-Type": "application/json",
  }
  try {
    if (request.method === "GET" && path === "models") {
      const response = await fetch(`${backendUrl()}/v1/models`, {
        headers,
        redirect: "error",
        signal: AbortSignal.any([request.signal, AbortSignal.timeout(15_000)]),
      })
      return json(await response.json(), response.status)
    }
    if (request.method !== "POST")
      return json({ detail: "Unknown studio operation." }, 404)
    const body = await request.text()
    if (body.length > 10_000_000)
      return json({ detail: "Studio requests are limited to 10 MB." }, 413)
    const payload: unknown = JSON.parse(body)
    if (path === "model/chat/completions") {
      const valid = z
        .object({
          model: z.string().min(1).max(512),
          messages: z.array(z.unknown()).max(20_000),
        })
        .passthrough()
        .parse(payload)
      const response = await fetch(`${backendUrl()}/v1/chat/completions`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          ...valid,
          include_mcp_tools: false,
          sampling_override: true,
        }),
        redirect: "error",
        signal: request.signal,
      })
      return new Response(response.body, {
        status: response.status,
        headers: {
          "Content-Type":
            response.headers.get("content-type") ?? "application/json",
          "Cache-Control": "no-store",
          "X-Accel-Buffering": "no",
        },
      })
    }
    if (path === "tools") {
      const valid = z
        .object({
          name: z.enum(["web_search", "fetch_url"]),
          input: z.unknown(),
          search: searchSchema,
        })
        .parse(payload)
      const response = await fetch(
        `${backendUrl()}/management/v1/server/settings`,
        {
          headers,
          redirect: "error",
          signal: AbortSignal.any([
            request.signal,
            AbortSignal.timeout(15_000),
          ]),
        }
      )
      if (!response.ok)
        return json(
          { detail: "Could not read server search configuration." },
          response.status
        )
      const config = z
        .object({
          sections: z
            .object({
              integrations: z
                .object({
                  web_search_brave_api_key: z.string().optional(),
                  web_search_searxng_url: z.string().optional(),
                })
                .passthrough(),
            })
            .passthrough(),
        })
        .parse(await response.json())
      const result = await executeWeb(
        valid.name,
        valid.input,
        valid.search,
        {
          braveApiKey: config.sections.integrations.web_search_brave_api_key,
          searxngUrl: config.sections.integrations.web_search_searxng_url,
        },
        request.signal
      )
      return json(result)
    }
    return json({ detail: "Unknown studio operation." }, 404)
  } catch (error) {
    if (request.signal.aborted) return new Response(null, { status: 499 })
    return json(
      {
        detail:
          error instanceof z.ZodError
            ? z.prettifyError(error)
            : error instanceof Error
              ? error.message
              : "Studio request failed.",
      },
      error instanceof z.ZodError || error instanceof SyntaxError ? 400 : 502
    )
  }
}
