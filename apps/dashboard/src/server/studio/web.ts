import { lookup } from "node:dns/promises"
import { Agent, fetch } from "undici"
import ipaddr from "ipaddr.js"
import { parseHTML } from "linkedom"
import TurndownService from "turndown"
import { z } from "zod"
import { engines, searchSchema } from "../../features/studio/agent/types.ts"

export type WebCredentials = { braveApiKey?: string; searxngUrl?: string }
type SearchSettings = z.infer<typeof searchSchema>
type Engine = (typeof engines)[number]
export type SearchResult = {
  title: string
  url: string
  snippet: string
  engine: string
  content?: string
  truncated?: boolean
  error?: string
}
const MAX_BYTES = 2 * 1024 * 1024
const textValue = (value: unknown): string =>
  typeof value === "string"
    ? value
    : typeof value === "number"
      ? `${value}`
      : ""
const clean = (text: string) => text.replace(/\s+/g, " ").trim()
const message = (error: unknown) =>
  error instanceof Error ? error.message : "Web request failed"
export function publicAddress(address: string): boolean {
  try {
    return ipaddr.process(address).range() === "unicast"
  } catch {
    return false
  }
}
export function validateUrl(value: string): URL {
  if (value.length > 2048) throw new Error("URL exceeds 2048 characters")
  const url = new URL(value)
  if (
    !["http:", "https:"].includes(url.protocol) ||
    url.username ||
    url.password
  )
    throw new Error("Only HTTP(S) URLs without credentials are allowed")
  return url
}
async function resolveAddresses(hostname: string, signal: AbortSignal) {
  signal.throwIfAborted()
  return new Promise<{ address: string; family: number }[]>(
    (resolve, reject) => {
      const aborted = () => reject(signal.reason)
      signal.addEventListener("abort", aborted, { once: true })
      void lookup(hostname, { all: true })
        .then(resolve, reject)
        .finally(() => signal.removeEventListener("abort", aborted))
    }
  )
}
/** DNS is validated and then pinned in the connector, including every redirect hop. */
export async function guardedText(
  value: string,
  signal: AbortSignal,
  options: { headers?: Record<string, string>; privateOrigin?: string } = {}
): Promise<{ text: string; url: string; contentType: string }> {
  let url = validateUrl(value)
  const firstOrigin = url.origin
  for (let hop = 0; hop <= 3; hop++) {
    signal.throwIfAborted()
    const hostname = url.hostname.replace(/^\[|\]$/g, "")
    const addresses = ipaddr.isValid(hostname)
      ? [
          {
            address: hostname,
            family: ipaddr.parse(hostname).kind() === "ipv6" ? 6 : 4,
          },
        ]
      : await resolveAddresses(hostname, signal)
    signal.throwIfAborted()
    if (
      !addresses.length ||
      (url.origin !== options.privateOrigin &&
        addresses.some(({ address }) => !publicAddress(address)))
    )
      throw new Error("URL resolves to a non-public network address")
    const pinned = addresses[0]!
    const dispatcher = new Agent({
      connect: {
        lookup: (_hostname, lookupOptions, callback) => {
          if (lookupOptions.all) callback(null, addresses)
          else callback(null, pinned.address, pinned.family)
        },
      },
    })
    try {
      const response = await fetch(url, {
        signal,
        dispatcher,
        redirect: "manual",
        headers: {
          "user-agent": "MoltoStudio/1.0",
          accept: "text/html,application/json,text/plain;q=0.9",
          ...(url.origin === firstOrigin ? options.headers : {}),
        },
      })
      if ([301, 302, 303, 307, 308].includes(response.status)) {
        const location = response.headers.get("location")
        await response.body?.cancel()
        if (!location || hop === 3)
          throw new Error(
            "Redirect limit exceeded or missing redirect destination"
          )
        url = validateUrl(new URL(location, url).href)
        continue
      }
      if (!response.ok) {
        await response.body?.cancel()
        throw new Error(`Web server returned HTTP ${response.status}`)
      }
      const contentType = (
        response.headers.get("content-type") ?? ""
      ).toLowerCase()
      if (contentType && !/text\/|json|xml|html/.test(contentType)) {
        await response.body?.cancel()
        throw new Error(`Unsupported content type: ${contentType}`)
      }
      if (Number(response.headers.get("content-length") ?? 0) > MAX_BYTES) {
        await response.body?.cancel()
        throw new Error("Response exceeds 2 MiB")
      }
      const reader = response.body?.getReader()
      if (!reader) return { text: "", url: url.href, contentType }
      const chunks: Uint8Array[] = []
      let length = 0
      try {
        while (true) {
          const chunk = await reader.read()
          if (chunk.done) break
          length += chunk.value.byteLength
          if (length > MAX_BYTES) throw new Error("Response exceeds 2 MiB")
          chunks.push(chunk.value)
        }
      } finally {
        await reader.cancel().catch(() => undefined)
      }
      return {
        text: Buffer.concat(chunks).toString("utf8"),
        url: url.href,
        contentType,
      }
    } finally {
      await dispatcher.close()
    }
  }
  throw new Error("Redirect limit exceeded")
}
export function markdown(html: string, base: string): string {
  const { document } = parseHTML(html)
  for (const node of document.querySelectorAll(
    "script,style,iframe,object,embed,noscript,nav,footer,form"
  ))
    node.remove()
  for (const node of document.querySelectorAll("[href],[src]"))
    for (const attribute of ["href", "src"]) {
      const value = node.getAttribute(attribute)
      if (value) {
        try {
          const url = new URL(value, base)
          if (["http:", "https:"].includes(url.protocol))
            node.setAttribute(attribute, url.href)
          else node.removeAttribute(attribute)
        } catch {
          node.removeAttribute(attribute)
        }
      }
    }
  const root = document.querySelector("main,article") ?? document.body
  return new TurndownService({ headingStyle: "atx", codeBlockStyle: "fenced" })
    .turndown(root?.innerHTML ?? html)
    .trim()
}
export function parseEngineHtml(
  engine: Engine,
  html: string,
  base: string
): SearchResult[] {
  const { document } = parseHTML(html)
  const rules: Partial<Record<Engine, [string, string, string]>> = {
    duckduckgo: [".result", ".result__a,h2 a", ".result__snippet"],
    brave: [
      '[data-type="web"]',
      "a:has(.title),a.heading-serpresult",
      ".snippet .content,.snippet",
    ],
    mojeek: ["ul.results > li", "h2 a", "p.s"],
    yahoo: [".relsrch", ".Title a,h3 a", ".Text"],
    yandex: ["li.serp-item", "h3 a", '[class*="text"]'],
  }
  const rule = rules[engine]
  if (!rule) return []
  return [...document.querySelectorAll(rule[0])].flatMap((node) => {
    const anchor = node.querySelector(rule[1])
    let href = anchor?.getAttribute("href")
    if (!href) return []
    try {
      const candidate = new URL(href, base)
      if (engine === "duckduckgo" && candidate.searchParams.has("uddg"))
        href = candidate.searchParams.get("uddg")!
      else if (engine === "yahoo" && href.includes("/RU="))
        href = decodeURIComponent(
          href.split("/RU=")[1]!.split("/RK=")[0]!.split("/RS=")[0]!
        )
      else href = candidate.href
      const url = validateUrl(href)
      if (
        url.hostname.endsWith("duckduckgo.com") ||
        url.pathname.includes("/aclick")
      )
        return []
      const title = clean(anchor?.textContent ?? "").slice(0, 160)
      if (!title) return []
      return [
        {
          title,
          url: url.href,
          snippet: clean(node.querySelector(rule[2])?.textContent ?? "").slice(
            0,
            500
          ),
          engine,
        },
      ]
    } catch {
      return []
    }
  })
}
function object(value: unknown): Record<string, unknown> {
  return value && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {}
}
function items(value: unknown): unknown[] {
  return Array.isArray(value) ? value : []
}
function jsonResults(data: unknown, engine: string): SearchResult[] {
  return items(data).flatMap((value) => {
    const item = object(value)
    try {
      return [
        {
          title: clean(textValue(item.title)).slice(0, 160),
          url: validateUrl(textValue(item.url)).href,
          snippet: clean(
            textValue(item.description ?? item.content ?? item.snippet)
          ).slice(0, 500),
          engine,
        },
      ]
    } catch {
      return []
    }
  })
}
async function searchEngine(
  engine: Engine,
  query: string,
  limit: number,
  signal: AbortSignal
): Promise<SearchResult[]> {
  const urls: Record<Engine, string> = {
    duckduckgo: "https://html.duckduckgo.com/html/?q=",
    brave: "https://search.brave.com/search?q=",
    mojeek: "https://www.mojeek.com/search?q=",
    yahoo: "https://search.yahoo.com/search?p=",
    yandex: "https://yandex.com/search/site/?web=1&text=",
    wikipedia: `https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=${limit}&srsearch=`,
    grokipedia: `https://grokipedia.com/api/typeahead?limit=${limit}&query=`,
  }
  const response = await guardedText(
    urls[engine] + encodeURIComponent(query),
    signal
  )
  if (engine === "wikipedia") {
    const data = object(JSON.parse(response.text))
    return items(object(data.query).search).map((value) => {
      const item = object(value)
      return {
        title: textValue(item.title),
        url: `https://en.wikipedia.org/?curid=${Number(item.pageid)}`,
        snippet: clean(
          parseHTML(`<div>${textValue(item.snippet)}</div>`).document
            .documentElement.textContent ?? ""
        ),
        engine,
      }
    })
  }
  if (engine === "grokipedia") {
    const data = object(JSON.parse(response.text))
    return items(data.results).map((value) => {
      const item = object(value)
      return {
        title: textValue(item.title),
        url: `https://grokipedia.com/page/${encodeURIComponent(textValue(item.slug))}`,
        snippet: clean(textValue(item.snippet)).slice(0, 500),
        engine,
      }
    })
  }
  const results = parseEngineHtml(engine, response.text, response.url)
  if (!results.length)
    throw new Error(
      "No parseable results; the engine may have blocked the request or returned no matches"
    )
  return results.slice(0, limit)
}
async function fetchContent(
  url: string,
  settings: SearchSettings,
  signal: AbortSignal
) {
  const response = await guardedText(url, signal)
  const content =
    response.contentType.includes("html") ||
    /^\s*(<!doctype html|<html)/i.test(response.text)
      ? markdown(response.text, response.url)
      : response.text
  const truncated = content.length > settings.maxChars
  if (truncated && !settings.truncate)
    throw new Error(
      `Page exceeds the ${settings.maxChars} character content limit`
    )
  return {
    url: response.url,
    content: content.slice(0, settings.maxChars),
    truncated,
  }
}
export async function executeWeb(
  name: "web_search" | "fetch_url",
  input: unknown,
  search: SearchSettings,
  credentials: WebCredentials = {},
  signal?: AbortSignal
): Promise<Record<string, unknown>> {
  const deadline = AbortSignal.timeout(20_000)
  const abort = signal ? AbortSignal.any([signal, deadline]) : deadline
  const settings = searchSchema.parse(search)
  try {
    if (name === "fetch_url") {
      const { url } = z
        .object({ url: z.string().min(1).max(2048) })
        .parse(input)
      return { ok: true, ...(await fetchContent(url, settings, abort)) }
    }
    const { query } = z
      .object({ query: z.string().trim().min(1).max(300) })
      .parse(input)
    let results: SearchResult[] = []
    const warnings: { engine: string; error: string }[] = []
    if (settings.provider === "brave") {
      if (!credentials.braveApiKey)
        throw new Error(
          "Brave Search API key is not configured in dashboard integrations"
        )
      const response = await guardedText(
        `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&count=${settings.maxResults}`,
        abort,
        { headers: { "X-Subscription-Token": credentials.braveApiKey } }
      )
      results = jsonResults(
        object(object(JSON.parse(response.text)).web).results,
        "brave"
      )
    } else if (settings.provider === "searxng") {
      if (!credentials.searxngUrl)
        throw new Error(
          "SearXNG URL is not configured in dashboard integrations"
        )
      const base = validateUrl(credentials.searxngUrl)
      const url = new URL(
        `${base.pathname.replace(/\/$/, "")}/search`,
        base.origin
      )
      url.searchParams.set("q", query)
      url.searchParams.set("format", "json")
      const response = await guardedText(url.href, abort, {
        privateOrigin: base.origin,
      })
      results = jsonResults(
        object(JSON.parse(response.text)).results,
        "searxng"
      )
    } else {
      const selected: readonly Engine[] =
        settings.provider === "duckduckgo"
          ? ["duckduckgo"]
          : settings.provider === "ddgs_custom"
            ? [...new Set(settings.engines)]
            : engines
      if (!selected.length) throw new Error("Select at least one search engine")
      const attempts = await Promise.allSettled(
        selected.map((engine) =>
          searchEngine(engine, query, settings.maxResults, abort)
        )
      )
      attempts.forEach((result, index) => {
        if (result.status === "fulfilled" && result.value.length)
          results.push(...result.value)
        else if (result.status === "fulfilled")
          warnings.push({
            engine: selected[index]!,
            error: "No matches returned",
          })
        else
          warnings.push({
            engine: selected[index]!,
            error: message(result.reason),
          })
      })
      // Interleave sources so the first selected engine cannot consume the entire result budget.
      const groups = selected.map((engine) =>
        results.filter((result) => result.engine === engine)
      )
      results = Array.from({ length: settings.maxResults }, (_, index) =>
        groups.flatMap((group) => (group[index] ? [group[index]!] : []))
      ).flat()
    }
    const seen = new Set<string>()
    results = results
      .filter((result) => !seen.has(result.url) && !!seen.add(result.url))
      .slice(0, settings.maxResults)
    if (settings.content === "full")
      results = await Promise.all(
        results.map(async (result) => {
          try {
            return {
              ...result,
              ...(await fetchContent(result.url, settings, abort)),
            }
          } catch (error) {
            return { ...result, error: message(error) }
          }
        })
      )
    abort.throwIfAborted()
    return {
      ok: results.length > 0,
      provider: settings.provider,
      results,
      ...(warnings.length ? { warnings } : {}),
      ...(!results.length ? { error: "No search results were available" } : {}),
    }
  } catch (error) {
    return { ok: false, error: message(error) }
  }
}
