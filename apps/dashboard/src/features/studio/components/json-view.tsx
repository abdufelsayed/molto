import { useEffect, useMemo, useState } from "react"

const previewLimit = 200_000
const highlightLimit = 50_000
const structuredKeys = new Set([
  "body",
  "data",
  "arguments",
  "input",
  "output",
  "result",
  "payload",
])

// Decode structured payloads for display only. Ordinary model prose stays verbatim.
function structured(value: unknown, key = "", depth = 0): unknown {
  if (depth > 24) return value
  if (typeof value === "string" && (!key || structuredKeys.has(key))) {
    const trimmed = value.trim()
    if (
      trimmed.startsWith("{") ||
      trimmed.startsWith("[") ||
      trimmed.startsWith('"')
    ) {
      try {
        const parsed: unknown = JSON.parse(trimmed)
        if (typeof parsed === "object" && parsed !== null)
          return structured(parsed, "", depth + 1)
        // Handle a JSON body encoded more than once without changing normal strings.
        if (typeof parsed === "string" && parsed !== value) {
          const decoded = structured(parsed, "", depth + 1)
          if (typeof decoded === "object" && decoded !== null) return decoded
        }
      } catch {
        /* Partial streams and plain text remain readable. */
      }
    }
    return value
  }
  if (Array.isArray(value))
    return value.map((entry) => structured(entry, key, depth + 1))
  if (value && typeof value === "object")
    return Object.fromEntries(
      Object.entries(value).map(([name, entry]) => [
        name,
        structured(entry, name, depth + 1),
      ])
    )
  return value
}

export function formatJson(value: unknown): { text: string; json: boolean } {
  if (value === undefined) return { text: "Not recorded", json: false }
  try {
    const display = structured(value)
    if (typeof display === "string") return { text: display, json: false }
    return {
      text: JSON.stringify(display, null, 2) ?? "Not recorded",
      json: true,
    }
  } catch {
    return { text: "Unable to serialize this value.", json: false }
  }
}

export function JsonView({
  value,
  ariaLabel,
}: {
  value: unknown
  ariaLabel?: string
}) {
  const formatted = useMemo(() => formatJson(value), [value])
  const preview = formatted.text.slice(0, previewLimit)
  const [highlight, setHighlight] = useState<{ text: string; html: string }>()
  useEffect(() => {
    if (!formatted.json || preview.length > highlightLimit) return
    let live = true
    const timer = setTimeout(() => {
      void import("shiki/bundle/web")
        .then(({ codeToHtml }) =>
          codeToHtml(preview, {
            lang: "json",
            themes: { light: "github-light", dark: "github-dark" },
          })
        )
        .then((html) => {
          if (live) setHighlight({ text: preview, html })
        })
        .catch(() => {
          if (live) setHighlight(undefined)
        })
    }, 120)
    return () => {
      live = false
      clearTimeout(timer)
    }
  }, [formatted.json, preview])
  const html =
    formatted.json && highlight?.text === preview ? highlight.html : undefined
  return (
    <div className="min-w-0">
      {html ? (
        <div
          aria-label={ariaLabel}
          className="studio-code max-h-96 overflow-auto py-2 font-mono text-xs leading-relaxed [&_pre]:break-all [&_pre]:whitespace-pre-wrap"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      ) : (
        <pre
          aria-label={ariaLabel}
          className="max-h-96 overflow-auto py-2 font-mono text-xs leading-relaxed break-all whitespace-pre-wrap"
        >
          {preview}
        </pre>
      )}
      {formatted.text.length > previewLimit && (
        <p className="text-xs text-muted-foreground">
          Preview limited to {previewLimit.toLocaleString()} characters. Copy
          includes the full formatted value.
        </p>
      )}
    </div>
  )
}
