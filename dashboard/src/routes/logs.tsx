import { createFileRoute } from "@tanstack/react-router"
import { useQuery } from "@tanstack/react-query"
import { useState } from "react"
import { toast } from "sonner"
import { managementQuery } from "@/features/management/request"
import { PageTitle, QueryState } from "@/components/page-state"
import { Choice } from "@/features/monitoring/components"
import type { Logs } from "@/features/monitoring/types"
import { Button } from "@/components/ui/button"
import { FieldGroup } from "@/components/ui/field"
import { Card, CardContent } from "@/components/ui/card"
export const Route = createFileRoute("/logs")({ component: LogsPage })
function LogsPage() {
  const [paused, setPaused] = useState(false)
  const [file, setFile] = useState("")
  const [level, setLevel] = useState("")
  const [lines, setLines] = useState("500")
  const query = useQuery(
    managementQuery<Logs>(
      ["logs", file, level, lines],
      `monitoring/logs?lines=${lines}${file ? `&file=${encodeURIComponent(file)}` : ""}${level ? `&level=${level}` : ""}`,
      paused ? undefined : 5000
    )
  )
  const content = query.data
    ? typeof query.data.logs === "string"
      ? query.data.logs
      : query.data.logs.join("\n")
    : ""
  function download() {
    const url = URL.createObjectURL(new Blob([content], { type: "text/plain" }))
    const link = document.createElement("a")
    link.href = url
    link.download = query.data?.log_file.split("/").pop() ?? "server.log"
    link.click()
    URL.revokeObjectURL(url)
  }
  return (
    <>
      <PageTitle
        title="Logs"
        description="Bounded server log tails with server-side level filtering."
      >
        <Button variant="outline" onClick={() => setPaused(!paused)}>
          {paused ? "Resume live updates" : "Pause live updates"}
        </Button>
        <Button
          variant="outline"
          disabled={query.isFetching}
          onClick={() => void query.refetch()}
        >
          Refresh
        </Button>
      </PageTitle>
      <FieldGroup className="grid sm:grid-cols-3">
        <Choice
          label="Log file"
          value={file}
          onChange={setFile}
          options={[
            { value: "", label: "Current log" },
            ...(query.data?.available_files ?? []).map((name) => ({
              value: name,
              label: name,
            })),
          ]}
        />
        <Choice
          label="Level"
          value={level}
          onChange={setLevel}
          options={[
            { value: "", label: "All levels" },
            ...["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].map(
              (value) => ({ value, label: value })
            ),
          ]}
        />
        <Choice
          label="Tail size"
          value={lines}
          onChange={setLines}
          options={["100", "500", "2000", "10000"].map((value) => ({
            value,
            label: `${value} lines`,
          }))}
        />
      </FieldGroup>
      <QueryState query={query}>
        {query.data && (
          <Card>
            <CardContent className="pt-4">
              <div className="mb-4 flex flex-wrap items-center gap-3">
                <p className="grow text-sm break-all text-muted-foreground">
                  {query.data.log_file} · {query.data.matched_lines} matched /{" "}
                  {query.data.total_lines} file lines ·{" "}
                  {paused ? "Paused" : "Updates every 5 seconds"}
                </p>
                <Button
                  variant="outline"
                  disabled={!content}
                  onClick={() => {
                    void navigator.clipboard.writeText(content).then(
                      () => toast.success("Log tail copied."),
                      () => toast.error("Clipboard access failed.")
                    )
                  }}
                >
                  Copy tail
                </Button>
                <Button
                  variant="outline"
                  disabled={!content}
                  onClick={download}
                >
                  Download tail
                </Button>
              </div>
              {content ? (
                <pre
                  aria-label="Server log tail"

                  className="max-h-[65vh] overflow-auto rounded-md bg-muted p-4 text-xs whitespace-pre-wrap"
                >
                  {content}
                </pre>
              ) : (
                <p>No log lines match these filters.</p>
              )}
            </CardContent>
          </Card>
        )}
      </QueryState>
    </>
  )
}
