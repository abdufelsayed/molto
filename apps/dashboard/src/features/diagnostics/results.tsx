import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { ApplyRecommendation } from "./recommendation"
import { resultRows, type Run } from "./types"
export function Results({ run, results }: { run: Run; results: unknown }) {
  const rows = resultRows(results)
  const eligible = rows.filter(
    (row) =>
      row.number !== undefined &&
      row.number >= 0 &&
      /(?:tok.*s|throughput|accuracy|score|prefill|decode)/i.test(row.metric)
  )
  const repeatedMetric = eligible
    .find(
      (row) =>
        eligible.filter(
          (other) =>
            other.metric.split(" / ").at(-1) === row.metric.split(" / ").at(-1)
        ).length > 1
    )
    ?.metric.split(" / ")
    .at(-1)
  const numeric = eligible.filter(
    (row) => row.metric.split(" / ").at(-1) === repeatedMetric
  )
  const maximum = Math.max(1, ...numeric.map((row) => row.number ?? 0))
  function exportRun() {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify({ ...run, results }, null, 2)], {
        type: "application/json",
      })
    )
    const link = document.createElement("a")
    link.href = url
    link.download = `diagnostic-${run.kind}-${run.id}.json`
    link.click()
    URL.revokeObjectURL(url)
  }
  return (
    <div className="grid gap-4">
      <div className="flex justify-between gap-4">
        <p className="text-sm text-muted-foreground">
          Reported measurements, including partial results. Compare values only
          when units and workloads match.
        </p>
        <Button variant="outline" onClick={exportRun}>
          Export run
        </Button>
      </div>
      <ApplyRecommendation run={run} results={results} />
      {run.recommendation != null && (
        <div className="rounded border p-3">
          <p className="text-sm font-medium">Recommended settings</p>
          <pre className="mt-2 overflow-auto text-xs">
            {JSON.stringify(run.recommendation, null, 2)}
          </pre>
          <p className="mt-2 text-xs text-muted-foreground">
            Review these settings in the model configuration before applying
            them.
          </p>
        </div>
      )}
      {rows.length ? (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Measurement</TableHead>
              <TableHead>Reported value</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row) => (
              <TableRow key={row.metric}>
                <TableCell className="break-all">{row.metric}</TableCell>
                <TableCell className="break-all tabular-nums">
                  {row.value}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      ) : (
        <p className="text-sm text-muted-foreground">
          No measurements recorded yet.
        </p>
      )}
      {numeric.length > 1 && (
        <details>
          <summary className="cursor-pointer text-sm">
            {repeatedMetric} comparison
          </summary>
          <div className="grid gap-3 pt-4" aria-label="Measurement bars">
            {numeric.slice(0, 24).map((row) => (
              <div key={row.metric} className="grid gap-1 text-xs">
                <span>
                  {row.metric}: {row.value}
                </span>
                <div className="h-2 rounded bg-muted">
                  <div
                    className="h-2 rounded bg-primary"
                    style={{ width: `${((row.number ?? 0) / maximum) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </details>
      )}
      <details>
        <summary className="cursor-pointer text-sm">
          Settings and request snapshot
        </summary>
        <pre className="mt-3 overflow-auto rounded bg-muted p-4 text-xs">
          {JSON.stringify(run.request, null, 2)}
        </pre>
        <p className="mt-2 text-xs text-muted-foreground">
          This snapshot records the run. Exporting does not apply settings or
          publish results.
        </p>
      </details>
    </div>
  )
}
