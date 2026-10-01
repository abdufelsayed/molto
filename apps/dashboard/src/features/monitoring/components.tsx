import { Field, FieldLabel } from "@/components/ui/field"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableRow } from "@/components/ui/table"
import { count } from "@/lib/format"
import type { Summary } from "./types"

export function Choice({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: string
  options: { value: string; label: string }[]
  onChange: (value: string) => void
}) {
  return (
    <Field>
      <FieldLabel>{label}</FieldLabel>
      <Select
        value={value}
        onValueChange={(next) => {
          if (next !== null) onChange(next)
        }}
      >
        <SelectTrigger aria-label={label} className="w-full">
          <SelectValue>
            {options.find((option) => option.value === value)?.label ?? value}
          </SelectValue>
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            {options.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectGroup>
        </SelectContent>
      </Select>
    </Field>
  )
}
export function SummaryTable({ data }: { data: Summary }) {
  const rate = (value: number | null | undefined, unit: string) =>
    value == null ? "Not recorded" : `${value.toFixed(2)} ${unit}`
  return (
    <Table>
      <TableBody>
        {[
          ["Requests", count(data.requests)],
          ["Prompt tokens", count(data.prompt_tokens)],
          ["Generated tokens", count(data.completion_tokens)],
          ["Cached tokens", count(data.cached_tokens)],
          ["Generation rate", rate(data.generation_tps, "tokens/s")],
          ["Prefill rate", rate(data.prefill_tps, "tokens/s")],
          ["Average request duration", rate(data.average_request_seconds, "s")],
        ].map(([label, value]) => (
          <TableRow key={label}>
            <TableCell>{label}</TableCell>
            <TableCell className="text-right tabular-nums">{value}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}
export function UsageChart({
  daily,
}: {
  daily: (Summary & { date: string })[]
}) {
  const max = Math.max(1, ...daily.map((day) => day.total_tokens))
  return (
    <Card>
      <CardHeader>
        <CardTitle>Tokens by day</CardTitle>
      </CardHeader>
      <CardContent>
        {daily.length ? (
          <>
            <svg
              role="img"
              aria-label="Daily total tokens. Values are listed below."
              viewBox={`0 0 ${Math.max(daily.length * 20, 120)} 120`}
              className="h-48 w-full"
              preserveAspectRatio="none"
            >
              {daily.map((day, index) => (
                <rect
                  key={day.date}
                  x={index * 20 + 2}
                  y={110 - (day.total_tokens / max) * 100}
                  width="16"
                  height={Math.max((day.total_tokens / max) * 100, 1)}
                  fill="currentColor"
                >
                  <title>
                    {day.date}: {count(day.total_tokens)} tokens
                  </title>
                </rect>
              ))}
            </svg>
            <details>
              <summary className="cursor-pointer text-sm">Daily values</summary>
              <Table>
                <TableBody>
                  {daily.map((day) => (
                    <TableRow key={day.date}>
                      <TableCell>{day.date}</TableCell>
                      <TableCell className="text-right">
                        {count(day.total_tokens)} tokens
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </details>
          </>
        ) : (
          <p>No recorded usage in this range.</p>
        )}
      </CardContent>
    </Card>
  )
}
