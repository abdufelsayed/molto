import { Table, TableBody, TableRow, TableCell } from "@/components/ui/table"

function rows(
  value: unknown,
  prefix = "",
  depth = 0
): { label: string; value: string }[] {
  if (depth > 6 || value === null || value === undefined) return []
  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  )
    return [
      {
        label: prefix.replaceAll("_", " "),
        value:
          typeof value === "boolean" ? (value ? "Yes" : "No") : String(value),
      },
    ]
  if (Array.isArray(value))
    return value
      .slice(0, 64)
      .flatMap((item, index) => rows(item, `${prefix} ${index + 1}`, depth + 1))
  if (typeof value === "object")
    return Object.entries(value).flatMap(([key, item]) =>
      rows(item, prefix ? `${prefix} / ${key}` : key, depth + 1)
    )
  return []
}
export function Evidence({ value }: { value: unknown }) {
  const values = rows(value)
  return values.length ? (
    <Table>
      <TableBody>
        {values.map((row, index) => (
          <TableRow key={index}>
            <TableCell className="text-muted-foreground">{row.label}</TableCell>
            <TableCell className="break-all whitespace-normal">
              {row.value}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  ) : (
    <p className="text-sm text-muted-foreground">No evidence reported.</p>
  )
}
