export function bytes(value: number) {
  if (value === 0) return "0 B"
  const units = ["B", "KiB", "MiB", "GiB", "TiB"]
  const index = Math.max(
    0,
    Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1)
  )
  return `${new Intl.NumberFormat(undefined, {
    maximumFractionDigits: index > 0 ? 1 : 0,
  }).format(value / 1024 ** index)} ${units[index]!}`
}

export function count(value: number) {
  return new Intl.NumberFormat().format(value)
}

export function percentage(value: number, total: number) {
  return total > 0 ? Math.round((value / total) * 100) : 0
}
