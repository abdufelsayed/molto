import { useId } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { managementRequest } from "@/features/management/request"
import { errorMessage } from "@/features/management/api"
import { Field, FieldLabel, FieldDescription } from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectGroup,
  SelectItem,
} from "@/components/ui/select"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
export function TextField({
  label,
  value,
  onChange,
  description,
  type = "text",
  min,
  max,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  description?: string
  type?: string
  min?: number
  max?: number
}) {
  const id = useId()
  return (
    <Field>
      <FieldLabel htmlFor={id}>{label}</FieldLabel>
      <Input
        id={id}
        type={type}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        min={min}
        max={max}
      />
      {description && <FieldDescription>{description}</FieldDescription>}
    </Field>
  )
}
export function Choice({
  label,
  value,
  onChange,
  options,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  options: { value: string; label: string }[]
}) {
  const id = useId()
  return (
    <Field>
      <FieldLabel htmlFor={id}>{label}</FieldLabel>
      <Select
        value={value}
        onValueChange={(next) => {
          if (next !== null) onChange(next)
        }}
        items={options}
      >
        <SelectTrigger id={id} className="w-full">
          <SelectValue />
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
export function Failure({ error }: { error: Error | null }) {
  return error ? (
    <Alert variant="destructive">
      <AlertTitle>Request failed</AlertTitle>
      <AlertDescription>{errorMessage(error)}</AlertDescription>
    </Alert>
  ) : null
}
export function useAction<T = unknown>() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: ({
      path,
      body,
      method = "POST",
    }: {
      path: string
      body?: unknown
      method?: "POST" | "DELETE" | "PATCH"
    }) => managementRequest<T>(path, { method, body }),
    retry: false,
    onSuccess: () => client.invalidateQueries({ queryKey: ["omlx"] }),
  })
}
