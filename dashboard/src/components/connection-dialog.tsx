import { useState } from "react"
import { KeyRoundIcon } from "lucide-react"
import { useRouter } from "@tanstack/react-router"
import { errorMessage } from "@/features/management/api"
import { managementKey, useManagement } from "@/features/management/queries"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import {
  Field,
  FieldDescription,
  FieldError,
  FieldGroup,
  FieldLabel,
} from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import { Spinner } from "@/components/ui/spinner"

export function ConnectionDialog({ compact = false }: { compact?: boolean }) {
  const [open, setOpen] = useState(false)
  const [key, setKey] = useState("")
  const [failure, setFailure] = useState<string>()
  const [pending, setPending] = useState(false)
  const { api, queryClient } = useManagement()
  const router = useRouter()

  async function connect(event: React.FormEvent) {
    event.preventDefault()
    setPending(true)
    setFailure(undefined)
    try {
      await api.connect(key)
      await queryClient.cancelQueries({ queryKey: managementKey })
      await queryClient.resetQueries({ queryKey: managementKey })
      await router.invalidate()
      setKey("")
      setOpen(false)
    } catch (error) {
      setFailure(
        errorMessage(
          error instanceof Error ? error : new Error("Connection failed.")
        )
      )
    } finally {
      setPending(false)
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(value) => {
        if (pending) return
        setOpen(value)
        if (!value) {
          setKey("")
          setFailure(undefined)
        }
      }}
    >
      <DialogTrigger
        render={<Button variant="outline" size={compact ? "sm" : "default"} />}
      >
        <KeyRoundIcon data-icon="inline-start" /> API access
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Connect to oMLX</DialogTitle>
          <DialogDescription>
            Enter the main API key configured on your oMLX server.
          </DialogDescription>
        </DialogHeader>
        <form
          onSubmit={(event) => void connect(event)}
          className="flex flex-col gap-4"
        >
          <FieldGroup>
            <Field data-invalid={!!failure}>
              <FieldLabel htmlFor="api-key">API key</FieldLabel>
              <Input
                id="api-key"
                type="password"
                autoComplete="off"
                value={key}
                aria-invalid={!!failure}
                onChange={(event) => setKey(event.target.value)}
              />
              <FieldDescription>
                You stay connected for up to eight hours. Inference subkeys
                cannot manage the server.
              </FieldDescription>
              {failure && <FieldError>{failure}</FieldError>}
            </Field>
          </FieldGroup>
          <DialogFooter>
            <Button type="submit" disabled={pending}>
              {pending && <Spinner data-icon="inline-start" />}
              {pending ? "Connecting…" : "Connect"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
