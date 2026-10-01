import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import {
  CopyIcon,
  EyeIcon,
  EyeOffIcon,
  KeyRoundIcon,
  RefreshCwIcon,
} from "lucide-react"
import { useRouter } from "@tanstack/react-router"
import { ApiError, detail, errorMessage } from "@/features/management/api"
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

type SetupStatus = {
  setup_required: boolean
  allowed: boolean
  reason?: string | null
}
async function setupRequest<T>(
  method: "GET" | "POST",
  body?: unknown,
  signal?: AbortSignal
) {
  const response = await fetch("/api/setup", {
    method,
    credentials: "same-origin",
    cache: "no-store",
    ...(method === "POST"
      ? {
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }
      : {}),
    signal: signal
      ? AbortSignal.any([signal, AbortSignal.timeout(20_000)])
      : AbortSignal.timeout(20_000),
  })
  const data: unknown = await response.json().catch(() => null)
  if (!response.ok)
    throw new ApiError(detail(data, response.status), response.status)
  if (!data || typeof data !== "object")
    throw new Error("The dashboard returned an invalid setup response.")
  return data as T
}

export function ConnectionDialog({ compact = false }: { compact?: boolean }) {
  const [open, setOpen] = useState(false)
  const [key, setKey] = useState("")
  const [confirmation, setConfirmation] = useState("")
  const [revealed, setRevealed] = useState(false)
  const [copied, setCopied] = useState("")
  const [failure, setFailure] = useState<string>()
  const [pending, setPending] = useState(false)
  const [setupCompleted, setSetupCompleted] = useState(false)
  const { api, queryClient } = useManagement()
  const router = useRouter()
  const setup = useQuery({
    queryKey: ["molto", "setup"],
    queryFn: async ({ signal }) => {
      const result = await setupRequest<SetupStatus>("GET", undefined, signal)
      if (
        typeof result.setup_required !== "boolean" ||
        typeof result.allowed !== "boolean"
      )
        throw new Error("The dashboard returned an invalid setup status.")
      return result
    },
    enabled: open,
    retry: false,
    staleTime: 0,
  })
  const initialSetup =
    !setupCompleted && setup.data?.setup_required === true && setup.data.allowed
  const validNewKey = /^[!-~]{4,4096}$/.test(key)
  const matching = confirmation === key && validNewKey

  function clearSecrets() {
    setKey("")
    setConfirmation("")
    setRevealed(false)
    setCopied("")
    setFailure(undefined)
    setSetupCompleted(false)
  }
  async function connect(event: React.FormEvent) {
    event.preventDefault()
    if (pending || (initialSetup && !matching)) return
    setPending(true)
    setFailure(undefined)
    let keyCreated = false
    try {
      if (initialSetup) {
        await setupRequest("POST", { key, confirmation })
        keyCreated = true
      }
      await api.connect(key)
      await queryClient.cancelQueries({ queryKey: managementKey })
      await queryClient.resetQueries({ queryKey: managementKey })
      await router.invalidate()
      clearSecrets()
      setOpen(false)
    } catch (error) {
      if (keyCreated) {
        setSetupCompleted(true)
        setRevealed(false)
        setConfirmation("")
      }
      const message = errorMessage(
        error instanceof Error ? error : new Error("Connection failed.")
      )
      setFailure(
        keyCreated
          ? `Your main key was created, but the dashboard connection failed. Use this new key with Connect. ${message}`
          : message
      )
      if (initialSetup) void setup.refetch()
    } finally {
      setPending(false)
    }
  }
  function generateKey() {
    const bytes = new Uint8Array(32)
    crypto.getRandomValues(bytes)
    const generated = `molto_${Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("")}`
    setKey(generated)
    setConfirmation(generated)
    setRevealed(false)
    setCopied("")
    setFailure(undefined)
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(value) => {
        if (pending) return
        setOpen(value)
        if (!value) clearSecrets()
      }}
    >
      <DialogTrigger
        render={<Button variant="outline" size={compact ? "sm" : "default"} />}
      >
        <KeyRoundIcon data-icon="inline-start" /> API access
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>
            {initialSetup ? "Set up Molto access" : "Connect to Molto"}
          </DialogTitle>
          <DialogDescription>
            {initialSetup
              ? "This server has no main key yet. Create an administrator key, keep a secure copy, and connect this dashboard."
              : "Enter the main API key configured on your Molto server."}
          </DialogDescription>
        </DialogHeader>
        <form
          onSubmit={(event) => void connect(event)}
          className="flex flex-col gap-4"
        >
          <FieldGroup>
            {setup.isFetching && !setup.data && (
              <p
                role="status"
                className="flex items-center gap-2 text-xs text-muted-foreground"
              >
                <Spinner />
                Checking initial setup…
              </p>
            )}
            {setup.isError && (
              <div className="flex items-center justify-between gap-2">
                <p className="text-xs text-muted-foreground">
                  Initial setup status could not be checked. You can still
                  connect with an existing main key.
                </p>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  disabled={pending || setup.isFetching}
                  onClick={() => void setup.refetch()}
                >
                  Retry check
                </Button>
              </div>
            )}
            {setup.data &&
              !setup.data.allowed &&
              (setup.data.setup_required || setup.data.reason) && (
                <p className="text-sm text-muted-foreground">
                  {setup.data.reason ??
                    "Create the first key from a local browser on the server machine."}
                </p>
              )}
            <Field data-invalid={!!failure}>
              <FieldLabel htmlFor="api-key">
                {initialSetup ? "New main API key" : "API key"}
              </FieldLabel>
              <div className="flex gap-2">
                <Input
                  id="api-key"
                  type={revealed ? "text" : "password"}
                  autoComplete={initialSetup ? "new-password" : "off"}
                  value={key}
                  aria-invalid={!!failure}
                  disabled={pending}
                  onChange={(event) => {
                    setKey(event.target.value)
                    setCopied("")
                  }}
                />
                {initialSetup && (
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    aria-label={
                      revealed ? "Hide new main key" : "Reveal new main key"
                    }
                    disabled={!key || pending}
                    onClick={() => setRevealed(!revealed)}
                  >
                    {revealed ? <EyeOffIcon /> : <EyeIcon />}
                  </Button>
                )}
              </div>
              <FieldDescription>
                {initialSetup
                  ? "Use at least four printable ASCII characters with no spaces. A generated key is recommended."
                  : "You stay connected for up to eight hours. Inference subkeys cannot manage the server."}
              </FieldDescription>
              {failure && <FieldError>{failure}</FieldError>}
            </Field>
            {initialSetup && (
              <>
                <div className="flex flex-wrap gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    disabled={pending}
                    onClick={generateKey}
                  >
                    <RefreshCwIcon />
                    Generate secure key
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    disabled={!key || pending}
                    onClick={() => {
                      void navigator.clipboard.writeText(key).then(
                        () =>
                          setCopied(
                            "Key copied. Save it in your password manager."
                          ),
                        () =>
                          setCopied(
                            "Clipboard unavailable. Reveal the key to copy it manually."
                          )
                      )
                    }}
                  >
                    <CopyIcon />
                    Copy key
                  </Button>
                </div>
                {copied && (
                  <p role="status" className="text-xs text-muted-foreground">
                    {copied}
                  </p>
                )}
                <Field>
                  <FieldLabel htmlFor="setup-key-confirmation">
                    Confirm main API key
                  </FieldLabel>
                  <Input
                    id="setup-key-confirmation"
                    type="password"
                    autoComplete="new-password"
                    value={confirmation}
                    disabled={pending}
                    onChange={(event) => setConfirmation(event.target.value)}
                  />
                  <FieldDescription>
                    The main key grants full server administration and inference
                    access. Setup saves it on the Molto server. This browser
                    receives an HttpOnly session cookie.
                  </FieldDescription>
                  {confirmation && confirmation !== key && (
                    <FieldError>The keys do not match.</FieldError>
                  )}
                </Field>
              </>
            )}
          </FieldGroup>
          <DialogFooter>
            <Button
              type="submit"
              disabled={
                pending ||
                (initialSetup ? !matching : setup.isFetching && !setup.data)
              }
            >
              {pending && <Spinner data-icon="inline-start" />}
              {pending
                ? initialSetup
                  ? "Setting up…"
                  : "Connecting…"
                : initialSetup
                  ? "Create main key and connect"
                  : "Connect"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
