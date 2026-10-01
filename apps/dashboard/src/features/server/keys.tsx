import { useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  CopyIcon,
  EyeIcon,
  EyeOffIcon,
  KeyRoundIcon,
  PlusIcon,
  ShieldCheckIcon,
} from "lucide-react"
import {
  managementQuery,
  managementRequest,
} from "@/features/management/request"
import { errorMessage } from "@/features/management/api"
import { QueryState } from "@/components/page-state"
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import {
  Field,
  FieldLabel,
  FieldDescription,
  FieldGroup,
} from "@/components/ui/field"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog"
import { Switch } from "@/components/ui/switch"
import type { AuthKeys, SubKey } from "./types"
import { labelFor } from "./types"

type KeyAction =
  | { kind: "create" }
  | { kind: "edit"; entry: SubKey }
  | { kind: "revoke"; entry: SubKey }
  | { kind: "rotate" }
function Secret({ value, label }: { value: string; label: string }) {
  const [visible, setVisible] = useState(false)
  const [feedback, setFeedback] = useState("")
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2">
        <code className="min-w-0 flex-1 rounded-md bg-muted px-3 py-2 text-xs break-all">
          {value
            ? visible
              ? value
              : "••••••••••••••••••••••••"
            : "No key configured"}
        </code>
        <Button
          variant="ghost"
          size="icon"
          aria-label={`${visible ? "Hide" : "Reveal"} ${label}`}
          disabled={!value}
          onClick={() => setVisible(!visible)}
        >
          {visible ? <EyeOffIcon /> : <EyeIcon />}
        </Button>
        <Button
          variant="ghost"
          size="icon"
          aria-label={`Copy ${label}`}
          disabled={!value}
          onClick={() => {
            void navigator.clipboard.writeText(value).then(
              () => setFeedback("Copied"),
              () =>
                setFeedback(
                  "Clipboard unavailable. Reveal and select the key to copy."
                )
            )
          }}
        >
          <CopyIcon />
        </Button>
      </div>
      {feedback && (
        <span role="status" className="text-xs text-muted-foreground">
          {feedback}
        </span>
      )}
    </div>
  )
}

export function KeyManagement() {
  const query = useQuery(
    managementQuery<AuthKeys>(["auth", "keys"], "auth/keys")
  )
  const queryClient = useQueryClient()
  const [action, setAction] = useState<KeyAction | null>(null)
  const [name, setName] = useState("")
  const [key, setKey] = useState("")
  const [confirmation, setConfirmation] = useState("")
  const [notice, setNotice] = useState("")
  const mutation = useMutation({
    mutationFn: async () => {
      if (!action) throw new Error("Select a key action first.")
      if (action.kind === "create")
        return managementRequest("auth/subkeys", {
          method: "POST",
          body: {
            name: name.trim(),
            ...(key.trim() ? { key: key.trim() } : {}),
          },
        })
      if (action.kind === "edit")
        return managementRequest(
          `auth/subkeys/${encodeURIComponent(action.entry.id)}`,
          {
            method: "PATCH",
            body: {
              name: name.trim(),
              ...(key.trim() ? { key: key.trim() } : {}),
            },
          }
        )
      if (action.kind === "revoke")
        return managementRequest(
          `auth/subkeys/${encodeURIComponent(action.entry.id)}`,
          { method: "DELETE" }
        )
      return managementRequest("auth/main-key", {
        method: "PATCH",
        body: { key: key.trim() },
      })
    },
    onSuccess: async () => {
      setNotice(
        action?.kind === "rotate"
          ? "Main key rotated. Use the new key in your API clients. This dashboard session has been updated."
          : action?.kind === "revoke"
            ? "Key revoked. Requests using it will be rejected."
            : "Key saved. Reveal or copy it below for your client."
      )
      setAction(null)
      setKey("")
      await queryClient.invalidateQueries({ queryKey: ["molto"] })
    },
  })
  function open(next: KeyAction) {
    setAction(next)
    setName(next.kind === "edit" ? next.entry.name : "")
    setKey("")
    setConfirmation("")
    mutation.reset()
  }
  const danger = action?.kind === "rotate" || action?.kind === "revoke"
  const allowed =
    action?.kind === "rotate"
      ? key.trim().length > 0 && confirmation === "ROTATE"
      : action?.kind === "revoke"
        ? confirmation === "REVOKE"
        : name.trim().length > 0
  return (
    <QueryState query={query}>
      {query.data && (
        <div className="flex flex-col gap-5">
          <div>
            <h2 className="text-lg font-semibold">API keys</h2>
            <p className="text-sm text-muted-foreground">
              Keep the main key for administration. Give each inference client
              its own key.
            </p>
          </div>
          {notice && (
            <Alert>
              <AlertTitle>Access updated</AlertTitle>
              <AlertDescription>{notice}</AlertDescription>
            </Alert>
          )}
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <ShieldCheckIcon className="size-4" />
                <CardTitle>Main key</CardTitle>
                <Badge>Administrator</Badge>
              </div>
              <CardDescription>
                Full access to server settings, models, and inference. Treat
                this key as a password.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Secret value={query.data.main_key} label="main key" />
              <Button
                variant="outline"
                onClick={() => open({ kind: "rotate" })}
              >
                Rotate main key
              </Button>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="space-y-1">
                  <CardTitle>Inference keys</CardTitle>
                  <CardDescription>
                    Subkeys can use inference APIs. They cannot manage this
                    server.
                  </CardDescription>
                </div>
                <Button onClick={() => open({ kind: "create" })}>
                  <PlusIcon />
                  Create key
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {query.data.sub_keys.length === 0 ? (
                <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed p-8 text-center">
                  <KeyRoundIcon className="size-6 text-muted-foreground" />
                  <p className="font-medium">No inference keys yet</p>
                  <p className="text-sm text-muted-foreground">
                    Create a named key for an app, device, or teammate.
                  </p>
                </div>
              ) : (
                query.data.sub_keys.map((entry) => (
                  <div
                    key={entry.id}
                    className="space-y-3 rounded-lg border p-4"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div>
                        <p className="text-sm font-medium">
                          {entry.name || "Unnamed key"}{" "}
                          <Badge variant="secondary">Inference only</Badge>
                        </p>
                        <p className="mt-1 text-xs text-muted-foreground">
                          Created{" "}
                          {entry.created_at
                            ? new Date(entry.created_at).toLocaleString()
                            : "at an unknown time"}
                        </p>
                      </div>
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => open({ kind: "edit", entry })}
                        >
                          Edit
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-destructive"
                          onClick={() => open({ kind: "revoke", entry })}
                        >
                          Revoke
                        </Button>
                      </div>
                    </div>
                    <Secret value={entry.key} label={`${entry.name} key`} />
                  </div>
                ))
              )}
            </CardContent>
          </Card>
          <Policy data={query.data} />
          <Dialog
            open={action !== null}
            onOpenChange={(next) => {
              if (!next && !mutation.isPending) {
                setAction(null)
                setKey("")
              }
            }}
          >
            <DialogContent>
              <DialogHeader>
                <DialogTitle>
                  {action?.kind === "rotate"
                    ? "Rotate main key"
                    : action?.kind === "revoke"
                      ? `Revoke ${action.entry.name || "key"}?`
                      : action?.kind === "edit"
                        ? "Edit inference key"
                        : "Create inference key"}
                </DialogTitle>
                <DialogDescription>
                  {action?.kind === "rotate"
                    ? "The old main key stops working immediately. Update every client that uses it. Your dashboard session will follow the new key."
                    : action?.kind === "revoke"
                      ? "This key stops working immediately. Clients using it will need a replacement."
                      : "Use a name that identifies the client. Leave the key blank to generate a secure key when creating, or keep the current key when editing."}
                </DialogDescription>
              </DialogHeader>
              <form
                className="space-y-5"
                onSubmit={(event) => {
                  event.preventDefault()
                  if (allowed && !mutation.isPending) mutation.mutate()
                }}
              >
                <FieldGroup>
                  {action?.kind !== "revoke" && (
                    <>
                      {action?.kind !== "rotate" && (
                        <Field>
                          <FieldLabel htmlFor="key-name">Name</FieldLabel>
                          <Input
                            id="key-name"
                            value={name}
                            onChange={(event) => setName(event.target.value)}
                            required
                            disabled={mutation.isPending}
                          />
                        </Field>
                      )}
                      <Field>
                        <FieldLabel htmlFor="key-value">
                          {action?.kind === "rotate"
                            ? "New main key"
                            : "Custom key"}
                        </FieldLabel>
                        <Input
                          id="key-value"
                          type="password"
                          autoComplete="new-password"
                          value={key}
                          onChange={(event) => setKey(event.target.value)}
                          required={action?.kind === "rotate"}
                          disabled={mutation.isPending}
                        />
                        <FieldDescription>
                          {action?.kind === "edit"
                            ? "Entering a new value invalidates the old key."
                            : "Store the key securely in your API client."}
                        </FieldDescription>
                      </Field>
                    </>
                  )}
                  {danger && (
                    <Field>
                      <FieldLabel htmlFor="key-confirmation">
                        Type {action?.kind === "rotate" ? "ROTATE" : "REVOKE"}{" "}
                        to confirm
                      </FieldLabel>
                      <Input
                        id="key-confirmation"
                        autoComplete="off"
                        value={confirmation}
                        onChange={(event) =>
                          setConfirmation(event.target.value)
                        }
                        disabled={mutation.isPending}
                      />
                    </Field>
                  )}
                </FieldGroup>
                {mutation.isError && (
                  <Alert variant="destructive">
                    <AlertDescription>
                      {errorMessage(mutation.error)}
                    </AlertDescription>
                  </Alert>
                )}
                <DialogFooter>
                  <Button
                    type="button"
                    variant="outline"
                    disabled={mutation.isPending}
                    onClick={() => {
                      setAction(null)
                      setKey("")
                    }}
                  >
                    Cancel
                  </Button>
                  <Button
                    type="submit"
                    variant={danger ? "destructive" : "default"}
                    disabled={!allowed || mutation.isPending}
                  >
                    {mutation.isPending
                      ? "Saving…"
                      : action?.kind === "rotate"
                        ? "Rotate key"
                        : action?.kind === "revoke"
                          ? "Revoke key"
                          : "Save key"}
                  </Button>
                </DialogFooter>
              </form>
            </DialogContent>
          </Dialog>
        </div>
      )}
    </QueryState>
  )
}

function Policy({ data }: { data: AuthKeys }) {
  const client = useQueryClient()
  const [edits, setEdits] = useState<Record<string, boolean>>({})
  const [saved, setSaved] = useState(false)
  const source =
    typeof data.policy === "object" && data.policy !== null
      ? (data.policy as Record<string, unknown>)
      : data
  const fields = Object.entries(source).filter(
    ([key, value]) =>
      typeof value === "boolean" && key !== "main_key" && key !== "sub_keys"
  )
  const mutation = useMutation({
    mutationFn: () =>
      managementRequest("auth/policy", { method: "PATCH", body: edits }),
    onSuccess: async () => {
      setEdits({})
      setSaved(true)
      await client.invalidateQueries({ queryKey: ["molto"] })
    },
  })
  if (!fields.length) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle>Authentication policy</CardTitle>
        <CardDescription>
          Control which requests must provide an API key.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form
          className="space-y-5"
          onSubmit={(event) => {
            event.preventDefault()
            if (Object.keys(edits).length && !mutation.isPending)
              mutation.mutate()
          }}
        >
          <FieldGroup>
            {fields.map(([key, value]) => (
              <Field key={key} orientation="horizontal">
                <FieldLabel htmlFor={`policy-${key}`}>
                  {key === "skip_api_key_verification"
                    ? "Skip API key verification"
                    : key === "allow_unauthenticated_inference"
                      ? "Allow inference without a key"
                      : labelFor(key)}
                </FieldLabel>
                <Switch
                  id={`policy-${key}`}
                  checked={edits[key] ?? value === true}
                  disabled={mutation.isPending}
                  onCheckedChange={(checked) => {
                    setEdits((current) => {
                      const next = { ...current }
                      if (checked === value) delete next[key]
                      else next[key] = checked
                      return next
                    })
                    setSaved(false)
                  }}
                />
              </Field>
            ))}
          </FieldGroup>
          {mutation.isError && (
            <Alert variant="destructive">
              <AlertDescription>
                {errorMessage(mutation.error)}
              </AlertDescription>
            </Alert>
          )}
          {saved && (
            <p role="status" className="text-sm text-muted-foreground">
              Authentication policy saved.
            </p>
          )}
          <Button
            type="submit"
            disabled={!Object.keys(edits).length || mutation.isPending}
          >
            Save policy
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}
