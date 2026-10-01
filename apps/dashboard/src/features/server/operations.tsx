import { useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  managementQuery,
  managementRequest,
} from "@/features/management/request"
import {
  connectionQuery,
  managementKey,
  useManagement,
} from "@/features/management/queries"
import { errorMessage } from "@/features/management/api"
import { ConnectionDialog } from "@/components/connection-dialog"
import { QueryState } from "@/components/page-state"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import {
  Field,
  FieldLabel,
  FieldDescription,
  FieldGroup,
} from "@/components/ui/field"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog"
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectGroup,
  SelectItem,
} from "@/components/ui/select"

type ServerInfo = {
  version: string
  base_path: string
  host: string
  port: number
  restart_supported: boolean
}
type UpdateInfo = {
  status: "checked" | "failed" | "unavailable"
  update_available: boolean | null
  update_channel: string
  current_version?: string
  latest_version?: string
  release_url?: string
  error?: string
}

function safeLink(value: string | undefined) {
  if (!value) return undefined
  try {
    const url = new URL(value)
    return ["https:", "http:"].includes(url.protocol) ? url.href : undefined
  } catch {
    return undefined
  }
}

export function ConnectionPanel() {
  const { api, queryClient } = useManagement()
  const connection = useQuery(connectionQuery(api))
  const info = useQuery(
    managementQuery<ServerInfo>(["server", "info"], "server/info")
  )
  const [restartOpen, setRestartOpen] = useState(false)
  const [confirmation, setConfirmation] = useState("")
  const [restartNotice, setRestartNotice] = useState(false)
  const [channel, setChannel] = useState("stable")
  const disconnect = useMutation({
    mutationFn: () => api.disconnect(),
    onSuccess: async () => {
      await queryClient.cancelQueries({ queryKey: managementKey })
      await queryClient.resetQueries({ queryKey: managementKey })
    },
  })
  const restart = useMutation({
    mutationFn: () => managementRequest("server/restart", { method: "POST" }),
    onSuccess: async () => {
      setRestartNotice(true)
      setRestartOpen(false)
      setConfirmation("")
      await queryClient.invalidateQueries({ queryKey: ["molto"] })
    },
  })
  const update = useMutation({
    mutationFn: () =>
      managementRequest<UpdateInfo>(`server/update?channel=${channel}`),
  })
  return (
    <div className="space-y-5">
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center gap-2">
            <CardTitle>Server connection</CardTitle>
            <Badge
              variant={connection.data?.connected ? "default" : "secondary"}
            >
              {connection.data?.connected ? "Connected" : "Disconnected"}
            </Badge>
          </div>
          <CardDescription>
            {connection.data?.server ??
              "Connect to your Molto server to manage its configuration."}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            {connection.data?.access === "local"
              ? "Connected locally on this machine. API keys are required for inference clients and remote administration."
              : "Use your main API key for administration. The dashboard stores the connection in its server session."}
          </p>
          <div className="flex flex-wrap gap-2">
            <ConnectionDialog />
            {connection.data?.connected && (
              <Button
                variant="outline"
                disabled={disconnect.isPending}
                onClick={() => disconnect.mutate()}
              >
                Disconnect
              </Button>
            )}
          </div>
          {disconnect.isError && (
            <Alert variant="destructive">
              <AlertDescription>
                {errorMessage(disconnect.error)}
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
      <QueryState query={info}>
        {info.data && (
          <Card>
            <CardHeader>
              <CardTitle>Running server</CardTitle>
              <CardDescription>
                Startup arguments may override the saved settings.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              <dl className="grid gap-4 sm:grid-cols-3">
                <div>
                  <dt className="text-xs text-muted-foreground">Version</dt>
                  <dd className="mt-1 font-medium">{info.data.version}</dd>
                </div>
                <div>
                  <dt className="text-xs text-muted-foreground">
                    Listening address
                  </dt>
                  <dd className="mt-1 font-mono text-sm break-all">
                    {info.data.host}:{info.data.port}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs text-muted-foreground">
                    Configuration directory
                  </dt>
                  <dd className="mt-1 font-mono text-xs break-all">
                    {info.data.base_path}
                  </dd>
                </div>
              </dl>
              <div className="flex flex-wrap items-center gap-3">
                <Button
                  variant="outline"
                  disabled={!info.data.restart_supported || restart.isPending}
                  onClick={() => {
                    setRestartOpen(true)
                    setConfirmation("")
                    restart.reset()
                  }}
                >
                  Restart server
                </Button>
                <p className="text-xs text-muted-foreground">
                  {info.data.restart_supported
                    ? "Restart interrupts inference requests and reloads saved settings."
                    : "This launch method cannot restart through the dashboard. Restart the process manually."}
                </p>
              </div>
              {restartNotice && (
                <Alert>
                  <AlertTitle>Restart requested</AlertTitle>
                  <AlertDescription>
                    The server is restarting. Wait for it to return, then
                    refresh server status.
                  </AlertDescription>
                </Alert>
              )}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => void info.refetch()}
              >
                Refresh status
              </Button>
            </CardContent>
          </Card>
        )}
      </QueryState>
      <Card>
        <CardHeader>
          <CardTitle>Check for updates</CardTitle>
          <CardDescription>
            Compare this server with the selected release channel. Checking does
            not install anything.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2">
            <Select
              items={[
                { value: "stable", label: "Stable" },
                { value: "beta", label: "Beta" },
              ]}
              value={channel}
              onValueChange={(value) => {
                if (value) {
                  setChannel(value)
                  update.reset()
                }
              }}
            >
              <SelectTrigger aria-label="Release channel">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  <SelectItem value="stable">Stable</SelectItem>
                  <SelectItem value="beta">Beta</SelectItem>
                </SelectGroup>
              </SelectContent>
            </Select>
            <Button
              variant="outline"
              disabled={update.isPending}
              onClick={() => update.mutate()}
            >
              {update.isPending ? "Checking…" : "Check for updates"}
            </Button>
          </div>
          {update.isError && (
            <Alert variant="destructive">
              <AlertDescription>{errorMessage(update.error)}</AlertDescription>
            </Alert>
          )}
          {update.data && (
            <Alert
              variant={
                update.data.status === "failed" ? "destructive" : "default"
              }
            >
              <AlertTitle>
                {update.data.status !== "checked"
                  ? "Update check unavailable"
                  : update.data.update_available
                    ? "Update available"
                    : "Server is up to date"}
              </AlertTitle>
              <AlertDescription>
                {update.data.error ??
                  (update.data.status !== "checked"
                    ? "The server could not verify this release channel."
                    : `${update.data.current_version ?? info.data?.version ?? "Current version"}${update.data.latest_version ? ` · Latest ${update.data.latest_version}` : ""}`)}
                {safeLink(update.data.release_url) && (
                  <a
                    href={safeLink(update.data.release_url)}
                    target="_blank"
                    rel="noreferrer"
                    className="ml-2 underline"
                  >
                    View release
                  </a>
                )}
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
      <Dialog
        open={restartOpen}
        onOpenChange={(open) => {
          if (!restart.isPending) setRestartOpen(open)
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Restart Molto?</DialogTitle>
            <DialogDescription>
              Active inference will be interrupted. Save your settings first.
              Restarting loads the configuration on disk.
            </DialogDescription>
          </DialogHeader>
          <Field>
            <FieldLabel htmlFor="restart-confirmation">
              Type RESTART to confirm
            </FieldLabel>
            <Input
              id="restart-confirmation"
              value={confirmation}
              onChange={(event) => setConfirmation(event.target.value)}
              disabled={restart.isPending}
            />
          </Field>
          {restart.isError && (
            <Alert variant="destructive">
              <AlertDescription>{errorMessage(restart.error)}</AlertDescription>
            </Alert>
          )}
          <DialogFooter>
            <Button
              variant="outline"
              disabled={restart.isPending}
              onClick={() => setRestartOpen(false)}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              disabled={confirmation !== "RESTART" || restart.isPending}
              onClick={() => restart.mutate()}
            >
              {restart.isPending ? "Restarting…" : "Restart server"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

type Integration = {
  id: string
  name: string
  model: string | null
  command: string
  description: string
}
type SearchResult = {
  title?: string
  url?: string
  snippet?: string
  body?: string
}
type SearchTest = {
  ok: boolean
  error?: string | { code?: string; message: string }
  results?: SearchResult[]
  provider?: string
  message?: string
}

export function IntegrationsPanel() {
  const query = useQuery(
    managementQuery<{ integrations: Integration[] }>(
      ["server", "integrations"],
      "server/integrations"
    )
  )
  const [copied, setCopied] = useState("")
  return (
    <div className="space-y-5">
      <QueryState query={query}>
        {query.data && (
          <Card>
            <CardHeader>
              <CardTitle>Client integrations</CardTitle>
              <CardDescription>
                Commands and connection examples use the server's current
                address and model. Review them before running them in your
                client.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {query.data.integrations.length ? (
                query.data.integrations.map((entry) => (
                  <div
                    key={entry.id}
                    className="space-y-2 rounded-lg border p-4"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="font-medium">{entry.name}</p>
                        <p className="mt-1 text-xs text-muted-foreground">
                          {entry.description}
                        </p>
                        {entry.model && (
                          <p className="mt-1 text-xs text-muted-foreground">
                            Model: {entry.model}
                          </p>
                        )}
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          void navigator.clipboard
                            .writeText(entry.command)
                            .then(
                              () => setCopied(entry.id),
                              () => setCopied("failed")
                            )
                        }}
                      >
                        {copied === entry.id ? "Copied" : "Copy"}
                      </Button>
                    </div>
                    <pre className="overflow-x-auto rounded-md bg-muted p-3 text-xs break-all whitespace-pre-wrap">
                      {entry.command}
                    </pre>
                  </div>
                ))
              ) : (
                <p className="text-sm text-muted-foreground">
                  No client integrations are available for this server.
                </p>
              )}
              {copied === "failed" && (
                <p role="status" className="text-sm text-destructive">
                  Clipboard unavailable. Select the command to copy it.
                </p>
              )}
            </CardContent>
          </Card>
        )}
      </QueryState>
      <WebSearchTest />
    </div>
  )
}

function WebSearchTest() {
  const queryClient = useQueryClient()
  const [provider, setProvider] = useState("saved")
  const [credential, setCredential] = useState("")
  const [url, setUrl] = useState("")
  const [backends, setBackends] = useState("")
  const mutation = useMutation({
    mutationFn: () =>
      managementRequest<SearchTest>("server/web-search/test", {
        method: "POST",
        body: {
          ...(provider === "saved" ? {} : { provider }),
          ...(provider === "brave" && credential
            ? { brave_api_key: credential }
            : {}),
          ...(provider === "searxng" && url ? { searxng_url: url } : {}),
          ...(provider === "ddgs_custom" ? { ddgs_backends: backends } : {}),
        },
      }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: ["molto", "server", "integrations"],
      })
    },
  })
  return (
    <Card>
      <CardHeader>
        <CardTitle>Test web search</CardTitle>
        <CardDescription>
          Send one test request using saved settings or temporary overrides.
          Temporary values are never saved.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form
          className="space-y-5"
          onSubmit={(event) => {
            event.preventDefault()
            if (!mutation.isPending) mutation.mutate()
          }}
        >
          <FieldGroup>
            <Field>
              <FieldLabel htmlFor="search-provider">Provider</FieldLabel>
              <Select
                items={[
                  { value: "saved", label: "Saved server settings" },
                  { value: "brave", label: "Brave" },
                  { value: "searxng", label: "SearXNG" },
                  { value: "ddgs", label: "Automatic metasearch" },
                  { value: "ddgs_custom", label: "Selected engines" },
                  { value: "duckduckgo", label: "DuckDuckGo" },
                ]}
                value={provider}
                onValueChange={(value) => {
                  if (value) {
                    setProvider(value)
                    mutation.reset()
                  }
                }}
              >
                <SelectTrigger
                  id="search-provider"
                  disabled={mutation.isPending}
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectItem value="saved">Saved server settings</SelectItem>
                    <SelectItem value="brave">Brave</SelectItem>
                    <SelectItem value="searxng">SearXNG</SelectItem>
                    <SelectItem value="ddgs">Automatic metasearch</SelectItem>
                    <SelectItem value="ddgs_custom">
                      Selected engines
                    </SelectItem>
                    <SelectItem value="duckduckgo">DuckDuckGo</SelectItem>
                  </SelectGroup>
                </SelectContent>
              </Select>
            </Field>
            {provider === "ddgs_custom" && (
              <Field>
                <FieldLabel htmlFor="test-search-engines">
                  Search engines
                </FieldLabel>
                <Input
                  id="test-search-engines"
                  value={backends}
                  onChange={(event) => setBackends(event.target.value)}
                  disabled={mutation.isPending}
                  placeholder="duckduckgo,wikipedia"
                />
                <FieldDescription>
                  Comma-separated: brave, duckduckgo, grokipedia, mojeek,
                  wikipedia, yahoo, yandex.
                </FieldDescription>
              </Field>
            )}
            {provider === "brave" && (
              <Field>
                <FieldLabel htmlFor="test-brave-key">
                  Temporary Brave API key
                </FieldLabel>
                <Input
                  id="test-brave-key"
                  type="password"
                  autoComplete="new-password"
                  value={credential}
                  disabled={mutation.isPending}
                  onChange={(event) => setCredential(event.target.value)}
                />
                <FieldDescription>
                  Leave blank to use the server's saved key.
                </FieldDescription>
              </Field>
            )}
            {provider === "searxng" && (
              <Field>
                <FieldLabel htmlFor="test-search-url">
                  Temporary SearXNG URL
                </FieldLabel>
                <Input
                  id="test-search-url"
                  type="url"
                  value={url}
                  disabled={mutation.isPending}
                  onChange={(event) => setUrl(event.target.value)}
                  placeholder="https://search.example.com"
                />
              </Field>
            )}
          </FieldGroup>
          <Button type="submit" variant="outline" disabled={mutation.isPending}>
            {mutation.isPending ? "Testing…" : "Test search"}
          </Button>
          {mutation.isError && (
            <Alert variant="destructive">
              <AlertDescription>
                {errorMessage(mutation.error)}
              </AlertDescription>
            </Alert>
          )}
          {mutation.data && (
            <Alert variant={mutation.data.ok ? "default" : "destructive"}>
              <AlertTitle>
                {mutation.data.ok
                  ? "Web search is working"
                  : "Web search test failed"}
              </AlertTitle>
              <AlertDescription>
                {(typeof mutation.data.error === "string"
                  ? mutation.data.error
                  : mutation.data.error?.message) ??
                  mutation.data.message ??
                  (mutation.data.ok
                    ? "The provider accepted the test request."
                    : "The provider did not accept the test request.")}
              </AlertDescription>
            </Alert>
          )}
          {mutation.data?.results && (
            <ul className="space-y-3">
              {mutation.data.results.map((result, index) => (
                <li key={result.url ?? index} className="rounded-lg border p-3">
                  {safeLink(result.url) ? (
                    <a
                      href={safeLink(result.url)}
                      target="_blank"
                      rel="noreferrer"
                      className="text-sm font-medium underline"
                    >
                      {result.title || result.url}
                    </a>
                  ) : (
                    <p className="text-sm font-medium">
                      {result.title || "Search result"}
                    </p>
                  )}
                  <p className="mt-1 text-xs text-muted-foreground">
                    {result.snippet ?? result.body}
                  </p>
                </li>
              ))}
            </ul>
          )}
        </form>
      </CardContent>
    </Card>
  )
}
