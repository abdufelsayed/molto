import { useCallback, useEffect, useRef, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import type { ModelMessage } from "ai"
import {
  ChevronDown,
  Copy,
  Download,
  FlaskConical,
  Folder,
  History,
  LoaderCircle,
  PanelRight,
  Play,
  Plus,
  RotateCcw,
  Send,
  Settings2,
  Square,
  Trash2,
  Upload,
  X,
} from "lucide-react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Field, FieldLabel } from "@/components/ui/field"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupTextarea,
} from "@/components/ui/input-group"
import {
  Attachment,
  AttachmentActions,
  AttachmentContent,
  AttachmentTitle,
  AttachmentAction,
} from "@/components/ui/attachment"
import {
  Message,
  MessageContent,
  MessageFooter,
  MessageHeader,
} from "@/components/ui/message"
import {
  MessageScroller,
  MessageScrollerButton,
  MessageScrollerContent,
  MessageScrollerItem,
  MessageScrollerProvider,
  MessageScrollerViewport,
} from "@/components/ui/message-scroller"
import { Marker, MarkerContent } from "@/components/ui/marker"
import { connectionQuery, useManagement } from "@/features/management/queries"
import { detail } from "@/features/management/api"
import { defaults, newSession, settingsSchema, modelBody } from "./agent/types"
import type { Run, StudioSession, Turn } from "./agent/types"
import { runAgent } from "./agent/run"
import type { Interaction } from "./agent/run"
import {
  loadSessions,
  saveSession,
  removeSession,
  importSession,
  download,
} from "./store"
import { SettingsPanel } from "./components/settings-panel"
import { Inspector } from "./components/inspector"
import { FilesPanel, filePath, putFile } from "./components/files"
import { Markdown, ResponseContent } from "./components/response"

async function jsonRequest(path: string, body?: unknown, signal?: AbortSignal) {
  const response = await fetch(`/api/studio/${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
    credentials: "same-origin",
    cache: "no-store",
  })
  const data: unknown = await response.json()
  if (!response.ok) throw new Error(detail(data, response.status))
  return data
}
function userMessage(turn: Turn, files: StudioSession["files"]): ModelMessage {
  const paths = turn.attachments ?? []
  const text = `${turn.text}${paths.length ? `\n\nAttached files in the virtual filesystem:\n${paths.join("\n")}` : ""}`
  const images = paths
    .filter((path) => /\.(png|jpe?g|webp|gif)$/i.test(path))
    .flatMap((path) => {
      const file = files.find((f) => f.path === path)
      return file?.content
        ? [
            {
              type: "image" as const,
              image: `data:image/${/\.jpe?g$/i.test(path) ? "jpeg" : path.split(".").at(-1)};base64,${file.content}`,
            },
          ]
        : []
    })
  return {
    role: "user",
    content: images.length ? [{ type: "text", text }, ...images] : text,
  }
}
function recordedPrompt(run: Run | undefined, turn: Turn) {
  if (run?.prompt) return run.prompt
  const input = run?.input.at(-1)
  if (input?.role !== "user") return turn
  const text =
    typeof input.content === "string"
      ? input.content
      : input.content
          .filter((part) => part.type === "text")
          .map((part) => part.text)
          .join("\n")
  const suffix = turn.attachments?.length
    ? `\n\nAttached files in the virtual filesystem:\n${turn.attachments.join("\n")}`
    : ""
  return {
    text:
      suffix && text.endsWith(suffix) ? text.slice(0, -suffix.length) : text,
    attachments: turn.attachments,
  }
}
export function Studio() {
  const { api } = useManagement()
  const access = useQuery(connectionQuery(api))
  const models = useQuery({
    queryKey: ["studio-models", access.data?.server],
    enabled: !!access.data?.connected,
    queryFn: async ({ signal }) => {
      const result = (await jsonRequest("models", undefined, signal)) as {
        data: { id: string }[]
      }
      return result.data.map((item) => item.id)
    },
  })
  const [sessions, setSessions] = useState<StudioSession[]>([])
  const sessionsRef = useRef(sessions)
  const [active, setActive] = useState("")
  const [ready, setReady] = useState(false)
  const [wide, setWide] = useState(true)
  const [storageError, setStorageError] = useState("")
  const [running, setRunning] = useState(false)
  const abort = useRef<AbortController | null>(null)
  const [selectedRun, setSelectedRun] = useState<string>()
  const [selectedEvent, setSelectedEvent] = useState<string>()
  const [selectedPath, setSelectedPath] = useState<string>()
  const [inspectorTab, setInspectorTab] = useState("run")
  const [mobile, setMobile] = useState<
    "sessions" | "settings" | "inspect" | null
  >(null)
  const [editing, setEditing] = useState<{ id: string; text: string }>()
  const [attachments, setAttachments] = useState<string[]>([])
  const [interaction, setInteraction] = useState<Interaction>()
  const reply = useRef<((answer: string) => void) | null>(null)
  const [answer, setAnswer] = useState("")
  const dirty = useRef(new Set<string>())
  const session = sessions.find((s) => s.id === active)
  const selected =
    session?.runs.find((r) => r.id === selectedRun) ?? session?.runs.at(-1)

  useEffect(() => {
    const media = window.matchMedia("(min-width: 1024px)")
    const change = () => setWide(media.matches)
    change()
    media.addEventListener("change", change)
    return () => media.removeEventListener("change", change)
  }, [])

  const updateSession = useCallback(
    (id: string, change: (session: StudioSession) => StudioSession) => {
      setSessions((previous) => {
        const next = previous.map((value) =>
          value.id === id ? { ...change(value), updated: Date.now() } : value
        )
        sessionsRef.current = next
        dirty.current.add(id)
        return next
      })
    },
    []
  )
  const add = useCallback((value: StudioSession) => {
    setSessions((previous) => {
      const next = [value, ...previous]
      sessionsRef.current = next
      return next
    })
    dirty.current.add(value.id)
    setActive(value.id)
    setSelectedRun(undefined)
    setSelectedEvent(undefined)
    setAttachments([])
    setEditing(undefined)
  }, [])
  useEffect(() => {
    let live = true
    void loadSessions()
      .then((saved) => {
        if (!live) return
        const values = saved.length ? saved : [newSession()]
        sessionsRef.current = values
        setSessions(values)
        setActive(values[0]!.id)
        setReady(true)
        if (!saved.length) dirty.current.add(values[0]!.id)
      })
      .catch((error: unknown) => {
        if (live) {
          setStorageError(String(error))
          add(newSession())
          setReady(true)
        }
      })
    // Throttled checkpoints still run while tokens arrive continuously.
    let saving = false
    const flush = async () => {
      if (saving) return
      saving = true
      // Failed writes are retried on the next checkpoint, not within this batch.
      const pendingIds = [...dirty.current]
      for (const id of pendingIds) {
        dirty.current.delete(id)
        const value = sessionsRef.current.find((s) => s.id === id)
        if (value)
          try {
            await saveSession(value)
            if (live) setStorageError("")
          } catch (error) {
            dirty.current.add(id)
            if (live)
              setStorageError(`Changes could not be saved: ${String(error)}`)
          }
      }
      saving = false
    }
    const timer = setInterval(() => {
      void flush()
    }, 500)
    const visibility = () => {
      if (document.visibilityState === "hidden") void flush()
    }
    document.addEventListener("visibilitychange", visibility)
    return () => {
      live = false
      clearInterval(timer)
      document.removeEventListener("visibilitychange", visibility)
      abort.current?.abort()
      void flush()
    }
  }, [add])
  useEffect(() => {
    if (session && !session.settings.model && models.data?.length)
      updateSession(session.id, (s) => ({
        ...s,
        settings: { ...s.settings, model: models.data![0]! },
      }))
  }, [models.data, session, updateSession])

  const start = useCallback(
    async (target?: Turn, mode: "live" | "replay" = "live", text?: string) => {
      const current = sessionsRef.current.find((s) => s.id === active)
      if (!current || abort.current || !access.data?.connected) return
      try {
        settingsSchema.parse(current.settings)
        modelBody(current.settings)
      } catch (error) {
        toast.error(error instanceof Error ? error.message : String(error))
        return
      }
      if (!current.settings.model) {
        toast.error("Select a local model.")
        return
      }
      const previousRun = target
        ? (current.runs.find((r) => r.id === target.selected) ??
          current.runs.find((r) => target.runIds.includes(r.id)))
        : undefined
      if (mode === "replay" && !previousRun) {
        toast.error("Select a recorded run first.")
        return
      }
      const turn: Turn = target
        ? {
            ...target,
            ...recordedPrompt(previousRun, target),
            text: text ?? recordedPrompt(previousRun, target).text,
          }
        : {
            id: crypto.randomUUID(),
            text: current.draft,
            runIds: [],
            attachments: [...attachments],
          }
      if (!turn.text.trim() && !turn.attachments?.length) return
      const targetIndex = target
        ? current.turns.findIndex((t) => t.id === target.id)
        : current.turns.length
      const preceding = current.turns.slice(0, targetIndex)
      const last = preceding.at(-1)
      const context = last
        ? (current.runs.find((r) => r.id === last.selected)?.messages ?? [])
        : []
      const files = previousRun?.before ?? current.files
      const messages: ModelMessage[] = previousRun
        ? [...previousRun.input.slice(0, -1), userMessage(turn, files)]
        : [...context, userMessage(turn, files)]
      const controller = new AbortController()
      abort.current = controller
      setRunning(true)
      setEditing(undefined)
      setAttachments([])
      updateSession(current.id, (s) => ({
        ...s,
        draft: target ? s.draft : "",
        branches:
          target && targetIndex < s.turns.length - 1
            ? [
                ...(s.branches ?? []),
                {
                  id: crypto.randomUUID(),
                  title: `Before rerunning turn ${targetIndex + 1} · ${new Date().toLocaleTimeString()}`,
                  turns: structuredClone(s.turns),
                  files: structuredClone(s.files),
                },
              ]
            : s.branches,
        turns: target
          ? s.turns
              .slice(0, targetIndex + 1)
              .map((t) => (t.id === turn.id ? turn : t))
          : [...s.turns, turn],
      }))
      let runId: string | undefined
      try {
        const inherited = await Promise.allSettled([
          api.settings(controller.signal),
          api.modelSettings(current.settings.model, controller.signal),
        ])
        controller.signal.throwIfAborted()
        const environment = {
          capturedAt: Date.now(),
          globalSampling:
            inherited[0].status === "fulfilled"
              ? inherited[0].value.sampling
              : { unavailable: true },
          modelSettings:
            inherited[1].status === "fulfilled"
              ? inherited[1].value.settings
              : { unavailable: true },
          note: "Saved configuration at run start. Unset controls inherit backend defaults; model templates can add their own defaults.",
        }
        await runAgent({
          settings: current.settings,
          messages,
          files,
          environment,
          signal: controller.signal,
          replay: mode === "replay" ? previousRun : undefined,
          modelUrl: `${window.location.origin}/api/studio/model`,
          web: (name, input, settings, signal) =>
            jsonRequest(
              "tools",
              { name, input, search: settings.search },
              signal
            ),
          interact: (value, signal) =>
            new Promise((resolve, reject) => {
              const cancel = () => {
                reply.current = null
                setInteraction(undefined)
                reject(signal.reason)
              }
              signal.addEventListener("abort", cancel, { once: true })
              setInteraction(value)
              setAnswer("")
              reply.current = (response) => {
                signal.removeEventListener("abort", cancel)
                reply.current = null
                setInteraction(undefined)
                resolve(response)
              }
            }),
          onUpdate: (run) => {
            const recorded = {
              ...run,
              prompt: {
                text: turn.text,
                attachments: turn.attachments
                  ? [...turn.attachments]
                  : undefined,
              },
            }
            if (!runId) {
              runId = run.id
              setSelectedRun(run.id)
              setSelectedEvent(undefined)
            }
            updateSession(current.id, (s) => {
              const stored = s.runs.find((r) => r.id === run.id)
              return {
                ...s,
                title:
                  s.title === "Untitled experiment"
                    ? turn.text.slice(0, 60) || "File experiment"
                    : s.title,
                files: run.after,
                runs: stored
                  ? s.runs.map((r) =>
                      r.id === run.id ? { ...recorded, note: r.note } : r
                    )
                  : [...s.runs, recorded],
                turns: s.turns.map((t) =>
                  t.id === turn.id
                    ? {
                        ...t,
                        runIds: t.runIds.includes(run.id)
                          ? t.runIds
                          : [...t.runIds, run.id],
                        selected: run.id,
                      }
                    : t
                ),
              }
            })
          },
        })
      } catch (error) {
        toast.error(error instanceof Error ? error.message : String(error))
      } finally {
        abort.current = null
        setRunning(false)
        setInteraction(undefined)
        reply.current = null
      }
    },
    [active, access.data?.connected, attachments, updateSession, api]
  )

  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      if (event.key === "Escape" && abort.current) {
        event.preventDefault()
        abort.current.abort()
      }
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault()
        if (!abort.current) void start()
      }
    }
    window.addEventListener("keydown", listener)
    return () => window.removeEventListener("keydown", listener)
  }, [start])
  async function uploadFiles(input: FileList, attach = false) {
    if (!session || running) return
    let files = session.files
    const paths: string[] = []
    try {
      for (const file of Array.from(input)) {
        if (file.size > session.settings.maxFileBytes)
          throw new Error(`${file.name} exceeds the filesystem limit.`)
        const path = filePath(file.name, session.settings.cwd)
        if (files.some((f) => f.path === path))
          throw new Error(`${path} already exists. Rename or delete it first.`)
        files = putFile(files, path, new Uint8Array(await file.arrayBuffer()))
        paths.push(path)
      }
      if (
        files.reduce((sum, f) => sum + (f.content?.length ?? 0) * 0.75, 0) >
        session.settings.maxFileBytes
      )
        throw new Error("Upload exceeds the filesystem size limit.")
      updateSession(session.id, (s) => ({ ...s, files }))
      if (attach) setAttachments((previous) => [...previous, ...paths])
      else {
        setSelectedPath(paths[0])
        setInspectorTab("files")
      }
    } catch (error) {
      toast.error(String(error))
    }
  }
  function saveCode(code: string, language: string) {
    if (!session || running) {
      toast.error("Stop the run before editing the filesystem.")
      return
    }
    const extension =
      (
        {
          javascript: "js",
          typescript: "ts",
          python: "py",
          bash: "sh",
          json: "json",
          html: "html",
          css: "css",
        } as Record<string, string>
      )[language] ?? "txt"
    const path = filePath(
      `snippet-${Date.now()}.${extension}`,
      session.settings.cwd
    )
    const files = putFile(session.files, path, code)
    if (
      files.reduce((sum, f) => sum + (f.content?.length ?? 0) * 0.75, 0) >
      session.settings.maxFileBytes
    ) {
      toast.error("Filesystem size limit exceeded.")
      return
    }
    updateSession(session.id, (s) => ({ ...s, files }))
    setSelectedPath(path)
    setInspectorTab("files")
    toast.success(`Saved ${path}`)
  }
  function inspect(run: Run, event?: string) {
    setSelectedRun(run.id)
    setSelectedEvent(event)
    setInspectorTab("run")
  }
  function duplicate(setupOnly = false) {
    if (!session) return
    const next = setupOnly
      ? newSession(structuredClone(session.settings))
      : structuredClone(session)
    next.id = crypto.randomUUID()
    next.updated = Date.now()
    next.title = `${session.title} (${setupOnly ? "setup" : "copy"})`
    next.files = structuredClone(session.files)
    add(next)
  }
  const sessionsPanel = (
    <div className="flex h-full min-h-0 flex-col border-r bg-muted/15">
      <div className="flex items-center justify-between border-b px-3 py-3">
        <span className="text-xs font-medium">Experiments</span>
        <Button
          size="icon-xs"
          variant="ghost"
          aria-label="New experiment"
          disabled={running}
          onClick={() =>
            add(
              newSession({
                ...defaults(),
                model: session?.settings.model ?? models.data?.[0] ?? "",
              })
            )
          }
        >
          <Plus />
        </Button>
      </div>
      <div className="flex-1 overflow-auto p-2">
        {sessions.map((item) => (
          <Button
            key={item.id}
            variant={active === item.id ? "secondary" : "ghost"}
            disabled={running && active !== item.id}
            className="mb-1 h-auto w-full justify-start py-2.5 text-left"
            onClick={() => {
              setActive(item.id)
              setSelectedRun(undefined)
              setSelectedEvent(undefined)
              setAttachments([])
              setEditing(undefined)
              setMobile(null)
            }}
          >
            <div className="min-w-0">
              <span className="block truncate text-xs">{item.title}</span>
              <span className="mt-1 block truncate text-[10px] text-muted-foreground">
                {item.runs.length} runs ·{" "}
                {item.settings.model.split("/").at(-1) || "No model"}
              </span>
            </div>
          </Button>
        ))}
      </div>
      <div className="border-t p-2">
        <Button
          size="sm"
          variant="ghost"
          className="w-full justify-start"
          disabled={running}
          render={
            <label
              aria-label="Import experiment file"
              htmlFor="studio-import-experiment"
            />
          }
        >
          <Upload />
          Import experiment
          <input
            id="studio-import-experiment"
            aria-label="Import experiment"
            type="file"
            accept="application/json,.json"
            className="sr-only"
            disabled={running}
            onChange={(event) => {
              const file = event.target.files?.[0]
              if (file) {
                if (file.size > 100_000_000)
                  toast.error("Import is limited to 100 MB.")
                else
                  void file
                    .text()
                    .then((text) => add(importSession(text)))
                    .catch((error: unknown) => toast.error(String(error)))
              }
              event.target.value = ""
            }}
          />
        </Button>
      </div>
    </div>
  )
  const inspector = session && (
    <Tabs
      value={inspectorTab}
      onValueChange={(value) => setInspectorTab(value)}
      className="flex h-full min-h-0 flex-col gap-0"
    >
      <div className="flex items-center gap-2 border-b px-3 py-1">
        <TabsList variant="line">
          <TabsTrigger value="run">
            <PanelRight />
            Inspector
          </TabsTrigger>
          <TabsTrigger value="files">
            <Folder />
            Files
          </TabsTrigger>
          <TabsTrigger value="history">
            <History />
            Runs
          </TabsTrigger>
        </TabsList>
        <span className="ml-auto truncate text-[10px] text-muted-foreground">
          {selected ? selected.id.slice(0, 8) : "Select a response"}
        </span>
      </div>
      <TabsContent value="run" className="min-h-0 flex-1 overflow-auto">
        <Inspector
          run={selected}
          selectedEvent={selectedEvent}
          onSelectEvent={setSelectedEvent}
          onNote={(note) => {
            if (selected)
              updateSession(session.id, (s) => ({
                ...s,
                runs: s.runs.map((r) =>
                  r.id === selected.id ? { ...r, note } : r
                ),
              }))
          }}
        />
      </TabsContent>
      <TabsContent value="files" className="min-h-0 flex-1">
        <FilesPanel
          files={session.files}
          onChange={(files) =>
            updateSession(session.id, (s) => ({ ...s, files }))
          }
          disabled={running}
          cwd={session.settings.cwd}
          maxBytes={session.settings.maxFileBytes}
          upload={(files) => {
            void uploadFiles(files)
          }}
          selectedPath={selectedPath}
          onSelectPath={setSelectedPath}
        />
      </TabsContent>
      <TabsContent value="history" className="min-h-0 flex-1 overflow-auto p-3">
        <div className="flex flex-col gap-2">
          {(session.branches ?? []).map((branch) => (
            <Button
              key={branch.id}
              size="sm"
              variant="outline"
              disabled={running}
              onClick={() => {
                updateSession(session.id, (s) => ({
                  ...s,
                  turns: structuredClone(branch.turns),
                  files: structuredClone(branch.files),
                  branches: [
                    ...(s.branches ?? []).filter((b) => b.id !== branch.id),
                    {
                      id: crypto.randomUUID(),
                      title: `Previous branch · ${new Date().toLocaleTimeString()}`,
                      turns: structuredClone(s.turns),
                      files: structuredClone(s.files),
                    },
                  ],
                }))
                setSelectedRun(undefined)
              }}
            >
              Restore branch: {branch.title}
            </Button>
          ))}
          {[...session.runs].reverse().map((run) => (
            <div
              key={run.id}
              className="flex flex-wrap items-center gap-2 rounded-lg border p-3"
            >
              <Button
                variant="ghost"
                size="sm"
                className="flex-1 justify-start"
                onClick={() => inspect(run)}
              >
                {new Date(run.created).toLocaleTimeString()} ·{" "}
                {run.settings.model.split("/").at(-1)}
                <Badge variant="outline">{run.status}</Badge>
                <Badge variant="secondary">{run.mode}</Badge>
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={running}
                onClick={() =>
                  updateSession(session.id, (s) => ({
                    ...s,
                    settings: structuredClone(run.settings),
                    files: structuredClone(run.before),
                  }))
                }
              >
                Restore setup + starting files
              </Button>
              <Button
                size="icon-sm"
                aria-label="Export run"
                variant="ghost"
                onClick={() =>
                  download(
                    `molto-run-${run.id}.json`,
                    JSON.stringify(run, null, 2)
                  )
                }
              >
                <Download />
              </Button>
            </div>
          ))}
        </div>
      </TabsContent>
    </Tabs>
  )

  if (!ready || !session)
    return (
      <div
        role="status"
        className="flex flex-1 items-center justify-center gap-2 text-sm text-muted-foreground"
      >
        <LoaderCircle className="size-4 animate-spin" />
        Opening studio
      </div>
    )
  const conversation = (
    <div className="flex h-full min-h-0 flex-col">
      <Collapsible className="shrink-0 border-b" defaultOpen={false}>
        <CollapsibleTrigger
          render={
            <Button
              variant="ghost"
              className="h-10 w-full justify-start rounded-none px-4"
            />
          }
        >
          <ChevronDown />
          <span className="text-xs">System prompt</span>
          <span className="ml-auto text-xs text-muted-foreground">
            {session.settings.system
              ? `${session.settings.system.length} characters`
              : "Empty"}
          </span>
        </CollapsibleTrigger>
        <CollapsibleContent className="p-4 pt-0">
          <Field>
            <FieldLabel htmlFor="studio-system" className="sr-only">
              System prompt
            </FieldLabel>
            <Textarea
              id="studio-system"
              value={session.settings.system}
              placeholder="No system prompt. Add one to experiment."
              className="max-h-64 min-h-32 resize-y font-mono text-xs"
              onChange={(event) =>
                updateSession(session.id, (s) => ({
                  ...s,
                  settings: { ...s.settings, system: event.target.value },
                }))
              }
            />
          </Field>
          {running && (
            <p className="mt-2 text-xs text-muted-foreground">
              Changes apply to the next run.
            </p>
          )}
        </CollapsibleContent>
      </Collapsible>
      <MessageScrollerProvider autoScroll>
        <MessageScroller className="flex-1">
          <MessageScrollerViewport>
            <MessageScrollerContent className="gap-6 p-4 lg:px-6">
              {!session.turns.length && (
                <MessageScrollerItem messageId="empty">
                  <div className="flex min-h-56 flex-col items-center justify-center gap-3 text-center">
                    <FlaskConical className="size-7 text-muted-foreground" />
                    <h1 className="text-lg font-medium">Start an experiment</h1>
                    <p className="max-w-sm text-sm text-muted-foreground">
                      An empty system prompt, inherited generation settings, and
                      all tools enabled. Change the setup, run a prompt, then
                      inspect what happened.
                    </p>
                    <p className="text-xs text-muted-foreground">
                      ⌘/Ctrl + Enter to run · Esc to stop
                    </p>
                  </div>
                </MessageScrollerItem>
              )}
              {session.turns.map((turn, index) => {
                const run =
                  session.runs.find((r) => r.id === turn.selected) ??
                  session.runs.find((r) => turn.runIds.includes(r.id))
                const prompt = recordedPrompt(run, turn)
                return (
                  <MessageScrollerItem
                    key={turn.id}
                    messageId={turn.id}
                    scrollAnchor
                  >
                    <div className="flex flex-col gap-5">
                      <Message>
                        <MessageContent>
                          <MessageHeader>
                            <span>USER</span>
                            <span className="ml-auto">Turn {index + 1}</span>
                          </MessageHeader>
                          {editing?.id === turn.id ? (
                            <div className="flex flex-col gap-2">
                              <Textarea
                                aria-label="Edit message"
                                value={editing.text}
                                onChange={(event) =>
                                  setEditing({
                                    id: turn.id,
                                    text: event.target.value,
                                  })
                                }
                              />
                              <div className="flex gap-2">
                                <Button
                                  size="sm"
                                  onClick={() => {
                                    void start(turn, "live", editing.text)
                                  }}
                                >
                                  <Play />
                                  Run edited message
                                </Button>
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  onClick={() => setEditing(undefined)}
                                >
                                  Cancel
                                </Button>
                              </div>
                            </div>
                          ) : (
                            <Markdown text={prompt.text} save={saveCode} />
                          )}
                          {!!prompt.attachments?.length && (
                            <div className="flex flex-wrap gap-1">
                              {prompt.attachments.map((path) => (
                                <Badge
                                  key={path}
                                  variant="outline"
                                  className="font-mono text-xs"
                                >
                                  {path}
                                </Badge>
                              ))}
                            </div>
                          )}
                          <MessageFooter>
                            <Button
                              size="xs"
                              variant="ghost"
                              disabled={running}
                              onClick={() =>
                                setEditing({ id: turn.id, text: prompt.text })
                              }
                            >
                              Edit + rerun
                            </Button>
                            <Button
                              size="xs"
                              variant="ghost"
                              disabled={running}
                              onClick={() => {
                                void start(turn)
                              }}
                            >
                              <RotateCcw />
                              Rerun live
                            </Button>
                            <Button
                              size="xs"
                              variant="ghost"
                              disabled={running || !run}
                              onClick={() => {
                                void start(turn, "replay")
                              }}
                            >
                              Replay tools
                            </Button>
                          </MessageFooter>
                        </MessageContent>
                      </Message>
                      {run && (
                        <Message>
                          <MessageContent>
                            <MessageHeader>
                              <span>
                                {run.settings.model.split("/").at(-1)}
                              </span>
                              <Badge className="ml-2" variant="outline">
                                {run.mode}
                              </Badge>
                              <Button
                                size="xs"
                                variant="ghost"
                                className="ml-auto"
                                onClick={() => inspect(run)}
                              >
                                Inspect
                              </Button>
                            </MessageHeader>
                            {turn.runIds.length > 1 && (
                              <div className="flex flex-wrap gap-1">
                                {turn.runIds.map((id, variant) => (
                                  <Button
                                    key={id}
                                    size="xs"
                                    variant={
                                      run.id === id ? "secondary" : "ghost"
                                    }
                                    disabled={running}
                                    onClick={() => {
                                      updateSession(session.id, (s) => {
                                        if (run.id === id) return s
                                        const chosen = s.runs.find(
                                          (r) => r.id === id
                                        )
                                        if (!chosen) return s
                                        return {
                                          ...s,
                                          branches: [
                                            ...(s.branches ?? []),
                                            {
                                              id: crypto.randomUUID(),
                                              title: `Before selecting run ${variant + 1} · ${new Date().toLocaleTimeString()}`,
                                              turns: structuredClone(s.turns),
                                              files: structuredClone(s.files),
                                            },
                                          ],
                                          files: structuredClone(chosen.after),
                                          turns: s.turns
                                            .slice(0, index + 1)
                                            .map((t) =>
                                              t.id === turn.id
                                                ? {
                                                    ...t,
                                                    ...recordedPrompt(
                                                      chosen,
                                                      t
                                                    ),
                                                    selected: id,
                                                  }
                                                : t
                                            ),
                                        }
                                      })
                                      setSelectedRun(id)
                                      setSelectedEvent(undefined)
                                      setEditing(undefined)
                                    }}
                                  >
                                    Run {variant + 1}
                                  </Button>
                                ))}
                              </div>
                            )}
                            <ResponseContent
                              run={run}
                              selectedEvent={selectedEvent}
                              inspect={(event) => inspect(run, event)}
                              save={saveCode}
                            />
                            <MessageFooter className="flex-wrap gap-2">
                              <span>
                                {run.status} · {run.steps.length} model calls
                              </span>
                              <Button
                                size="xs"
                                variant="ghost"
                                aria-label="Copy response"
                                onClick={() => {
                                  void navigator.clipboard.writeText(
                                    run.events
                                      .filter((e) => e.type === "text")
                                      .map((e) => e.text)
                                      .join("\n")
                                  )
                                }}
                              >
                                <Copy />
                              </Button>
                              <Button
                                size="xs"
                                variant="ghost"
                                disabled={running}
                                onClick={() => {
                                  const clone = newSession(
                                    structuredClone(run.settings)
                                  )
                                  clone.title = `${session.title} (fork)`
                                  clone.files = structuredClone(run.after)
                                  clone.turns = structuredClone(
                                    session.turns.slice(0, index + 1)
                                  )
                                  const ids = new Set(
                                    clone.turns.flatMap((t) => t.runIds)
                                  )
                                  clone.runs = structuredClone(
                                    session.runs.filter((r) => ids.has(r.id))
                                  )
                                  add(clone)
                                }}
                              >
                                Fork here
                              </Button>
                            </MessageFooter>
                          </MessageContent>
                        </Message>
                      )}
                      {index < session.turns.length - 1 && (
                        <Marker variant="separator">
                          <MarkerContent>Turn {index + 2}</MarkerContent>
                        </Marker>
                      )}
                    </div>
                  </MessageScrollerItem>
                )
              })}
            </MessageScrollerContent>
          </MessageScrollerViewport>
          <MessageScrollerButton />
        </MessageScroller>
      </MessageScrollerProvider>
      {interaction && (
        <div
          className="border-t bg-muted/30 p-4"
          role="region"
          aria-label={
            interaction.kind === "question"
              ? "Agent question"
              : "Tool confirmation"
          }
        >
          <p className="mb-2 text-sm font-medium">
            {interaction.kind === "question"
              ? String(
                  (interaction.event.input as { question: string }).question
                )
              : `Allow ${interaction.event.tool}?`}
          </p>
          {interaction.kind === "confirm" ? (
            <>
              <pre className="mb-2 max-h-32 overflow-auto text-xs">
                {JSON.stringify(interaction.event.input, null, 2)}
              </pre>
              <div className="flex gap-2">
                <Button size="sm" onClick={() => reply.current?.("allow")}>
                  Allow tool
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => reply.current?.("deny")}
                >
                  Deny tool
                </Button>
              </div>
            </>
          ) : (
            <>
              <div className="mb-2 flex flex-wrap gap-2">
                {(
                  (interaction.event.input as { options?: string[] }).options ??
                  []
                ).map((option) => (
                  <Button
                    key={option}
                    size="sm"
                    variant="outline"
                    onClick={() => reply.current?.(option)}
                  >
                    {option}
                  </Button>
                ))}
              </div>
              <div className="flex gap-2">
                <Input
                  aria-label="Answer the agent"
                  placeholder="Your answer"
                  value={answer}
                  onChange={(event) => setAnswer(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && answer.trim())
                      reply.current?.(answer)
                  }}
                />
                <Button
                  size="sm"
                  disabled={!answer.trim()}
                  onClick={() => reply.current?.(answer)}
                >
                  Answer
                </Button>
              </div>
            </>
          )}
        </div>
      )}
      <div className="shrink-0 border-t p-3">
        {attachments.map((path) => (
          <Attachment key={path} size="xs" state="done" className="mb-2">
            <AttachmentContent>
              <AttachmentTitle>{path}</AttachmentTitle>
            </AttachmentContent>
            <AttachmentActions>
              <AttachmentAction
                aria-label={`Remove attachment ${path}`}
                onClick={() =>
                  setAttachments((previous) =>
                    previous.filter((p) => p !== path)
                  )
                }
              >
                <X />
              </AttachmentAction>
            </AttachmentActions>
          </Attachment>
        ))}
        <InputGroup>
          <InputGroupTextarea
            aria-label="Prompt"
            placeholder="Enter a prompt to test the model…"
            className="max-h-48 min-h-20"
            value={session.draft}
            onChange={(event) =>
              updateSession(session.id, (s) => ({
                ...s,
                draft: event.target.value,
              }))
            }
            onKeyDown={(event) => {
              if (
                event.key === "Enter" &&
                !event.shiftKey &&
                !event.metaKey &&
                !event.ctrlKey &&
                !event.nativeEvent.isComposing
              ) {
                event.preventDefault()
                if (!running) void start()
              }
            }}
          />
          <InputGroupAddon align="block-end">
            <InputGroupButton
              size="icon-sm"
              variant="ghost"
              aria-label="Attach files"
              disabled={running}
              render={
                <label
                  aria-label="Attach files to prompt"
                  htmlFor="studio-attach-prompt"
                />
              }
            >
              <Plus />
              <input
                id="studio-attach-prompt"
                type="file"
                aria-label="Attach prompt files"
                multiple
                disabled={running}
                className="sr-only"
                onChange={(event) => {
                  if (event.target.files)
                    void uploadFiles(event.target.files, true)
                  event.target.value = ""
                }}
              />
            </InputGroupButton>
            <span className="min-w-0 flex-1 truncate text-[10px] text-muted-foreground">
              {session.settings.model.split("/").at(-1) || "Choose a model"} ·{" "}
              {session.settings.tools.length} tools ·{" "}
              {running
                ? "Settings apply next run"
                : "Enter to run · Shift+Enter for newline"}
            </span>
            {running ? (
              <InputGroupButton
                aria-label="Stop run"
                variant="destructive"
                size="sm"
                onClick={() => abort.current?.abort()}
              >
                <Square />
                Stop
              </InputGroupButton>
            ) : (
              <InputGroupButton
                aria-label="Run prompt"
                variant="default"
                size="sm"
                disabled={
                  !access.data?.connected ||
                  !session.settings.model ||
                  (!session.draft.trim() && !attachments.length)
                }
                onClick={() => {
                  void start()
                }}
              >
                <Send />
                Run
              </InputGroupButton>
            )}
          </InputGroupAddon>
        </InputGroup>
      </div>
    </div>
  )
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex h-12 shrink-0 items-center gap-2 border-b px-3">
        <Button
          className="lg:hidden"
          size="icon-sm"
          variant="ghost"
          aria-label="Show experiments"
          onClick={() => setMobile("sessions")}
        >
          <History />
        </Button>
        <Input
          aria-label="Experiment title"
          className="h-7 w-auto min-w-0 flex-1 border-transparent bg-transparent text-xs shadow-none"
          value={session.title}
          onChange={(event) =>
            updateSession(session.id, (s) => ({
              ...s,
              title: event.target.value,
            }))
          }
        />
        <Badge variant={running ? "secondary" : "outline"}>
          {running ? "Running" : storageError ? "Unsaved" : "Local studio"}
        </Badge>
        <Button
          aria-label="Duplicate experiment"
          size="icon-sm"
          variant="ghost"
          disabled={running}
          onClick={() => duplicate()}
        >
          <Copy />
        </Button>
        <Button
          size="sm"
          variant="ghost"
          className="hidden md:inline-flex"
          disabled={running}
          onClick={() => duplicate(true)}
        >
          Copy setup
        </Button>
        <Button
          aria-label="Export experiment"
          size="icon-sm"
          variant="ghost"
          onClick={() =>
            download(
              `molto-${session.id}.json`,
              JSON.stringify(session, null, 2)
            )
          }
        >
          <Download />
        </Button>
        <Button
          aria-label="Delete experiment"
          size="icon-sm"
          variant="ghost"
          disabled={running}
          onClick={() => {
            dirty.current.delete(session.id)
            void removeSession(session.id).catch((error: unknown) =>
              toast.error(String(error))
            )
            const next = sessions.filter((s) => s.id !== session.id)
            setSessions(next)
            sessionsRef.current = next
            if (next.length) setActive(next[0]!.id)
            else add(newSession())
          }}
        >
          <Trash2 />
        </Button>
        <Button
          size="icon-sm"
          variant="ghost"
          className="lg:hidden"
          aria-label="Show controls"
          onClick={() => setMobile("settings")}
        >
          <Settings2 />
        </Button>
        <Button
          size="icon-sm"
          variant="ghost"
          className="lg:hidden"
          aria-label="Show inspector"
          onClick={() => setMobile("inspect")}
        >
          <PanelRight />
        </Button>
      </div>
      {storageError && (
        <p role="alert" className="border-b px-4 py-2 text-xs text-destructive">
          {storageError} Export your experiment to retain a copy.
        </p>
      )}
      {models.error && (
        <p role="alert" className="border-b px-4 py-2 text-xs text-destructive">
          {models.error.message}
        </p>
      )}
      {!access.data?.connected && (
        <p
          role="alert"
          className="border-b px-4 py-2 text-xs text-muted-foreground"
        >
          Connect to Molto through API access to run a model.
        </p>
      )}
      {wide ? (
        <div className="min-h-0 flex-1">
          <ResizablePanelGroup orientation="horizontal">
            <ResizablePanel defaultSize="17%" minSize="150px" maxSize="25%">
              {sessionsPanel}
            </ResizablePanel>
            <ResizableHandle />
            <ResizablePanel defaultSize="59%" minSize="300px">
              <ResizablePanelGroup orientation="vertical">
                <ResizablePanel defaultSize="70%" minSize="220px">
                  {conversation}
                </ResizablePanel>
                <ResizableHandle />
                <ResizablePanel defaultSize="30%" minSize="100px">
                  {inspector}
                </ResizablePanel>
              </ResizablePanelGroup>
            </ResizablePanel>
            <ResizableHandle />
            <ResizablePanel defaultSize="24%" minSize="260px" maxSize="40%">
              <div className="h-full overflow-auto">
                <SettingsPanel
                  settings={session.settings}
                  onChange={(settings) =>
                    updateSession(session.id, (s) => ({ ...s, settings }))
                  }
                  models={models.data ?? []}
                  running={running}
                />
              </div>
            </ResizablePanel>
          </ResizablePanelGroup>
        </div>
      ) : (
        <div className="min-h-0 flex-1">{conversation}</div>
      )}
      <Sheet
        open={mobile !== null}
        onOpenChange={(open) => {
          if (!open) setMobile(null)
        }}
      >
        <SheetContent className="w-full sm:max-w-lg">
          <SheetHeader>
            <SheetTitle>
              {mobile === "sessions"
                ? "Experiments"
                : mobile === "settings"
                  ? "Session controls"
                  : "Run inspector"}
            </SheetTitle>
          </SheetHeader>
          <div className="min-h-0 flex-1 overflow-auto">
            {mobile === "sessions" ? (
              sessionsPanel
            ) : mobile === "settings" ? (
              <SettingsPanel
                settings={session.settings}
                onChange={(settings) =>
                  updateSession(session.id, (s) => ({ ...s, settings }))
                }
                models={models.data ?? []}
                running={running}
              />
            ) : (
              inspector
            )}
          </div>
        </SheetContent>
      </Sheet>
    </div>
  )
}
