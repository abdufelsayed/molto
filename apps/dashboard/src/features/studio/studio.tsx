import { useCallback, useEffect, useRef, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import type { ModelMessage } from "ai"
import {
  Copy,
  ChevronDown,
  ChevronUp,
  GitBranch,
  Pencil,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Download,
  FlaskConical,
  LoaderCircle,
  ListTree,
  MoreHorizontal,
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
import type { PanelImperativeHandle } from "react-resizable-panels"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { Bubble, BubbleContent } from "@/components/ui/bubble"
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu"
import {
  Empty,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
  EmptyDescription,
} from "@/components/ui/empty"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
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
import { Inspector, type InspectorProps } from "./components/inspector"
import { IconAction } from "./components/icon-action"
import { FilesPanel, filePath, putFile } from "./components/files"
import { Markdown, ResponseContent } from "./components/response"

const messageActions =
  "flex items-center gap-0.5 [@media(hover:hover)]:opacity-0 group-hover/message:opacity-100 group-focus-within/message:opacity-100"

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
  const [inspectorTab, setInspectorTab] = useState("trace")
  const [settingsTab, setSettingsTab] = useState("model")
  const [showSettings, setShowSettings] = useState(true)
  const observabilityPanel = useRef<PanelImperativeHandle | null>(null)
  const [showInspector, setShowInspector] = useState(false)
  const [showSessions, setShowSessions] = useState(true)
  const [mobile, setMobile] = useState<"sessions" | "settings" | null>(null)
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
    const change = () => {
      setWide(media.matches)
      if (media.matches) setMobile(null)
    }
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
        openFiles()
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
      `snippet-${crypto.randomUUID()}.${extension}`,
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
    openFiles()
    toast.success(`Saved ${path}`)
  }
  function inspect(run: Run, event?: string) {
    setSelectedRun(run.id)
    setSelectedEvent(event)
    setInspectorTab("trace")
    observabilityPanel.current?.expand()
  }
  function toggleObservability() {
    const panel = observabilityPanel.current
    if (panel?.isCollapsed()) panel.expand()
    else panel?.collapse()
  }
  function openFiles() {
    setSettingsTab("sandbox")
    if (wide) setShowSettings(true)
    else setMobile("settings")
  }
  function fork(run: Run, index: number) {
    if (!session) return
    const clone = newSession(structuredClone(run.settings))
    clone.title = `${session.title} (fork)`
    clone.files = structuredClone(run.after)
    clone.turns = structuredClone(session.turns.slice(0, index + 1))
    const ids = new Set(clone.turns.flatMap((turn) => turn.runIds))
    clone.runs = structuredClone(
      session.runs.filter((value) => ids.has(value.id))
    )
    add(clone)
  }
  function activateVariant(turn: Turn, index: number, id: string) {
    if (!session || turn.selected === id) return
    updateSession(session.id, (s) => {
      const chosen = s.runs.find((run) => run.id === id)
      if (!chosen) return s
      return {
        ...s,
        branches: [
          ...(s.branches ?? []),
          {
            id: crypto.randomUUID(),
            title: `Before selecting run ${turn.runIds.indexOf(id) + 1} · ${new Date().toLocaleTimeString()}`,
            turns: structuredClone(s.turns),
            files: structuredClone(s.files),
          },
        ],
        files: structuredClone(chosen.after),
        turns: s.turns
          .slice(0, index + 1)
          .map((value) =>
            value.id === turn.id
              ? { ...value, ...recordedPrompt(chosen, value), selected: id }
              : value
          ),
      }
    })
    setSelectedRun(id)
    setSelectedEvent(undefined)
    setEditing(undefined)
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
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center justify-between px-3 py-3">
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
          nativeButton={false}
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
  const sandboxFiles = session && (
    <FilesPanel
      files={session.files}
      onChange={(files) => updateSession(session.id, (s) => ({ ...s, files }))}
      disabled={running}
      cwd={session.settings.cwd}
      maxBytes={session.settings.maxFileBytes}
      upload={(files) => {
        void uploadFiles(files)
      }}
      selectedPath={selectedPath}
      onSelectPath={setSelectedPath}
    />
  )
  const settingsPanel = session && (
    <SettingsPanel
      settings={session.settings}
      onChange={(settings) =>
        updateSession(session.id, (s) => ({ ...s, settings }))
      }
      models={models.data ?? []}
      running={running}
      tab={settingsTab}
      onTabChange={setSettingsTab}
      sandboxFiles={sandboxFiles}
    />
  )
  const inspector = session && (
    <Tabs
      value={inspectorTab}
      onValueChange={(tab) => {
        setInspectorTab(tab)
        observabilityPanel.current?.expand()
      }}
      className="flex h-full min-h-0 flex-col gap-0"
    >
      <div className="flex h-9 shrink-0 items-center gap-1 px-3">
        <TabsList
          variant="line"
          aria-label="Observability views"
          className="min-w-0 flex-1"
        >
          {[
            ["trace", "Trace"],
            ["request", "Request"],
            ["response", "Response"],
            ["usage", "Usage"],
            ["history", "Runs"],
          ].map(([value, label]) => (
            <TabsTrigger
              key={value}
              value={value!}
              onClick={() => observabilityPanel.current?.expand()}
            >
              {label}
            </TabsTrigger>
          ))}
        </TabsList>
        <IconAction
          label={
            showInspector ? "Minimize observability" : "Expand observability"
          }
          aria-expanded={showInspector}
          aria-controls="observability-content"
          onClick={toggleObservability}
        >
          {showInspector ? <ChevronDown /> : <ChevronUp />}
        </IconAction>
      </div>
      <div
        id="observability-content"
        role="region"
        aria-label="Observability content"
        aria-hidden={!showInspector}
        inert={!showInspector}
        className="flex min-h-0 flex-1 flex-col overflow-hidden"
      >
        {(["trace", "request", "response", "usage"] as const).map((tab) => (
          <TabsContent
            key={tab}
            value={tab}
            className="min-h-0 flex-1 overflow-auto"
          >
            <Inspector
              tab={tab satisfies InspectorProps["tab"]}
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
        ))}
        <TabsContent
          value="history"
          className="min-h-0 flex-1 overflow-auto p-3"
        >
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
                className="group/run flex flex-wrap items-center gap-2 rounded-md px-2 py-1 focus-within:bg-muted/40 hover:bg-muted/40"
              >
                <Button
                  variant="ghost"
                  size="sm"
                  className="flex-1 justify-start"
                  onClick={() => inspect(run)}
                >
                  {new Date(run.created).toLocaleTimeString()} ·{" "}
                  {run.settings.model.split("/").at(-1)}
                  <span className="ml-auto text-xs text-muted-foreground">
                    {run.status}
                    {run.mode === "replay" ? " · replay" : ""}
                  </span>
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
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
      </div>
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
      <MessageScrollerProvider autoScroll>
        <MessageScroller className="flex-1">
          <MessageScrollerViewport>
            <MessageScrollerContent className="gap-8 p-4 lg:p-6">
              {!session.turns.length && (
                <MessageScrollerItem messageId="empty">
                  <Empty className="min-h-72">
                    <EmptyHeader>
                      <EmptyMedia variant="icon">
                        <FlaskConical />
                      </EmptyMedia>
                      <EmptyTitle>Start an experiment</EmptyTitle>
                      <EmptyDescription>
                        Choose a model, adjust its settings, and run a prompt.
                      </EmptyDescription>
                    </EmptyHeader>
                  </Empty>
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
                    className="mx-auto w-full max-w-3xl"
                  >
                    <div className="flex flex-col gap-6">
                      <Message
                        align="end"
                        role="article"
                        aria-label={`User message, turn ${index + 1}`}
                      >
                        <MessageContent>
                          <MessageHeader className="gap-2">
                            <span className="text-foreground">You</span>
                          </MessageHeader>
                          <Bubble
                            variant="tinted"
                            align="end"
                            className={
                              editing?.id === turn.id ? "w-full" : undefined
                            }
                          >
                            <BubbleContent
                              className={
                                editing?.id === turn.id ? "w-full" : undefined
                              }
                            >
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
                                  <div className="flex justify-end gap-2">
                                    <Button
                                      size="sm"
                                      variant="ghost"
                                      onClick={() => setEditing(undefined)}
                                    >
                                      Cancel
                                    </Button>
                                    <Button
                                      size="sm"
                                      onClick={() => {
                                        void start(turn, "live", editing.text)
                                      }}
                                    >
                                      <Play data-icon="inline-start" />
                                      Run edited message
                                    </Button>
                                  </div>
                                </div>
                              ) : (
                                <Markdown text={prompt.text} save={saveCode} />
                              )}
                            </BubbleContent>
                          </Bubble>
                          {!!prompt.attachments?.length && (
                            <div className="flex flex-wrap justify-end gap-2">
                              {prompt.attachments.map((path) => (
                                <Attachment key={path} size="xs" state="done">
                                  <AttachmentContent>
                                    <AttachmentTitle title={path}>
                                      {path.split("/").at(-1)}
                                    </AttachmentTitle>
                                  </AttachmentContent>
                                </Attachment>
                              ))}
                            </div>
                          )}
                          <MessageFooter className={messageActions}>
                            <IconAction
                              label="Copy message"
                              onClick={() => {
                                void navigator.clipboard.writeText(prompt.text)
                              }}
                            >
                              <Copy />
                            </IconAction>
                            <IconAction
                              label="Edit message"
                              disabled={running}
                              onClick={() =>
                                setEditing({ id: turn.id, text: prompt.text })
                              }
                            >
                              <Pencil />
                            </IconAction>
                            <IconAction
                              label="Rerun live"
                              disabled={running}
                              onClick={() => {
                                void start(turn)
                              }}
                            >
                              <RotateCcw />
                            </IconAction>
                            <IconAction
                              label="Fork here"
                              disabled={running || !run}
                              onClick={() => {
                                if (run) fork(run, index)
                              }}
                            >
                              <GitBranch />
                            </IconAction>
                            <DropdownMenu>
                              <Tooltip>
                                <TooltipTrigger
                                  render={
                                    <DropdownMenuTrigger
                                      render={
                                        <Button
                                          variant="ghost"
                                          size="icon-xs"
                                          aria-label={`Turn ${index + 1} actions`}
                                          disabled={running}
                                        />
                                      }
                                    />
                                  }
                                >
                                  <MoreHorizontal />
                                </TooltipTrigger>
                                <TooltipContent>
                                  More message actions
                                </TooltipContent>
                              </Tooltip>
                              <DropdownMenuContent
                                align="end"
                                className="min-w-44"
                              >
                                <DropdownMenuGroup>
                                  <DropdownMenuItem
                                    disabled={!run}
                                    onClick={() => {
                                      void start(turn, "replay")
                                    }}
                                  >
                                    Replay tools
                                  </DropdownMenuItem>
                                </DropdownMenuGroup>
                              </DropdownMenuContent>
                            </DropdownMenu>
                          </MessageFooter>
                        </MessageContent>
                      </Message>
                      {run && (
                        <Message
                          role="article"
                          aria-label={`Assistant message, turn ${index + 1}`}
                        >
                          <MessageContent>
                            <MessageHeader className="gap-2">
                              <span className="text-foreground">Assistant</span>
                              <span className="min-w-0 truncate">
                                {run.settings.model.split("/").at(-1)}
                              </span>
                              {run.mode === "replay" && (
                                <Badge variant="outline">Replay</Badge>
                              )}
                            </MessageHeader>
                            {turn.runIds.length > 1 && (
                              <ToggleGroup
                                aria-label="Response variants"
                                value={[run.id]}
                                size="sm"
                                variant="default"
                                spacing={1}
                                disabled={running}
                                className="flex-wrap"
                                onValueChange={(ids) => {
                                  if (ids[0])
                                    activateVariant(turn, index, ids[0])
                                }}
                              >
                                {turn.runIds.map((id, variant) => (
                                  <ToggleGroupItem
                                    key={id}
                                    value={id}
                                    aria-label={`Run ${variant + 1}`}
                                  >
                                    Run {variant + 1}
                                  </ToggleGroupItem>
                                ))}
                              </ToggleGroup>
                            )}
                            <Bubble variant="ghost" className="w-full">
                              <BubbleContent className="w-full">
                                <ResponseContent
                                  run={run}
                                  selectedEvent={selectedEvent}
                                  inspect={(event) => inspect(run, event)}
                                  save={saveCode}
                                />
                              </BubbleContent>
                            </Bubble>
                            <MessageFooter className={messageActions}>
                              <IconAction
                                label="Copy response"
                                onClick={() => {
                                  void navigator.clipboard.writeText(
                                    run.events
                                      .filter((event) => event.type === "text")
                                      .map((event) => event.text)
                                      .join("\n")
                                  )
                                }}
                              >
                                <Copy />
                              </IconAction>
                              <IconAction
                                label="Rerun response"
                                disabled={running}
                                onClick={() => {
                                  void start(turn)
                                }}
                              >
                                <RotateCcw />
                              </IconAction>
                              <IconAction
                                label="Fork response"
                                disabled={running}
                                onClick={() => fork(run, index)}
                              >
                                <GitBranch />
                              </IconAction>
                              <IconAction
                                label="Inspect this run"
                                onClick={() => inspect(run)}
                              >
                                <ListTree />
                              </IconAction>
                              <span className="ml-1 text-[10px] font-normal text-muted-foreground">
                                {run.status} · {run.steps.length} model calls
                              </span>
                            </MessageFooter>
                          </MessageContent>
                        </Message>
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
              nativeButton={false}
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
              {running
                ? "Settings apply next run"
                : `${session.settings.tools.length} tools enabled`}
              <span className="ml-2 hidden sm:inline">
                Shift+Enter for newline
              </span>
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
  const workspace = (
    <ResizablePanelGroup orientation="vertical" id="studio-center">
      <ResizablePanel
        id="studio-conversation"
        defaultSize="100%"
        minSize="220px"
      >
        {conversation}
      </ResizablePanel>
      <ResizableHandle
        withHandle
        disableDoubleClick
        aria-label="Resize observability"
        aria-valuetext={
          showInspector ? "Observability open" : "Observability closed"
        }
        title="Drag to resize observability. Double-click or press Enter to minimize or restore its content."
        className="hover:bg-muted-foreground/30"
        onDoubleClick={toggleObservability}
        onKeyDownCapture={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault()
            toggleObservability()
          }
        }}
      />
      <ResizablePanel
        id="studio-observability"
        panelRef={observabilityPanel}
        defaultSize="36px"
        collapsedSize="36px"
        minSize="180px"
        maxSize="65%"
        collapsible
        onResize={(size) => setShowInspector(size.inPixels > 36.5)}
      >
        <section className="h-full min-h-0" aria-label="Observability">
          {inspector}
        </section>
      </ResizablePanel>
    </ResizablePanelGroup>
  )
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex h-12 shrink-0 items-center gap-2 border-b px-3">
        <IconAction
          label="Show experiments"
          size="icon-sm"
          aria-expanded={wide ? showSessions : mobile === "sessions"}
          onClick={() => {
            if (wide) setShowSessions((value) => !value)
            else setMobile("sessions")
          }}
        >
          {showSessions && wide ? <PanelLeftClose /> : <PanelLeftOpen />}
        </IconAction>
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
        {(running || storageError) && (
          <Badge variant="secondary">{running ? "Running" : "Unsaved"}</Badge>
        )}
        <IconAction
          label="Show controls"
          size="icon-sm"
          aria-expanded={wide ? showSettings : mobile === "settings"}
          onClick={() => {
            if (wide) setShowSettings((value) => !value)
            else setMobile("settings")
          }}
        >
          {wide ? (
            showSettings ? (
              <PanelRightClose />
            ) : (
              <PanelRightOpen />
            )
          ) : (
            <Settings2 />
          )}
        </IconAction>
        <DropdownMenu>
          <Tooltip>
            <TooltipTrigger
              render={
                <DropdownMenuTrigger
                  render={
                    <Button
                      aria-label="Experiment actions"
                      size="icon-sm"
                      variant="ghost"
                    />
                  }
                />
              }
            >
              <MoreHorizontal />
            </TooltipTrigger>
            <TooltipContent>Experiment actions</TooltipContent>
          </Tooltip>
          <DropdownMenuContent align="end" className="min-w-48">
            <DropdownMenuGroup>
              <DropdownMenuItem disabled={running} onClick={() => duplicate()}>
                <Copy />
                Duplicate experiment
              </DropdownMenuItem>
              <DropdownMenuItem
                disabled={running}
                onClick={() => duplicate(true)}
              >
                Copy setup
              </DropdownMenuItem>
              <DropdownMenuItem
                onClick={() =>
                  download(
                    `molto-${session.id}.json`,
                    JSON.stringify(session, null, 2)
                  )
                }
              >
                <Download />
                Export experiment
              </DropdownMenuItem>
            </DropdownMenuGroup>
            <DropdownMenuSeparator />
            <DropdownMenuGroup>
              <DropdownMenuItem
                disabled={running}
                variant="destructive"
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
                Delete experiment
              </DropdownMenuItem>
            </DropdownMenuGroup>
          </DropdownMenuContent>
        </DropdownMenu>
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
            {showSessions && (
              <>
                <ResizablePanel defaultSize="15%" minSize="150px" maxSize="25%">
                  {sessionsPanel}
                </ResizablePanel>
                <ResizableHandle />
              </>
            )}
            <ResizablePanel defaultSize="60%" minSize="300px">
              {workspace}
            </ResizablePanel>
            {showSettings && (
              <>
                <ResizableHandle />
                <ResizablePanel defaultSize="28%" minSize="280px" maxSize="45%">
                  {settingsPanel}
                </ResizablePanel>
              </>
            )}
          </ResizablePanelGroup>
        </div>
      ) : (
        <div className="min-h-0 flex-1">{workspace}</div>
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
              {mobile === "sessions" ? "Experiments" : "Session controls"}
            </SheetTitle>
          </SheetHeader>
          <div className="min-h-0 flex-1 overflow-hidden">
            {mobile === "sessions" ? sessionsPanel : settingsPanel}
          </div>
        </SheetContent>
      </Sheet>
    </div>
  )
}
