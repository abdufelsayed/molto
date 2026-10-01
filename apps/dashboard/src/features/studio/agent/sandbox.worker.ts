import { VirtualSandbox } from "./sandbox"
import type { AgentSettings, Snapshot, ToolName } from "./types"

const sandbox = new VirtualSandbox()
const pending = new Map<
  string,
  {
    resolve: (value: { content: string; url: string }) => void
    reject: (error: Error) => void
  }
>()
const scope = globalThis as unknown as {
  postMessage: (value: unknown) => void
  onmessage: ((event: MessageEvent) => void) | null
}
scope.onmessage = (event: MessageEvent) => {
  const message = event.data as {
    id: string
    action: string
    files: Snapshot
    name: ToolName
    args: Record<string, unknown>
    settings: AgentSettings
    data?: { content: string; url: string }
    error?: string
  }
  if (message.action === "network-result") {
    const wait = pending.get(message.id)
    if (message.error) wait?.reject(new Error(message.error))
    else if (message.data) wait?.resolve(message.data)
    pending.delete(message.id)
    return
  }
  void (async () => {
    try {
      let data: unknown
      if (message.action === "restore")
        await sandbox.restore(
          message.files,
          message.settings.maxFileBytes,
          message.settings.cwd
        )
      else if (message.action === "execute")
        data = await sandbox.execute(
          message.name,
          message.args,
          message.settings,
          (url) =>
            new Promise((resolve, reject) => {
              const id = crypto.randomUUID()
              pending.set(id, { resolve, reject })
              scope.postMessage({ action: "network", id, url })
            })
        )
      scope.postMessage({
        id: message.id,
        data,
        files: await sandbox.snapshot(),
      })
    } catch (error) {
      scope.postMessage({
        id: message.id,
        error: error instanceof Error ? error.message : String(error),
        files: await sandbox.snapshot(),
      })
    }
  })()
}
