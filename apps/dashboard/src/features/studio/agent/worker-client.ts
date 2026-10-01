import type { AgentSettings, Snapshot, ToolName } from "./types"

export class SandboxClient {
  private worker: Worker
  private pending = new Map<
    string,
    { resolve: (value: unknown) => void; reject: (error: Error) => void }
  >()
  private networkController = new AbortController()
  private closed = false
  get available() {
    return !this.closed
  }
  files: Snapshot = []
  constructor(
    private settings: AgentSettings,
    private network: (
      url: string,
      signal: AbortSignal
    ) => Promise<{ content: string; url: string }>
  ) {
    this.worker = new Worker(new URL("./sandbox.worker.ts", import.meta.url), {
      type: "module",
    })
    this.worker.onmessage = (event: MessageEvent) => {
      const message = event.data as {
        id: string
        action?: string
        url?: string
        data?: unknown
        error?: string
        files?: Snapshot
      }
      if (message.action === "network" && message.url) {
        void this.network(message.url, this.networkController.signal).then(
          (data) => {
            if (!this.closed)
              this.worker.postMessage({
                action: "network-result",
                id: message.id,
                data,
              })
          },
          (error: unknown) => {
            if (!this.closed)
              this.worker.postMessage({
                action: "network-result",
                id: message.id,
                error: String(error),
              })
          }
        )
        return
      }
      if (message.files) this.files = message.files
      const wait = this.pending.get(message.id)
      if (message.error) wait?.reject(new Error(message.error))
      else wait?.resolve(message.data)
      this.pending.delete(message.id)
    }
    this.worker.onerror = (event) =>
      this.close(new Error(event.message || "Sandbox worker failed."))
  }
  private request(
    action: string,
    extra: Record<string, unknown>,
    signal?: AbortSignal
  ): Promise<unknown> {
    if (this.closed)
      return Promise.reject(
        new Error(
          "The sandbox was terminated; start a new run to restore its last completed snapshot."
        )
      )
    signal?.throwIfAborted()
    return new Promise((resolve, reject) => {
      const id = crypto.randomUUID()
      const timer = setTimeout(
        () =>
          this.close(
            new Error(
              "Sandbox execution timed out. Changes since the last completed tool were discarded."
            )
          ),
        this.settings.toolTimeout
      )
      const abort = () =>
        this.close(
          new Error(
            "Sandbox stopped. Changes since the last completed tool were discarded."
          )
        )
      signal?.addEventListener("abort", abort, { once: true })
      const cleanup = () => {
        clearTimeout(timer)
        signal?.removeEventListener("abort", abort)
      }
      this.pending.set(id, {
        resolve: (value) => {
          cleanup()
          resolve(value)
        },
        reject: (error) => {
          cleanup()
          reject(error)
        },
      })
      this.worker.postMessage({ id, action, settings: this.settings, ...extra })
    })
  }
  restore(files: Snapshot, signal?: AbortSignal) {
    return this.request("restore", { files }, signal)
  }
  execute(name: ToolName, args: Record<string, unknown>, signal?: AbortSignal) {
    return this.request("execute", { name, args }, signal)
  }
  close(error = new Error("Sandbox closed.")) {
    this.closed = true
    this.networkController.abort(error)
    this.worker.terminate()
    for (const wait of this.pending.values()) wait.reject(error)
    this.pending.clear()
  }
}
