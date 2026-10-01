import { Bash, InMemoryFs, getCommandNames } from "just-bash/browser"
import type { CommandName } from "just-bash/browser"
import type { AgentSettings, Snapshot, ToolName } from "./types"
import { decodeBytes, encodeBytes } from "./encoding"

export { decodeBytes, encodeBytes } from "./encoding"
export class VirtualSandbox {
  fs = new InMemoryFs({}, { maxTotalBytes: 20_000_000 })
  async restore(files: Snapshot, maxBytes = 20_000_000, cwd = "/workspace") {
    this.fs = new InMemoryFs({}, { maxTotalBytes: maxBytes })
    for (const file of files
      .filter((f) => f.type !== "symlink")
      .sort((a, b) => a.path.length - b.path.length)) {
      if (file.type === "directory")
        await this.fs.mkdir(file.path, { recursive: true })
      else {
        await this.fs.mkdir(
          file.path.slice(0, file.path.lastIndexOf("/")) || "/",
          { recursive: true }
        )
        await this.fs.writeFile(file.path, decodeBytes(file.content ?? ""))
      }
      if (file.mode !== undefined) await this.fs.chmod(file.path, file.mode)
    }
    for (const file of files.filter((f) => f.type === "symlink")) {
      await this.fs.mkdir(
        file.path.slice(0, file.path.lastIndexOf("/")) || "/",
        { recursive: true }
      )
      await this.fs.symlink(file.target ?? "", file.path)
    }
    await this.fs.mkdir("/workspace", { recursive: true })
    await this.fs.mkdir(cwd, { recursive: true })
  }
  async snapshot(): Promise<Snapshot> {
    return Promise.all(
      this.fs
        .getAllPaths()
        .sort()
        .map(async (path) => {
          const stat = await this.fs.lstat(path)
          if (stat.isSymbolicLink)
            return {
              path,
              type: "symlink" as const,
              target: await this.fs.readlink(path),
              mode: stat.mode,
            }
          if (stat.isDirectory)
            return { path, type: "directory" as const, mode: stat.mode }
          return {
            path,
            type: "file" as const,
            content: encodeBytes(await this.fs.readFileBuffer(path)),
            mode: stat.mode,
          }
        })
    )
  }
  async execute(
    name: ToolName,
    args: Record<string, unknown>,
    settings: AgentSettings,
    network?: (url: string) => Promise<{ content: string; url: string }>
  ) {
    const path = this.fs.resolvePath(
      settings.cwd,
      typeof args.path === "string" ? args.path : ""
    )
    if (["write", "edit"].includes(name) && settings.readOnly)
      throw new Error("The session filesystem is read-only.")
    if (name === "read") {
      const lines = (await this.fs.readFile(path)).split("\n")
      const offset = Number(args.offset ?? 1) - 1
      const selected = lines.slice(
        offset,
        args.limit ? offset + Number(args.limit) : undefined
      )
      return {
        path,
        content: selected.join("\n").slice(0, settings.outputLimit),
        totalLines: lines.length,
        truncated: selected.join("\n").length > settings.outputLimit,
      }
    }
    if (name === "write") {
      await this.fs.mkdir(path.slice(0, path.lastIndexOf("/")) || "/", {
        recursive: true,
      })
      await this.fs.writeFile(path, String(args.content))
      return { path, written: true }
    }
    if (name === "edit") {
      const content = await this.fs.readFile(path)
      const old = String(args.oldText)
      if (!old || !content.includes(old))
        throw new Error("Exact text was not found.")
      if (content.indexOf(old) !== content.lastIndexOf(old))
        throw new Error("Text occurs more than once; provide a unique match.")
      await this.fs.writeFile(path, content.replace(old, String(args.newText)))
      return { path, edited: true }
    }
    if (name !== "bash") throw new Error(`Unknown sandbox tool: ${name}`)
    const mutations = new Set([
      "writeFile",
      "appendFile",
      "mkdir",
      "createExclusive",
      "rm",
      "cp",
      "mv",
      "chmod",
      "symlink",
      "link",
      "utimes",
    ])
    const fs = settings.readOnly
      ? new Proxy(this.fs, {
          get(target, key) {
            if (mutations.has(String(key)))
              return () => {
                throw new Error("The session filesystem is read-only.")
              }
            const value = Reflect.get(target, key)
            return typeof value === "function" ? value.bind(target) : value
          },
        })
      : this.fs
    // No host JS, Python or SQLite execution. curl can only use the same fetch_url
    // broker as the explicit tool, and only when both session switches permit it.
    const commands = getCommandNames().filter(
      (name): name is CommandName =>
        ![
          "gzip",
          "gunzip",
          "zcat",
          "python",
          "python3",
          "sqlite3",
          "js-exec",
          "curl",
          "wget",
        ].includes(name)
    )
    const bash = new Bash({
      fs,
      cwd: settings.cwd,
      env: settings.env,
      commands,
      executionLimitProfile: "hardened",
      executionLimits: {
        maxExecutionTimeMs: settings.toolTimeout,
        maxOutputSize: settings.outputLimit,
        maxFileSystemBytes: settings.maxFileBytes,
        maxCommandCount: settings.maxCommands,
        maxLoopIterations: settings.maxLoopIterations,
        maxLiveBytes: 64_000_000,
      },
      ...(network && settings.network && settings.tools.includes("fetch_url")
        ? {
            fetch: async (url: string, options?: { method?: string }) => {
              if (options?.method && options.method !== "GET")
                throw new Error("Sandbox network allows GET only.")
              const result = await network(url)
              return {
                status: 200,
                statusText: "OK",
                headers: { "content-type": "text/plain; charset=utf-8" },
                body: new TextEncoder().encode(result.content),
                url: result.url,
              }
            },
          }
        : {}),
    })
    const result = await bash.exec(String(args.command), {
      cwd: settings.cwd,
      env: settings.env,
    })
    return {
      stdout: result.stdout,
      stderr: result.stderr,
      exitCode: result.exitCode,
    }
  }
}
