import { useState } from "react"
import { Download, File, Plus, RotateCcw, Trash2, Upload } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Field, FieldLabel } from "@/components/ui/field"
import type { Snapshot, VirtualFile } from "../agent/types"
import { decodeBytes, encodeBytes } from "../agent/encoding"
import { download } from "../store"

export function filePath(value: string, cwd: string) {
  const parts = (value.startsWith("/") ? value : `${cwd}/${value}`).split("/")
  const result: string[] = []
  for (const part of parts) {
    if (part === "..") result.pop()
    else if (part && part !== ".") result.push(part)
  }
  return `/${result.join("/")}`
}
export function putFile(
  files: Snapshot,
  path: string,
  content: string | Uint8Array
): Snapshot {
  const result = files.filter((f) => f.path !== path)
  const parts = path.split("/").slice(1, -1)
  let parent = ""
  for (const part of parts) {
    parent += `/${part}`
    if (!result.some((f) => f.path === parent))
      result.push({ path: parent, type: "directory", mode: 493 })
  }
  result.push({
    path,
    type: "file",
    content: encodeBytes(
      typeof content === "string" ? new TextEncoder().encode(content) : content
    ),
    mode: 420,
  })
  return result.sort((a, b) => a.path.localeCompare(b.path))
}
export function FilesPanel({
  files,
  onChange,
  disabled,
  cwd,
  maxBytes,
  upload,
  selectedPath,
  onSelectPath,
}: {
  files: Snapshot
  onChange: (files: Snapshot) => void
  disabled: boolean
  cwd: string
  maxBytes: number
  upload: (files: FileList) => void
  selectedPath?: string
  onSelectPath: (path: string) => void
}) {
  const [draft, setDraft] = useState<{ path: string; text: string }>()
  const [newPath, setNewPath] = useState("")
  const [error, setError] = useState("")
  const file = files.find((f) => f.path === selectedPath)
  const text =
    file?.type === "file"
      ? new TextDecoder().decode(decodeBytes(file.content ?? ""))
      : ""
  const choose = (item: VirtualFile) => {
    setDraft(undefined)
    onSelectPath(item.path)
  }
  const save = (path: string, value: string) => {
    const next = putFile(files, path, value)
    if (
      next.reduce((sum, f) => sum + (f.content?.length ?? 0) * 0.75, 0) >
      maxBytes
    ) {
      setError("The filesystem size limit would be exceeded.")
      return
    }
    setError("")
    onChange(next)
    onSelectPath(path)
    setDraft(undefined)
  }
  return (
    <div className="flex min-w-0 flex-col gap-3">
      <div className="flex shrink-0 items-center justify-between">
        <span className="text-xs font-medium">Virtual filesystem</span>
        <div className="flex gap-1">
          <Tooltip>
            <TooltipTrigger
              render={
                <Button
                  nativeButton={false}
                  size="icon-xs"
                  variant="ghost"
                  disabled={disabled}
                  render={
                    <label
                      aria-label="Upload files"
                      htmlFor="studio-upload-files"
                    />
                  }
                >
                  <Upload />
                  <input
                    id="studio-upload-files"
                    aria-label="Upload sandbox files"
                    type="file"
                    multiple
                    className="sr-only"
                    disabled={disabled}
                    onChange={(event) => {
                      if (event.target.files) upload(event.target.files)
                      event.target.value = ""
                    }}
                  />
                </Button>
              }
            />
            <TooltipContent>Upload files</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger
              render={
                <Button
                  aria-label="Reset sandbox files"
                  size="icon-xs"
                  variant="ghost"
                  disabled={disabled}
                  onClick={() => {
                    onChange([])
                    setDraft(undefined)
                    onSelectPath("")
                  }}
                >
                  <RotateCcw />
                </Button>
              }
            />
            <TooltipContent>Reset sandbox files</TooltipContent>
          </Tooltip>
        </div>
      </div>
      <div className="flex min-w-0 shrink-0 gap-2">
        <Input
          aria-label="New file path"
          placeholder={`${cwd}/notes.txt`}
          className="min-w-0 flex-1"
          value={newPath}
          disabled={disabled}
          onChange={(event) => setNewPath(event.target.value)}
        />
        <Tooltip>
          <TooltipTrigger
            render={
              <Button
                aria-label="Create file"
                size="icon"
                variant="outline"
                disabled={disabled || !newPath.trim()}
                onClick={() => {
                  save(filePath(newPath, cwd), "")
                  setNewPath("")
                }}
              >
                <Plus />
              </Button>
            }
          />
          <TooltipContent>Create file</TooltipContent>
        </Tooltip>
      </div>
      {error && (
        <p role="alert" className="text-xs text-destructive">
          {error}
        </p>
      )}
      <div className="flex min-w-0 flex-col gap-3">
        <div className="max-h-48 overflow-auto">
          {files
            .filter((f) => f.type !== "directory")
            .map((item) => (
              <Button
                key={item.path}
                size="sm"
                variant={selectedPath === item.path ? "secondary" : "ghost"}
                className="w-full justify-start font-mono text-xs"
                onClick={() => choose(item)}
              >
                <File className="shrink-0" />
                <span className="truncate">{item.path}</span>
                {item.type === "symlink" && <span>↗</span>}
              </Button>
            ))}
          {!files.some((f) => f.type !== "directory") && (
            <p className="py-3 text-xs text-muted-foreground">
              Upload files or let the agent create them.
            </p>
          )}
        </div>
        {file && (
          <div className="flex min-h-0 flex-col gap-2">
            <div className="flex items-center gap-1">
              <span className="min-w-0 flex-1 truncate font-mono text-xs">
                {file.path}
              </span>
              <Tooltip>
                <TooltipTrigger
                  render={
                    <Button
                      aria-label="Download file"
                      size="icon-xs"
                      variant="ghost"
                      disabled={file.type !== "file"}
                      onClick={() =>
                        download(
                          file.path.split("/").at(-1) ?? "file",
                          decodeBytes(file.content ?? ""),
                          "application/octet-stream"
                        )
                      }
                    >
                      <Download />
                    </Button>
                  }
                />
                <TooltipContent>Download file</TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger
                  render={
                    <Button
                      aria-label="Delete file"
                      size="icon-xs"
                      variant="ghost"
                      disabled={disabled}
                      onClick={() =>
                        onChange(files.filter((f) => f.path !== file.path))
                      }
                    >
                      <Trash2 />
                    </Button>
                  }
                />
                <TooltipContent>Delete file</TooltipContent>
              </Tooltip>
            </div>
            {file.type === "symlink" ? (
              <p className="text-xs">Symlink → {file.target}</p>
            ) : (
              <>
                <Field className="min-h-0 flex-1">
                  <FieldLabel htmlFor="sandbox-editor" className="sr-only">
                    File content
                  </FieldLabel>
                  <Textarea
                    id="sandbox-editor"
                    value={draft?.path === file.path ? draft.text : text}
                    disabled={disabled}
                    onChange={(event) =>
                      setDraft({ path: file.path, text: event.target.value })
                    }
                    className="min-h-48 resize-y font-mono text-xs"
                  />
                </Field>
                <Button
                  size="sm"
                  disabled={disabled || draft?.path !== file.path}
                  onClick={() => save(file.path, draft?.text ?? text)}
                >
                  Save file
                </Button>
              </>
            )}
          </div>
        )}
      </div>
      <p className="text-xs text-muted-foreground">
        {Math.round(
          files.reduce((sum, f) => sum + (f.content?.length ?? 0) * 0.75, 0) /
            1024
        )}{" "}
        KB · {cwd}
        {disabled ? " · Files are locked during a run" : ""}
      </p>
    </div>
  )
}
