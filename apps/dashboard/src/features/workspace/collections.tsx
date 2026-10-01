import { useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  managementQuery,
  managementRequest,
} from "@/features/management/request"
import { QueryState } from "@/components/page-state"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog"
import { Failure } from "./primitives"
import type { Collection, Collections } from "./types"
export function CollectionPanel({ selected }: { selected: string[] }) {
  const client = useQueryClient()
  const query = useQuery(
    managementQuery<Collections>(
      ["workspace", "collections"],
      "workspace/collections"
    )
  )
  const [edit, setEdit] = useState<Collection | null>(null)
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [preload, setPreload] = useState(false)
  const [remove, setRemove] = useState<Collection | null>(null)
  const action = useMutation({
    mutationFn: ({
      path,
      method,
      body,
    }: {
      path: string
      method: "PUT" | "POST" | "DELETE"
      body?: unknown
    }) =>
      managementRequest<unknown>(`workspace/collections${path}`, {
        method,
        body,
      }),
    onSuccess: () => {
      setEdit(null)
      setRemove(null)
      void client.invalidateQueries({ queryKey: ["omlx"] })
    },
  })
  function open(collection: Collection) {
    setEdit(collection)
    setName(collection.name)
    setDescription(collection.description ?? "")
    setPreload(collection.preload)
    action.reset()
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle>Collections</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          Collections organize models. Startup preload pins members for loading
          when the server starts. Use Load collection to load them now.
        </p>
        <Button
          disabled={!selected.length || action.isPending}
          onClick={() =>
            open({
              id: crypto.randomUUID(),
              name: "",
              description: "",
              model_ids: selected,
              preload: false,
            })
          }
        >
          Create collection from {selected.length} selected
        </Button>
        <QueryState query={query}>
          {!query.data?.collections.length && (
            <p className="text-sm">No collections saved.</p>
          )}
          {query.data?.collections.map((c) => (
            <div key={c.id} className="space-y-2 rounded-lg border p-3">
              <p className="font-medium">{c.name}</p>
              <p className="text-sm text-muted-foreground">{c.description}</p>
              <p className="text-sm">
                {c.model_ids.join(", ") || "No models"} •{" "}
                {c.preload
                  ? "Preloads on startup"
                  : "Loads only when requested"}
              </p>
              <div className="flex flex-wrap gap-2">
                <Button
                  variant="outline"
                  disabled={action.isPending}
                  onClick={() => open(c)}
                >
                  Edit
                </Button>
                <Button
                  disabled={action.isPending || !c.model_ids.length}
                  onClick={() =>
                    action.mutate({
                      path: `/${encodeURIComponent(c.id)}/load`,
                      method: "POST",
                    })
                  }
                >
                  Load collection
                </Button>
                <Button
                  variant="outline"
                  disabled={action.isPending}
                  onClick={() => setRemove(c)}
                >
                  Remove collection
                </Button>
              </div>
            </div>
          ))}
        </QueryState>
        <Failure error={action.error} />
        <Dialog
          open={edit !== null}
          onOpenChange={(open) => {
            if (!open && !action.isPending) setEdit(null)
          }}
        >
          <DialogContent>
            <DialogHeader>
              <DialogTitle>{"Save collection"}</DialogTitle>
              <DialogDescription>
                Members: {edit?.model_ids.join(", ")}. Saving does not load
                models now.
              </DialogDescription>
            </DialogHeader>
            <FieldGroup>
              <Field>
                <FieldLabel htmlFor="collection-name">Name</FieldLabel>
                <Input
                  id="collection-name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </Field>
              <Field>
                <FieldLabel htmlFor="collection-description">
                  Description
                </FieldLabel>
                <Input
                  id="collection-description"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                />
              </Field>
              <Field orientation="horizontal">
                <Checkbox
                  id="collection-preload"
                  checked={preload}
                  disabled={action.isPending}
                  onCheckedChange={(checked) => setPreload(checked === true)}
                />
                <FieldLabel htmlFor="collection-preload">
                  Preload models on server startup
                </FieldLabel>
              </Field>
              <p className="text-sm text-muted-foreground">
                Enabling preload pins every member so it loads on startup and
                stays resident. Check the memory plan first. Disabling preload
                or removing the collection keeps existing model pins; change
                pins in each model's settings.
              </p>
            </FieldGroup>
            {selected.length > 0 && (
              <Button
                variant="outline"
                onClick={() =>
                  setEdit((c) => (c ? { ...c, model_ids: selected } : c))
                }
              >
                Replace members with selected models
              </Button>
            )}
            <Failure error={action.error} />
            <Button
              disabled={
                action.isPending || !name.trim() || !edit?.model_ids.length
              }
              onClick={() =>
                action.mutate({
                  path: `/${encodeURIComponent(edit?.id ?? "")}`,
                  method: "PUT",
                  body: {
                    name: name.trim(),
                    description,
                    model_ids: edit?.model_ids,
                    preload,
                  },
                })
              }
            >
              Save collection
            </Button>
          </DialogContent>
        </Dialog>
        <Dialog
          open={remove !== null}
          onOpenChange={(open) => {
            if (!open && !action.isPending) setRemove(null)
          }}
        >
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Remove {remove?.name}?</DialogTitle>
              <DialogDescription>
                This removes the collection definition. Model files stay in
                storage.
              </DialogDescription>
            </DialogHeader>
            <Failure error={action.error} />
            <Button
              variant="destructive"
              disabled={action.isPending}
              onClick={() =>
                action.mutate({
                  path: `/${encodeURIComponent(remove?.id ?? "")}`,
                  method: "DELETE",
                })
              }
            >
              Confirm removal
            </Button>
          </DialogContent>
        </Dialog>
      </CardContent>
    </Card>
  )
}
