import { useState } from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { toast } from "sonner"
import type { Profile } from "@/features/management/api"
import { errorMessage, settingsMessage } from "@/features/management/api"
import {
  profilesQuery,
  modelSettingsQuery,
  managementKey,
  useManagement,
} from "@/features/management/queries"
import { SettingsForm } from "@/features/settings/form"
import {
  modelOptionsQuery,
  optionFields,
} from "@/features/models/model-options"
import type { ModelOptions } from "@/features/models/model-options"
import { QueryState } from "@/components/page-state"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Field, FieldLabel } from "@/components/ui/field"
import { Switch } from "@/components/ui/switch"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogCancel,
} from "@/components/ui/alert-dialog"

export function ProfilesPanel({ modelId }: { modelId: string }) {
  const { api, queryClient } = useManagement()
  const query = useQuery(profilesQuery(api, modelId))
  const options = useQuery(modelOptionsQuery(modelId))
  const settings = useQuery(modelSettingsQuery(api, modelId))
  const [editing, setEditing] = useState<Profile | "new" | null>(null)
  const [deleting, setDeleting] = useState<Profile | null>(null)
  const [notice, setNotice] = useState("")
  const invalidate = () =>
    queryClient.invalidateQueries({ queryKey: managementKey })
  const apply = useMutation({
    mutationFn: (name: string) => api.applyProfile(modelId, name),
    onSuccess: (result) => setNotice(settingsMessage(result)),
    onSettled: invalidate,
  })
  const remove = useMutation({
    mutationFn: (name: string) => api.deleteProfile(modelId, name),
    onSuccess: () => {
      setDeleting(null)
      toast.success("Profile deleted.")
    },
    onSettled: invalidate,
  })
  return (
    <QueryState query={query}>
      <div className="flex justify-end">
        <Button
          disabled={!settings.data || !options.data || apply.isPending}
          onClick={() => setEditing("new")}
        >
          Create profile
        </Button>
      </div>
      {notice && (
        <Alert className="mt-4">
          <AlertTitle>Profile applied</AlertTitle>
          <AlertDescription>{notice}</AlertDescription>
        </Alert>
      )}
      {apply.isError && (
        <Alert variant="destructive" className="mt-4">
          <AlertTitle>Profile application needs attention</AlertTitle>
          <AlertDescription>{errorMessage(apply.error)}</AlertDescription>
        </Alert>
      )}
      {editing && (
        <ProfileEditor
          key={editing === "new" ? "new" : editing.name}
          modelId={modelId}
          profile={editing === "new" ? undefined : editing}
          current={settings.data?.settings ?? {}}
          options={options.data}
          done={() => {
            setEditing(null)
            void invalidate()
          }}
        />
      )}
      <div className="mt-4 grid gap-4">
        {query.data?.profiles.map((profile) => (
          <Card key={profile.name}>
            <CardHeader>
              <CardTitle className="flex flex-wrap gap-2">
                {profile.display_name || profile.name}
                {profile.expose_as_model && (
                  <Badge variant="outline">Exposed as API model</Badge>
                )}
              </CardTitle>
              <CardDescription>
                {typeof profile.description === "string"
                  ? profile.description
                  : profile.name}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <dl className="mb-4 grid gap-2 text-sm sm:grid-cols-2">
                {Object.entries(profile.settings).map(([key, value]) => (
                  <div key={key} className="flex gap-2">
                    <dt className="text-muted-foreground">
                      {key.replaceAll("_", " ")}
                    </dt>
                    <dd className="break-all">
                      {typeof value === "string"
                        ? value
                        : (JSON.stringify(value) ?? "Default")}
                    </dd>
                  </div>
                ))}
              </dl>
              <div className="flex flex-wrap gap-2">
                <Button
                  size="sm"
                  disabled={apply.isPending}
                  onClick={() => {
                    if (
                      window.confirm(
                        `Apply ${profile.name}? Omitted universal fields reset to defaults; specified engine settings may unload or reload this model.`
                      )
                    )
                      apply.mutate(profile.name)
                  }}
                >
                  Apply profile
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  disabled={apply.isPending}
                  onClick={() => setEditing(profile)}
                >
                  Edit profile
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={apply.isPending}
                  onClick={() => setDeleting(profile)}
                >
                  Delete profile
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
      {!query.data?.profiles.length && !editing && (
        <p className="py-12 text-center text-muted-foreground">
          No saved profiles. Create one to keep a named model configuration.
        </p>
      )}
      <AlertDialog
        open={!!deleting}
        onOpenChange={(open) => {
          if (!open && !remove.isPending) setDeleting(null)
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              Delete profile {deleting?.name}?
            </AlertDialogTitle>
            <AlertDialogDescription>
              The saved profile will be removed. The model's current settings
              remain applied.
            </AlertDialogDescription>
          </AlertDialogHeader>
          {remove.isError && (
            <Alert variant="destructive">
              <AlertDescription>{errorMessage(remove.error)}</AlertDescription>
            </Alert>
          )}
          <AlertDialogFooter>
            <AlertDialogCancel disabled={remove.isPending}>
              Cancel
            </AlertDialogCancel>
            <Button
              variant="destructive"
              disabled={remove.isPending}
              onClick={() => {
                if (deleting) remove.mutate(deleting.name)
              }}
            >
              Confirm delete
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </QueryState>
  )
}
function ProfileEditor({
  modelId,
  profile,
  current,
  options,
  done,
}: {
  modelId: string
  profile?: Profile
  options?: ModelOptions
  current: Record<string, unknown>
  done: () => void
}) {
  const { api } = useManagement()
  const [name, setName] = useState(profile?.name ?? "")
  const [display, setDisplay] = useState(profile?.display_name ?? "")
  const [description, setDescription] = useState(
    typeof profile?.description === "string" ? profile.description : ""
  )
  const [expose, setExpose] = useState(profile?.expose_as_model ?? false)
  const [apiName, setApiName] = useState(
    typeof profile?.api_name === "string" ? profile.api_name : ""
  )
  const [values] = useState(
    () =>
      profile?.settings ??
      Object.fromEntries(
        (options?.fields.filter((field) => field.profile) ?? []).map(
          (field) => [field.key, current[field.key] ?? null]
        )
      )
  )
  async function save(patch: Record<string, unknown>) {
    if (!name.trim()) throw new Error("Enter a profile name.")
    const settings = { ...values, ...patch }
    const metadata = {
      display_name: display,
      description: description || null,
      expose_as_model: expose,
      api_name: expose ? apiName || null : null,
      settings,
    }
    if (profile)
      return api.updateProfile(modelId, profile.name, {
        ...metadata,
        new_name: name.trim(),
      })
    return api.createProfile(modelId, {
      ...metadata,
      name: name.trim(),
    })
  }
  return (
    <Card className="mt-4">
      <CardHeader>
        <CardTitle>{profile ? "Edit profile" : "Create profile"}</CardTitle>
        <CardDescription>
          New profiles capture all eligible current settings. Applying a profile
          restores omitted universal settings to their defaults.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-5">
        <div className="grid gap-4 sm:grid-cols-2">
          <Field>
            <FieldLabel htmlFor="profile-name">Profile name</FieldLabel>
            <Input
              id="profile-name"
              value={name}
              onChange={(event) => setName(event.target.value)}
            />
          </Field>
          <Field>
            <FieldLabel htmlFor="profile-display">Display name</FieldLabel>
            <Input
              id="profile-display"
              value={display}
              onChange={(event) => setDisplay(event.target.value)}
            />
          </Field>
          <Field>
            <FieldLabel htmlFor="profile-description">Description</FieldLabel>
            <Input
              id="profile-description"
              value={description}
              onChange={(event) => setDescription(event.target.value)}
            />
          </Field>
          <Field>
            <FieldLabel htmlFor="profile-expose">
              Expose as API model
            </FieldLabel>
            <Switch
              id="profile-expose"
              checked={expose}
              onCheckedChange={setExpose}
            />
          </Field>
          {expose && (
            <Field>
              <FieldLabel htmlFor="profile-api-name">API model name</FieldLabel>
              <Input
                id="profile-api-name"
                placeholder="Server default"
                value={apiName}
                onChange={(event) => setApiName(event.target.value)}
              />
            </Field>
          )}
        </div>
        <Alert>
          <AlertTitle>Application readback</AlertTitle>
          <AlertDescription>
            Omitted universal settings reset to defaults when this profile is
            applied. All eligible fields are shown below; Default means an
            explicit reset. Engine fields omitted from older profiles remain
            unchanged.
          </AlertDescription>
        </Alert>
        <SettingsForm
          values={values}
          fields={options ? optionFields(options, "profile") : []}
          save={save}
          describe={() => "Profile saved."}
          alwaysSave
          saved={async () => {
            toast.success("Profile saved.")
            done()
          }}
        />
        <Button
          variant="ghost"
          onClick={() => {
            if (
              window.confirm(
                "Close this editor and discard unsaved profile changes?"
              )
            )
              done()
          }}
        >
          Close editor
        </Button>
      </CardContent>
    </Card>
  )
}
