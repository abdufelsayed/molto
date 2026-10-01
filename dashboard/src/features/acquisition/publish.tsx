import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { managementQuery } from "@/features/management/request"
import { QueryState } from "@/components/page-state"
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { FieldGroup } from "@/components/ui/field"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog"
import { Choice, TextField, useAction, Failure } from "./controls"
import { Toggle } from "./prepare"
import type { LocalCatalog } from "./types"
import type { Operation } from "@/features/operations/page"
export function Publish() {
  const query = useQuery(
    managementQuery<LocalCatalog>(
      ["prepare-models"],
      "acquisition/prepare/models"
    )
  )
  const [path, setPath] = useState("")
  const [token, setToken] = useState("")
  const [namespace, setNamespace] = useState("")
  const [name, setName] = useState("")
  const [privateRepo, setPrivate] = useState(true)
  const [autoReadme, setAutoReadme] = useState(true)
  const [notice, setNotice] = useState(false)
  const [readmeSource, setReadmeSource] = useState("")
  const [confirmation, setConfirmation] = useState(false)
  const validate = useAction<{
    username: string
    orgs?: { name: string }[]
    can_write?: boolean
    valid?: boolean
  }>()
  const publish = useAction<Operation>()
  const repo = `${namespace}/${name}`
  const valid =
    validate.data?.valid !== false &&
    validate.data?.can_write !== false &&
    Boolean(validate.data)
  return (
    <Card>
      <CardHeader>
        <CardTitle>Publish a prepared model</CardTitle>
        <CardDescription>
          Upload a local checkpoint to Hugging Face with a write token. Review
          the exact destination and visibility before publishing.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-5">
        <QueryState query={query}>
          {query.data && (
            <>
              <FieldGroup>
                <Choice
                  label="Local model to publish"
                  value={path}
                  onChange={(next) => {
                    setPath(next)
                    validate.reset()
                    publish.reset()
                  }}
                  options={query.data.models.map((item) => ({
                    value: item.path,
                    label: item.name,
                  }))}
                />
                <TextField
                  label="Hugging Face write token"
                  type="password"
                  value={token}
                  onChange={(next) => {
                    setToken(next)
                    validate.reset()
                  }}
                  description="Kept in this form until submission. It is never saved in operation history."
                />
                <Failure error={validate.error} />
                <Button
                  variant="outline"
                  disabled={
                    !path ||
                    !token.trim() ||
                    validate.isPending ||
                    publish.isPending ||
                    query.isError
                  }
                  onClick={() =>
                    validate.mutate(
                      {
                        path: "acquisition/publish/validate",
                        body: { token, model_path: path },
                      },
                      { onSuccess: (result) => setNamespace(result.username) }
                    )
                  }
                >
                  {validate.isPending
                    ? "Validating…"
                    : "Validate token and source"}
                </Button>
                {valid && (
                  <>
                    <Choice
                      label="Repository namespace"
                      value={namespace}
                      onChange={setNamespace}
                      options={[
                        {
                          value: validate.data!.username,
                          label: validate.data!.username,
                        },
                        ...(validate.data!.orgs ?? []).map((org) => ({
                          value: org.name,
                          label: org.name,
                        })),
                      ]}
                    />
                    <TextField
                      label="Repository name"
                      value={name}
                      onChange={setName}
                    />
                    <Toggle
                      label="Private repository"
                      value={privateRepo}
                      onChange={setPrivate}
                    />
                    <Toggle
                      label="Generate model README"
                      value={autoReadme}
                      onChange={setAutoReadme}
                    />
                    <Choice
                      label="README source model, optional"
                      value={readmeSource}
                      onChange={setReadmeSource}
                      options={[
                        { value: "", label: "Use selected model" },
                        ...query.data.models.map((item) => ({
                          value: item.path,
                          label: item.name,
                        })),
                      ]}
                    />
                    <Toggle
                      label="Include redownload notice"
                      value={notice}
                      onChange={setNotice}
                    />
                    <Button
                      disabled={
                        publish.isPending ||
                        query.isError ||
                        !/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(name) ||
                        name.includes("..") ||
                        !namespace
                      }
                      onClick={() => {
                        publish.reset()
                        setConfirmation(true)
                      }}
                    >
                      Review publication
                    </Button>
                  </>
                )}
              </FieldGroup>
              {publish.data && (
                <Alert>
                  <AlertTitle>Publication queued</AlertTitle>
                  <AlertDescription>
                    Follow operation {publish.data.id} in Activity.
                  </AlertDescription>
                </Alert>
              )}
            </>
          )}
        </QueryState>
      </CardContent>
      <Dialog
        open={confirmation}
        onOpenChange={(open) => {
          if (!publish.isPending) setConfirmation(open)
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Publish {repo}?</DialogTitle>
            <DialogDescription>
              This uploads files from {path} to the{" "}
              {privateRepo ? "private" : "public"} Hugging Face repository{" "}
              {repo}. Existing repository files may be updated.{" "}
              {autoReadme
                ? "The server generates a README."
                : "The existing model README is used."}{" "}
              Uploads continue until the worker finishes.
            </DialogDescription>
          </DialogHeader>
          <Failure error={publish.error} />
          <DialogFooter>
            <Button
              variant="outline"
              disabled={publish.isPending}
              onClick={() => setConfirmation(false)}
            >
              Back
            </Button>
            <Button
              disabled={publish.isPending}
              onClick={() =>
                publish.mutate(
                  {
                    path: "acquisition/publish/start",
                    body: {
                      model_path: path,
                      repo_id: repo,
                      token,
                      private: privateRepo,
                      auto_readme: autoReadme,
                      redownload_notice: notice,
                      readme_source_path: readmeSource,
                    },
                  },
                  {
                    onSuccess: () => {
                      setToken("")
                      validate.reset()
                      setConfirmation(false)
                    },
                  }
                )
              }
            >
              {publish.isPending ? "Starting upload…" : "Confirm publication"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  )
}
