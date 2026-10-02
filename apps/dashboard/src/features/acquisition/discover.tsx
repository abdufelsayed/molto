import { usePreference } from "@/features/preferences/provider"
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
import { FieldGroup } from "@/components/ui/field"
import { Button } from "@/components/ui/button"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { TextField, Choice, Failure, useAction } from "./controls"
import type { HubInfo, HubResults } from "./types"
import { bytes, count } from "@/lib/format"
import type { Operation } from "@/features/operations/page"
export function Discover() {
  const [provider, setProvider] = usePreference("discover.provider")
  const [search, setSearch] = usePreference("discover.search")
  const [sort, setSort] = usePreference("discover.sort")
  const [mlx, setMlx] = usePreference("discover.mlx")
  const [minimumSize, setMinimumSize] = usePreference("discover.minimumSize")
  const [maximumSize, setMaximumSize] = usePreference("discover.maximumSize")
  const [minimumParams, setMinimumParams] = usePreference(
    "discover.minimumParams"
  )
  const [maximumParams, setMaximumParams] = usePreference(
    "discover.maximumParams"
  )
  const [sizeSort, setSizeSort] = usePreference("discover.sizeSort")
  const [budget, setBudget] = usePreference("discover.budget")
  const [request, setRequest] = useState<{
    provider: string
    path: string
  } | null>(null)
  const [repo, setRepo] = useState("")
  const [selected, setSelected] = useState<{
    provider: string
    repo: string
  } | null>(null)
  const [token, setToken] = useState("")
  const results = useQuery({
    ...managementQuery<HubResults>(
      ["hub-search", request],
      request?.path ?? "acquisition/hf/recommended"
    ),
    enabled: request !== null,
  })
  const info = useQuery({
    ...managementQuery<HubInfo>(
      ["hub-info", selected],
      `acquisition/${selected?.provider ?? "hf"}/info?repo_id=${encodeURIComponent(selected?.repo ?? "")}`
    ),
    enabled: selected !== null,
  })
  const download = useAction<Operation>()
  const searchHub = (recommended: boolean) => {
    const parameters = new URLSearchParams(
      recommended
        ? {
            max_memory_bytes: String(Math.round(Number(budget) * 1024 ** 3)),
            mlx_only: mlx,
          }
        : { query: search, sort, mlx_only: mlx }
    )
    if (!recommended && provider === "hf") {
      for (const [name, value, multiplier] of [
        ["min_size", minimumSize, 1024 ** 3],
        ["max_size", maximumSize, 1024 ** 3],
        ["min_params", minimumParams, 1e9],
        ["max_params", maximumParams, 1e9],
      ] as const) {
        if (value.trim())
          parameters.set(name, String(Math.round(Number(value) * multiplier)))
      }
      if (sizeSort !== "off") {
        parameters.set("sort_by_size", "true")
        parameters.set("sort_ascending", String(sizeSort === "ascending"))
      }
    }
    setRequest({
      provider,
      path: `acquisition/${provider}/${recommended ? "recommended" : "search"}?${parameters}`,
    })
  }
  const validFilters =
    provider !== "hf" ||
    [minimumSize, maximumSize, minimumParams, maximumParams].every(
      (value) =>
        !value.trim() || (Number.isFinite(Number(value)) && Number(value) >= 0)
    )

  return (
    <div className="grid gap-6">
      <Card>
        <CardHeader>
          <CardTitle>1. Find a checkpoint</CardTitle>
          <CardDescription>
            Search a hub or inspect a repository ID. Hub metadata helps you
            choose; local compatibility is checked after downloading.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          <FieldGroup className="grid gap-4 sm:grid-cols-2">
            <Choice
              label="Hub"
              value={provider}
              onChange={setProvider}
              options={[
                { value: "hf", label: "Hugging Face" },
                { value: "ms", label: "ModelScope" },
              ]}
            />
            <TextField
              label="Search models"
              value={search}
              onChange={setSearch}
            />
            <Choice
              label="Sort"
              value={sort}
              onChange={setSort}
              options={[
                "trending",
                "downloads",
                "created",
                "updated",
                "likes",
              ].map((value) => ({ value, label: value }))}
            />
            <Choice
              label="Checkpoint format"
              value={mlx}
              onChange={setMlx}
              options={[
                { value: "true", label: "MLX checkpoints" },
                {
                  value: "false",
                  label: "All checkpoints, may require conversion",
                },
              ]}
            />
            <TextField
              label="Recommendation size limit in GiB"
              value={budget}
              onChange={setBudget}
              type="number"
              min={0.1}
            />
          </FieldGroup>
          {provider === "hf" && (
            <details>
              <summary className="cursor-pointer text-sm font-medium">
                Hugging Face size and parameter filters
              </summary>
              <FieldGroup className="mt-4 grid gap-4 sm:grid-cols-2">
                <TextField
                  label="Minimum checkpoint size in GiB"
                  value={minimumSize}
                  onChange={setMinimumSize}
                  type="number"
                  min={0}
                />
                <TextField
                  label="Maximum checkpoint size in GiB"
                  value={maximumSize}
                  onChange={setMaximumSize}
                  type="number"
                  min={0}
                />
                <TextField
                  label="Minimum parameters in billions"
                  value={minimumParams}
                  onChange={setMinimumParams}
                  type="number"
                  min={0}
                />
                <TextField
                  label="Maximum parameters in billions"
                  value={maximumParams}
                  onChange={setMaximumParams}
                  type="number"
                  min={0}
                />
                <Choice
                  label="Sort by checkpoint size"
                  value={sizeSort}
                  onChange={setSizeSort}
                  options={[
                    { value: "off", label: "Use selected sort" },
                    { value: "ascending", label: "Smallest first" },
                    { value: "descending", label: "Largest first" },
                  ]}
                />
              </FieldGroup>
            </details>
          )}
          <div className="flex flex-wrap gap-2">
            <Button
              disabled={results.isFetching || !validFilters}
              onClick={() => searchHub(false)}
            >
              Search
            </Button>
            <Button
              variant="outline"
              disabled={results.isFetching || !(Number(budget) > 0)}
              onClick={() => searchHub(true)}
            >
              Find recommended models
            </Button>
          </div>
          {request && (
            <QueryState query={results}>
              {results.data && (
                <>
                  <p className="text-sm text-muted-foreground">
                    Results from{" "}
                    {request.provider === "hf" ? "Hugging Face" : "ModelScope"}
                  </p>
                  {results.data.hf_token_invalid && (
                    <Alert>
                      <AlertTitle>Configured hub token was rejected</AlertTitle>
                      <AlertDescription>
                        Public results may still be available. Update the token
                        in server settings or enter a token for this download.
                      </AlertDescription>
                    </Alert>
                  )}
                  <div className="grid gap-3 sm:grid-cols-2">
                    {(
                      results.data.models ?? [
                        ...(results.data.trending ?? []),
                        ...(results.data.popular ?? []),
                      ]
                    )
                      .filter(
                        (item, index, items) =>
                          items.findIndex(
                            (other) => other.repo_id === item.repo_id
                          ) === index
                      )
                      .map((model) => (
                        <Card key={model.repo_id}>
                          <CardContent className="grid gap-2 pt-4">
                            <p className="font-medium break-all">
                              {model.repo_id}
                            </p>
                            <p className="text-sm text-muted-foreground">
                              {model.size ? bytes(model.size) : "Size unknown"}{" "}
                              · {count(model.downloads ?? 0)} downloads
                              {model.params_formatted
                                ? ` · ${model.params_formatted} parameters`
                                : ""}
                            </p>
                            <Button
                              variant="outline"
                              onClick={() => {
                                setRepo(model.repo_id)
                                setSelected({
                                  provider: request.provider,
                                  repo: model.repo_id,
                                })
                                download.reset()
                              }}
                            >
                              Inspect checkpoint
                            </Button>
                          </CardContent>
                        </Card>
                      ))}
                  </div>
                  {!(
                    results.data.models?.length ||
                    results.data.trending?.length ||
                    results.data.popular?.length
                  ) && (
                    <p className="text-sm text-muted-foreground">
                      No models matched. Try another query or include all
                      checkpoint formats.
                    </p>
                  )}
                </>
              )}
            </QueryState>
          )}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>2. Inspect and download</CardTitle>
          <CardDescription>
            The server downloads into its configured model storage. A full
            precision source may need conversion before it can run.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          <TextField
            label="Repository ID"
            value={repo}
            onChange={setRepo}
            description="Use owner/model, for example mlx-community/Qwen3-4B-4bit."
          />
          <Button
            variant="outline"
            disabled={!/^[^/\s]+\/[^/\s]+$/.test(repo) || info.isFetching}
            onClick={() => {
              setSelected({ provider, repo })
              download.reset()
            }}
          >
            Inspect repository
          </Button>
          {selected && (
            <QueryState query={info}>
              {info.data && (
                <>
                  <p className="font-medium break-all">
                    {selected.provider === "hf" ? "Hugging Face" : "ModelScope"}{" "}
                    / {info.data.repo_id}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    <Badge variant="outline">
                      {info.data.size ? bytes(info.data.size) : "Size unknown"}
                    </Badge>
                    {info.data.pipeline_tag && (
                      <Badge variant="outline">{info.data.pipeline_tag}</Badge>
                    )}
                    <Badge variant="outline">
                      {info.data.mlx_compatible === true
                        ? "MLX compatible"
                        : info.data.requires_conversion === true
                          ? "Needs conversion"
                          : "Compatibility checked locally after download"}
                    </Badge>
                  </div>
                  {info.data.is_adapter && (
                    <Alert>
                      <AlertTitle>Adapter checkpoint</AlertTitle>
                      <AlertDescription>
                        This repository requires a separate base model.
                        Downloading it does not produce a standalone model.
                      </AlertDescription>
                    </Alert>
                  )}
                  {info.data.files?.length ? (
                    <details>
                      <summary className="cursor-pointer text-sm">
                        Repository files
                      </summary>
                      <ul className="max-h-64 overflow-auto text-sm">
                        {info.data.files.map((file) => (
                          <li
                            key={file.name}
                            className="flex justify-between gap-4 py-1"
                          >
                            <span className="break-all">{file.name}</span>
                            <span className="shrink-0">{bytes(file.size)}</span>
                          </li>
                        ))}
                      </ul>
                    </details>
                  ) : null}
                  {info.data.model_card && (
                    <details>
                      <summary className="cursor-pointer text-sm">
                        Model card
                      </summary>
                      <div className="max-h-80 overflow-auto text-sm whitespace-pre-wrap">
                        {info.data.model_card}
                      </div>
                    </details>
                  )}
                  <TextField
                    label="Download token, optional"
                    type="password"
                    value={token}
                    onChange={setToken}
                    description="Used for this request only. Leave empty to use the server's current hub configuration."
                  />
                  <Failure error={download.error} />
                  <Button
                    disabled={download.isPending || info.isError}
                    onClick={() =>
                      download.mutate(
                        {
                          path: `acquisition/${selected.provider}/downloads`,
                          body: { repo_id: selected.repo, token },
                        },
                        { onSuccess: () => setToken("") }
                      )
                    }
                  >
                    {download.isPending
                      ? "Starting download…"
                      : "Download checkpoint"}
                  </Button>
                  {download.data && (
                    <Alert>
                      <AlertTitle>Download queued</AlertTitle>
                      <AlertDescription>
                        Follow operation {download.data.id} in Activity. You can
                        leave this page while the server downloads.
                      </AlertDescription>
                    </Alert>
                  )}
                </>
              )}
            </QueryState>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
