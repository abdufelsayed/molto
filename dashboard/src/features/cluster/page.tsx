import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useState, type ReactNode } from "react"
import { z } from "zod"
import { toast } from "sonner"
import { ApiError, errorMessage } from "@/features/management/api"
import { PageTitle } from "@/components/page-state"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field"
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectGroup,
  SelectItem,
} from "@/components/ui/select"
import { Evidence } from "./evidence"
import {
  clusterRequest,
  devicesSchema,
  inventorySchema,
  catalogueSchema,
  budgetsSchema,
  proposalSchema,
  stageSchema,
  deploymentsSchema,
  joinStatusSchema,
  rdmaSchema,
  evidenceSchema,
  inventoryHosts,
  membershipReplanBody,
  membershipHosts,
  replanPreviewSchema,
  type Proposal,
} from "./contracts"

function Section({
  title,
  description,
  children,
}: {
  title: string
  description: string
  children: ReactNode
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">{children}</CardContent>
    </Card>
  )
}
function Choice({
  label,
  value,
  onChange,
  options,
  disabled = false,
}: {
  disabled?: boolean
  label: string
  value: string
  onChange: (value: string) => void
  options: { value: string; label: string; disabled?: boolean }[]
}) {
  return (
    <Field>
      <FieldLabel>{label}</FieldLabel>
      <Select
        disabled={disabled || !options.length}
        value={value}
        onValueChange={(value) => {
          if (value !== null) onChange(value)
        }}
        items={options}
      >
        <SelectTrigger aria-label={label} className="w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            {options.map((option) => (
              <SelectItem
                key={option.value}
                value={option.value}
                disabled={option.disabled}
              >
                {option.label}
              </SelectItem>
            ))}
          </SelectGroup>
        </SelectContent>
      </Select>
    </Field>
  )
}
export function ClusterPage() {
  const client = useQueryClient()
  const devices = useQuery({
    queryKey: ["omlx", "cluster", "devices"],
    queryFn: ({ signal }) =>
      clusterRequest("devices", devicesSchema, { signal }),
    retry: false,
    refetchInterval: (query) => (query.state.error ? false : 5000),
  })
  const available = devices.isSuccess
  const deployments = useQuery({
    queryKey: ["omlx", "cluster", "deployments"],
    queryFn: ({ signal }) =>
      clusterRequest("deployments", deploymentsSchema, { signal }),
    enabled: available,
    retry: false,
    refetchInterval: 5000,
  })
  const runtime = useQuery({
    queryKey: ["omlx", "cluster", "runtime"],
    queryFn: ({ signal }) =>
      clusterRequest("runtime", evidenceSchema, { signal }),
    enabled: available,
    retry: false,
    refetchInterval: 5000,
  })
  const joining = useQuery({
    queryKey: ["omlx", "cluster", "join"],
    queryFn: ({ signal }) =>
      clusterRequest("pair/join", evidenceSchema, { signal }),
    enabled: available,
    retry: false,
    refetchInterval: 5000,
  })
  const enrollment = useQuery({
    queryKey: ["omlx", "cluster", "enrollment"],
    queryFn: ({ signal }) =>
      clusterRequest("join-status", joinStatusSchema, { signal }),
    enabled: available,
    retry: false,
  })
  const rdma = useQuery({
    queryKey: ["omlx", "cluster", "rdma"],
    queryFn: ({ signal }) =>
      clusterRequest("rdma-links", rdmaSchema, { signal }),
    enabled: available,
    retry: false,
  })
  const [ip, setIp] = useState("")
  const [port, setPort] = useState("8000")
  const [coordinator, setCoordinator] = useState("")
  const [codes, setCodes] = useState<Record<string, string>>({})
  const [excluded, setExcluded] = useState<string[]>([])
  const [roles, setRoles] = useState<Record<string, string>>({})
  const [strategy, setStrategy] = useState("auto")
  const [profile, setProfile] = useState("balanced")
  const [context, setContext] = useState("8192")
  const [inventory, setInventory] = useState<z.infer<
    typeof inventorySchema
  > | null>(null)
  const [catalogue, setCatalogue] = useState<z.infer<
    typeof catalogueSchema
  > | null>(null)
  const [modelId, setModelId] = useState("")
  const [proposal, setProposal] = useState<Proposal | null>(null)
  const [jobId, setJobId] = useState("")
  const [feedback, setFeedback] = useState<Record<string, unknown> | null>(null)
  const [joinIp, setJoinIp] = useState("")
  const [joinPort, setJoinPort] = useState("8000")
  const [scheme, setScheme] = useState("http")
  const [command, setCommand] = useState("")
  const [cudaA, setCudaA] = useState("")
  const [cudaB, setCudaB] = useState("")
  const [confirmation, setConfirmation] = useState("")
  const [deploymentMembers, setDeploymentMembers] = useState<
    Record<string, string[]>
  >({})
  const [replanStageJob, setReplanStageJob] = useState("")
  const [replan, setReplan] = useState<{
    body: Record<string, unknown>
    result: z.infer<typeof replanPreviewSchema>
  } | null>(null)
  const stage = useQuery({
    queryKey: ["omlx", "cluster", "stage", jobId],
    queryFn: ({ signal }) =>
      clusterRequest(`stage/${encodeURIComponent(jobId)}`, stageSchema, {
        signal,
      }),
    enabled: !!jobId,
    retry: false,
    refetchInterval: (query) =>
      ["completed", "failed"].includes(query.state.data?.status ?? "")
        ? false
        : 2000,
  })
  const membershipStage = useQuery({
    queryKey: ["omlx", "cluster", "stage", replanStageJob],
    queryFn: ({ signal }) =>
      clusterRequest(
        `stage/${encodeURIComponent(replanStageJob)}`,
        stageSchema,
        { signal }
      ),
    enabled: !!replanStageJob,
    retry: false,
    refetchInterval: (query) =>
      ["completed", "failed"].includes(query.state.data?.status ?? "")
        ? false
        : 2000,
  })
  const action = useMutation({
    mutationFn: async (work: () => Promise<void>) => work(),
    retry: false,
    onSuccess: async () => {
      toast.success("Cluster request completed.")
      await client.invalidateQueries({ queryKey: ["omlx"] })
    },
  })
  function run(work: () => Promise<void>) {
    action.reset()
    action.mutate(work)
  }
  function invalidatePlan() {
    setProposal(null)
    setJobId("")
    setReplan(null)
    setReplanStageJob("")
  }
  const selected = [
    ...(devices.data?.self ? [devices.data.self] : []),
    ...(devices.data?.paired ?? []),
  ].filter((device) => !excluded.includes(device.node_id))
  const hosts = inventoryHosts(selected, devices.data?.self?.node_id)
  const model = inventory?.models.find((item) => item.id === modelId)
  const modelFit = catalogue?.models.find((item) => item.model_id === modelId)
  const strategyUnavailable =
    strategy === "tensor"
      ? modelFit?.supports_tensor_parallel === false
      : strategy === "pipeline"
        ? modelFit?.supports_pipeline === false
        : false
  const plannerReady =
    selected.length >= 2 &&
    hosts.length === selected.length &&
    hosts.every(
      (host) =>
        selected.find((node) => node.node_id === host.node_id)?.addrs.length
    )
  const busy = action.isPending
  const post = async (path: string, body: unknown) => {
    const result = await clusterRequest(path, evidenceSchema, {
      method: "POST",
      body,
    })
    setFeedback(result)
    return result
  }
  async function measure(
    candidateHosts = hosts,
    priorRoles: Record<string, string> = {}
  ) {
    return (
      await clusterRequest("node-budgets", budgetsSchema, {
        method: "POST",
        body: {
          hosts: candidateHosts,
          roles: Object.fromEntries(
            candidateHosts.map((host) => [
              host.node_id,
              roles[host.node_id] ??
                priorRoles[host.node_id] ??
                (host.node_id === devices.data?.self?.node_id
                  ? "workstation"
                  : "headless"),
            ])
          ),
        },
      })
    ).nodes
  }
  async function preview() {
    if (!model) throw new Error("Select an inventoried model first.")
    const target = z.coerce.number().int().min(1).max(1048576).parse(context)
    const nodes = await measure()
    const source = model.locations.find(
      (location) => location.ssh === model.model_source
    )
    const result = await clusterRequest("autoconfigure", proposalSchema, {
      method: "POST",
      body: {
        nodes,
        hosts: hosts.map((host) => ({
          ...host,
          ips: selected
            .find((node) => node.node_id === host.node_id)
            ?.addrs.map((address) => address.ip),
          rdma: [],
        })),
        model_path: model.model_path,
        model_source: model.model_source,
        model_source_python:
          source?.python_executable ?? model.python_executable,
        path_map: Object.fromEntries(
          model.locations.map((location) => [
            location.node_id,
            location.model_path,
          ])
        ),
        strategy,
        execution_profile: profile,
        target_context_tokens: target,
        detect_transports: true,
        preflight: true,
        measure_performance: false,
      },
    })
    setProposal(result)
    setJobId("")
  }
  return (
    <>
      <PageTitle
        title="Cluster"
        description="Experimental distributed inference. Discover and trust nodes, preview a placement, then explicitly stage and activate it."
      />
      {devices.isPending && <p role="status">Checking cluster availability…</p>}
      {devices.isError && (
        <Alert>
          <AlertTitle>
            {devices.error instanceof ApiError &&
            [404, 503].includes(devices.error.status)
              ? "Cluster unavailable on this server"
              : "Cluster connection failed"}
          </AlertTitle>
          <AlertDescription>
            {errorMessage(devices.error)}
            <p>
              Standalone servers do not expose cluster management. Enable
              distributed inference on the server before enrolling nodes.
            </p>
            <Button variant="outline" onClick={() => void devices.refetch()}>
              Check availability
            </Button>
          </AlertDescription>
        </Alert>
      )}
      {available && (
        <fieldset disabled={busy} className="min-w-0 space-y-6">
          {action.isError && (
            <Alert variant="destructive">
              <AlertTitle>Request failed</AlertTitle>
              <AlertDescription>{errorMessage(action.error)}</AlertDescription>
            </Alert>
          )}
          {busy && <p role="status">Waiting for the cluster operation…</p>}
          <Section
            title="1. Discover and trust nodes"
            description="Discovery does not establish trust or prove liveness. Approve the six-digit code displayed on the joining node."
          >
            <div className="flex flex-wrap gap-2">
              <Button
                variant="outline"
                disabled={busy}
                onClick={() =>
                  run(async () => {
                    setFeedback(
                      await clusterRequest("discovery/health", evidenceSchema)
                    )
                  })
                }
              >
                Check discovery health
              </Button>
              <Button variant="outline" onClick={() => void devices.refetch()}>
                Refresh nodes
              </Button>
            </div>
            {!devices.data.paired.length && !devices.data.discovered.length && (
              <p>
                No peers found. Check Local Network permission in macOS System
                Settings or add a peer by IP.
              </p>
            )}
            {[
              ...(devices.data.self ? [devices.data.self] : []),
              ...devices.data.paired,
              ...devices.data.discovered,
            ].map((node) => {
              const self = node.node_id === devices.data.self?.node_id
              const paired =
                self ||
                devices.data.paired.some(
                  (item) => item.node_id === node.node_id
                )
              return (
                <div
                  key={node.node_id}
                  className="space-y-3 rounded-lg border p-4"
                >
                  <p className="font-medium">
                    {node.friendly_name ?? node.node_id}{" "}
                    {self
                      ? "· This server"
                      : paired
                        ? "· Paired"
                        : "· Discovered"}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {node.state ?? "Liveness unknown"} ·{" "}
                    {node.addrs
                      .map(
                        (address) =>
                          `${address.ip} (${address.if_type ?? "unknown link"})`
                      )
                      .join(", ") || "No address reported"}
                  </p>
                  <Evidence value={node.caps} />
                  {paired ? (
                    <div className="flex flex-wrap gap-2">
                      <Button
                        variant="outline"
                        disabled={busy}
                        onClick={() => {
                          setExcluded(
                            excluded.includes(node.node_id)
                              ? excluded.filter((id) => id !== node.node_id)
                              : [...excluded, node.node_id]
                          )
                          invalidatePlan()
                        }}
                      >
                        {excluded.includes(node.node_id)
                          ? "Include in plan"
                          : "Exclude from plan"}
                      </Button>
                      <Button
                        variant="outline"
                        disabled={
                          busy ||
                          !hosts.find((host) => host.node_id === node.node_id)
                        }
                        onClick={() =>
                          run(async () => {
                            await post("peer-probe", {
                              ssh: hosts.find(
                                (host) => host.node_id === node.node_id
                              )?.ssh,
                            })
                          })
                        }
                      >
                        Probe node
                      </Button>
                      {!self && (
                        <Button
                          variant="destructive"
                          disabled={busy}
                          onClick={() => {
                            const key = `unpair:${node.node_id}`
                            if (confirmation !== key) {
                              setConfirmation(key)
                              return
                            }
                            run(async () => {
                              setFeedback(
                                await clusterRequest(
                                  `devices/${encodeURIComponent(node.node_id)}`,
                                  evidenceSchema,
                                  { method: "DELETE" }
                                )
                              )
                              setConfirmation("")
                              invalidatePlan()
                            })
                          }}
                        >
                          {confirmation === `unpair:${node.node_id}`
                            ? "Confirm unpair"
                            : "Unpair"}
                        </Button>
                      )}
                    </div>
                  ) : (
                    <FieldGroup>
                      <Field>
                        <FieldLabel htmlFor={`code-${node.node_id}`}>
                          Pairing code for {node.friendly_name ?? node.node_id}
                        </FieldLabel>
                        <Input
                          id={`code-${node.node_id}`}
                          value={codes[node.node_id] ?? ""}
                          inputMode="numeric"
                          maxLength={6}
                          onChange={(event) =>
                            setCodes({
                              ...codes,
                              [node.node_id]: event.target.value,
                            })
                          }
                        />
                      </Field>
                      <div className="flex gap-2">
                        <Button
                          disabled={
                            busy || !/^\d{6}$/.test(codes[node.node_id] ?? "")
                          }
                          onClick={() =>
                            run(async () => {
                              await post("pair/approve", {
                                node_id: node.node_id,
                                code: codes[node.node_id],
                              })
                              setCodes({ ...codes, [node.node_id]: "" })
                            })
                          }
                        >
                          Approve pairing
                        </Button>
                        <Button
                          variant="outline"
                          disabled={busy}
                          onClick={() =>
                            run(async () => {
                              await post("pair/deny", { node_id: node.node_id })
                            })
                          }
                        >
                          Deny
                        </Button>
                      </div>
                    </FieldGroup>
                  )}
                </div>
              )
            })}
            <FieldGroup className="grid sm:grid-cols-2">
              <Field>
                <FieldLabel htmlFor="peer-ip">Peer IP</FieldLabel>
                <Input
                  id="peer-ip"
                  value={ip}
                  onChange={(event) => setIp(event.target.value)}
                />
              </Field>
              <Field>
                <FieldLabel htmlFor="peer-port">Peer HTTP port</FieldLabel>
                <Input
                  id="peer-port"
                  type="number"
                  min={1}
                  max={65535}
                  value={port}
                  onChange={(event) => setPort(event.target.value)}
                />
              </Field>
            </FieldGroup>
            <Button
              disabled={busy || !ip}
              onClick={() =>
                run(async () => {
                  await post("devices/manual", {
                    ip: z.ipv4().or(z.ipv6()).parse(ip),
                    port: z.coerce.number().int().min(1).max(65535).parse(port),
                  })
                })
              }
            >
              Add peer by IP
            </Button>
            <Field>
              <FieldLabel htmlFor="coordinator">
                Coordinator address to join
              </FieldLabel>
              <Input
                id="coordinator"
                placeholder="192.168.1.10:8000"
                value={coordinator}
                onChange={(event) => setCoordinator(event.target.value)}
              />
            </Field>
            <div className="flex gap-2">
              <Button
                disabled={busy || !coordinator}
                onClick={() =>
                  run(async () => {
                    await post("pair/join", { coordinator_addr: coordinator })
                  })
                }
              >
                Request to join
              </Button>
              <Button
                variant="outline"
                disabled={busy}
                onClick={() =>
                  run(async () => {
                    await post("pair/join/cancel", {})
                  })
                }
              >
                Cancel join
              </Button>
            </div>
            {joining.isError ? (
              <p>{errorMessage(joining.error)}</p>
            ) : (
              <Evidence value={joining.data} />
            )}
          </Section>
          <Section
            title="2. Inventory and placement"
            description="Choose trusted members and a model from their inventory. Measure live memory ceilings before calculating fit or generating a signed placement."
          >
            <FieldGroup>
              {selected.map((node) => (
                <Choice
                  disabled={busy}
                  key={node.node_id}
                  label={`Role for ${node.friendly_name ?? node.node_id}`}
                  value={
                    roles[node.node_id] ??
                    (node.node_id === devices.data.self?.node_id
                      ? "workstation"
                      : "headless")
                  }
                  onChange={(value) => {
                    setRoles({ ...roles, [node.node_id]: value })
                    invalidatePlan()
                  }}
                  options={[
                    { value: "workstation", label: "Workstation" },
                    { value: "headless", label: "Headless" },
                  ]}
                />
              ))}
            </FieldGroup>
            {!plannerReady && (
              <p className="text-sm text-muted-foreground">
                Planning requires at least two selected trusted nodes with
                reported SSH targets and collective addresses.
              </p>
            )}
            <Button
              disabled={busy || !hosts.length}
              onClick={() =>
                run(async () => {
                  setInventory(
                    await clusterRequest("models", inventorySchema, {
                      method: "POST",
                      body: { hosts },
                    })
                  )
                  setCatalogue(null)
                  invalidatePlan()
                })
              }
            >
              Read model inventory
            </Button>
            {inventory?.errors.map((item) => (
              <p key={item.node_id} className="text-destructive">
                {item.node_id}: {item.detail}
              </p>
            ))}
            {inventory && !inventory.models.length && (
              <p>No complete models reported by the selected nodes.</p>
            )}
            <FieldGroup>
              <Choice
                disabled={busy}
                label="Cluster model"
                value={modelId}
                onChange={(value) => {
                  setModelId(value)
                  invalidatePlan()
                }}
                options={(inventory?.models ?? []).map((item) => ({
                  value: item.id,
                  label: item.display_name ?? item.id,
                }))}
              />
              <Choice
                disabled={busy}
                label="Split strategy"
                value={strategy}
                onChange={(value) => {
                  setStrategy(value)
                  invalidatePlan()
                }}
                options={[
                  { value: "auto", label: "Automatic" },
                  {
                    value: "tensor",
                    label: "Tensor",
                    disabled: modelFit?.supports_tensor_parallel === false,
                  },
                  {
                    value: "pipeline",
                    label: "Pipeline",
                    disabled: modelFit?.supports_pipeline === false,
                  },
                ]}
              />
              <Choice
                disabled={busy}
                label="Execution profile"
                value={profile}
                onChange={(value) => {
                  setProfile(value)
                  invalidatePlan()
                }}
                options={[
                  { value: "interactive", label: "Interactive" },
                  { value: "balanced", label: "Balanced" },
                  { value: "throughput", label: "Throughput" },
                ]}
              />
              <Field>
                <FieldLabel htmlFor="cluster-context">
                  Target context tokens
                </FieldLabel>
                <Input
                  id="cluster-context"
                  type="number"
                  min={1}
                  max={1048576}
                  value={context}
                  onChange={(event) => {
                    setContext(event.target.value)
                    invalidatePlan()
                  }}
                />
              </Field>
            </FieldGroup>
            <p className="text-sm text-muted-foreground">
              The server reports unsupported model strategies and fabric
              blockers in the preview. Performance measurement is off; previews
              do not run an inference benchmark.
            </p>
            <div className="flex gap-2">
              <Button
                variant="outline"
                disabled={busy || !plannerReady || !inventory?.models.length}
                onClick={() =>
                  run(async () => {
                    setCatalogue(
                      await clusterRequest("catalogue", catalogueSchema, {
                        method: "POST",
                        body: {
                          nodes: await measure(),
                          models: inventory?.models.map((item) => ({
                            id: item.id,
                            model_path: item.model_path,
                            model_source: item.model_source ?? "127.0.0.1",
                            source_node_id: item.source_node_id ?? "",
                            model_source_python:
                              item.locations.find(
                                (location) => location.ssh === item.model_source
                              )?.python_executable ?? item.python_executable,
                          })),
                          execution_profile: profile,
                        },
                      })
                    )
                  })
                }
              >
                Assess model catalogue
              </Button>
              <Button
                disabled={
                  busy || !plannerReady || !model || strategyUnavailable
                }
                onClick={() => run(preview)}
              >
                Preview placement
              </Button>
            </div>
            {model && (
              <Evidence
                value={{
                  source: model.model_source,
                  locations: model.locations,
                }}
              />
            )}
            {strategyUnavailable && (
              <p className="text-destructive">
                The catalogue reports this split strategy is unsupported for
                this model. Select Automatic or another supported strategy.
              </p>
            )}
            {catalogue && <Evidence value={catalogue} />}
            {proposal && (
              <>
                <Evidence value={proposal} />
                <p>
                  Review the placement, fabric, preflight and staging evidence
                  above. Activation starts distributed workers and loads model
                  weights.
                </p>
                <div className="flex gap-2">
                  <Button
                    disabled={
                      busy ||
                      !(
                        proposal.ready_to_stage || proposal.ready_to_activate
                      ) ||
                      (!!jobId && stage.data?.status !== "failed")
                    }
                    onClick={() =>
                      run(async () => {
                        const job = await clusterRequest("stage", stageSchema, {
                          method: "POST",
                          body: {
                            activation: proposal.activation,
                            parallel: 2,
                          },
                        })
                        setJobId(job.job_id)
                      })
                    }
                  >
                    Stage approved placement
                  </Button>
                  <Button
                    disabled={
                      busy ||
                      (!proposal.ready_to_activate &&
                        stage.data?.status !== "completed")
                    }
                    onClick={() =>
                      run(async () => {
                        await post("deployments", proposal.activation)
                        invalidatePlan()
                      })
                    }
                  >
                    Activate reviewed placement
                  </Button>
                </div>
                {!proposal.ready_to_activate && !proposal.ready_to_stage && (
                  <p className="text-destructive">
                    Resolve the reported blockers and generate a new preview
                    before proceeding.
                  </p>
                )}
              </>
            )}
            {stage.isError && (
              <p className="text-destructive">
                Staging status unavailable: {errorMessage(stage.error)}. The
                server may have restarted. Re-preview the placement before
                staging again.
              </p>
            )}
            {stage.data && <Evidence value={stage.data} />}
          </Section>
          <Section
            title="3. Deployments and runtime"
            description="Unload keeps placement registered. Deactivate removes it. Replanning uses the included nodes and roles selected above, preserving existing member paths and fabric endpoints. Review the membership changes before approval reloads workers."
          >
            {deployments.isError && (
              <p className="text-destructive">
                {errorMessage(deployments.error)}
              </p>
            )}
            {!!deployments.data?.load_error && (
              <Evidence
                value={{ registry_error: deployments.data.load_error }}
              />
            )}
            {deployments.data?.deployments.length === 0 && (
              <p>No registered deployments.</p>
            )}
            {deployments.data?.deployments.map((deployment) => (
              <div
                key={deployment.deployment_id}
                className="space-y-3 rounded-lg border p-4"
              >
                <Evidence value={deployment} />
                <p className="text-sm">Members for this deployment replan</p>
                <div className="flex flex-wrap gap-2">
                  {[
                    ...(devices.data.self ? [devices.data.self] : []),
                    ...devices.data.paired,
                  ].map((node) => {
                    const ids =
                      deploymentMembers[deployment.deployment_id] ??
                      deployment.hosts.map((host) => host.node_id)
                    const included = ids.includes(node.node_id)
                    return (
                      <Button
                        key={node.node_id}
                        variant={included ? "secondary" : "outline"}
                        aria-pressed={included}
                        disabled={busy}
                        onClick={() => {
                          setDeploymentMembers({
                            ...deploymentMembers,
                            [deployment.deployment_id]: included
                              ? ids.filter((id) => id !== node.node_id)
                              : [...ids, node.node_id],
                          })
                          setReplan(null)
                          setReplanStageJob("")
                        }}
                      >
                        {included ? "Included" : "Add"}{" "}
                        {node.friendly_name ?? node.node_id}
                      </Button>
                    )
                  })}
                </div>
                <div className="flex flex-wrap gap-2">
                  {["load", "unload"].map((verb) => (
                    <Button
                      key={verb}
                      variant="outline"
                      disabled={busy}
                      onClick={() => {
                        const key = `${verb}:${deployment.deployment_id}`
                        if (confirmation !== key) {
                          setConfirmation(key)
                          return
                        }
                        run(async () => {
                          await post(
                            `deployments/${encodeURIComponent(deployment.deployment_id)}/${verb}`,
                            {}
                          )
                          setConfirmation("")
                        })
                      }}
                    >
                      {confirmation === `${verb}:${deployment.deployment_id}`
                        ? `Confirm ${verb}`
                        : verb === "load"
                          ? "Load weights"
                          : "Unload weights"}
                    </Button>
                  ))}
                  <Button
                    variant="outline"
                    disabled={
                      busy ||
                      (
                        deploymentMembers[deployment.deployment_id] ??
                        deployment.hosts.map((host) => host.node_id)
                      ).length < 2
                    }
                    onClick={() =>
                      run(async () => {
                        const allDevices = [
                          ...(devices.data.self ? [devices.data.self] : []),
                          ...devices.data.paired,
                        ]
                        const ids =
                          deploymentMembers[deployment.deployment_id] ??
                          deployment.hosts.map((host) => host.node_id)
                        const candidateHosts = membershipHosts(
                          deployment,
                          allDevices,
                          devices.data.self?.node_id,
                          ids
                        )
                        const body = membershipReplanBody({
                          deployment,
                          nodes: await measure(
                            candidateHosts,
                            Object.fromEntries(
                              deployment.assignments
                                .filter((assignment) => assignment.role)
                                .map((assignment) => [
                                  assignment.node_id,
                                  assignment.role!,
                                ])
                            )
                          ),
                          hosts: candidateHosts,
                          inventory: inventory?.models ?? [],
                          target_context_tokens: z.coerce
                            .number()
                            .int()
                            .min(1)
                            .max(1048576)
                            .parse(context),
                          execution_profile: profile,
                        })
                        setReplanStageJob("")
                        setReplan({
                          body,
                          result: await clusterRequest(
                            "replan",
                            replanPreviewSchema,
                            { method: "POST", body }
                          ),
                        })
                      })
                    }
                  >
                    Preview replan
                  </Button>
                  <Button
                    variant="destructive"
                    disabled={busy}
                    onClick={() => {
                      const key = `delete:${deployment.deployment_id}`
                      if (confirmation !== key) {
                        setConfirmation(key)
                        return
                      }
                      run(async () => {
                        setFeedback(
                          await clusterRequest(
                            `deployments/${encodeURIComponent(deployment.deployment_id)}`,
                            evidenceSchema,
                            { method: "DELETE" }
                          )
                        )
                        setConfirmation("")
                      })
                    }}
                  >
                    {confirmation === `delete:${deployment.deployment_id}`
                      ? "Confirm deactivate"
                      : "Deactivate"}
                  </Button>
                </div>
              </div>
            ))}
            {replan && (
              <>
                <Evidence value={replan.result} />
                <p>
                  Review added and removed members, memory roles, retained paths
                  and placement changes. Stage the signed placement when new
                  members need model files, then approve the reload.
                </p>
                <Button
                  variant="outline"
                  disabled={
                    busy ||
                    (!!replanStageJob &&
                      membershipStage.data?.status !== "failed")
                  }
                  onClick={() =>
                    run(async () => {
                      const activation = proposalSchema.parse({
                        plan: replan.result.plan,
                        activation: {
                          ...replan.body,
                          backend: replan.result.backend,
                          approved_placement:
                            replan.result.plan.placement_signature,
                          preflight: true,
                        },
                      }).activation
                      const job = await clusterRequest("stage", stageSchema, {
                        method: "POST",
                        body: { activation, parallel: 2 },
                      })
                      setReplanStageJob(job.job_id)
                    })
                  }
                >
                  Stage reviewed replan
                </Button>
                {membershipStage.data && (
                  <Evidence value={membershipStage.data} />
                )}
                {membershipStage.isError && (
                  <p className="text-destructive">
                    Replan staging status unavailable:{" "}
                    {errorMessage(membershipStage.error)}. Generate a new
                    preview before staging again.
                  </p>
                )}
                <Button
                  disabled={
                    busy ||
                    (!!replanStageJob &&
                      membershipStage.data?.status !== "completed")
                  }
                  onClick={() =>
                    run(async () => {
                      const plan = z
                        .object({
                          plan: z.object({
                            placement_signature: z.string().min(16),
                          }),
                        })
                        .parse(replan.result)
                      await post("replan", {
                        ...replan.body,
                        approved_placement: plan.plan.placement_signature,
                      })
                      setReplan(null)
                    })
                  }
                >
                  Approve replan and reload
                </Button>
              </>
            )}
            {runtime.isError ? (
              <p className="text-destructive">{errorMessage(runtime.error)}</p>
            ) : (
              <Evidence value={runtime.data} />
            )}
            <Button
              variant="outline"
              disabled={busy}
              onClick={() =>
                run(async () => {
                  const report = await clusterRequest(
                    "diagnostics",
                    evidenceSchema
                  )
                  setFeedback(report)
                  const url = URL.createObjectURL(
                    new Blob([JSON.stringify(report, null, 2)], {
                      type: "application/json",
                    })
                  )
                  const link = document.createElement("a")
                  link.href = url
                  link.download = "omlx-cluster-diagnostics.json"
                  link.click()
                  URL.revokeObjectURL(url)
                })
              }
            >
              Download cluster diagnostics
            </Button>
          </Section>
          <Section
            title="4. CUDA enrollment and fabric verification"
            description="Create a short-lived enrollment command for a private controller IP. Run it yourself on the intended worker. Verification performs an explicit bounded fabric probe."
          >
            <FieldGroup>
              <Field>
                <FieldLabel htmlFor="join-ip">
                  Controller private IPv4
                </FieldLabel>
                <Input
                  id="join-ip"
                  value={joinIp}
                  onChange={(event) => setJoinIp(event.target.value)}
                />
              </Field>
              <Field>
                <FieldLabel htmlFor="join-port">Controller port</FieldLabel>
                <Input
                  id="join-port"
                  type="number"
                  value={joinPort}
                  onChange={(event) => setJoinPort(event.target.value)}
                />
              </Field>
              <Choice
                disabled={busy}
                label="Controller scheme"
                value={scheme}
                onChange={setScheme}
                options={[
                  { value: "http", label: "HTTP" },
                  { value: "https", label: "HTTPS" },
                ]}
              />
            </FieldGroup>
            <Button
              disabled={busy || !joinIp}
              onClick={() =>
                run(async () => {
                  const result = await clusterRequest(
                    "join-keys",
                    z
                      .object({
                        command: z.string(),
                        join_id: z.string(),
                        expires_at: z.number(),
                      })
                      .passthrough(),
                    {
                      method: "POST",
                      body: {
                        controller_ip: z.ipv4().parse(joinIp),
                        controller_port: z.coerce
                          .number()
                          .int()
                          .min(1)
                          .max(65535)
                          .parse(joinPort),
                        scheme,
                        ttl_seconds: 1800,
                      },
                    }
                  )
                  setCommand(result.command)
                  setFeedback({
                    join_id: result.join_id,
                    expires_at: result.expires_at,
                  })
                })
              }
            >
              Generate enrollment command
            </Button>
            {command && (
              <div className="space-y-2">
                <p className="text-sm">
                  This command contains a temporary credential. It expires
                  within 30 minutes.
                </p>
                <code className="block rounded bg-muted p-3 break-all">
                  {command}
                </code>
                <Button
                  variant="outline"
                  onClick={() => {
                    void navigator.clipboard.writeText(command).then(
                      () => toast.success("Command copied."),
                      () => toast.error("Copy failed.")
                    )
                  }}
                >
                  Copy enrollment command
                </Button>
                <Button variant="outline" onClick={() => setCommand("")}>
                  Hide command
                </Button>
              </div>
            )}
            {enrollment.isError && (
              <p className="text-destructive">
                {errorMessage(enrollment.error)}
              </p>
            )}
            {!!enrollment.data?.load_error && (
              <Evidence
                value={{ enrollment_error: enrollment.data.load_error }}
              />
            )}
            {enrollment.data?.join_keys.map((key) => (
              <div key={key.join_id} className="flex items-center gap-3">
                <span>
                  {key.join_id} · {key.status} ·{" "}
                  {key.expires_at
                    ? new Date(key.expires_at * 1000).toLocaleString()
                    : "Expiry unknown"}
                </span>
                <Button
                  variant="outline"
                  disabled={busy || key.status !== "pending"}
                  onClick={() =>
                    run(async () => {
                      setFeedback(
                        await clusterRequest(
                          `join-keys/${encodeURIComponent(key.join_id)}`,
                          evidenceSchema,
                          { method: "DELETE" }
                        )
                      )
                      setCommand("")
                    })
                  }
                >
                  Revoke key
                </Button>
              </div>
            ))}
            <FieldGroup>
              <Choice
                disabled={busy}
                label="First CUDA worker"
                value={cudaA}
                onChange={setCudaA}
                options={(enrollment.data?.nodes ?? []).map((node) => ({
                  value: node.node_id,
                  label: node.node_id,
                }))}
              />
              <Choice
                disabled={busy}
                label="Second CUDA worker"
                value={cudaB}
                onChange={setCudaB}
                options={(enrollment.data?.nodes ?? []).map((node) => ({
                  value: node.node_id,
                  label: node.node_id,
                  disabled: node.node_id === cudaA,
                }))}
              />
            </FieldGroup>
            <Button
              disabled={busy || !cudaA || !cudaB || cudaA === cudaB}
              onClick={() =>
                run(async () => {
                  await post("cuda-fabric/verify", {
                    hosts: enrollment.data?.nodes
                      .filter(
                        (node) =>
                          node.node_id === cudaA || node.node_id === cudaB
                      )
                      .map((node) => ({
                        node_id: node.node_id,
                        ssh: node.ssh,
                      })),
                  })
                })
              }
            >
              Verify CUDA fabric
            </Button>
            {rdma.isError && (
              <p className="text-destructive">{errorMessage(rdma.error)}</p>
            )}
            {rdma.data && (
              <>
                <p>
                  RDMA helper:{" "}
                  {rdma.data.helper.available
                    ? "Available"
                    : (rdma.data.helper.reason ?? "Unavailable")}
                </p>
                {rdma.data.links.length === 0 && <p>No RDMA links reported.</p>}
                {rdma.data.links.map((link) => (
                  <div key={link.name} className="space-y-2 rounded border p-3">
                    <Evidence value={link} />
                    <Button
                      variant="outline"
                      disabled={
                        busy ||
                        !rdma.data.helper.available ||
                        !!link.in_use_by ||
                        link.verifying
                      }
                      onClick={() =>
                        run(async () => {
                          await post("rdma-links/verify", { link: link.name })
                        })
                      }
                    >
                      Verify RDMA link {link.name}
                    </Button>
                  </div>
                ))}
              </>
            )}
          </Section>
          {feedback && (
            <Section
              title="Last operation evidence"
              description="The server response describes the outcome. A completed request can still report failed checks."
            >
              <Evidence value={feedback} />
              <Button variant="outline" onClick={() => setFeedback(null)}>
                Dismiss evidence
              </Button>
            </Section>
          )}
        </fieldset>
      )}
    </>
  )
}
