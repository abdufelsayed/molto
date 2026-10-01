export type Summary = {
  requests: number
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  cached_tokens: number
  generation_tps: number | null
  prefill_tps: number | null
  average_request_seconds: number | null
}
export type Usage = {
  enabled: boolean
  available: boolean
  state: "available" | "disabled" | "unavailable"
  start: string
  end: string
  timezone: string
  dropped_requests: number
  totals: Summary | null
  models: (Summary & { model_id: string })[]
  daily?: (Summary & { date: string })[]
}
export type LiveRequest = {
  request_id?: string
  stage?: string
  kind?: string
  detail?: string
  elapsed?: number
  elapsed_seconds?: number
  queue_position?: number
  processed?: number
  total?: number
  generated_tokens?: number
  tokens_per_second?: number | null
}
export type Activity = {
  models: {
    id: string
    active_requests: number
    waiting_requests: number
    prefilling: LiveRequest[]
    generating: LiveRequest[]
    waiting: LiveRequest[]
    stage: string
    loading_elapsed_seconds?: number | null
    activities: LiveRequest[]
  }[]
  total_active_requests: number
  total_waiting_requests: number
  model_memory_used: number
  model_memory_max: number
  memory_pressure: Record<string, unknown> | null
  uptime_seconds: number | null
}
export type Versions = {
  engines: Record<
    string,
    {
      version: string | null
      commit: string | null
      url: string | null
      source: string | null
    }
  >
  python: string
  platform: string
}
export type Logs = {
  logs: string[] | string
  total_lines: number
  matched_lines: number
  log_file: string
  available_files: string[]
}
