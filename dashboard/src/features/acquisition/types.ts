export type HubModel = {
  repo_id: string
  name: string
  size?: number
  size_formatted?: string
  downloads?: number
  likes?: number
  params_formatted?: string | null
}
export type HubResults = {
  models?: HubModel[]
  trending?: HubModel[]
  popular?: HubModel[]
  total?: number
  hf_token_invalid?: boolean
}
export type HubInfo = HubModel & {
  model_card?: string
  description?: string
  files?: { name: string; size: number }[]
  tags?: string[]
  is_adapter?: boolean
  pipeline_tag?: string
  mlx_compatible?: boolean
  requires_conversion?: boolean
}
export type Capability = {
  available: boolean
  reason?: string | null
  adapter?: string
  output_name?: string
  target_format?: string
  requires_conversion?: boolean
  bits?: number[]
}
export type LocalModel = {
  id: string
  name: string
  path: string
  model_type: string
  format: string
  precision: string
  size: number
  conversion: Capability
  quantization: Capability
}
export type LocalCatalog = {
  models: LocalModel[]
  options: {
    oq_levels: number[]
    group_sizes: number[]
    dtypes: string[]
    exclusive_busy: boolean
  }
}
