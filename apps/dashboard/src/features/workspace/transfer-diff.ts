export type ConfigurationBundle = {
  models: {
    id: string
    settings: Record<string, unknown>
    profiles: unknown[]
    policy: unknown
  }[]
  collections: unknown[]
}
export function configurationDiff(
  current: ConfigurationBundle,
  incoming: ConfigurationBundle
) {
  const changes: {
    model_id?: string
    field: string
    before?: unknown
    after?: unknown
  }[] = []
  for (const model of incoming.models) {
    const old = current.models.find((m) => m.id === model.id)
    for (const [field, after] of Object.entries(model.settings)) {
      const before = old?.settings[field]
      if (JSON.stringify(before) !== JSON.stringify(after))
        changes.push({ model_id: model.id, field, before, after })
    }
    for (const field of ["profiles", "policy"] as const) {
      if (JSON.stringify(old?.[field]) !== JSON.stringify(model[field]))
        changes.push({
          model_id: model.id,
          field,
          before: old?.[field],
          after: model[field],
        })
    }
  }
  if (
    JSON.stringify(current.collections) !== JSON.stringify(incoming.collections)
  )
    changes.push({
      field: "collections",
      before: current.collections,
      after: incoming.collections,
    })
  return changes
}
