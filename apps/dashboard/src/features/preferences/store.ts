import { createStore } from "@tanstack/react-store"
import type { z } from "zod"
import { appPreferences } from "./app-schema"
import { studioPreferences } from "./studio-schema"

export const storageKey = "molto.ui.preferences"
export const preferenceSchemas = { ...appPreferences, ...studioPreferences }
export type PreferenceKey = keyof typeof preferenceSchemas
export type PreferenceValue<K extends PreferenceKey> = z.output<
  (typeof preferenceSchemas)[K]
>
type Values = Record<string, unknown>
type StorageAccess = Pick<Storage, "getItem" | "setItem">
const defaults = Object.fromEntries(
  Object.entries(preferenceSchemas).map(([key, schema]) => [
    key,
    schema.parse(undefined),
  ])
) as Values
const maxScopedEntries = 200
const maxStorageChars = 1_000_000

function address(key: PreferenceKey, scope?: string) {
  return scope === undefined ? key : `${key}:${scope}`
}

export function decodePreferences(raw: string | null): Values {
  if (!raw || raw.length > maxStorageChars) return {}
  try {
    const saved: unknown = JSON.parse(raw)
    if (
      !saved ||
      typeof saved !== "object" ||
      !("version" in saved) ||
      saved.version !== 1 ||
      !("values" in saved) ||
      !saved.values ||
      typeof saved.values !== "object" ||
      Array.isArray(saved.values)
    )
      return {}
    const values: Values = {}
    for (const [id, value] of Object.entries(saved.values)) {
      const key = id.split(":", 1)[0] as PreferenceKey
      if (!Object.hasOwn(preferenceSchemas, key) || id.length > 16_384) continue
      const result = preferenceSchemas[key].safeParse(value)
      if (result.success) values[id] = result.data
    }
    return prune(values)
  } catch {
    return {}
  }
}

function prune(values: Values): Values {
  const counts = new Map<string, number>()
  let chars = JSON.stringify({ version: 1, values: {} }).length
  let entries = 0
  return Object.fromEntries(
    Object.entries(values)
      .reverse()
      .filter(([id]) => {
        const size =
          JSON.stringify(id).length +
          1 +
          JSON.stringify(values[id]).length +
          (entries ? 1 : 0)
        if (chars + size > maxStorageChars) return false
        if (id.includes(":")) {
          const key = id.split(":", 1)[0]!
          const count = (counts.get(key) ?? 0) + 1
          counts.set(key, count)
          if (count > maxScopedEntries) return false
        }
        chars += size
        entries++
        return true
      })
      .reverse()
  )
}

// Each root creates its own store. Server requests never share browser state.
export function createPreferenceStore() {
  const store = createStore({ ready: false, values: {} as Values })
  let storage: StorageAccess | undefined
  function get<K extends PreferenceKey>(
    key: K,
    scope?: string
  ): PreferenceValue<K> {
    const id = address(key, scope)
    const values = store.get().values
    return (
      Object.hasOwn(values, id) ? values[id] : defaults[key]
    ) as PreferenceValue<K>
  }
  return {
    store,
    get,
    hydrate(access?: StorageAccess) {
      if (store.get().ready) return
      storage = access
      let values: Values = {}
      try {
        values = decodePreferences(storage?.getItem(storageKey) ?? null)
      } catch {
        /* Browser storage can be disabled. */
      }
      // Nothing is written while hydrating, so defaults cannot replace saved preferences.
      store.setState(() => ({ ready: true, values }))
    },
    set<K extends PreferenceKey>(
      key: K,
      change:
        | PreferenceValue<K>
        | ((value: PreferenceValue<K>) => PreferenceValue<K>),
      scope?: string
    ) {
      if (!store.get().ready || scope === "") return
      const previous = get(key, scope)
      const next = preferenceSchemas[key].parse(
        typeof change === "function" ? change(previous) : change
      )
      if (Object.is(previous, next)) return
      const id = address(key, scope)
      const values = { ...store.get().values }
      delete values[id]
      if (next !== undefined) values[id] = next
      const bounded = prune(values)
      store.setState((state) => ({ ...state, values: bounded }))
      try {
        storage?.setItem(
          storageKey,
          JSON.stringify({ version: 1, values: bounded })
        )
      } catch {
        /* Keep the current session usable if storage is full. */
      }
    },
    removeScope(scope: string) {
      if (!scope || !store.get().ready) return
      const values = Object.fromEntries(
        Object.entries(store.get().values).filter(
          ([id]) => !id.endsWith(`:${scope}`) && !id.includes(`:${scope}:`)
        )
      )
      store.setState((state) => ({ ...state, values }))
      try {
        storage?.setItem(storageKey, JSON.stringify({ version: 1, values }))
      } catch {
        /* Best-effort preference cleanup. */
      }
    },
  }
}
