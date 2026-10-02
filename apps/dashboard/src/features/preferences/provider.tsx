import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react"
import type { Dispatch, ReactNode, SetStateAction } from "react"
import { useSelector } from "@tanstack/react-store"
import { createPreferenceStore } from "./store"
import type { PreferenceKey, PreferenceValue } from "./store"

const PreferencesContext = createContext<ReturnType<
  typeof createPreferenceStore
> | null>(null)

export function PreferencesProvider({ children }: { children: ReactNode }) {
  const [preferences] = useState(createPreferenceStore)
  useEffect(() => {
    let storage: Storage | undefined
    try {
      storage = window.localStorage
    } catch {
      /* Some browsers deny storage access. */
    }
    preferences.hydrate(storage)
  }, [preferences])
  return <PreferencesContext value={preferences}>{children}</PreferencesContext>
}

export function usePreferences() {
  const preferences = useContext(PreferencesContext)
  if (!preferences) throw new Error("PreferencesProvider is required")
  return preferences
}

export function usePreferencesReady() {
  const { store } = usePreferences()
  return useSelector(store, (state) => state.ready)
}

export function usePreference<K extends PreferenceKey>(
  key: K,
  scope?: string
): [PreferenceValue<K>, Dispatch<SetStateAction<PreferenceValue<K>>>] {
  const preferences = usePreferences()
  const value = useSelector(preferences.store, () =>
    preferences.get(key, scope)
  )
  const setValue = useCallback(
    (change: SetStateAction<PreferenceValue<K>>) => {
      preferences.set(key, change, scope)
    },
    [preferences, key, scope]
  )
  return [value, setValue]
}
