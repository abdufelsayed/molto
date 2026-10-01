import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { PageTitle, QueryState } from "@/components/page-state"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { managementQuery } from "@/features/management/request"
import { ServerEditor, settingsGroups } from "./editor"
import { KeyManagement } from "./keys"
import { ConnectionPanel, IntegrationsPanel } from "./operations"
import type { ServerSettings } from "./types"

export function ServerSettingsPage() {
  const [active, setActive] = useState("connection")
  const settings = useQuery(
    managementQuery<ServerSettings>(["server", "settings"], "server/settings")
  )
  const defaults = useQuery(
    managementQuery<ServerSettings>(["server", "defaults"], "server/defaults")
  )
  return (
    <>
      <PageTitle
        title="Server settings"
        description="Configure oMLX, manage API access, and connect your clients."
      />
      <Tabs
        value={active}
        onValueChange={(value) => {
          if (typeof value === "string") setActive(value)
        }}
      >
        <div className="overflow-x-auto pb-1">
          <TabsList
            className="h-auto min-w-max"
            aria-label="Server settings sections"
          >
            <TabsTrigger value="connection" className="px-3 py-2">
              Connection
            </TabsTrigger>
            <TabsTrigger value="keys" className="px-3 py-2">
              API keys
            </TabsTrigger>
            {settingsGroups.map((group) => (
              <TabsTrigger
                key={group.id}
                value={group.id}
                className="px-3 py-2"
              >
                {group.title}
              </TabsTrigger>
            ))}
          </TabsList>
        </div>
        <TabsContent value="connection" keepMounted>
          <ConnectionPanel />
        </TabsContent>
        <TabsContent value="keys" keepMounted>
          <KeyManagement />
        </TabsContent>
        <TabsContent
          value={
            active === "connection" || active === "keys" ? "__settings" : active
          }
          keepMounted
          className="space-y-5"
        >
          <QueryState query={settings}>
            {settings.data && (
              <ServerEditor
                data={settings.data}
                defaults={defaults.data}
                active={active}
              />
            )}
          </QueryState>
          <div hidden={active !== "integrations"}>
            <IntegrationsPanel />
          </div>
        </TabsContent>
      </Tabs>
    </>
  )
}
