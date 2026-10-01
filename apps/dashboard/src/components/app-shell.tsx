import { Link, useRouterState } from "@tanstack/react-router"
import { useQuery } from "@tanstack/react-query"
import { useTheme } from "next-themes"
import {
  BoxesIcon,
  CpuIcon,
  LayoutDashboardIcon,
  MoonIcon,
  RefreshCwIcon,
  Settings2Icon,
  DatabaseIcon,
  SunIcon,
  ActivityIcon,
  NetworkIcon,
  FlaskConicalIcon,
  ScrollTextIcon,
  ChartNoAxesCombinedIcon,
  PackagePlusIcon,
} from "lucide-react"
import { ConnectionDialog } from "@/components/connection-dialog"
import {
  stateQuery,
  connectionQuery,
  managementKey,
  useManagement,
} from "@/features/management/queries"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { managementQuery } from "@/features/management/request"
import { Separator } from "@/components/ui/separator"
import { QueryState } from "@/components/page-state"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar"

const navigation = [
  { title: "Overview", to: "/", icon: LayoutDashboardIcon },
  { title: "Studio", to: "/studio", icon: FlaskConicalIcon },
  { title: "Models", to: "/models", icon: BoxesIcon },
  { title: "Add model", to: "/add-model", icon: PackagePlusIcon },
  { title: "Activity", to: "/activity", icon: ActivityIcon },
  { title: "Monitoring", to: "/monitoring", icon: ChartNoAxesCombinedIcon },
  { title: "Logs", to: "/logs", icon: ScrollTextIcon },
  { title: "Cache", to: "/cache", icon: DatabaseIcon },
  { title: "Settings", to: "/settings", icon: Settings2Icon },
  { title: "Diagnostics", to: "/diagnostics", icon: FlaskConicalIcon },
  { title: "Cluster", to: "/cluster", icon: NetworkIcon },
] as const

export function AppShell({ children }: { children: React.ReactNode }) {
  const { api, queryClient } = useManagement()
  const access = useQuery(connectionQuery(api))
  const status = useQuery({
    ...stateQuery(api),
    enabled: !!access.data?.connected,
  })
  const pathname = useRouterState({
    select: (state) => state.location.pathname,
  })
  const title =
    navigation.find((item) => item.to === pathname)?.title ??
    (pathname.startsWith("/models/") ? "Model details" : "Molto")
  const { resolvedTheme, setTheme } = useTheme()
  const connected = !!status.data && !status.isError
  const connectionLabel =
    access.isPending || (access.data?.connected && status.isPending)
      ? "Connecting"
      : connected
        ? "Connected"
        : "Disconnected"

  return (
    <SidebarProvider>
      <DashboardSidebar
        pathname={pathname}
        connected={connected}
        connectionLabel={connectionLabel}
      />
      <SidebarInset className="min-w-0">
        <header className="flex h-16 shrink-0 items-center justify-between gap-2 border-b px-4 md:px-8">
          <div className="flex items-center gap-3">
            <SidebarTrigger />
            <Separator orientation="vertical" className="h-4" />
            <span className="text-sm">{title}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="hidden text-xs text-muted-foreground sm:inline">
              Refreshes while open
            </span>
            <Button
              variant="ghost"
              size="icon"
              aria-label="Refresh dashboard"
              onClick={() =>
                void queryClient.invalidateQueries({ queryKey: managementKey })
              }
            >
              <RefreshCwIcon data-icon="inline-start" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              aria-label="Toggle color theme"
              onClick={() =>
                setTheme(resolvedTheme === "dark" ? "light" : "dark")
              }
            >
              {resolvedTheme === "dark" ? (
                <SunIcon data-icon="inline-start" />
              ) : (
                <MoonIcon data-icon="inline-start" />
              )}
            </Button>
          </div>
        </header>
        <div
          className={
            pathname === "/studio"
              ? "flex h-[calc(100dvh-4rem)] min-h-0 w-full flex-col overflow-hidden"
              : "mx-auto flex w-full max-w-7xl flex-1 flex-col gap-6 p-4 md:p-8"
          }
        >
          <QueryState query={access}>{children}</QueryState>
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}

function DashboardSidebar({
  pathname,
  connected,
  connectionLabel,
}: {
  pathname: string
  connected: boolean
  connectionLabel: string
}) {
  const { setOpenMobile } = useSidebar()
  const server = useQuery({
    ...managementQuery<{ distributed_inference_active?: boolean }>(
      ["server-info"],
      "server/info"
    ),
    enabled: connected,
  })
  return (
    <Sidebar>
      <SidebarHeader className="px-4 py-6">
        <Link
          to="/"
          onClick={() => setOpenMobile(false)}
          className="flex items-center gap-3"
        >
          <div className="flex size-9 items-center justify-center rounded-xl bg-primary text-primary-foreground">
            <CpuIcon className="size-5" />
          </div>
          <div className="flex flex-col gap-0.5">
            <span className="text-sm font-semibold">Molto</span>
            <span className="text-xs text-muted-foreground">
              Local inference
            </span>
          </div>
        </Link>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Management</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {navigation
                .filter(
                  (item) =>
                    item.to !== "/cluster" ||
                    server.data?.distributed_inference_active
                )
                .map((item) => (
                  <SidebarMenuItem key={item.to}>
                    <SidebarMenuButton
                      isActive={
                        pathname === item.to ||
                        (item.to === "/models" &&
                          pathname.startsWith("/models/"))
                      }
                      render={
                        <Link
                          to={item.to}
                          onClick={() => setOpenMobile(false)}
                        />
                      }
                    >
                      <item.icon />
                      <span>{item.title}</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter className="gap-3 p-4">
        <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
          <span>Server connection</span>
          <Badge variant={connected ? "secondary" : "outline"}>
            {connectionLabel}
          </Badge>
        </div>
        <ConnectionDialog compact />
      </SidebarFooter>
    </Sidebar>
  )
}
