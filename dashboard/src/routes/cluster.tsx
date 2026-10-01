import { createFileRoute } from "@tanstack/react-router"
import { ClusterPage } from "@/features/cluster/page"
export const Route = createFileRoute("/cluster")({ component: ClusterPage })
