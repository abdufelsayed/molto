import openapi from "@molto/contracts/openapi.json"

const managementRoutes = Object.entries(openapi.paths)
  .filter(([path]) => path.startsWith("/management/v1/"))
  .map(([path, methods]) => {
    const pattern = path
      .replace(/^\/management\/v1\//, "")
      .split(/(\{[^}]+\})/)
      .map((part) =>
        part.startsWith("{")
          ? ".+"
          : part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
      )
      .join("")
    return {
      pattern: new RegExp(`^${pattern}$`),
      methods: Object.keys(methods),
    }
  })

const clusterManagementRoutes: {
  method: string
  pattern: RegExp
  base: string
}[] = [
  {
    method: "GET",
    pattern: /^(devices|discovery\/health|pair\/join)$/,
    base: "/api/cluster",
  },
  {
    method: "POST",
    pattern: /^(devices\/manual|pair\/(approve|deny|join|join\/cancel))$/,
    base: "/api/cluster",
  },
  { method: "DELETE", pattern: /^devices\/[^/]+$/, base: "/api/cluster" },
  {
    method: "GET",
    pattern:
      /^(deployments|runtime|join-status|rdma-links|diagnostics|stage\/[^/]+)$/,
    base: "/admin/api/cluster",
  },
  {
    method: "POST",
    pattern:
      /^(node-budgets|models|catalogue|autoconfigure|peer-probe|stage|deployments|replan|join-keys|cuda-fabric\/verify|rdma-links\/verify|deployments\/[^/]+\/(load|unload))$/,
    base: "/admin/api/cluster",
  },
  {
    method: "DELETE",
    pattern: /^(deployments|join-keys)\/[^/]+$/,
    base: "/admin/api/cluster",
  },
]

export function operationTarget(method: string, path: string) {
  let decoded: string
  try {
    decoded = decodeURIComponent(path)
  } catch {
    return undefined
  }
  if (
    /[\\%?#]/.test(decoded) ||
    decoded
      .split("")
      .some(
        (character) =>
          character.charCodeAt(0) <= 32 || character.charCodeAt(0) === 127
      ) ||
    decoded.split("/").some((part) => !part || part === "." || part === "..")
  )
    return undefined
  if (decoded === "setup" || decoded.startsWith("setup/")) return undefined
  if (path.startsWith("cluster/")) {
    const suffix = path.slice("cluster/".length)
    const route = clusterManagementRoutes.find(
      (candidate) =>
        candidate.method === method && candidate.pattern.test(suffix)
    )
    return route ? `${route.base}/${suffix}` : undefined
  }
  return managementRoutes.some(
    (route) =>
      route.methods.includes(method.toLowerCase()) && route.pattern.test(path)
  )
    ? `/management/v1/${path}`
    : undefined
}
