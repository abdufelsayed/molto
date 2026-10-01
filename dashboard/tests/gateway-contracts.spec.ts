import { readFile } from "node:fs/promises"
import ts from "typescript"
import { expect, test } from "@playwright/test"

type Gateway = {
  connect: (request: Request) => Promise<Response>
  forward: (request: Request, path: string) => Promise<Response>
  session: (request: Request) => { key: string } | undefined
}

test("gateway forwards retained cluster paths and rejects unsupported operations", async () => {
  const source = await readFile("src/server/connection.server.ts", "utf8")
  const schema = await readFile("src/lib/openapi.json", "utf8")
  const code = ts.transpileModule(
    source.replace(
      'import openapi from "@/lib/openapi.json"',
      `const openapi = ${schema}`
    ),
    {
      compilerOptions: {
        target: ts.ScriptTarget.ES2022,
        module: ts.ModuleKind.ES2022,
      },
    }
  ).outputText
  const gateway = (await import(
    `data:text/javascript;base64,${Buffer.from(code).toString("base64")}`
  )) as Gateway
  const originalFetch = globalThis.fetch
  const calls: { url: string; authorization: string | null }[] = []
  globalThis.fetch = (url, options) => {
    calls.push({
      url:
        typeof url === "string" ? url : url instanceof URL ? url.href : url.url,
      authorization: new Headers(options?.headers).get("Authorization"),
    })
    return Promise.resolve(Response.json({ ok: true }))
  }
  const origin = "http://localhost:3000"
  try {
    const login = await gateway.connect(
      new Request(`${origin}/api/connection`, {
        method: "POST",
        headers: { origin },
        body: JSON.stringify({ key: "contract-key" }),
      })
    )
    const cookie = login.headers.get("set-cookie")!.split(";")[0]!
    const cases: [string, string, string][] = [
      ["GET", "cluster/devices", "/api/cluster/devices"],
      ["GET", "cluster/pair/join", "/api/cluster/pair/join"],
      ["POST", "cluster/pair/approve", "/api/cluster/pair/approve"],
      ["DELETE", "cluster/devices/node-a", "/api/cluster/devices/node-a"],
      ["GET", "cluster/deployments", "/admin/api/cluster/deployments"],
      ["GET", "cluster/stage/job-a", "/admin/api/cluster/stage/job-a"],
      ["POST", "cluster/catalogue", "/admin/api/cluster/catalogue"],
      ["POST", "cluster/replan", "/admin/api/cluster/replan"],
      [
        "POST",
        "cluster/deployments/plan-a/load",
        "/admin/api/cluster/deployments/plan-a/load",
      ],
      [
        "POST",
        "cluster/rdma-links/verify",
        "/admin/api/cluster/rdma-links/verify",
      ],
      [
        "DELETE",
        "cluster/join-keys/key-a",
        "/admin/api/cluster/join-keys/key-a",
      ],
      [
        "GET",
        "models/org%2Fmodel/options",
        "/management/v1/models/org%2Fmodel/options",
      ],
    ]
    for (const [method, path, target] of cases) {
      calls.length = 0
      const response = await gateway.forward(
        new Request(`${origin}/api/omlx/${path}`, {
          method,
          headers: { origin, cookie },
          ...(method !== "GET" ? { body: "{}" } : {}),
        }),
        path
      )
      expect(response.status, path).toBe(200)
      expect(calls).toHaveLength(1)
      expect(new URL(calls[0]!.url).pathname).toBe(target)
      expect(calls[0]!.authorization).toBe("Bearer contract-key")
    }
    for (const [method, path] of [
      ["GET", "server/restart"],
      ["POST", "state"],
      ["GET", "models/foo/%2e%2e/options"],
      ["GET", "models/%5c..%5cstate/options"],
      ["POST", "cluster/execute"],
    ]) {
      calls.length = 0
      const response = await gateway.forward(
        new Request(`${origin}/api/omlx/${path}`, {
          method,
          headers: { origin, cookie },
          ...(method !== "GET" ? { body: "{}" } : {}),
        }),
        path!
      )
      expect(response.status).toBe(404)
      expect(calls).toHaveLength(0)
    }
  } finally {
    globalThis.fetch = originalFetch
  }
})
