import { expect, test } from "@playwright/test"

const authorization = "Bearer inference-sub-key"
const upstream = "http://127.0.0.1:8765"

test.beforeEach(async ({ request }) => {
  expect((await request.post(`${upstream}/__test__/reset`)).ok()).toBe(true)
})

test("built public listener forwards client inference auth without a dashboard session", async ({
  request,
  baseURL,
}) => {
  expect((await request.get("/v1/models")).status()).toBe(401)
  const models = await request.get("/v1/models?fixture=client", {
    headers: { Authorization: authorization },
  })
  expect(models.status()).toBe(200)
  expect(await models.json()).toMatchObject({
    authorization,
    cookie: null,
    query: "fixture=client",
  })
  expect((await request.get("/api/molto/state")).status()).toBe(401)
  expect(
    (
      await request.get("/management/v1/state", {
        headers: { Authorization: authorization },
      })
    ).status()
  ).toBe(404)
  expect(
    (
      await request.post("/api/connection", {
        headers: { Origin: baseURL! },
        data: { key: "dashboard-test-key" },
      })
    ).ok()
  ).toBe(true)
  const withSession = await request.get("/v1/models", {
    headers: { Authorization: authorization },
  })
  expect(await withSession.json()).toMatchObject({
    authorization,
    cookie: null,
  })
  expect((await request.get("/v1/models")).status()).toBe(401)
})

test("built public listener preserves multipart bytes, binary responses and HTTP errors", async ({
  request,
}) => {
  const multipart = Buffer.concat([
    Buffer.from(
      '--fixture-boundary\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n'
    ),
    Buffer.from([0, 255, 1, 128]),
    Buffer.from("\r\n--fixture-boundary--\r\n"),
  ])
  const echoed = await request.post("/v1/audio/transcriptions", {
    headers: {
      Authorization: authorization,
      "Content-Type": "multipart/form-data; boundary=fixture-boundary",
    },
    data: multipart,
  })
  expect(echoed.status()).toBe(200)
  expect(echoed.headers()["x-fixture-upstream"]).toBe("multipart")
  expect(await echoed.body()).toEqual(multipart)
  const audio = await request.post("/v1/audio/speech", {
    headers: { Authorization: authorization },
    data: { input: "fixture" },
  })
  expect(audio.headers()["content-type"]).toContain("audio/wav")
  expect(await audio.body()).toEqual(Buffer.from([0, 255, 1, 128]))
  const error = await request.post("/v1/embeddings", {
    headers: { Authorization: authorization },
    data: { input: "fixture" },
  })
  expect(error.status()).toBe(422)
  expect(error.headers()["x-fixture-upstream"]).toBe("error")
  expect(await error.json()).toEqual({
    error: { message: "Synthetic model unavailable", type: "fixture_error" },
  })
  const metrics = await (
    await request.get(`${upstream}/__test__/proxy-metrics`)
  ).json()
  expect(metrics.requests[0]).toMatchObject({
    authorization,
    content_type: "multipart/form-data; boundary=fixture-boundary",
    bytes: [...multipart],
  })
})

test("built public listener streams the first SSE chunk and propagates cancellation", async ({
  page,
  request,
}) => {
  await page.goto("/")
  const chunk = await page.evaluate(async () => {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 5000)
    try {
      const response = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: "Bearer inference-sub-key",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ stream: true }),
        signal: controller.signal,
      })
      if (!response.ok || !response.body)
        throw new Error(`Stream HTTP ${response.status}`)
      const first = await response.body.getReader().read()
      controller.abort()
      return new TextDecoder().decode(first.value)
    } finally {
      clearTimeout(timeout)
      controller.abort()
    }
  })
  expect(chunk).toContain('data: {"fixture":"first"}\n\n')
  await expect
    .poll(
      async () =>
        (await (await request.get(`${upstream}/__test__/proxy-metrics`)).json())
          .streams_cancelled
    )
    .toBe(1)
})

test("built public realtime route preserves immediate start auth, binary frames and both close directions", async ({
  page,
  request,
}) => {
  await page.goto("/")
  const result = await page.evaluate(async () => {
    async function exchange(key: string, upstreamClose: boolean) {
      return await new Promise<{
        frames: (string | number[])[]
        code: number
        reason: string
      }>((resolve, reject) => {
        const socket = new WebSocket(
          `${location.origin.replace(/^http/, "ws")}/v1/audio/transcriptions/realtime`
        )
        socket.binaryType = "arraybuffer"
        const frames: (string | number[])[] = []
        const timer = setTimeout(() => {
          socket.close()
          reject(new Error("Realtime fixture timeout"))
        }, 5000)
        socket.onopen = () => {
          socket.send(
            JSON.stringify({
              type: "start",
              api_key: key,
              model: "fixture",
              language: "en",
            })
          )
          if (key === "inference-sub-key")
            socket.send(new Uint8Array([0, 255, 1, 128]))
        }
        socket.onmessage = (event) => {
          frames.push(
            typeof event.data === "string"
              ? event.data
              : [...new Uint8Array(event.data as ArrayBuffer)]
          )
          if (frames.length === 2) {
            if (upstreamClose) socket.send("upstream-close")
            else socket.close(1000, "Fixture client close")
          }
        }
        socket.onerror = () => {
          clearTimeout(timer)
          reject(new Error("Realtime proxy error"))
        }
        socket.onclose = (event) => {
          clearTimeout(timer)
          resolve({ frames, code: event.code, reason: event.reason })
        }
      })
    }
    return {
      client: await exchange("inference-sub-key", false),
      upstream: await exchange("inference-sub-key", true),
      rejected: await exchange("wrong-key", false),
    }
  })
  expect(result.client.frames).toEqual([
    JSON.stringify({
      type: "start",
      api_key: "inference-sub-key",
      model: "fixture",
      language: "en",
    }),
    [0, 255, 1, 128],
  ])
  expect(result.client.code).toBe(1000)
  expect(result.upstream.code).toBe(1008)
  expect(result.upstream.reason).toBe("Fixture upstream close")
  expect(result.rejected.frames).toEqual([])
  expect(result.rejected.code).toBe(1008)
  await expect
    .poll(
      async () =>
        (await (await request.get(`${upstream}/__test__/proxy-metrics`)).json())
          .ws_closed
    )
    .toContain(1000)
})

test("built public inference proxy rejects unknown methods and encoded unsafe paths", async ({
  request,
}) => {
  for (const path of [
    "/v1/not-a-route",
    "/v1/models/a%2F..%2Fb/load",
    "/v1/models/a%5Cb/load",
    "/v1/models/a%252Fb/load",
  ]) {
    const response = path.endsWith("/load")
      ? await request.post(path, { headers: { Authorization: authorization } })
      : await request.get(path, { headers: { Authorization: authorization } })
    expect(response.status(), path).toBe(404)
    expect(await response.text(), path).toContain("Unknown inference operation")
  }
  expect(
    (
      await request.post("/v1/models", {
        headers: { Authorization: authorization },
      })
    ).status()
  ).toBe(404)
})
