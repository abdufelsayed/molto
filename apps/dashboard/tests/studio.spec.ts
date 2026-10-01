import { test, expect, type Page } from "@playwright/test"

const fixture = "http://127.0.0.1:8765"
async function connect(page: Page) {
  await page.goto("/studio")
  await page.getByRole("button", { name: "API access" }).first().click()
  await page.getByLabel("API key", { exact: true }).fill("dashboard-test-key")
  await page.getByRole("button", { name: "Connect", exact: true }).click()
  await expect(
    page.getByRole("dialog", { name: "Connect to Molto" })
  ).toHaveCount(0)
  await expect(page.getByRole("button", { name: "Run prompt" })).toBeDisabled()
  await expect(page.getByLabel("Model", { exact: true })).toContainText(
    "test-model"
  )
}
async function run(page: Page, prompt: string) {
  await page.getByRole("textbox", { name: "Prompt", exact: true }).fill(prompt)
  await page.getByRole("button", { name: "Run prompt" }).click()
}
async function finished(page: Page) {
  await expect(page.getByRole("button", { name: "Stop run" })).toHaveCount(0)
  await expect(page.getByText(/complete · \d+ model calls/)).toBeVisible()
}
async function stored(page: Page) {
  return page.evaluate(async () => {
    const db = await new Promise<IDBDatabase>((resolve, reject) => {
      const req = indexedDB.open("molto-studio", 1)
      req.onsuccess = () => resolve(req.result)
      req.onerror = () => reject(req.error)
    })
    const values = await new Promise<any[]>((resolve, reject) => {
      const req = db.transaction("sessions").objectStore("sessions").getAll()
      req.onsuccess = () => resolve(req.result)
      req.onerror = () => reject(req.error)
    })
    db.close()
    return values
  })
}

test.beforeEach(async ({ request }) => {
  await request.post(`${fixture}/__test__/reset`)
})

test("streams local inference with empty system, inherited sampling and exactly the enabled tools", async ({
  page,
  request,
}) => {
  const errors: string[] = []
  page.on("pageerror", (error) => errors.push(error.message))
  await connect(page)
  await run(page, "basic experiment")
  await expect(
    page.getByText("Fixture response: basic experiment", { exact: true })
  ).toBeVisible()
  await finished(page)
  const metrics = await (
    await request.get(`${fixture}/__test__/proxy-metrics`)
  ).json()
  const body = metrics.requests.find(
    (entry: any) => entry.path === "/v1/chat/completions"
  ).body
  expect(body.model).toBe("mlx-community/test-model")
  expect(body.include_mcp_tools).toBe(false)
  expect(body.sampling_override).toBe(true)
  expect(body.messages).toEqual([{ role: "user", content: "basic experiment" }])
  for (const key of ["temperature", "top_p", "top_k", "max_tokens", "seed"])
    expect(body).not.toHaveProperty(key)
  expect(body.tools.map((entry: any) => entry.function.name).sort()).toEqual([
    "ask_question",
    "bash",
    "edit",
    "fetch_url",
    "read",
    "web_search",
    "write",
  ])
  await expect
    .poll(() => stored(page).then((sessions) => sessions[0]?.runs[0]?.status))
    .toBe("complete")
  const session = (await stored(page))[0]
  expect(session.runs[0].events.map((event: any) => event.type)).toContain(
    "reasoning"
  )
  expect(session.runs[0].steps[0].usage.totalTokens).toBe(20)
  expect(errors).toEqual([])
})

test("exact numeric and system settings are recorded immutably across reruns", async ({
  page,
  request,
}) => {
  await connect(page)
  await page
    .getByRole("button", { name: "System prompt", exact: false })
    .click()
  await page
    .getByRole("textbox", { name: "System prompt", exact: true })
    .fill("Be precise.")
  await page.getByLabel("Temperature", { exact: true }).fill("0.27")
  await page.getByRole("switch", { name: "read", exact: true }).uncheck()
  await run(page, "numeric experiment")
  await finished(page)
  await page.getByLabel("Temperature", { exact: true }).fill("0.81")
  await page.getByRole("button", { name: "Rerun live" }).click()
  await expect(
    page.getByRole("button", { name: "Run 2", exact: true })
  ).toBeVisible()
  await finished(page)
  await expect
    .poll(() => stored(page).then((sessions) => sessions[0]?.runs.length))
    .toBe(2)
  const runs = (await stored(page))[0].runs
  expect(runs.map((entry: any) => entry.settings.temperature)).toEqual([
    0.27, 0.81,
  ])
  const metrics = await (
    await request.get(`${fixture}/__test__/proxy-metrics`)
  ).json()
  expect(
    metrics.requests
      .filter((entry: any) => entry.body)
      .map((entry: any) => entry.body.temperature)
  ).toEqual([0.27, 0.81])
  expect(
    metrics.requests[0].body.tools.map((entry: any) => entry.function.name)
  ).not.toContain("read")
  expect(metrics.requests[0].body.messages[0]).toEqual({
    role: "system",
    content: "Be precise.",
  })
})

test("real browser worker executes bash and persists its filesystem after reload", async ({
  page,
}) => {
  const workers: string[] = []
  page.on("worker", (worker) => workers.push(worker.url()))
  await connect(page)
  await run(page, "sandbox experiment")
  await finished(page)
  expect(workers.some((url) => url.includes("worker"))).toBe(true)
  await page
    .getByRole("button", { name: "Recorded settings for this run" })
    .click()
  await page.screenshot({ path: "/tmp/molto-studio.png", fullPage: true })
  await expect
    .poll(() =>
      stored(page).then((sessions) =>
        sessions[0]?.files.some((file: any) =>
          file.path.endsWith("/experiment.txt")
        )
      )
    )
    .toBe(true)
  await page.reload()
  await page.getByRole("tab", { name: "Files", exact: true }).first().click()
  await page.getByRole("button", { name: /\/experiment.txt/ }).click()
  await expect(page.getByRole("textbox", { name: "File content" })).toHaveValue(
    "worker persisted"
  )
})

test("read-only filesystem rejects writes performed through bash", async ({
  page,
}) => {
  await connect(page)
  await page.getByRole("button", { name: "Sandbox", exact: true }).click()
  await page.getByRole("switch", { name: "Read-only filesystem" }).check()
  await run(page, "sandbox experiment")
  await finished(page)
  await expect
    .poll(() => stored(page).then((sessions) => sessions[0]?.runs.length))
    .toBe(1)
  const session = (await stored(page))[0]
  expect(
    session.files.some((file: any) => file.path.endsWith("/experiment.txt"))
  ).toBe(false)
  expect(JSON.stringify(session.runs[0].events)).toMatch(/read.only/i)
})

test("question options resume the agent with the actual answer", async ({
  page,
}) => {
  await connect(page)
  await run(page, "question experiment")
  const question = page.getByRole("region", { name: "Agent question" })
  await expect(
    question.getByText("Which direction?", { exact: true })
  ).toBeVisible()
  await question.getByRole("button", { name: "Verify", exact: true }).click()
  await finished(page)
  await expect
    .poll(() =>
      stored(page).then((sessions) => sessions[0]?.runs[0]?.steps.length)
    )
    .toBe(2)
  const session = (await stored(page))[0]
  expect(JSON.stringify(session.runs[0].events)).toContain("Verify")
})

test("denied confirmation resumes with a denial and never writes the file", async ({
  page,
}) => {
  await connect(page)
  await page.getByLabel("Tool execution", { exact: true }).click()
  await page.getByRole("option", { name: /confirm/i }).click()
  await run(page, "confirmation experiment")
  const confirmation = page.getByRole("region", { name: "Tool confirmation" })
  await expect(confirmation).toBeVisible()
  await confirmation.getByRole("button", { name: "Deny tool" }).click()
  await finished(page)
  await expect
    .poll(() => stored(page).then((sessions) => sessions[0]?.runs[0]?.status))
    .toBe("complete")
  const session = (await stored(page))[0]
  expect(
    session.files.some((file: any) => file.path.endsWith("/denied.txt"))
  ).toBe(false)
  expect(
    session.runs[0].events.find((entry: any) => entry.tool === "write").state
  ).toBe("denied")
})

test("stopping a streamed model call propagates cancellation and preserves partial output", async ({
  page,
  request,
}) => {
  await connect(page)
  await run(page, "long stream experiment")
  await expect(
    page.getByText("Fixture response: long stream experiment", { exact: true })
  ).toBeVisible()
  await page.getByRole("button", { name: "Stop run" }).click()
  await expect(page.getByRole("button", { name: "Stop run" })).toHaveCount(0)
  await expect
    .poll(
      async () =>
        (await (await request.get(`${fixture}/__test__/proxy-metrics`)).json())
          .streams_cancelled
    )
    .toBeGreaterThan(0)
  await expect
    .poll(() => stored(page).then((sessions) => sessions[0]?.runs[0]?.status))
    .toBe("stopped")
  expect(JSON.stringify((await stored(page))[0].runs[0].events)).toContain(
    "long stream experiment"
  )
})

test("editing retains prior variants and exported experiments can be imported independently", async ({
  page,
}) => {
  await connect(page)
  await run(page, "original experiment")
  await finished(page)
  await page.getByRole("button", { name: "Edit + rerun" }).click()
  await page
    .getByRole("textbox", { name: "Edit message" })
    .fill("edited experiment")
  await page.getByRole("button", { name: "Run edited message" }).click()
  await expect(
    page.getByText("Fixture response: edited experiment", { exact: true })
  ).toBeVisible()
  await finished(page)
  await page.getByRole("button", { name: "Run 1", exact: true }).click()
  await expect(
    page.getByText("Fixture response: original experiment", { exact: true })
  ).toBeVisible()
  const downloadPromise = page.waitForEvent("download")
  await page.getByRole("button", { name: "Export experiment" }).click()
  const download = await downloadPromise
  const path = await download.path()
  expect(path).toBeTruthy()
  await page
    .getByLabel("Import experiment", { exact: true })
    .setInputFiles(path!)
  await expect(page.getByLabel("Experiment title")).toHaveValue(/imported/i)
  await expect
    .poll(() => stored(page).then((sessions) => sessions.length))
    .toBe(2)
})

test("studio endpoints require authentication and reject cross-origin mutations", async ({
  request,
}) => {
  expect((await request.get("/api/studio/models")).status()).toBe(401)
  await request.post("/api/connection", {
    data: { key: "dashboard-test-key" },
    headers: { Origin: "http://127.0.0.1:3007" },
  })
  expect(
    (
      await request.post("/api/studio/model/chat/completions", {
        data: { model: "mlx-community/test-model", messages: [] },
        headers: { Origin: "https://evil.example" },
      })
    ).status()
  ).toBe(403)
})

test("recorded web results replay without another live web request", async ({
  page,
}) => {
  let searches = 0
  await page.route("**/api/studio/tools", async (route) => {
    searches += 1
    await route.fulfill({
      json: {
        ok: true,
        provider: "duckduckgo",
        results: [
          {
            title: "Recorded source",
            url: "https://example.com/research",
            snippet: "A source from the recorded search.",
          },
        ],
      },
    })
  })
  await connect(page)
  await run(page, "web replay experiment")
  await finished(page)
  expect(searches).toBe(1)
  await page.getByRole("button", { name: "Replay tools" }).click()
  await expect(
    page.getByRole("button", { name: "Run 2", exact: true })
  ).toBeVisible()
  await finished(page)
  expect(searches).toBe(1)
  await expect
    .poll(() => stored(page).then((sessions) => sessions[0]?.runs.length))
    .toBe(2)
  const runs = (await stored(page))[0].runs
  expect(runs[1].mode).toBe("replay")
  expect(
    runs[1].events.find((entry: any) => entry.tool === "web_search").replayed
  ).toBe(true)
  expect(runs[1].sources[0].url).toBe("https://example.com/research")
})

test("duplicate experiment and copy setup retain independent filesystem snapshots", async ({
  page,
}) => {
  await connect(page)
  await run(page, "sandbox experiment")
  await finished(page)
  await page.getByRole("button", { name: "Duplicate experiment" }).click()
  await expect(page.getByLabel("Experiment title")).toHaveValue(/\(copy\)$/)
  await expect
    .poll(() => stored(page).then((sessions) => sessions.length))
    .toBe(2)
  let sessions = await stored(page)
  expect(
    sessions.every((session: any) =>
      session.files.some((file: any) => file.path.endsWith("/experiment.txt"))
    )
  ).toBe(true)
  await page.getByRole("tab", { name: "Files", exact: true }).first().click()
  await page.getByRole("button", { name: "Reset files" }).click()
  await expect
    .poll(() =>
      stored(page).then(
        (values) =>
          values.filter((session: any) => session.files.length === 0).length
      )
    )
    .toBe(1)
  sessions = await stored(page)
  expect(
    sessions
      .find((session: any) => !session.title.endsWith("(copy)"))
      .files.some((file: any) => file.path.endsWith("/experiment.txt"))
  ).toBe(true)
  await page.getByRole("button", { name: "Copy setup", exact: true }).click()
  await expect(page.getByLabel("Experiment title")).toHaveValue(/\(setup\)$/)
  await expect(page.getByRole("button", { name: "Rerun live" })).toHaveCount(0)
})

test("editing an earlier turn archives and restores its original continuation", async ({
  page,
}) => {
  await connect(page)
  await run(page, "sandbox first turn")
  await finished(page)
  await run(page, "original second turn")
  await expect(
    page.getByText("Fixture response: original second turn", { exact: true })
  ).toBeVisible()
  await page.getByRole("button", { name: "Edit + rerun" }).first().click()
  await page
    .getByRole("textbox", { name: "Edit message" })
    .fill("sandbox revised first turn")
  await page.getByRole("button", { name: "Run edited message" }).click()
  await expect(
    page.getByRole("button", { name: "Run 2", exact: true })
  ).toBeVisible()
  await finished(page)
  await expect(
    page.getByText("original second turn", { exact: true })
  ).toHaveCount(0)
  await expect
    .poll(() => stored(page).then((sessions) => sessions[0]?.branches?.length))
    .toBe(1)
  let session = (await stored(page))[0]
  expect(session.branches[0].turns.map((turn: any) => turn.text)).toEqual([
    "sandbox first turn",
    "original second turn",
  ])
  expect(
    session.branches[0].files.some((file: any) =>
      file.path.endsWith("/experiment.txt")
    )
  ).toBe(true)
  await page.getByRole("tab", { name: "Runs", exact: true }).click()
  await page.getByRole("button", { name: /^Restore branch:/ }).click()
  await expect(
    page.getByText("original second turn", { exact: true })
  ).toBeVisible()
  await expect
    .poll(() => stored(page).then((sessions) => sessions[0]?.turns.length))
    .toBe(2)
  session = (await stored(page))[0]
  expect(session.turns[0].text).toBe("sandbox first turn")
  expect(session.runs).toHaveLength(3)
})

test("mobile studio exposes session controls in a usable sheet and runs a prompt", async ({
  page,
}) => {
  await connect(page)
  await page.setViewportSize({ width: 390, height: 844 })
  await page.getByRole("button", { name: "Show controls" }).click()
  const sheet = page.getByRole("dialog", { name: "Session controls" })
  await expect(sheet).toBeVisible()
  await sheet.getByLabel("Temperature", { exact: true }).fill("0.39")
  await page.keyboard.press("Escape")
  await expect(sheet).toHaveCount(0)
  await run(page, "mobile experiment")
  await finished(page)
  await expect
    .poll(() =>
      stored(page).then(
        (sessions) => sessions[0]?.runs[0]?.settings.temperature
      )
    )
    .toBe(0.39)
  await page.getByRole("button", { name: "Show inspector" }).click()
  await expect(
    page.getByRole("dialog", { name: "Run inspector" })
  ).toBeVisible()
  await expect(
    page.getByRole("textbox", { name: "Research note" })
  ).toBeVisible()
})

test("activating a historical variant restores its prompt, attachments, files and continuation context", async ({
  page,
  request,
}) => {
  await connect(page)
  await run(page, "seed experiment")
  await finished(page)
  await expect
    .poll(() => stored(page).then((sessions) => sessions[0]?.runs[0]?.status))
    .toBe("complete")
  const original = (await stored(page))[0]
  await page.evaluate(async (value) => {
    const file = (name: string, content: string) => ({
      path: `/workspace/${name}.txt`,
      type: "file",
      content: btoa(content),
    })
    const oldFile = file("old", "old variant filesystem")
    const newFile = file("new", "new variant filesystem")
    const descendantFile = file("descendant", "descendant filesystem")
    const base = value.runs[0]
    const makeRun = (
      id: string,
      prompt: string,
      attachment: typeof oldFile,
      answer: string
    ) => {
      const input = [
        {
          role: "user",
          content: `${prompt}\n\nAttached files in the virtual filesystem:\n${attachment.path}`,
        },
      ]
      return {
        ...base,
        id,
        prompt: { text: prompt, attachments: [attachment.path] },
        input,
        messages: [...input, { role: "assistant", content: answer }],
        before: [],
        after: [attachment],
        events: [
          {
            id: `${id}-text`,
            type: "text",
            at: Date.now(),
            step: 1,
            text: answer,
          },
        ],
      }
    }
    const oldRun = makeRun(
      "old-variant",
      "old historical prompt",
      oldFile,
      "Old historical answer"
    )
    const newRun = makeRun(
      "new-variant",
      "new historical prompt",
      newFile,
      "New historical answer"
    )
    const descendant = {
      ...base,
      id: "descendant-run",
      prompt: { text: "descendant prompt" },
      input: [
        ...newRun.messages,
        { role: "user", content: "descendant prompt" },
      ],
      messages: [
        ...newRun.messages,
        { role: "user", content: "descendant prompt" },
        { role: "assistant", content: "Descendant answer" },
      ],
      after: [descendantFile],
      events: [
        {
          id: "descendant-text",
          type: "text",
          at: Date.now(),
          step: 1,
          text: "Descendant answer",
        },
      ],
    }
    value.turns = [
      {
        id: "historical-turn",
        text: "new historical prompt",
        attachments: [newFile.path],
        runIds: [oldRun.id, newRun.id],
        selected: newRun.id,
      },
      {
        id: "descendant-turn",
        text: "descendant prompt",
        runIds: [descendant.id],
        selected: descendant.id,
      },
    ]
    value.runs = [oldRun, newRun, descendant]
    value.files = [descendantFile]
    value.branches = []
    const db = await new Promise<IDBDatabase>((resolve, reject) => {
      const opening = indexedDB.open("molto-studio", 1)
      opening.onsuccess = () => resolve(opening.result)
      opening.onerror = () => reject(opening.error)
    })
    await new Promise<void>((resolve, reject) => {
      const transaction = db.transaction("sessions", "readwrite")
      transaction.objectStore("sessions").put(value)
      transaction.oncomplete = () => resolve()
      transaction.onerror = () => reject(transaction.error)
    })
    db.close()
  }, original)
  await page.reload()
  await expect(
    page.getByText("descendant prompt", { exact: true })
  ).toBeVisible()
  await page.getByRole("button", { name: "Run 1", exact: true }).click()
  await expect(
    page.getByText("old historical prompt", { exact: true })
  ).toBeVisible()
  await expect(
    page.getByText("new historical prompt", { exact: true })
  ).toHaveCount(0)
  await expect(
    page.getByText("descendant prompt", { exact: true })
  ).toHaveCount(0)
  await expect(
    page.getByText("/workspace/old.txt", { exact: true })
  ).toBeVisible()
  await expect
    .poll(() => stored(page).then((sessions) => sessions[0]?.turns.length))
    .toBe(1)
  let session = (await stored(page))[0]
  expect(session.files.map((file: any) => file.path)).toEqual([
    "/workspace/old.txt",
  ])
  expect(session.branches[0].turns).toHaveLength(2)
  expect(session.branches[0].files[0].path).toBe("/workspace/descendant.txt")
  // Selecting a variant at the last turn also swaps its filesystem.
  await page.getByRole("button", { name: "Run 2", exact: true }).click()
  await expect(
    page.getByText("new historical prompt", { exact: true })
  ).toBeVisible()
  await expect
    .poll(() => stored(page).then((sessions) => sessions[0]?.files[0]?.path))
    .toBe("/workspace/new.txt")
  await page.getByRole("button", { name: "Run 1", exact: true }).click()
  await run(page, "continue selected historical variant")
  await expect(
    page.getByText("Fixture response: continue selected historical variant", {
      exact: true,
    })
  ).toBeVisible()
  const metrics = await (
    await request.get(`${fixture}/__test__/proxy-metrics`)
  ).json()
  const latest = metrics.requests.filter((entry: any) => entry.body).at(-1).body
  expect(latest.messages).toEqual([
    {
      role: "user",
      content:
        "old historical prompt\n\nAttached files in the virtual filesystem:\n/workspace/old.txt",
    },
    { role: "assistant", content: "Old historical answer" },
    { role: "user", content: "continue selected historical variant" },
  ])
  await expect
    .poll(() => stored(page).then((sessions) => sessions[0]?.turns.length))
    .toBe(2)
  session = (await stored(page))[0]
  expect(
    session.files.some((file: any) => file.path === "/workspace/old.txt")
  ).toBe(true)
  expect(
    session.files.some((file: any) => file.path === "/workspace/new.txt")
  ).toBe(false)
})

test("storage quota failures show an error, retry periodically and recover without a busy loop", async ({
  page,
}) => {
  await connect(page)
  await expect
    .poll(() => stored(page).then((sessions) => sessions.length))
    .toBe(1)
  await page.evaluate(() => {
    const state = window as unknown as {
      quotaAttempts: number
      originalPut: IDBObjectStore["put"]
    }
    state.quotaAttempts = 0
    state.originalPut = Object.getOwnPropertyDescriptor(
      IDBObjectStore.prototype,
      "put"
    )!.value as IDBObjectStore["put"]
    IDBObjectStore.prototype.put = function (
      this: IDBObjectStore,
      ...args: Parameters<IDBObjectStore["put"]>
    ) {
      state.quotaAttempts += 1
      // The cap makes a broken busy loop observable without freezing Chromium forever.
      if (state.quotaAttempts <= 1000)
        throw new DOMException(
          "Synthetic storage quota exceeded",
          "QuotaExceededError"
        )
      return state.originalPut.apply(this, args)
    }
  })
  await page.getByLabel("Experiment title").fill("pending quota recovery")
  await expect(page.getByRole("alert")).toContainText("QuotaExceededError")
  const attempts = await page.evaluate(async () => {
    await new Promise((resolve) => setTimeout(resolve, 1200))
    return (window as unknown as { quotaAttempts: number }).quotaAttempts
  })
  expect(attempts).toBeGreaterThanOrEqual(2)
  expect(attempts).toBeLessThanOrEqual(5)
  await page.evaluate(() => {
    const state = window as unknown as { originalPut: IDBObjectStore["put"] }
    IDBObjectStore.prototype.put = state.originalPut
  })
  await expect(page.getByRole("alert")).toHaveCount(0)
  await expect
    .poll(() => stored(page).then((sessions) => sessions[0]?.title))
    .toBe("pending quota recovery")
})

test("citations link recorded sources without rewriting code, unknown citations or existing links", async ({
  page,
}) => {
  await connect(page)
  await run(page, "citation seed experiment")
  await finished(page)
  await expect
    .poll(() => stored(page).then((sessions) => sessions[0]?.runs[0]?.status))
    .toBe("complete")
  const original = (await stored(page))[0]
  await page.evaluate(async (value) => {
    const run = value.runs[0]
    run.sources = [
      { id: "1", title: "Source", url: "https://example.com/research" },
    ]
    run.events = [
      {
        id: "citation-text",
        type: "text",
        at: Date.now(),
        step: 1,
        text: "Citation [1]. Inline `[1]`.\n\n```text\n[1]\n```\n\nUnknown [99].\n\n[Existing link [1]](https://example.com/other)",
      },
    ]
    const db = await new Promise<IDBDatabase>((resolve, reject) => {
      const opening = indexedDB.open("molto-studio", 1)
      opening.onsuccess = () => resolve(opening.result)
      opening.onerror = () => reject(opening.error)
    })
    await new Promise<void>((resolve, reject) => {
      const transaction = db.transaction("sessions", "readwrite")
      transaction.objectStore("sessions").put(value)
      transaction.oncomplete = () => resolve()
      transaction.onerror = () => reject(transaction.error)
    })
    db.close()
  }, original)
  await page.reload()
  const messages = page.getByRole("region", { name: "Messages" })
  await expect(
    messages.getByRole("link", { name: "[1]", exact: true })
  ).toHaveAttribute("href", "https://example.com/research")
  await expect(messages.locator("p code")).toHaveText("[1]")
  await expect(messages.locator("pre")).toHaveText("[1]")
  await expect(messages.locator("pre a, p code a")).toHaveCount(0)
  await expect(
    messages.getByText("Unknown [99].", { exact: true })
  ).toBeVisible()
  await expect(
    messages.getByRole("link", { name: "Existing link [1]", exact: true })
  ).toHaveAttribute("href", "https://example.com/other")
})
