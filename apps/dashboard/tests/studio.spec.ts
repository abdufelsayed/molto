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
  await expect(
    page.getByRole("combobox", { name: "Model", exact: true })
  ).toContainText("test-model")
}
async function configuration(
  page: Page,
  tab: "Prompt" | "Model" | "Tools" | "Sandbox"
) {
  await page
    .getByRole("tablist", { name: "Session configuration" })
    .getByRole("tab", { name: tab, exact: true })
    .click()
}
async function userAction(page: Page, label: string, turn = 1) {
  const article = page.getByRole("article", {
    name: `User message, turn ${turn}`,
    exact: true,
  })
  await article.hover()
  await article.getByRole("button", { name: label, exact: true }).click()
}
async function experimentAction(page: Page, label: string) {
  await page
    .getByRole("button", { name: "Experiment actions", exact: true })
    .click()
  await page.getByRole("menuitem", { name: label, exact: true }).click()
}
async function resizeObservability(
  page: Page,
  direction: "collapse" | "restore"
) {
  const divider = page.getByRole("separator", {
    name: "Resize observability",
    exact: true,
  })
  const box = await divider.boundingBox()
  if (!box) throw new Error("The observability divider is unavailable.")
  const x = box.x + box.width / 2
  await page.mouse.move(x, box.y + box.height / 2)
  await page.mouse.down()
  await page.mouse.move(
    x,
    direction === "collapse" ? page.viewportSize()!.height - 1 : box.y - 220,
    { steps: 8 }
  )
  await page.mouse.up()
}
async function observability(page: Page, tab: string) {
  await page
    .getByRole("tablist", { name: "Observability views" })
    .getByRole("tab", { name: tab, exact: true })
    .click()
}
async function run(page: Page, prompt: string) {
  await page.getByRole("textbox", { name: "Prompt", exact: true }).fill(prompt)
  await page.getByRole("button", { name: "Run prompt" }).click()
}
async function finished(page: Page) {
  await expect(page.getByRole("button", { name: "Stop run" })).toHaveCount(0)
  await expect(
    page.getByText(/complete · \d+ model calls/).last()
  ).toBeVisible()
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
  await configuration(page, "Prompt")
  await page
    .getByRole("textbox", { name: "System prompt", exact: true })
    .fill("Be precise.")
  await configuration(page, "Model")
  await page.getByLabel("Temperature", { exact: true }).fill("0.27")
  await configuration(page, "Tools")
  await page.getByRole("switch", { name: "read", exact: true }).uncheck()
  await run(page, "numeric experiment")
  await finished(page)
  await configuration(page, "Model")
  await page.getByLabel("Temperature", { exact: true }).fill("0.81")
  await userAction(page, "Rerun live")
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
  await observability(page, "Usage")
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
  await configuration(page, "Sandbox")
  await page.getByRole("button", { name: /\/experiment.txt/ }).click()
  await expect(page.getByRole("textbox", { name: "File content" })).toHaveValue(
    "worker persisted"
  )
})

test("read-only filesystem rejects writes performed through bash", async ({
  page,
}) => {
  await connect(page)
  await configuration(page, "Sandbox")
  await page
    .getByRole("button", { name: "Sandbox settings", exact: true })
    .click()
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
  await configuration(page, "Tools")
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
  await userAction(page, "Edit message")
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
  await experimentAction(page, "Export experiment")
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
  await userAction(page, "Turn 1 actions")
  await page
    .getByRole("menuitem", { name: "Replay tools", exact: true })
    .click()
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
  await experimentAction(page, "Duplicate experiment")
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
  await configuration(page, "Sandbox")
  await page.getByRole("button", { name: "Reset sandbox files" }).click()
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
  await experimentAction(page, "Copy setup")
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
  await userAction(page, "Edit message")
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
  await observability(page, "Runs")
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
  const body = page.locator('div[aria-label="Observability content"]')
  await expect
    .poll(() =>
      body.evaluate((element) => element.getBoundingClientRect().height)
    )
    .toBeLessThanOrEqual(1)
  await expect(
    page.getByRole("tablist", { name: "Observability views" }).getByRole("tab")
  ).toHaveCount(5)
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
  await observability(page, "Usage")
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
  await expect(page.getByText("old.txt", { exact: true })).toBeVisible()
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

test("activity collapses after completion and preserves manual streaming visibility", async ({
  page,
}) => {
  await connect(page)
  await run(page, "sandbox activity experiment")
  await finished(page)
  const first = page.getByRole("article", {
    name: "Assistant message, turn 1",
    exact: true,
  })
  const activity = first.getByRole("button", { name: /^Activity/ })
  await expect(activity).toHaveAttribute("aria-expanded", "false")
  await activity.click()
  await first
    .getByRole("button", { name: "Model reasoning", exact: true })
    .first()
    .click()
  await expect(
    first.getByText("Inspecting the experiment.", { exact: true }).first()
  ).toBeVisible()
  await expect(activity).toHaveAttribute("aria-expanded", "true")
  await activity.click()
  await run(page, "long stream activity experiment")
  const second = page.getByRole("article", {
    name: "Assistant message, turn 2",
    exact: true,
  })
  const streamingActivity = second.getByRole("button", { name: /^Activity/ })
  await expect(streamingActivity).toHaveAttribute("aria-expanded", "true")
  await streamingActivity.click()
  await expect(
    second.getByText("Fixture response: long stream activity experiment", {
      exact: true,
    })
  ).toBeVisible()
  await expect(streamingActivity).toHaveAttribute("aria-expanded", "false")
  await page.getByRole("button", { name: "Stop run" }).click()
  await expect(page.getByRole("button", { name: "Stop run" })).toHaveCount(0)
  await expect(streamingActivity).toHaveAttribute("aria-expanded", "false")
})

test("observability collapses to zero by dragging, restores its state, and opens historical inspection", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 1100 })
  await connect(page)
  const region = page.locator('div[aria-label="Observability content"]')
  await expect(
    page.getByRole("button", { name: "Observability", exact: true })
  ).toHaveCount(0)
  await expect(
    page.getByRole("button", { name: "Select observed run", exact: true })
  ).toHaveCount(0)
  const views = page.getByRole("tablist", { name: "Observability views" })
  await expect(views.getByRole("tab")).toHaveText([
    "Trace",
    "Request",
    "Response",
    "Usage",
    "Runs",
  ])
  await expect
    .poll(() =>
      region.evaluate((element) => element.getBoundingClientRect().height)
    )
    .toBeLessThanOrEqual(1)
  await expect(region).toHaveAttribute("aria-hidden", "true")
  const handle = page.getByRole("separator", {
    name: "Resize observability",
    exact: true,
  })
  await expect(handle).toBeVisible()
  await expect(handle.locator(":scope > div")).toBeVisible()
  await views.getByRole("tab", { name: "Trace", exact: true }).click()
  await expect
    .poll(() =>
      region.evaluate((element) => element.getBoundingClientRect().height)
    )
    .toBeGreaterThan(100)
  await page
    .getByRole("button", { name: "Minimize observability", exact: true })
    .click()
  await expect
    .poll(() =>
      region.evaluate((element) => element.getBoundingClientRect().height)
    )
    .toBeLessThanOrEqual(1)
  await expect(views.getByRole("tab")).toHaveCount(5)
  await views.getByRole("tab", { name: "Trace", exact: true }).click()
  await expect
    .poll(() =>
      region.evaluate((element) => element.getBoundingClientRect().height)
    )
    .toBeGreaterThan(100)
  const prompt = await page
    .getByRole("textbox", { name: "Prompt", exact: true })
    .boundingBox()
  const divider = await page
    .getByRole("separator", { name: "Resize observability", exact: true })
    .boundingBox()
  expect(divider!.y).toBeGreaterThan(prompt!.y + prompt!.height)
  await run(page, "first observability experiment")
  await finished(page)
  await run(page, "second observability experiment")
  await finished(page)
  await observability(page, "Usage")
  await page
    .getByRole("textbox", { name: "Research note", exact: true })
    .fill("Preserve this note and selected tab.")
  await resizeObservability(page, "collapse")
  await expect
    .poll(() =>
      region.evaluate((element) => element.getBoundingClientRect().height)
    )
    .toBeLessThanOrEqual(1)
  await expect(
    page.getByRole("separator", { name: "Resize observability", exact: true })
  ).toBeVisible()
  await resizeObservability(page, "restore")
  await expect(
    views.getByRole("tab", { name: "Usage", exact: true })
  ).toHaveAttribute("aria-selected", "true")
  await expect(
    page.getByRole("textbox", { name: "Research note", exact: true })
  ).toHaveValue("Preserve this note and selected tab.")
  await expect(
    page.getByRole("textbox", { name: "Prompt", exact: true })
  ).toBeVisible()
  await handle.focus()
  await page.keyboard.press("Enter")
  await expect(views.getByRole("tab")).toHaveCount(5)
  await expect
    .poll(() =>
      region.evaluate((element) => element.getBoundingClientRect().height)
    )
    .toBeLessThanOrEqual(1)
  await page.keyboard.press("Space")
  await expect(
    views.getByRole("tab", { name: "Usage", exact: true })
  ).toHaveAttribute("aria-selected", "true")
  await expect(
    page.getByRole("textbox", { name: "Research note", exact: true })
  ).toHaveValue("Preserve this note and selected tab.")
  await resizeObservability(page, "collapse")
  await expect
    .poll(() =>
      region.evaluate((element) => element.getBoundingClientRect().height)
    )
    .toBeLessThanOrEqual(1)
  const first = page.getByRole("article", {
    name: "Assistant message, turn 1",
    exact: true,
  })
  await first.hover()
  await first
    .getByRole("button", { name: "Inspect this run", exact: true })
    .click()
  await expect
    .poll(() =>
      region.evaluate((element) => element.getBoundingClientRect().height)
    )
    .toBeGreaterThan(100)
  await expect(
    views.getByRole("tab", { name: "Trace", exact: true })
  ).toHaveAttribute("aria-selected", "true")
  await observability(page, "Request")
  await page.getByRole("button", { name: "Model call 1", exact: true }).click()
  await expect(page.getByLabel("Request 1", { exact: true })).toContainText(
    "first observability experiment"
  )
  await observability(page, "Trace")
  await page.screenshot({
    path: "/tmp/molto-studio-minimal-observability.png",
    fullPage: true,
    animations: "disabled",
  })
})

test("minimal studio distinguishes roles, exposes keyboard and touch actions, and keeps panels independent", async ({
  page,
  browser,
}) => {
  await page.setViewportSize({ width: 1440, height: 1100 })
  await connect(page)
  await run(page, "sandbox: create a file and inspect its contents")
  await finished(page)
  await run(page, "Summarize the result and show the command.")
  await expect(
    page.getByText(
      "Fixture response: Summarize the result and show the command.",
      { exact: true }
    )
  ).toBeVisible()
  await expect
    .poll(() => stored(page).then((sessions) => sessions[0]?.runs.length))
    .toBe(2)
  const value = (await stored(page))[0]
  value.title = "Browser sandbox research"
  const firstText = value.runs[0].events
    .filter((event: any) => event.type === "text")
    .at(-1)
  firstText.text =
    "Created `/workspace/experiment.txt` and verified its contents: **worker persisted**."
  const lastText = value.runs[1].events
    .filter((event: any) => event.type === "text")
    .at(-1)
  lastText.text =
    "The file remains in the session's virtual filesystem. The command wrote the text, then read it back.\n\n```bash\nprintf 'worker persisted' > experiment.txt\ncat experiment.txt\n```"
  await page.evaluate(async (session) => {
    const opening = indexedDB.open("molto-studio", 1)
    const db = await new Promise<IDBDatabase>((resolve) => {
      opening.onsuccess = () => resolve(opening.result)
    })
    await new Promise<void>((resolve) => {
      const tx = db.transaction("sessions", "readwrite")
      tx.objectStore("sessions").put(session)
      tx.oncomplete = () => resolve()
    })
    db.close()
  }, value)
  await page.reload()
  const user = page.getByRole("article", {
    name: "User message, turn 2",
    exact: true,
  })
  const assistant = page.getByRole("article", {
    name: "Assistant message, turn 2",
    exact: true,
  })
  await expect(user).toHaveAttribute("data-align", "end")
  await expect(assistant).toHaveAttribute("data-align", "start")
  await expect(user.getByText("You", { exact: true })).toBeVisible()
  await expect(assistant.getByText("Assistant", { exact: true })).toBeVisible()
  await expect(
    page
      .getByRole("region", { name: "Messages" })
      .locator('[data-slot="avatar"]')
  ).toHaveCount(0)
  await expect(
    page.getByRole("button", { name: "Observability", exact: true })
  ).toHaveCount(0)
  await resizeObservability(page, "collapse")
  await expect
    .poll(() =>
      page
        .locator('div[aria-label="Observability content"]')
        .evaluate((element) => element.getBoundingClientRect().height)
    )
    .toBeLessThanOrEqual(1)
  await expect(
    page
      .getByRole("tablist", { name: "Session configuration" })
      .getByRole("tab")
  ).toHaveCount(4)
  await expect(page.getByRole("menuitem")).toHaveCount(0)
  await user.hover()
  await user.getByRole("button", { name: "Copy message", exact: true }).focus()
  await page.keyboard.press("Tab")
  const edit = user.getByRole("button", { name: "Edit message", exact: true })
  await expect(edit).toBeFocused()
  await expect(
    page
      .locator('[data-slot="tooltip-content"]')
      .filter({ hasText: /^Edit message$/ })
  ).toBeVisible()
  await page.keyboard.press("Escape")
  await page.getByRole("textbox", { name: "Prompt", exact: true }).focus()
  await page.mouse.move(0, 0)
  await expect(page.locator('[data-slot="tooltip-content"]')).toHaveCount(0)
  await page.screenshot({
    path: "/tmp/molto-studio-minimal-desktop.png",
    fullPage: true,
    animations: "disabled",
  })
  await page
    .getByRole("button", { name: "Show experiments", exact: true })
    .click()
  await expect(
    page.getByRole("button", { name: "New experiment", exact: true })
  ).toHaveCount(0)
  await expect(
    page.getByRole("tablist", { name: "Session configuration" })
  ).toBeVisible()
  await page.getByRole("button", { name: "Show controls", exact: true }).click()
  await expect(
    page.getByRole("tablist", { name: "Session configuration" })
  ).toHaveCount(0)
  await page
    .getByRole("button", { name: "Show experiments", exact: true })
    .click()
  await expect(
    page.getByRole("button", { name: "New experiment", exact: true })
  ).toBeVisible()
  const touchContext = await browser.newContext({
    viewport: { width: 390, height: 844 },
    isMobile: true,
    hasTouch: true,
  })
  await touchContext.addCookies(await page.context().cookies())
  const touchPage = await touchContext.newPage()
  await touchPage.goto("/studio")
  await touchPage
    .getByRole("button", { name: "Show experiments", exact: true })
    .tap()
  await touchPage
    .getByLabel("Import experiment", { exact: true })
    .setInputFiles({
      name: "research.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(value)),
    })
  await touchPage.keyboard.press("Escape")
  const touchUser = touchPage.getByRole("article", {
    name: "User message, turn 2",
    exact: true,
  })
  await expect(
    touchUser.getByRole("button", { name: "Edit message", exact: true })
  ).toBeVisible()
  await touchUser
    .getByRole("button", { name: "Turn 2 actions", exact: true })
    .tap()
  await expect(
    touchPage.getByRole("menuitem", { name: "Replay tools", exact: true })
  ).toBeVisible()
  await touchPage.keyboard.press("Escape")
  await expect(touchPage.getByRole("menuitem")).toHaveCount(0)
  await touchPage.getByRole("textbox", { name: "Prompt", exact: true }).tap()
  await expect(touchPage.locator('[data-slot="tooltip-content"]')).toHaveCount(
    0
  )
  expect(
    await touchPage.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth
    )
  ).toBe(true)
  await touchPage.screenshot({
    path: "/tmp/molto-studio-minimal-mobile.png",
    fullPage: true,
    animations: "disabled",
  })
  await touchContext.close()
})

test("keyboard file controls open attachment, sandbox upload and experiment import choosers", async ({
  page,
}) => {
  await connect(page)
  const attach = page.getByRole("button", {
    name: "Attach files to prompt",
    exact: true,
  })
  await attach.focus()
  const chooserPromise = page.waitForEvent("filechooser")
  await page.keyboard.press("Enter")
  const chooser = await chooserPromise
  await chooser.setFiles({
    name: "keyboard.txt",
    mimeType: "text/plain",
    buffer: Buffer.from("keyboard attachment"),
  })
  await expect(
    page.getByRole("button", {
      name: "Remove attachment /workspace/keyboard.txt",
      exact: true,
    })
  ).toBeVisible()
  await expect
    .poll(() =>
      stored(page).then((sessions) =>
        sessions[0]?.files.some(
          (file: any) => file.path === "/workspace/keyboard.txt"
        )
      )
    )
    .toBe(true)
  await configuration(page, "Sandbox")
  await page.getByRole("button", { name: "Upload files", exact: true }).focus()
  const uploadPromise = page.waitForEvent("filechooser")
  await page.keyboard.press("Enter")
  const upload = await uploadPromise
  await upload.setFiles({
    name: "keyboard-upload.txt",
    mimeType: "text/plain",
    buffer: Buffer.from("keyboard sandbox upload"),
  })
  await expect
    .poll(() =>
      stored(page).then((sessions) =>
        sessions[0]?.files.some(
          (file: any) => file.path === "/workspace/keyboard-upload.txt"
        )
      )
    )
    .toBe(true)
  const exported = (await stored(page))[0]
  await page
    .getByRole("button", { name: "Import experiment file", exact: true })
    .focus()
  const importPromise = page.waitForEvent("filechooser")
  await page.keyboard.press("Enter")
  const importing = await importPromise
  await importing.setFiles({
    name: "keyboard-experiment.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(exported)),
  })
  await expect(page.getByLabel("Experiment title")).toHaveValue(/\(imported\)$/)
  await expect
    .poll(() => stored(page).then((sessions) => sessions.length))
    .toBe(2)
})

test("compact tool disclosures render highlighted JSON and keep step inspection working", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 1400 })
  await connect(page)
  await run(page, "sandbox JSON presentation experiment")
  await finished(page)
  await expect
    .poll(() => stored(page).then((sessions) => sessions[0]?.runs[0]?.status))
    .toBe("complete")
  const value = (await stored(page))[0]
  const tool = value.runs[0].events.find((event: any) => event.tool === "bash")
  tool.input = {
    command: "printf 'structured JSON'",
    arguments: JSON.stringify({ nested: { enabled: true, count: 2 } }),
  }
  tool.output = {
    ok: true,
    payload: JSON.stringify({ files: ["experiment.txt"], value: 42 }),
  }
  value.runs[0].steps[0].request = {
    method: "POST",
    body: JSON.stringify(
      JSON.stringify({
        model: "mlx-community/test-model",
        messages: [{ role: "user", content: "a readable request body" }],
        options: { temperature: 0.27 },
      })
    ),
  }
  await page.evaluate(async (session) => {
    const opening = indexedDB.open("molto-studio", 1)
    const db = await new Promise<IDBDatabase>((resolve) => {
      opening.onsuccess = () => resolve(opening.result)
    })
    await new Promise<void>((resolve) => {
      const tx = db.transaction("sessions", "readwrite")
      tx.objectStore("sessions").put(session)
      tx.oncomplete = () => resolve()
    })
    db.close()
  }, value)
  await page.reload()
  const assistant = page.getByRole("article", {
    name: "Assistant message, turn 1",
    exact: true,
  })
  await assistant.getByRole("button", { name: /^Activity/ }).click()
  const step = assistant.getByRole("button", {
    name: /bash.*printf 'structured JSON'/,
  })
  await step.hover()
  await expect(step).toHaveCSS("background-color", "rgba(0, 0, 0, 0)")
  await expect(step.locator("..")).toHaveCSS(
    "background-color",
    "rgba(0, 0, 0, 0)"
  )
  await step.click()
  await expect(
    assistant.locator("p").filter({ hasText: tool.text })
  ).toHaveCount(0)
  expect(
    (await stored(page))[0].runs[0].events.find(
      (event: any) => event.tool === "bash"
    ).text
  ).toBe(tool.text)
  const argumentsToggle = assistant.getByRole("button", {
    name: /^(Args|Arguments)$/,
  })
  if ((await argumentsToggle.getAttribute("aria-expanded")) === "false")
    await argumentsToggle.click()
  const argumentsContent = assistant.getByLabel("Arguments", { exact: true })
  await expect(argumentsContent.locator("pre")).toContainText('"nested": {')
  await expect(argumentsContent.locator("pre")).toHaveClass(/shiki/)
  expect(
    JSON.parse((await argumentsContent.locator("pre").textContent())!).arguments
      .nested
  ).toEqual({ enabled: true, count: 2 })
  const resultToggle = assistant.getByRole("button", {
    name: /^(Result|Results)$/,
  })
  if ((await resultToggle.getAttribute("aria-expanded")) === "false")
    await resultToggle.click()
  const resultContent = assistant.getByLabel("Result", { exact: true })
  await expect(resultContent.locator("pre")).toHaveClass(/shiki/)
  expect(
    JSON.parse((await resultContent.locator("pre").textContent())!).payload
  ).toEqual({ files: ["experiment.txt"], value: 42 })
  const changesLabel = `File changes (${tool.changes.length})`
  const changesToggle = assistant.getByRole("button", {
    name: changesLabel,
    exact: true,
  })
  await expect(changesToggle).toHaveAttribute("aria-expanded", "false")
  await expect(
    assistant.getByRole("button", {
      name: "Model-visible result (limited)",
      exact: true,
    })
  ).toHaveAttribute("aria-expanded", "false")
  await changesToggle.click()
  const changesContent = assistant.getByLabel(changesLabel, { exact: true })
  await expect(changesContent).toContainText('"/workspace/experiment.txt"')
  const formattedChanges = JSON.stringify(tool.changes, null, 2)
  await expect(changesContent).toHaveText(formattedChanges.slice(0, 200_000))
  if (formattedChanges.length <= 50_000)
    await expect(changesContent.locator("pre")).toHaveClass(/shiki/)
  await changesToggle.click()
  await expect(changesContent).toBeHidden()
  expect(
    (await stored(page))[0].runs[0].events.find(
      (event: any) => event.tool === "bash"
    ).changes
  ).toEqual(tool.changes)
  await resizeObservability(page, "collapse")
  await step.hover()
  const inspect = assistant.getByRole("button", {
    name: "Inspect bash",
    exact: true,
  })
  await inspect.focus()
  await page.keyboard.press("Tab")
  await page.keyboard.press("Shift+Tab")
  await expect(inspect).toBeFocused()
  await expect(
    page
      .locator('[data-slot="tooltip-content"]')
      .filter({ hasText: /^Inspect bash$/ })
  ).toBeVisible()
  await inspect.click()
  await expect(
    page
      .getByRole("tablist", { name: "Observability views" })
      .getByRole("tab", { name: "Trace", exact: true })
  ).toHaveAttribute("aria-selected", "true")
  await expect(page.getByLabel("Tool input", { exact: true })).toContainText(
    '"nested": {'
  )
  await observability(page, "Request")
  await page.getByRole("button", { name: "Model call 1", exact: true }).click()
  const request = page.getByLabel("Request 1", { exact: true })
  await expect(request.locator("pre")).toHaveClass(/shiki/)
  const displayed = JSON.parse((await request.textContent())!)
  expect(displayed.body.options.temperature).toBe(0.27)
  expect(displayed.body.messages).toEqual([
    { role: "user", content: "a readable request body" },
  ])
  await page.mouse.move(0, 0)
  await expect(page.locator('[data-slot="tooltip-content"]')).toHaveCount(0)
  const divider = page.getByRole("separator", {
    name: "Resize observability",
  })
  const dividerBox = (await divider.boundingBox())!
  await page.mouse.move(dividerBox.x + dividerBox.width / 2, dividerBox.y + 2)
  await page.mouse.down()
  await page.mouse.move(
    dividerBox.x + dividerBox.width / 2,
    dividerBox.y - 160,
    { steps: 8 }
  )
  await page.mouse.up()
  await request.scrollIntoViewIfNeeded()
  await page.screenshot({
    path: "/tmp/molto-studio-compact-json.png",
    fullPage: true,
    animations: "disabled",
  })
})
