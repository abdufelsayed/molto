import { test, expect, type Page } from "@playwright/test"

const key = "molto.ui.preferences"
async function connect(page: Page, path = "/studio") {
  await page.goto(path)
  await page.getByRole("button", { name: "API access" }).first().click()
  await page.getByLabel("API key", { exact: true }).fill("dashboard-test-key")
  await page.getByRole("button", { name: "Connect", exact: true }).click()
  await expect(
    page.getByRole("dialog", { name: "Connect to Molto" })
  ).toHaveCount(0)
}
async function saved(page: Page) {
  return page.evaluate(
    (key) => JSON.parse(localStorage.getItem(key) || "{}"),
    key
  )
}
async function editorState(page: Page, id: string) {
  return page.evaluate(async (id) => {
    const request = indexedDB.open("molto-studio", 1)
    const db = await new Promise<IDBDatabase>((resolve) => {
      request.onsuccess = () => resolve(request.result)
    })
    const reading = db.transaction("sessions").objectStore("sessions").get(id)
    const session = await new Promise<any>((resolve) => {
      reading.onsuccess = () => resolve(reading.result)
    })
    db.close()
    return session?.fileEditor
  }, id)
}
async function tab(page: Page, name: string, list = "Session configuration") {
  await page
    .getByRole("tablist", { name: list })
    .getByRole("tab", { name, exact: true })
    .click()
}
async function run(page: Page, prompt: string) {
  await page.getByRole("textbox", { name: "Prompt", exact: true }).fill(prompt)
  await page.getByRole("button", { name: "Run prompt" }).click()
  await expect(
    page.getByRole("button", { name: "Stop", exact: true })
  ).toHaveCount(0)
  await expect(
    page.getByText(/complete · \d+ model calls/).last()
  ).toBeVisible()
}
test.beforeEach(async ({ request }) => {
  await request.post("http://127.0.0.1:8765/__test__/reset")
})

test("older active experiment, tabs and disclosure choices survive reload and navigation", async ({
  page,
}) => {
  await connect(page)
  await run(page, "older sandbox preference experiment")
  const active = (await saved(page)).values["studio.active"]
  await tab(page, "Prompt")
  await page
    .getByLabel("System prompt", { exact: true })
    .fill("Preference experiment instructions")
  const assistant = page.getByRole("article", {
    name: "Assistant message, turn 1",
    exact: true,
  })
  await assistant.getByRole("button", { name: /^Activity/ }).click()
  await assistant.getByRole("button", { name: /bash.*printf/ }).click()
  await assistant
    .getByRole("button", { name: "Inspect bash", exact: true })
    .click()
  const selection = (await saved(page)).values
  expect(selection[`studio.run:${active}`]).toBeTruthy()
  expect(selection[`studio.event:${active}`]).toBeTruthy()
  await tab(page, "Usage", "Observability views")
  await page
    .getByRole("textbox", { name: "Research note", exact: true })
    .fill("Persistent note")
  await tab(page, "Sandbox")
  await page
    .getByRole("button", { name: "/workspace/experiment.txt", exact: true })
    .click()
  await page
    .getByLabel("File content", { exact: true })
    .fill("Unsaved editor draft")
  const attachment = page.waitForEvent("filechooser")
  await page
    .getByRole("button", { name: "Attach files to prompt", exact: true })
    .click()
  await (
    await attachment
  ).setFiles({
    name: "context.txt",
    mimeType: "text/plain",
    buffer: Buffer.from("research context"),
  })
  await expect(
    page.getByRole("button", {
      name: "Remove attachment /workspace/context.txt",
      exact: true,
    })
  ).toBeVisible()
  await expect
    .poll(async () => (await editorState(page, active))?.draft?.text)
    .toBe("Unsaved editor draft")
  await page
    .getByRole("button", { name: "New experiment", exact: true })
    .click()
  await run(page, "newer preference experiment")
  await page
    .getByRole("button", {
      name: /older sandbox preference experiment.*1 runs/,
    })
    .click()
  await page.reload()
  await tab(page, "Prompt")
  await expect
    .poll(async () => (await saved(page)).values["studio.active"])
    .toBe(active)
  await expect(page.getByLabel("System prompt", { exact: true })).toHaveValue(
    "Preference experiment instructions"
  )
  await expect(
    assistant.getByRole("button", { name: /^Activity/ })
  ).toHaveAttribute("aria-expanded", "true")
  await expect(
    assistant.getByRole("button", { name: /bash.*printf/ })
  ).toHaveAttribute("aria-expanded", "true")
  await expect(
    page.getByRole("textbox", { name: "Research note", exact: true })
  ).toHaveValue("Persistent note")
  await page.goto("/models")
  await page.goto("/studio")
  await expect(page.getByLabel("System prompt", { exact: true })).toHaveValue(
    "Preference experiment instructions"
  )
  const restored = (await saved(page)).values
  expect(restored[`studio.run:${active}`]).toBe(
    selection[`studio.run:${active}`]
  )
  expect(restored[`studio.event:${active}`]).toBe(
    selection[`studio.event:${active}`]
  )
  expect(restored[`studio.path:${active}`]).toBe("/workspace/experiment.txt")
  await tab(page, "Sandbox")
  await expect(page.getByLabel("File content", { exact: true })).toHaveValue(
    "Unsaved editor draft"
  )
  await expect(
    page.getByRole("button", { name: /Remove.*context.txt/ })
  ).toBeVisible()
  expect(JSON.stringify(await saved(page))).not.toContain("dashboard-test-key")
})

test("studio sidebars and observability size restore while minimized tabs stay accessible", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 1100 })
  await connect(page)
  await page
    .getByRole("textbox", { name: "Prompt", exact: true })
    .fill("Immediate reload draft")
  await page.reload()
  await expect(
    page.getByRole("textbox", { name: "Prompt", exact: true })
  ).toHaveValue("Immediate reload draft")
  await tab(page, "Usage", "Observability views")
  const divider = page.getByRole("separator", {
    name: "Resize observability",
    exact: true,
  })
  const box = (await divider.boundingBox())!
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
  await page.mouse.down()
  await page.mouse.move(box.x + box.width / 2, box.y - 100, { steps: 8 })
  await page.mouse.up()
  const body = page.getByRole("region", {
    name: "Observability content",
    exact: true,
  })
  const sessionsPanel = page.locator(
    '[data-slot="resizable-panel"][id="studio-sessions"]'
  )
  const workspaceHandle = page
    .locator(
      '[data-slot="resizable-panel-group"][id="studio-workspace"] > [role="separator"]'
    )
    .first()
  const horizontalBox = (await workspaceHandle.boundingBox())!
  await page.mouse.move(
    horizontalBox.x + horizontalBox.width / 2,
    horizontalBox.y + 100
  )
  await page.mouse.down()
  await page.mouse.move(horizontalBox.x + 60, horizontalBox.y + 100, {
    steps: 8,
  })
  await page.mouse.up()
  const width = (await sessionsPanel.boundingBox())!.width
  await page.reload()
  await expect
    .poll(async () =>
      Math.abs((await sessionsPanel.boundingBox())!.width - width)
    )
    .toBeLessThan(4)
  await page
    .getByRole("button", { name: "Show experiments", exact: true })
    .click()
  await page.getByRole("button", { name: "Show controls", exact: true }).click()
  const collapsedSidebarsHeight = (await body.boundingBox())!.height
  await page.reload()
  await expect(
    page.getByRole("button", { name: "Show experiments", exact: true })
  ).toBeVisible()
  await expect(
    page.getByRole("button", { name: "Show controls", exact: true })
  ).toBeVisible()
  await expect
    .poll(async () =>
      Math.abs((await body.boundingBox())!.height - collapsedSidebarsHeight)
    )
    .toBeLessThan(4)
  await page
    .getByRole("button", { name: "Minimize observability", exact: true })
    .click()
  await page.reload()
  await expect(
    page.getByRole("tablist", { name: "Observability views" }).getByRole("tab")
  ).toHaveCount(5)
  await expect(
    page.locator('[aria-label="Observability content"]')
  ).toHaveAttribute("aria-hidden", "true")
  await tab(page, "Usage", "Observability views")
  await expect(body).toBeVisible()
})

test("library filters restore on bare navigation and explicit URL filters win", async ({
  page,
}) => {
  await connect(page, "/models")
  await page.getByLabel("Search library").fill("unloaded")
  await expect(page).toHaveURL(/q=unloaded/)
  await expect
    .poll(async () => (await saved(page)).values["models.filters"]?.q)
    .toBe("unloaded")
  console.log(
    "LIBRARY before leave",
    await saved(page),
    new URL(page.url()).search
  )
  await page.goto("/studio")
  const bareLibrary = await page.goto("/models")
  expect(bareLibrary?.status()).toBe(200)
  expect(bareLibrary?.request().redirectedFrom()).toBeNull()
  await expect(page.getByLabel("Search library")).toHaveValue("unloaded")
  await page.getByLabel("Search library").fill("123")
  await expect
    .poll(async () => (await saved(page)).values["models.filters"]?.q)
    .toBe("123")
  await page.reload()
  await expect(page.getByLabel("Search library")).toHaveValue("123")
  await page.goto("/models?q=")
  await expect(page.getByLabel("Search library")).toHaveValue("")
  await expect
    .poll(async () => (await saved(page)).values["models.filters"]?.q)
    .toBe("")
  await page.goto("/models?q=test-model")
  await expect(page.getByLabel("Search library")).toHaveValue("test-model")
  await page.goto("/models/mlx-community%2Ftest-model?tab=settings")
  await expect(
    page.getByRole("tab", { name: "Settings", exact: true })
  ).toHaveAttribute("aria-selected", "true")
  await page.goto("/studio")
  await page.goto("/models/mlx-community%2Ftest-model")
  await expect(
    page.getByRole("tab", { name: "Settings", exact: true })
  ).toHaveAttribute("aria-selected", "true")
  await page.goto("/models/mlx-community%2Ftest-model?tab=summary")
  await expect(
    page.getByRole("tab", { name: "Summary", exact: true })
  ).toHaveAttribute("aria-selected", "true")
})

for (const mode of ["corrupt", "denied"] as const) {
  test(`${mode} browser preferences fall back without hydration errors`, async ({
    page,
  }) => {
    const errors: string[] = []
    page.on("pageerror", (error) => errors.push(error.message))
    await page.addInitScript(
      ({ mode, key }) => {
        if (mode === "corrupt") localStorage.setItem(key, "{broken JSON")
        else {
          // oxlint-disable-next-line typescript/unbound-method
          const original = Storage.prototype.getItem
          Storage.prototype.getItem = function (name) {
            if (name === key) throw new DOMException("Denied", "SecurityError")
            return original.call(this, name)
          }
          // oxlint-disable-next-line typescript/unbound-method
          const write = Storage.prototype.setItem
          Storage.prototype.setItem = function (name, value) {
            if (name === key) throw new DOMException("Denied", "SecurityError")
            write.call(this, name, value)
          }
        }
      },
      { mode, key }
    )
    await connect(page)
    await tab(page, "Prompt")
    await page
      .getByLabel("System prompt", { exact: true })
      .fill("Usable fallback")
    await tab(page, "Trace", "Observability views")
    await expect(
      page.getByRole("region", { name: "Observability content", exact: true })
    ).toBeVisible()
    expect(errors).toEqual([])
  })
}

test("page filters and server section preferences survive navigation without replaying operations", async ({
  page,
}) => {
  await connect(page, "/logs")
  const select = async (label: string, option: string) => {
    await page.getByRole("combobox", { name: label, exact: true }).click()
    await page.getByRole("option", { name: option, exact: true }).click()
  }
  await select("Level", "ERROR")
  await select("Tail size", "100 lines")
  await page.goto("/monitoring")
  await select("Date range", "Last 30 days")
  await page.goto("/activity")
  await select("Show operations", "Failed")
  await page.goto("/settings")
  await page.getByRole("tab", { name: "Logging", exact: true }).click()
  await page.getByLabel("Search all server settings").fill("log")
  await page.goto("/logs")
  await expect(
    page.getByRole("combobox", { name: "Level", exact: true })
  ).toContainText("ERROR")
  await expect(
    page.getByRole("combobox", { name: "Tail size", exact: true })
  ).toContainText("100 lines")
  await page.goto("/monitoring")
  await expect(
    page.getByRole("combobox", { name: "Date range", exact: true })
  ).toContainText("Last 30 days")
  await page.goto("/activity")
  await expect(
    page.getByRole("combobox", { name: "Show operations", exact: true })
  ).toContainText("Failed")
  await page.goto("/settings")
  await expect(
    page.getByRole("tab", { name: "Logging", exact: true })
  ).toHaveAttribute("aria-selected", "true")
  await expect(page.getByLabel("Search all server settings")).toHaveValue("log")
  expect(
    Object.keys((await saved(page)).values).some((name) =>
      /confirm|operation|credential|token|api.?key/i.test(name)
    )
  ).toBe(false)
})

test("discovery inputs and workflow restore without an automatic search or download", async ({
  page,
}) => {
  let searches = 0
  let mutations = 0
  page.on("request", (request) => {
    if (/acquisition\/.*\/(search|recommended)/.test(request.url())) searches++
    if (/acquisition/.test(request.url()) && request.method() !== "GET")
      mutations++
  })
  await connect(page, "/add-model")
  await page
    .getByLabel("Search models", { exact: true })
    .fill("remember this search")
  await page
    .getByRole("button", { name: "2. Prepare local model", exact: true })
    .click()
  await page.reload()
  await expect(
    page.getByRole("button", { name: "2. Prepare local model", exact: true })
  ).toHaveAttribute("aria-pressed", "true")
  await page
    .getByRole("button", { name: "1. Find and download", exact: true })
    .click()
  await expect(page.getByLabel("Search models", { exact: true })).toHaveValue(
    "remember this search"
  )
  await page.goto("/studio")
  await page.goto("/add-model")
  await expect(page.getByLabel("Search models", { exact: true })).toHaveValue(
    "remember this search"
  )
  expect(searches).toBe(0)
  expect(mutations).toBe(0)
})

test("global desktop sidebar retains its collapsed preference across reload and pages", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 1000 })
  await connect(page)
  await page
    .getByRole("button", { name: "Toggle Sidebar", exact: true })
    .click()
  await expect
    .poll(async () => (await saved(page)).values["app.sidebar"])
    .toBe(false)
  await page.reload()
  await expect(page.locator('[data-slot="sidebar"]')).toHaveAttribute(
    "data-state",
    "collapsed"
  )
  await page.goto("/models")
  await expect(page.locator('[data-slot="sidebar"]')).toHaveAttribute(
    "data-state",
    "collapsed"
  )
  await page
    .getByRole("button", { name: "Toggle Sidebar", exact: true })
    .click()
  await expect
    .poll(async () => (await saved(page)).values["app.sidebar"])
    .toBe(true)
})

test("overview statistics scope remembers navigation and explicit URL scope wins", async ({
  page,
}) => {
  await connect(page, "/")
  await page
    .getByRole("combobox", { name: "Statistics scope", exact: true })
    .click()
  await page.getByRole("option", { name: "All time", exact: true }).click()
  await expect
    .poll(async () => (await saved(page)).values["overview.scope"])
    .toBe("alltime")
  console.log(
    "OVERVIEW before leave",
    await saved(page),
    new URL(page.url()).search
  )
  await page.goto("/models")
  const bareOverview = await page.goto("/")
  expect(bareOverview?.status()).toBe(200)
  expect(bareOverview?.request().redirectedFrom()).toBeNull()
  await expect(
    page.getByRole("combobox", { name: "Statistics scope", exact: true })
  ).toContainText("All time")
  await page.goto("/?scope=session")
  await expect(
    page.getByRole("combobox", { name: "Statistics scope", exact: true })
  ).toContainText("This session")
  await expect
    .poll(async () => (await saved(page)).values["overview.scope"])
    .toBe("session")
})
