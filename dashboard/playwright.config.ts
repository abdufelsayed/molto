import { defineConfig, devices } from "@playwright/test"

export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  use: {
    baseURL: "http://127.0.0.1:3007",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command:
        'uv run --project "${OMLX_SOURCE:-..}" python tests/omlx_server.py 8765',
      url: "http://127.0.0.1:8765/health",
      reuseExistingServer: false,
      timeout: 30_000,
    },
    {
      command:
        "OMLX_API_URL=http://127.0.0.1:8765 HOST=127.0.0.1 PORT=3007 pnpm start",
      url: "http://127.0.0.1:3007/api/connection",
      reuseExistingServer: false,
      timeout: 30_000,
    },
  ],
})
