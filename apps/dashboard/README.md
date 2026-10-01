# Molto dashboard

The Molto dashboard lives in `apps/dashboard/` in the Molto repository. It uses TanStack Start, TanStack Query, and shadcn/ui. Preset `b43fOHkIM` supplies the Nova style, mauve base, pink accents, Geist font, and Lucide icons.

`molto serve` starts the dashboard and inference backend together. Nitro owns the public host and port; FastAPI listens on a separate private loopback socket. Clients use the same public origin for the dashboard, inference, health, and the dashboard session gateway. Raw `/management/v1` and `/admin` routes remain private and are blocked by the public proxy. Dashboard management requests use `/api/molto` with an opaque session cookie.

## Run

A complete installed wheel includes dashboard assets and a standalone Node runtime. It needs neither pnpm nor a separate dashboard command:

```sh
molto serve --model-dir ~/models
```

Open <http://127.0.0.1:8000>. If no main key exists, a directly connected local browser can create one with confirmation. Initial setup requires a loopback bind and rejects forwarded requests. If a key already exists, select **API access** and connect with it. Inference subkeys do not grant management access.

Source checkouts require Node and pnpm. From the repository root:

```sh
uv sync --all-packages --inexact
pnpm install --frozen-lockfile
pnpm --filter molto-dashboard build
uv run --all-packages --inexact molto serve --model-dir ~/models
```

The launcher uses `apps/dashboard/.output/server/index.mjs` in a source checkout and finds Node on PATH. `MOLTO_NODE` can select another Node executable. It fails with an actionable error if assets or the runtime are missing.

Use one public bind address, such as `--host 127.0.0.1` or `--host 0.0.0.0`; comma-separated addresses are rejected. Non-loopback binds require a main key configured before startup. Use HTTPS through a trusted reverse proxy for network access. First-run key creation is a direct-local operation and is unavailable through that proxy.

For dashboard development, install the frontend dependencies, then run from the repository root:

```sh
pnpm install --frozen-lockfile
uv run --all-packages --inexact molto serve --dashboard-dev --model-dir ~/models
```

This source-only mode starts native Vite/Nitro on the configured public host and port, paired with the private FastAPI child. It needs Node and pnpm but no production dashboard build. Vite uses strict port binding; it will not silently select another port. The flag is not saved to settings and installed distributions reject it. Do not point a separate frontend's `MOLTO_API_URL` at the public application port: raw management routes are deliberately blocked there.

## Packaging

`tooling/release/build_dashboard_bundle.py` builds the dashboard, verifies the standalone macOS ARM64 Node archive against its SHA256 checksum, and stages the server, public assets, runtime, and Node license under `apps/cli/src/molto_cli/_dashboard`. The root `pnpm build` command calls `tooling/release/build.py` to stage all six Python members and produce one `molto` wheel. Release CI checks the installed bundle. Packaging does not start inference or prove native model operations.

## Pages

- Overview shows residency, memory accounting, defaults, and serving counters.
- Models manages the complete local registry, including incomplete artifacts, memory planning, startup collections, storage, and configuration import/export. Model pages provide capability-aware settings, profiles, reusable templates, presets, generation-config import, recipes, MTPLX sidecar import, verification, updates, revisions, moves, and deletion previews.
- Add model searches Hugging Face and ModelScope, starts downloads, converts and quantizes local checkpoints, prepares supported diffusion models, and publishes to Hugging Face after explicit confirmation.
- Activity shows operation history, progress, errors, and the cancellation or retry actions the backend supports for each job.
- Monitoring shows active requests, queues, usage history, serving counters, and package provenance. Logs provides bounded file tails with severity filters. Cache provides metrics, separate hot/SSD clears, and a read-only prefix probe.
- Settings provides grouped server configuration, defaults, key creation/editing/revocation, main-key rotation, integration commands, provider checks, update information, and restart when the server supports it.
- Diagnostics runs throughput, accuracy, context-window, and ANE checks, with progress, history, result export, and explicit recommendation application. Local runs wait for inference to drain and can unload models or leave the tested model warm. Review the displayed impact before starting. Results are not automatically published.
- Cluster appears when distributed inference is enabled. It manages discovery, pairing, peers, verified deployment plans, staging, deployments, join keys, and CUDA/RDMA checks through the retained experimental protocol.

The server remains authoritative. An accepted unload can remain pending while requests finish; the dashboard polls until the model actually unloads. Model patches and profile application report saved settings separately from unload, reload, deferred transitions, and reload errors. A failed reload can follow a successful save.

Forms preserve edited fields during refresh and after failed saves. Settings patches send edited fields only. A model setting reset sends explicit `null` to restore the backend default. Applying a profile resets omitted universal settings; model-specific fields use an overlay. Explicit null profile values restore those fields' defaults.

Server settings use nested section patches. The response identifies `live_applied` and `restart_required` fields. Saving a restart-required setting does not restart the server. CLI and environment overrides can take precedence again on the next start. Only explicit edits are persisted; unrelated effective overrides are not copied into saved configuration. The integrated launcher provides supervisor support. A backend restart keeps the dashboard alive. If saved host/port edits change the effective binding, restart relaunches both processes; explicit CLI and environment overrides still take precedence.

Operation history survives a backend restart; native workers do not. Interrupted work is recorded as interrupted or failed, and does not resume automatically. Retry is explicit and may require a fresh provider token. Cancellation waits for native work to drain. The backend reserves files used by downloads, conversion, quantization, and publishing, and blocks conflicting moves or deletion. Deletion requires a current preview token and rechecks loaded/default/pinned/dependency guards.

Collections can mark selected models for startup preload through the persisted pinned-model mechanism. Saving a collection does not load models immediately. Clearing the collection's preload flag does not automatically unpin models.

Configuration import previews list field changes, `affected_model_ids`, blockers, and `can_apply`. Only changed records need to be idle and unloaded; unchanged loaded models do not block an import. The backend checks again when applying.

MTPLX sidecar import requires an unloaded compatible checkpoint within configured roots. It validates and stages changes, then replaces checkpoint files with rollback on failure. If rollback itself fails, the error identifies retained recovery files. Inspect those files before retrying.

Server resource inspection shows hardware memory, the Metal allocation cap, active limits, saved limits, and a draft tier preview without applying it. Any suggested OS wired-memory command is copy-only; the dashboard never executes it.

File maintenance preserves model identity when moving between configured roots. Moving a cached model moves its whole repository, including revisions, refs, and shared blobs. External path dependencies and active operations block the move. Virtual profile models do not have physical checkpoint actions. STT and speech-to-speech models can receive structural verification, but have no inference smoke probe.

Chat is outside this dashboard. Downloads and publishing contact their selected providers; model preparation and local diagnostics require the corresponding Molto runtime dependencies and compatible checkpoints.

## Connection and hosting

The browser talks only to the dashboard's same-origin `/api` routes. The server forwards allowlisted operations to Molto with the main bearer key. The connected main key stays in server memory; the browser receives an opaque HTTP-only, SameSite Strict session cookie, with Secure enabled for HTTPS. Sessions last eight hours. An inference-backend restart keeps Nitro and these sessions alive. A dashboard restart, full application restart, or effective public host/port change clears them. Rotating the main key through this dashboard updates the current session and invalidates other sessions connected with the previous key. The key-management page can deliberately reveal keys returned by the backend; treat an unlocked dashboard as privileged access. API responses use `Cache-Control: no-store`, and mutations require a matching request origin.

Run a single dashboard server process. Its session store is process-local and bounded to 128 active sessions. A load-balanced deployment would need a shared session store. Never put the main key in a `VITE_*` variable or client-side storage.

Live pages poll while visible; the logs page also supports pausing refresh. Authentication failures stop those polls until reconnection. Failed refreshes show the last received data with an error. Mutations refresh related state and are not retried automatically.

## Regenerate API types

The schema and types come from the actual Molto FastAPI management and cluster routers in `apps/server`. With the repository's `uv` environment available:

```sh
pnpm --filter molto-dashboard generate-api
pnpm typecheck
```

The root `tooling/export_management_schema.py` exporter writes `packages/contracts/generated/openapi.json`; OpenAPI TypeScript generates `packages/contracts/generated/api-types.ts`. Dashboard code imports both through the `@molto/contracts` workspace package without starting a server or loading model weights. Regenerate both after backend contract changes; do not edit generated types by hand.

## Verify

Oxlint runs with type-aware linting and compiler diagnostics through
`oxlint-tsgolint`. Oxfmt handles formatting and Tailwind class ordering.
Root `pnpm check` checks formatting and linting for both Python and TypeScript;
`pnpm typecheck` runs native TypeScript 7 and the selected maintained Python interface/build modules. The separate TypeScript 5 dependency supplies
the compiler API used by OpenAPI generation, while checks use the Go compiler.

```sh
pnpm check
pnpm typecheck
pnpm --filter molto-dashboard exec playwright install chromium
pnpm --filter molto-dashboard test
```

Browser tests start the production Node build on port 3007 and a Molto management test server on port 8765. The test server uses the actual routes, management service, authentication, and profile/settings persistence with synthetic engines and metrics. The suite combines actual management routes and persistence with mocked browser contracts for optional integrations. Synthetic engines do not prove native conversion, quantization, diagnostic quality, provider uploads, or remote cluster deployment. It does not load real model weights. `MOLTO_SOURCE` can override the default workspace root when testing another checkout.
