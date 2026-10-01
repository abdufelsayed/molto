# oMLX dashboard

The oMLX dashboard lives in `dashboard/` in the oMLX repository. It uses TanStack Start, TanStack Query, and shadcn/ui. Preset `b43fOHkIM` supplies the Nova style, mauve base, pink accents, Geist font, and Lucide icons.

The bundled application will serve the dashboard and proxy inference through one public port, with the Python backend kept private. The commands below describe the current development setup while that runtime integration is being completed.

## Run

Start your oMLX server separately with its main API key configured. The dashboard connects to its `/management/v1` API; inference subkeys do not grant management access.

From this directory:

```sh
pnpm install --frozen-lockfile
OMLX_API_URL=http://127.0.0.1:8000 pnpm dev
```

Open <http://127.0.0.1:3000>, select **API access**, and enter the main key. Set `OMLX_API_URL` to your server's actual HTTP(S) origin, without a path. It defaults to `http://127.0.0.1:8000` and is read by the dashboard server at runtime.

For production:

```sh
pnpm build
OMLX_API_URL=http://127.0.0.1:8000 PORT=3000 pnpm start
```

The build produces a Node server in `.output/server` and browser assets in `.output/public`. Development and production default to loopback. To serve on another interface, set `HOST` for production or pass `--host` to the development command. Use HTTPS when accessing the dashboard across a network.

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

Server settings use nested section patches. The response identifies `live_applied` and `restart_required` fields. Saving a restart-required setting does not restart the server. CLI and environment overrides can take precedence again on the next start. Only explicit edits are persisted; unrelated effective overrides are not copied into saved configuration. Restart is offered only when oMLX advertises supervisor support.

Operation history survives a backend restart; native workers do not. Interrupted work is recorded as interrupted or failed, and does not resume automatically. Retry is explicit and may require a fresh provider token. Cancellation waits for native work to drain. The backend reserves files used by downloads, conversion, quantization, and publishing, and blocks conflicting moves or deletion. Deletion requires a current preview token and rechecks loaded/default/pinned/dependency guards.

Collections can mark selected models for startup preload through the persisted pinned-model mechanism. Saving a collection does not load models immediately. Clearing the collection's preload flag does not automatically unpin models.

Configuration import previews list field changes, `affected_model_ids`, blockers, and `can_apply`. Only changed records need to be idle and unloaded; unchanged loaded models do not block an import. The backend checks again when applying.

MTPLX sidecar import requires an unloaded compatible checkpoint within configured roots. It validates and stages changes, then replaces checkpoint files with rollback on failure. If rollback itself fails, the error identifies retained recovery files. Inspect those files before retrying.

Server resource inspection shows hardware memory, the Metal allocation cap, active limits, saved limits, and a draft tier preview without applying it. Any suggested OS wired-memory command is copy-only; the dashboard never executes it.

File maintenance preserves model identity when moving between configured roots. Moving a cached model moves its whole repository, including revisions, refs, and shared blobs. External path dependencies and active operations block the move. Virtual profile models do not have physical checkpoint actions. STT and speech-to-speech models can receive structural verification, but have no inference smoke probe.

Chat is outside this dashboard. Downloads and publishing contact their selected providers; model preparation and local diagnostics require the corresponding oMLX runtime dependencies and compatible checkpoints.

## Connection and hosting

The browser talks only to the dashboard's same-origin `/api` routes. The server forwards allowlisted operations to oMLX with the main bearer key. The connected main key stays in server memory; the browser receives an opaque HTTP-only, SameSite Strict session cookie, with Secure enabled for HTTPS. Sessions last eight hours and expire on dashboard server restart. Rotating the main key through this dashboard updates the current session and invalidates other sessions connected with the previous key. The key-management page can deliberately reveal keys returned by the backend; treat an unlocked dashboard as privileged access. API responses use `Cache-Control: no-store`, and mutations require a matching request origin.

Run a single dashboard server process. Its session store is process-local and bounded to 128 active sessions. A load-balanced deployment would need a shared session store. Never put the main key in a `VITE_*` variable or client-side storage.

Live pages poll while visible; the logs page also supports pausing refresh. Authentication failures stop those polls until reconnection. Failed refreshes show the last received data with an error. Mutations refresh related state and are not retried automatically.

## Regenerate API types

The schema and types come from the actual oMLX FastAPI management router in the parent directory. With the repository's `uv` environment available:

```sh
pnpm generate-api
pnpm typecheck
```

For another source location, use `OMLX_SOURCE=/path/to/omlx pnpm generate-api`. The script writes `src/lib/openapi.json` and `src/lib/api-types.ts` without starting a server or loading model weights. Regenerate both after backend contract changes; do not edit generated types by hand.

## Verify

Oxlint runs with type-aware linting and compiler diagnostics through
`oxlint-tsgolint`. Oxfmt handles formatting and Tailwind class ordering.
`pnpm check` checks both formatting and linting; `pnpm typecheck` runs the
native TypeScript 7 compiler. The separate TypeScript 5 dependency supplies
the compiler API used by OpenAPI generation, while checks use the Go compiler.

```sh
pnpm lint
pnpm check
pnpm typecheck
pnpm exec playwright install chromium
pnpm test
```

Browser tests start the production Node build on port 3007 and an oMLX management test server on port 8765. The test server uses the actual routes, management service, authentication, and profile/settings persistence with synthetic engines and metrics. The suite combines actual management routes and persistence with mocked browser contracts for optional integrations. Synthetic engines do not prove native conversion, quantization, diagnostic quality, provider uploads, or remote cluster deployment. It does not load real model weights. `OMLX_SOURCE` can override the default parent repository when testing another checkout.
