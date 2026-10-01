# Backend architecture

This document is for developers adding an inference feature or a client for
the management API. oMLX combines a Python inference server with a TanStack Start dashboard. The
former macOS app and old admin pages remain removed.

## Request paths

```text
HTTP client --> public Nitro server
                  |-- dashboard and same-origin session routes
                  |-- inference HTTP/stream/WebSocket proxy --> private FastAPI
                  `-- management gateway --------------------> private FastAPI
                                                                   |
                                                             engine pool
```

`apps/cli/src/omlx_cli/application.py` owns both child processes. It starts Nitro and verifies
its instance readiness before launching inference on an inherited ephemeral
loopback socket. Public port conflicts fail before model startup. Nitro uses
native HTTP proxying for inference and a bounded WebSocket proxy for realtime
transcription. The launcher supports one public bind address and shuts down
child process groups together.

The CLI starts the application supervisor. The supervisor invokes
`apps/server/src/omlx_server/bootstrap.py` for the private inference child;
bootstrap initializes configuration and runtime resources. The CLI does not own
FastAPI route initialization.

`apps/server/src/omlx_server/server.py` exposes `create_app()`, constructing an
application with its own `ServerState` and protocol controller. Requests resolve
that controller from application state. Protocol handling is split into
OpenAI, Anthropic, Responses, streaming, inventory, transport, and error modules
inside `apps/server/src/omlx_server/`; instances do not share a global server
singleton.

Model adapters under `packages/runtime/src/omlx_runtime/engine/` handle the
supported tasks. `packages/runtime/src/omlx_runtime/engine_pool.py` discovers
models, loads engines, tracks memory, and handles unloads. The scheduler and
cache implementations serve inference requests. Management reads immutable
`ModelView` records from `get_model_view()` and calls public load/unload,
resource, cache, and admission operations. It does not access the pool's private
entry dictionaries or scheduler state. See [repository architecture](architecture.md)
for the allowed package dependency graph.

Image checkpoint identity and operation policy live in `packages/runtime/src/omlx_runtime/diffusion`.
Preparation owns bounded acquisition and atomic saved-checkpoint publication;
adapters own native class selection and request translation. Image routes
validate before pool acquisition, and the image engine owns serialized MLX
lifecycle work. See [image models](image-models.md) for supported operations and
verification limits. This boundary is separate from DiffusionGemma text
generation and DFlash drafting.

`packages/runtime/src/omlx_runtime/diffusion/cache.py` owns bounded, materialized prompt and reference embeddings,
cache keys, and native predictor residency. `packages/runtime/src/omlx_runtime/diffusion/batching.py` translates
matching FLUX.2 seed variants into a native tensor batch through request-local
hooks. The native denoising loop remains in mflux. The image engine owns batch
memory admission and executes generation, cache clearing, switching, and
release under one lifecycle lock on the shared MLX executor.

`apps/server/src/omlx_server/api/management_routes.py` mounts the core routes and domain routers for
model options, workspace, acquisition, server settings, monitoring, and diagnostics.
Services take a `ManagementContext` with explicit pool/settings references and
runtime callbacks. `apps/server/src/omlx_server/api/management_dependencies.py` retrieves that context from
application state and lazily creates a process-owned `ManagementRuntime`.
Importing the routes does not start the server, download weights, or create workers.

`ManagementRuntime` shares operation admission, mutation coordination, and file
reservations across acquisition and workspace maintenance. Its managers use the
existing downloader, converter, quantizer, and publishing implementations. Durable
records describe work; native workers and provider credentials remain process-owned.
Restart marks abandoned work rather than resuming it. Server shutdown drains runtime
work before releasing its pool resources.

Local diagnostics use the pool's `exclusive_management` gate. It closes external
engine admission, waits for leases and scheduler work, then allows the owning run
to acquire its engine. It does not hold the pool lock throughout inference. Native
preparation uses its separate exclusive gate and retains ownership through cleanup.
File readers and writers reserve paths until their workers drain, so maintenance
cannot remove or move an active source or output. Workspace operations recheck
state and deletion preview tokens under mutation coordination. Configuration
imports coordinate only changed model records and expose blockers before apply.
MTPLX import stages checkpoint changes and rolls back failed replacement; an
incomplete rollback retains recovery files. Startup collections use persisted
pinning rather than starting a second preload mechanism.

`packages/management/src/omlx_management/diffusion_jobs.py` owns local calibration and quantization jobs,
progress, durable history, and cooperative cancellation. It runs preparation
on the shared MLX executor under the pool's exclusive admission gate, waiting
for inference to drain and retaining ownership through worker cleanup.

`apps/server/src/omlx_server/auth.py` checks bearer keys for both API families. All management routes
require the main key. The retained inference load route accepts a subkey. On a
loopback-only bind, only an explicit `skip_api_key_verification` setting can
bypass the management check. Inference retains its own loopback behavior.
The server rejects non-loopback binding without a main key. See the
[management API](management-api.md) for exact access rules.

## State and persistence

`packages/config/` owns persistence and validation; runtime converts settings
into scheduler configuration through `settings_adapter.py`. Configuration imports
do not load MLX or the server. Global configuration is stored under the selected base path in
`settings.json`; model-specific settings and profiles have their own persisted
store. The default base path is `~/.omlx`, while an explicit base path and an
existing macOS app base-path pointer can select another directory. A CLI
option overrides a saved setting for startup. Model discovery reads model
directories; it does not copy checkpoints into the data directory.

The engine pool owns loaded models only for the current server process.
Session metrics reset on restart. All-time counters and hourly usage history
have separate persisted stores. The optional SSD KV cache stores reusable
blocks and can rebuild its index from compatible saved blocks after restart.
These stores are separate from model settings. The legacy flat settings route reports `requires_restart`. The full nested
server settings route reports field lists in `live_applied` and
`restart_required`. It persists explicit edits without saving unrelated CLI or
environment overrides. Process restart is available only through a supported
supervisor callback.

## Dashboard and backend boundary

Nitro serves the dashboard, browser sessions, and public API proxy. FastAPI
provides `/management/v1/*` and `/v1/*` on its private listener. The browser management
gateway uses the session's main bearer key. Native clients use
`/api/management/v1/*` with their explicit main bearer key; inference clients
retain their own credentials. Raw `/management/v1/*` and `/admin/*` are blocked by the public
proxy; browser management uses `/api/omlx/*` with its opaque session cookie.
FastAPI does not serve dashboard HTML or browser sessions. The management API covers inventory and library maintenance, model configuration,
profiles/templates/presets, acquisition and preparation, publishing, operation
history, server settings and keys, monitoring/logs/cache, and diagnostics.
Experimental cluster management remains under its retained protocol routes.
See [model control](model-control.md) and the [management API](management-api.md).
The backend serves no browser chat or dashboard assets.

The TanStack Start dashboard maintains a process-local main-key session store.
Its browser receives an opaque HttpOnly cookie and sends same-origin requests;
the dashboard server forwards only allowed methods and paths. Main-key rotation
updates the initiating session and invalidates sessions using the old key.
Multiple dashboard processes would need shared session state. Backend restarts
retain the Nitro process and its sessions. Restart requests reload saved settings
with CLI/environment precedence; a changed effective public host or port restarts
both processes and clears sessions. Guarded first-run key creation requires a
loopback public bind, a direct local client, no forwarding headers, matching key
confirmation, and no existing main key.

Installed release wheels include Nitro assets and a standalone macOS ARM64 Node
runtime under `apps/cli/src/omlx_cli/_dashboard`. Source checkouts use `apps/dashboard/.output` and Node
on PATH or `OMLX_NODE`. The bundle script verifies the Node archive checksum;
wheel CI builds and inspects the installed bundle. Source-only
`omlx serve --dashboard-dev` substitutes native Vite/Nitro on the same configured
public binding, retains the private backend child, and uses strict port checks.
It requires Node and installed pnpm dependencies, but no production build; the
flag is not persisted and installed distributions reject it. These packaging checks do not
exercise native model operations. Resource inspection
reads hardware and active memory limits and previews draft settings. Suggested OS
commands remain copy-only.

## Validation limits

The unit tests can check route validation and service behavior with synthetic
engines. They do not prove throughput, memory limits, or output parity for a
real checkpoint. An engine or kernel change needs a representative model on
Apple Silicon, with the hardware and settings recorded. See
[testing](TESTING.md).
