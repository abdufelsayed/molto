# Dashboard integration

Recorded on 2026-10-01. This records the accepted scope, delivered architecture,
and verification for the non-chat dashboard migration.

## Accepted scope

The dashboard belongs in `apps/dashboard/` in the Molto repository. One `molto serve`
command runs the public TanStack Start/Nitro web application and a separate
private FastAPI inference process. Inference clients and the dashboard use the
same public host and port. The user selected a command distribution.

Keep shadcn/ui preset `b43fOHkIM`, Oxfmt, type-aware Oxlint, and the native
TypeScript compiler. Transfer useful old admin functionality with clearer
workflows, explicit effects, preserved drafts, and truthful failure states.
Browser chat is outside this scope. The reference checkout remains at
`/tmp/molto-old-admin-0b07cdd`.

## Delivered architecture

- Native Nitro owns the public listener and proxies the supported inference and
  peer protocol paths to an inherited ephemeral loopback FastAPI socket.
- HTTP proxying preserves caller credentials, streaming, cancellation, uploads,
  binary responses, errors, and status codes. Realtime audio uses native CrossWS.
- Raw management and admin routes remain private. The public dashboard gateway
  uses allowlisted methods and paths with an opaque HttpOnly administrator session.
- Local first-run setup creates and persists a main key, then uses the ordinary
  login endpoint to establish the session. It requires a loopback public bind,
  an actual local peer, matching confirmation, and no existing main key.
- After setup, the bundled loopback dashboard opens local sessions automatically.
  A launcher-generated capability permits a private current-key handoff, with
  loopback binding, verified peer, URL and origin checks and no forwarded requests.
  Remote and standalone dashboards retain manual main-key sign-in. The browser
  receives an opaque session; explicit disconnect suppresses automatic sign-in.
  Nitro handlers and TanStack SSR share one process-local session store.
- The launcher owns readiness, crash recovery, shutdown, and restart. A backend
  restart retains Nitro and its sessions; effective public host/port changes
  restart both processes, respecting CLI/environment precedence.
- Release wheels include the production dashboard and verified standalone Node
  runtime. Installed usage requires neither pnpm nor a separate Node installation.
- Source-only `--dashboard-dev` runs native Vite/Nitro on the configured public
  port, supervised with the same private backend. No custom public HTTP server.
- One public bind address is supported. Use `0.0.0.0` for all IPv4 interfaces.
  Multiple addresses are rejected before saving or starting.

## Functionality coverage

- [x] Repository integration, tooling, generated contracts and navigation.
- [x] Complete server settings, directories, networking, resource previews,
      cache settings, generation defaults, integrations and tools.
- [x] First-key setup, main-key rotation, subkey reveal/create/edit/revoke,
      authentication policy and session handling.
- [x] Full model settings, capability gates, profiles, templates, presets,
      generation import, helper recipes, reset and sidecar import.
- [x] Library filters, health checks, memory planning, collections and startup
      preload; configuration import/export; storage, revisions, moves and deletion.
- [x] Hub discovery/downloads, conversion/quantization, diffusion preparation,
      explicit publishing and durable operation activity.
- [x] Live request activity, usage history, bounded rotated logs, engine
      provenance, statistics reset and cache inspection/clear/probes.
- [x] Throughput, accuracy, context and ANE diagnostics with durable history,
      exclusive admission and cancellation that drains native work.
- [x] Conditional experimental cluster management and public peer protocols.
- [x] Independent reviews of model moves, import rollback, monitoring,
      diagnostics, resource detection, authentication, proxy and launcher.
- [x] Desktop/mobile HCI inspection, including visible library filters and
      distinct hub labels with explicit boolean states.
- [x] Final installed-bundle verification and milestone delivery.

## Verification

The final integrated backend run passed 1,358 tests. Independent launcher/CLI
review passed 127 tests, including readiness identity, crash recovery, late
shutdown children, and safe process ownership.
All 64 browser tests pass against production Nitro and actual management routes
with disposable persistence and synthetic engines. Six native HTTP/WebSocket
proxy tests pass. Formatting, type-aware lint, native TypeScript and production
build pass. The first-run browser test covers persisted keys, normal sessions,
reload, rejected repeat setup, and failed persistence with retained drafts.

The local-access fix passes all 70 browser tests, eight proxy tests, and 80
targeted authentication, setup and launcher tests. It covers automatic local
entry, inference-key creation, reload, explicit disconnect, external key rotation,
public-bind login and rejection of forged origins, peers and forwarding headers.
The rebuilt native wheel was installed and the local application restarted.
A fresh browser reached the live key-creation form without entering a key,
retained access after reload and discovered all nine existing models. Credentials
and all 62 safetensor files retained their prior values, sizes and inodes.

The final wheel was built and installed outside the checkout. With Node and
pnpm absent from PATH, one command launched Nitro and private FastAPI. Public
health/models, initial key persistence, normal sessions, main/subkey authentication,
subkey edit/revocation, backend-only restart/session retention, public-port
restart/session invalidation, and shutdown passed. Disabled cluster support
created no helper. Shutdown exited 0 with no surviving children. Native Vite
development also passed live checks and three consecutive clean shutdowns.

No real model downloads, weight loading, model preparation, provider publishing,
remote cluster deployments, or OS memory-limit mutations were used as dashboard
validation. Native jobs and experimental cluster execution require their runtime
extras and compatible hardware/checkpoints. Release CI and the updated Homebrew
formula are implemented; a new release has not been published. The formula's
stable URL predates the dashboard, so this source version requires HEAD until
an actual release updates the URL/checksum.

## Milestones

- `3489e45` — dashboard moved into Molto; management features, model moves and
  native public inference proxy restored and verified.
- Bundled startup, local setup and final integration — verified. The delivery
  commit includes the launcher, distribution build, guarded setup, development
  mode, final HCI fixes and this verification record.
