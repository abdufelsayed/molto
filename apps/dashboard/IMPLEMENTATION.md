# Complete non-chat management dashboard

Carry over the old admin's useful capabilities with workflows organized around
models, server configuration, operations, monitoring, diagnostics, and clusters.
Keep TanStack Start, shadcn Base UI preset b43fOHkIM, type-aware Oxlint, Oxfmt,
and the native TypeScript compiler. The old reference remains at
/tmp/molto-old-admin-0b07cdd. Do not restore bundled HTML or chat.

## Shared contracts and ownership

Authenticated backend feature routers have no prefix of their own and export `router`.
The supervisor mounts them under the existing authenticated `/management/v1`
router. First-run setup is mounted separately under `/management/v1/setup`; its public BFF requires a direct local transport and creates a key before normal login. Use `molto_server.api.management_dependencies.get_context`, `get_service`, and
`get_runtime`, never import the server to acquire state. New domain services
must be importable without starting jobs or loading weights.

ManagementContext now has optional get_api_key, set_api_key, get_bind_host,
get_server_info callbacks and runtime_state. Existing test contexts remain valid.
The supervisor owns server.py wiring, management_routes.py inclusion, generated
API types, the dashboard gateway, navigation, shared request utilities and final
integration. Domain owners must not edit those files.

The acquisition owner implements management_runtime.py. ManagementRuntime takes
a ManagementContext and exposes context, settings, control (ModelControl),
operation_lock (asyncio.Lock), mutation_lock (asyncio.Lock), and lazily created
download/quantization/upload managers. Workspace and diagnostics owners share
runtime.control for operation history and operation_lock for exclusive GPU work.
Do not persist credentials in operation payloads. Cancellation must retain
exclusive admission until native work has stopped. Shutdown releases workers.

Frontend domain modules use `managementRequest<T>(path, options)` from
src/features/management/request.ts. Paths are relative to /management/v1 and
the request utility adds /api/molto/. Mutations accept options.method and
options.body; options.signal controls queries. API errors use existing ApiError.
Domain clients may define explicit wire types initially; supervisor regenerates
the OpenAPI schema after integration. No raw JSON-only product workflows.
Reuse existing FieldGroup/Field and Base UI components; preserve edits across
refresh, render error/empty/loading states, disable duplicate submissions,
invalidate ['molto'] after successful mutations. No automatic mutation retries.

## Feature slices

1. Server/auth backend owns services/management_server.py,
   api/management_server_routes.py and corresponding tests. GET server/settings
   returns sections plus field metadata (key, section, label, description, type,
   default, choices, minimum, maximum, secret, restart_required). PATCH accepts
   nested section patches; validate a candidate before saving/applying effects.
   GET server/defaults, GET server/info, POST server/restart, GET server/update.
   GET auth/keys returns main_key string and sub_keys [{id,key,name,created_at}],
   POST auth/subkeys {name,key?}, PATCH auth/subkeys/{id} {name?,key?}, DELETE,
   PATCH auth/main-key {key}, PATCH auth/policy. Opaque subkey IDs, duplicate
   checks, rollback on failed persistence and live main-key rotation are required.
   Full server/network/model roots/resource/cache/default/integration settings,
   launch commands and web-search testing must be covered.
2. Model backend owns management_models.py and ManagementService model sections,
   new api/management_model_routes.py and service modules/tests. Restore every
   useful persisted model field with validation and compatibility gates. Add
   settings field metadata/capabilities, templates CRUD/apply, presets refresh,
   recipes, full reset and generation-config import. Profile application must
   reconcile loaded engines, preserve explicit resets, and report transitions.
   GET model-options, GET models/{id}/options; template and helper contracts are
   coordinated directly with the model UI owner.
3. Workspace backend owns api/management_workspace_routes.py,
   services/management_workspace.py and tests. /workspace/registry, /storage,
   /plan, /collections CRUD/load, /export and /import preview/apply; per-model
   /workspace/models/{id}/verify, check-update, stage-update, revision, move,
   delete-plan and delete. Canonical IDs, server-derived destinations, loaded/
   pinned/default/dependency checks enforced at mutation, safe drain/no abort,
   structural versus smoke verification distinguished, import validation,
   one lifecycle authority, truthful operation capabilities are required.
4. Acquisition backend owns management_runtime.py, api/management_acquisition_routes.py,
   services/management_acquisition.py and tests. /acquisition/{hf|ms}/search,
   recommended, info, downloads; /acquisition/prepare/models, convert, estimate,
   quantize; /acquisition/publish/validate and start; /operations list/get/cancel/
   retry/delete. Retained services do real work only on explicit requests.
   Progress and history persist, secrets never appear in jobs, retry/cancel are
   advertised only where implemented. Diffusion jobs remain separately supported.
5. Monitoring backend owns management_monitoring.py and its router/tests.
   /monitoring/activity, usage, logs, versions; stats reset and cache probe.
   Canonical model filters, date ranges, disabled/unavailable history states,
   rotated logs with bounded tails and server-side level filtering, reliable
   engine coverage, safe probe and main-key authorization are required.
6. Diagnostics backend owns services/diagnostics/ plus management_diagnostics.py,
   its router/tests. Restore throughput, accuracy, context and ANE tuning using
   real runners. /diagnostics/capabilities, runs GET/POST, runs/{id} GET/cancel/
   results. One exclusivity boundary, cooperative cancellation, durable records,
   meaningful progress, local/external target support, no unsolicited result
   publication, no fake success. Coordinate with UI owner on request schema.
7. Cluster frontend covers retained experimental APIs through a fixed gateway
   mapping. Discovery, enrollment/join keys, peers, inventory/catalogue, probes,
   plans, staging, deployment, load/unload, replan and diagnostics have guided
   explicit actions, truthful gates and no arbitrary endpoint/command execution.

## Proof and acceptance

Every backend owner runs focused tests including permission, validation,
failed persistence, busy state and success cases at their boundary. Every UI
owner runs formatting, type-aware lint and native typecheck for their files and
provides tests or exact browser steps. The supervisor checks full feature parity,
reviews integrated code, regenerates types, runs build and browser tests against
actual management routes with disposable state and synthetic engines, then
obtains independent review. Real model, network publication, system settings,
remote cluster changes are not executed as validation. Disposable launcher processes are started and terminated to verify supervision and shutdown.

## Coverage ledger

- [x] Server settings, resources, directories, network and integrations
- [x] Main key rotation, subkey reveal/create/edit/revoke and auth policy
- [x] Full model configuration, capabilities, profile/template/preset helpers
- [x] Library, health, memory planning, collections, storage/revisions/deletion
- [x] Discovery/downloads, conversion/quantization/publishing and durable activity
- [x] Cache settings, clearing and diagnostics; local diffusion preparation UI
- [x] Live request activity, historical usage, logs and engine provenance
- [x] Throughput/accuracy/context diagnostics and ANE tuning
- [x] Conditional cluster management
- [x] Integrated gateway/session handling, generated contracts and navigation
- [x] Backend tests, browser proof, type-aware lint, format, typecheck and build
- [x] Independent review findings resolved and documentation current

The bundled command uses native Nitro proxying with a separate private FastAPI process. Installed wheels carry Node and dashboard assets; source development uses `molto serve --dashboard-dev`. The integration record in `../docs/dashboard-integration-plan.md` contains milestone and verification evidence.
