# Backend architecture

This document is for developers adding an inference feature or a client for
the management API. oMLX is a Python server. The former macOS app and web
admin pages are outside this repository's backend boundary.

## Request paths

```text
HTTP client
    |-- /v1/* inference routes --> engine pool --> task engine --> MLX runtime
    |-- /management/v1/* -------> management service --> engine pool and settings
    `-- /health ----------------> startup readiness
```

`omlx/cli.py` starts the foreground server. `omlx/server.py` owns the FastAPI
application, inference routes, and runtime state. Model adapters under
`omlx/engine/` handle the supported tasks. `omlx/engine_pool.py` discovers
models, loads engines, tracks memory, and handles unloads. The scheduler and
cache implementations serve inference requests; the management API calls the
same engine pool so its load state matches the running server.

`omlx/api/management_routes.py` maps HTTP requests to
`omlx/services/management.py`. The service takes a `ManagementContext` with
explicit references to the pool, model settings manager, global settings, and
callbacks for the current default model and sampling configuration. Importing
the service does not start the server or create a downloader. The route module
gets its context from application state during a request.

`omlx/auth.py` checks bearer keys for both API families. All management routes
require the main key. The retained inference load route accepts a subkey. On a
loopback-only bind, only an explicit `skip_api_key_verification` setting can
bypass the management check. Inference retains its own loopback behavior.
The server rejects non-loopback binding without a main key. See the
[management API](management-api.md) for exact access rules.

## State and persistence

Global configuration is stored under the selected base path in
`settings.json`; model-specific settings and profiles have their own persisted
store. The default base path is `~/.omlx`, while an explicit base path and an
existing macOS app base-path pointer can select another directory. A CLI
option overrides a saved setting for startup. Model discovery reads model
directories; it does not copy checkpoints into the data directory.

The engine pool owns loaded models only for the current server process.
Session metrics reset on restart. All-time counters and hourly usage history
have separate persisted stores. The optional SSD KV cache stores reusable
blocks and can rebuild its index from compatible saved blocks after restart.
These stores are separate from model settings. A management request that
changes scheduler construction options can require a server restart; check
the `requires_restart` response field.

## Boundary for a separate dashboard

A dashboard is a client of `/management/v1/*` and `/v1/*`. It should keep its
own UI state and pass the main bearer key for management. The backend does
not serve dashboard HTML or provide a browser session. The management API
currently covers inventory, load and unload, settings, profiles, stats, and
cache inspection or clearing. It does not replace every operation from the
old `/admin/api/*` routes. See [model control](model-control.md) for the
available operations and [README](../README.md#migration-from-the-earlier-app-and-web-ui)
for migration notes.

## Validation limits

The unit tests can check route validation and service behavior with synthetic
engines. They do not prove throughput, memory limits, or output parity for a
real checkpoint. An engine or kernel change needs a representative model on
Apple Silicon, with the hardware and settings recorded. See
[testing](TESTING.md).
