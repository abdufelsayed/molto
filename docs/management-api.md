# Management API

This is the HTTP contract for a script or separate dashboard that controls a
running oMLX server. The base URL is `http://127.0.0.1:8000` by default;
all routes below start with `/management/v1`. The management API controls the
same engine pool used by inference requests. It does not manage the server
process. Use `omlx serve` or the Homebrew service for that.

## Authentication

Set a main key before exposing the server to a network. Pass it as a bearer
token:

```bash
curl http://127.0.0.1:8000/management/v1/models \
  -H "Authorization: Bearer $OMLX_API_KEY"
```

The main key grants all management routes, including model load. Inference
subkeys cannot use `/management/v1/*`; the retained
`POST /v1/models/{model_id}/load` endpoint accepts them for CLI
compatibility. The retained `POST /v1/models/{model_id}/unload` endpoint now
requires the main key. A browser cookie from the former admin UI is not a
management credential. Configure a main key through
`OMLX_API_KEY`, `omlx serve --api-key`, or saved settings. The management API
requires that key even on loopback. An explicit `skip_api_key_verification`
setting permits keyless management only on a loopback bind. Non-loopback
binding requires a main key and does not allow that bypass. A missing or
wrong key returns HTTP 401 with `WWW-Authenticate: Bearer`.

An explicit `auth.allow_unauthenticated_inference` setting affects inference,
not management permissions.

## Routes

| Method and path | Result or effect |
| --- | --- |
| `GET /models` | Discovered models, load state, and supported persisted model settings |
| `GET /state` | Default model, memory ceiling, counts, and compact load state |
| `POST /models/refresh` | Re-read model settings and rescan configured model directories |
| `POST /models/{model_id}/load` | Load a discovered model; requires main key |
| `POST /models/{model_id}/unload` | Request unload; may return HTTP 202 while unloading |
| `GET /settings` | Current global sampling and selected scheduler settings |
| `PATCH /settings` | Persist supported global setting changes |
| `GET /models/{model_id}/settings` | Model-specific settings |
| `PATCH /models/{model_id}/settings` | Persist a subset of model-specific settings |
| `GET /models/{model_id}/profiles` | Saved profiles for a discovered model |
| `POST /models/{model_id}/profiles` | Create a profile |
| `PUT /models/{model_id}/profiles/{name}` | Update or rename a profile |
| `DELETE /models/{model_id}/profiles/{name}` | Delete a profile |
| `POST /models/{model_id}/profiles/{name}/apply` | Apply a profile to model settings |
| `GET /stats` | Session or all-time counters; optional `model_id` query |
| `GET /cache` | Cache statistics for loaded models and the SSD cache path |
| `POST /cache/{hot|ssd}/clear` | Clear the selected cache tier |
| `POST /diffusion/calibrations` | Queue calibration of a discovered local diffusion checkpoint; HTTP 202 |
| `POST /diffusion/quantizations` | Queue calibrated transformer quantization from local floating-point weights; HTTP 202 |
| `GET /diffusion/jobs` | Preparation job history and current progress |
| `GET /diffusion/jobs/{job_id}` | One preparation job |
| `POST /diffusion/jobs/{job_id}/cancel` | Request cancellation; running work drains before reaching `cancelled` |

`model_id` is a discovered model ID. Read it from `GET /models`; do not
derive it from a display alias. All write bodies are JSON. Unknown patch
fields are rejected. Responses use JSON and report errors in `detail`.

## Load and settings example

```bash
BASE=http://127.0.0.1:8000/management/v1
MODEL=my-model

curl -X POST "$BASE/models/$MODEL/load" \
  -H "Authorization: Bearer $OMLX_API_KEY"

curl -X PATCH "$BASE/models/$MODEL/settings" \
  -H "Authorization: Bearer $OMLX_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"temperature":0.7,"ttl_seconds":300}'
```

For a model setting, a supplied `null` resets that field to its default.
The response includes `requires_reload` when a loaded engine's runtime
configuration differs from the new saved settings. The backend tries to
unload the old engine without aborting active requests. It returns
`auto_unloaded`, `auto_reloaded`, `reload_deferred`, and `reload_error` to
describe that transition. An idle pinned model reloads immediately. An
in-use model unloads after active work drains, so check `/state` before
sending more work. `PATCH /settings` accepts global sampling and selected
scheduler fields such as `temperature` and `max_concurrent_requests`; its
response includes `requires_restart` for scheduler construction settings.
An omitted field stays unchanged.

Profiles contain a `name`, a `settings` object with allowed model setting
fields, and optional display name, description, exposed model flag, and API
name. For example:

```bash
curl -X POST "$BASE/models/$MODEL/profiles" \
  -H "Authorization: Bearer $OMLX_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"name":"focused","settings":{"temperature":0.2}}'
```

The accepted fields and validation bounds come from
`omlx/services/management_models.py`. Inspect `GET /settings` or
`GET /models/{model_id}/settings` for the current values before patching.

## Cache and metrics

`GET /stats` defaults to `scope=session`; use `scope=alltime` for persisted
all-time counters. It does not return hourly usage history; see
[local usage history](usage-analytics.md). `GET /cache` reports live engine
cache statistics, where available, and the configured SSD cache directory.
Clearing `hot` reclaims loaded engines' hot cache. Clearing `ssd` also removes known saved cache
blocks for unloaded models. A clear may fail with HTTP 503 if a cache file
cannot be removed. These operations affect reuse and may make a later request
recompute a prefix.

Diffusion models report bounded prompt/reference embedding entries, tensor
bytes, hits, misses, evictions, and native prediction-factory reuse in the same
`GET /cache` response. They retain no SSD entries. Clearing `hot` releases their
embeddings and prediction factories on the shared MLX executor, while keeping
the loaded model. Active requests prevent management cache clearing. See
[image performance](image-models.md#residency-and-performance) for pipeline limits.

## Migration and limits

The old `/admin` browser pages, login cookie, and most `/admin/api/*` routes
are removed. Update management clients to use bearer authorization and the
routes above. Experimental cluster protocol routes under `/admin/api/cluster`
remain for cluster peers and are outside this management API; they have
separate enrollment rules and a main-key check for management actions.

The diffusion preparation routes below replace none of the old admin job
contracts. There are no management routes here for downloads, general model
conversion, text quantization, update staging, benchmark jobs, storage moves, or browser chat.
The backend may retain CLI or service utilities for some of those tasks, but
their old admin HTTP contracts are not available. See
[backend architecture](backend-architecture.md) and
[model control](model-control.md).

## Local diffusion preparation

These jobs accept discovered model IDs and complete local checkpoints,
including downloaded Hugging Face cache snapshots. They never fetch missing
files. Calibration and calibrated quantization currently support the exact
identities `flux2-klein-4b` and `qwen-image-2.1`. `GET /models` exposes their
`diffusion.calibration` and `diffusion.calibrated_quantization` availability.
See [image preparation](image-models.md#calibrate-and-quantize-offline) for
component policy and verification limits.

Start calibration with 1–32 text-to-image tasks:

```bash
curl -X POST "$BASE/diffusion/calibrations" \
  -H "Authorization: Bearer $OMLX_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"local-flux2-klein-4b","tasks":[{"prompt":"A red teapot on a table","width":256,"height":256,"steps":4,"seed":17}],"max_rows":64}'
```

Each task accepts `prompt`, `width`, `height`, `steps`, `seed`, `guidance`,
and `negative_prompt`; pipeline validation rejects unsupported values.
Task dimensions default to 256×256. `max_rows` defaults to 256 and is bounded
to 1–4096 activation rows per linear call.

Use the returned `id` to poll `GET /diffusion/jobs/{id}`. Jobs report `status`,
`phase`, fractional `progress` from 0 to 1, `detail`, timestamps, and a result
or error. Statuses are `queued`, `waiting`, `running`, `cancelling`,
`completed`, `failed`, and `cancelled`. Calibration reports are written under
`<base_path>/preparation/diffusion/<job_id>.json`; successful results include
`output_path`. History survives restart; interrupted jobs become failed and
are not automatically resumed.

Quantize a complete floating-point checkpoint of the same identity using a
completed calibration job:

```bash
curl -X POST "$BASE/diffusion/quantizations" \
  -H "Authorization: Bearer $OMLX_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"local-flux2-klein-4b-float","calibration_job_id":"CALIBRATION_JOB_ID","bits":4,"group_size":64,"budget_ratio":1.10}'
```

Supported bits are 3, 4, 5, 6, and 8; group sizes are 32, 64, and 128. Supply
either `budget_bytes` or `budget_ratio`, and optionally `protected` layer
patterns. Packed checkpoints can provide calibration but cannot be used as
fresh quantization sources. Completed checkpoints are published into the
first configured model directory as `<identity>-oq-<job_id>`, then discovered
automatically. Output paths are chosen by the server; clients cannot override
them or overwrite source checkpoints.

Preparation waits for existing inference leases and scheduler work to drain,
then holds exclusive pool admission through native cleanup. New engine
acquisitions and model refresh return busy while this gate is active;
`GET /state` reports `preparation_active`. Resident and pinned models stay
loaded. Admission checks remaining pool ceiling, system memory, and Metal
capacity with room reserved for temporary allocations; if a job cannot fit,
unload idle models before retrying.

Cancellation is cooperative at denoising forward, packing chunk, and workflow
boundaries. Loading, an individual kernel, and native saving can delay it.
The gate remains held until the worker stops and releases its allocations.
A checkpoint or report already published remains a completed result if a
cancellation request arrives afterward. Unpublished staging is removed.
