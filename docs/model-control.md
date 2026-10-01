# Model control without the web admin app

Use this guide to inspect and control models on a running oMLX server. The
backend no longer serves the Models workspace. A separate dashboard can use
the [management API](management-api.md); these examples use `curl`.

Start `omlx serve` with a model directory as shown in the
[quickstart](../README.md#quickstart), then set the same main key in the shell
where you send requests:

```bash
export OMLX_API_KEY=replace-with-your-main-key
BASE=http://127.0.0.1:8000/management/v1
curl "$BASE/models" -H "Authorization: Bearer $OMLX_API_KEY"
```

The inventory lists discovered models, load state, and their persisted model
settings. `GET /state` gives a compact view of the default model, model
count, memory ceiling, and loading states. After adding or removing a model
directory, call `POST /models/refresh` to rescan it.

## Load and unload

Use a model ID from the inventory:

```bash
MODEL=my-model
curl -X POST "$BASE/models/$MODEL/load" \
  -H "Authorization: Bearer $OMLX_API_KEY"
curl -X POST "$BASE/models/$MODEL/unload" \
  -H "Authorization: Bearer $OMLX_API_KEY"
```

Load can return an error if the model is unknown or already loading. Unload
can return HTTP 202 while active use drains. The server's existing engine
pool enforces memory admission and eviction. `GET /state` shows the later
load state. All management routes, including load, require the main key when
authentication is active. Inference subkeys can load through the retained
`POST /v1/models/{model_id}/load` endpoint. The retained inference unload
endpoint now requires the main key.

## Change settings and profiles

Read `GET /settings` for global sampling and selected scheduler settings.
`PATCH /settings` accepts a flat JSON object with supported fields. A
response with `requires_restart: true` means a scheduler construction change
will take effect after a server restart. A model-specific patch returns
`requires_reload` if a loaded engine's runtime configuration differs from
the saved settings:

```bash
curl -X PATCH "$BASE/models/$MODEL/settings" \
  -H "Authorization: Bearer $OMLX_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"is_pinned":true,"ttl_seconds":300}'
```

An omitted field stays unchanged. For model settings, an explicit `null`
restores that field's default. `GET /models/{model_id}/settings` returns the
current values. Use `GET /models/{model_id}/options` for field metadata and capability reasons.
The accepted patch schema is in `omlx/services/management_models.py`; unsupported
model-specific options are rejected. Templates, presets, generation-config import,
recipes, and optimal snapshots provide additional ways to select settings.

When `requires_reload` is true, the backend requests an unload without
aborting active requests. If the model is idle, `auto_unloaded` is true; a
pinned model is then loaded again and `auto_reloaded` is true. If requests
are still running, `reload_deferred` is true and unload waits for them to
finish. The response also includes `reload_error` if that transition fails.
The saved settings may already have changed in that case, so inspect
`GET /state` before sending another request. A deferred pinned model is not
immediately reloaded by this PATCH response; load it again after it unloads
if needed.

Profiles save a named set of allowed model settings. Create one with
`POST /models/{model_id}/profiles`, list them with `GET`, update with `PUT`
at `/profiles/{name}`, apply with `POST /profiles/{name}/apply`, and delete
with `DELETE /profiles/{name}`. The body for creation includes `name` and
`settings`. Profiles can optionally expose a separate API model ID. See the
[management API](management-api.md#load-and-settings-example) for a request.

Applying a profile follows the same unload/reload lifecycle and returns the
same transition flags as a model settings patch. Omitted universal fields
reset to their defaults; model-specific fields retain their existing values
unless the profile supplies an override. Active requests are allowed to
finish before a required unload.

## Inspect and clear cache

`GET /cache` reports cache statistics for loaded models and the configured
SSD cache directory. `POST /cache/hot/clear` reclaims hot cache. `POST
/cache/ssd/clear` deletes known saved SSD blocks, including blocks for
unloaded models. A later matching prompt may have to recompute its prefix.
`GET /stats` reports session counters by default and accepts
`scope=alltime` or a `model_id` query.

## Maintain the local library

Use `GET /workspace/registry` for complete and incomplete local artifacts and
`GET /workspace/storage` for configured roots. `POST /workspace/plan` estimates
residency and eviction for selected models. Collections save reusable model lists.
Preload collections persist pinning for the existing startup loader; saving does
not load immediately, and removing preload does not automatically unpin.
Configuration export/import moves settings and profiles, not checkpoint weights;
preview an import with `dry_run: true` before applying it. The preview includes
changed fields, affected model IDs, blockers, and `can_apply`. Only changed model
records require unload; application rechecks their state.

Each model supports structural or smoke verification, update checking and staging,
revision activation, storage moves, and deletion previews under
`/workspace/models/{model_id}`. File changes are guarded against active work,
default or pinned models, dependencies, and overlapping operation paths. Request
`GET /delete-plan` before `DELETE /delete`, and include its `plan_token`. If the
model must unload, explicitly request draining and wait for completion. Moves
preserve model IDs and move a cached model's whole repository, including refs,
revisions, and shared blobs. Explicit external path dependencies block a move.
Virtual profile
models have no physical-file actions. STT and speech-to-speech verification is
structural only because no smoke probe is available.

`POST /models/{model_id}/import-mtplx` imports a compatible local MTPLX sidecar
into an unloaded checkpoint. Changes are staged and rollback restores previous
files on failure. An incomplete rollback reports retained recovery files for
inspection before retry.

## Acquire and prepare models

The acquisition API searches Hugging Face and ModelScope and starts downloads.
Local preparation supports conversion, quantization estimates, and oQ quantization.
Publishing requires a provider token and an explicit destination. Availability
depends on installed optional dependencies and checkpoint compatibility. Inspect
preparation options before starting work.

Track jobs through `/operations`; the response advertises supported cancel and
retry actions. History persists, but workers cannot survive a server restart.
Interrupted work needs an explicit retry and may need a fresh provider token.
Cancellation waits for native work to stop before releasing file reservations.
Local [diffusion preparation](management-api.md#local-diffusion-preparation) has
its own job endpoint and compatible-checkpoint requirements.

Local diagnostic runs can unload models while benchmarking or probing limits.
They do not apply context recommendations or publish accuracy results by default.
Review the impact and result before applying a recommendation. Experimental
cluster management uses the separate retained cluster protocol.
