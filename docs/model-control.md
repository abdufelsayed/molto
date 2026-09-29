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
current values. The API does not offer every field from the old web admin
pages; its accepted patch schema is in
`omlx/services/management_models.py`.

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

## Inspect and clear cache

`GET /cache` reports cache statistics for loaded models and the configured
SSD cache directory. `POST /cache/hot/clear` reclaims hot cache. `POST
/cache/ssd/clear` deletes known saved SSD blocks, including blocks for
unloaded models. A later matching prompt may have to recompute its prefix.
`GET /stats` reports session counters by default and accepts
`scope=alltime` or a `model_id` query.

## Limits of this API

The former workspace also offered checkpoint verification, downloads,
conversion, quantization, update staging, storage moves, collections, and a
shared job queue. They are not part of the current management API. Do not
assume an operation record is a resumable background job. Model loading and
settings use the running engine pool and persisted settings; restarting the
server does not keep a model resident. Experimental cluster protocol routes
remain separate from this API.
