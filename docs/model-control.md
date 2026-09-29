# Web model workspace

The model workspace is available under **Models** in the web admin app. It has
three views:

- **Library** is the single registry and entry point for model management.
- **Add model** contains download, a shared Convert & Quantize view, and
  publishing.
- **Activity** contains the shared operation queue.

The workspace uses the same engine pool as the OpenAI-compatible APIs, so a
policy or collection changed in the browser affects the running server.

## Registry and capabilities

`GET /admin/api/control/registry` returns one normalized record for every
physical or virtual model. Each record includes:

- model, engine, and checkpoint type;
- accepted inputs, produced outputs, supported tasks, and API endpoints;
- local path, Hugging Face repository, active revision, cached revisions,
  license metadata, architecture, and quantization metadata;
- estimated, observed, and resident memory;
- load state, policy, health, update state, cluster placement, profiles, and
  helper or variant relationships.

The server derives this record at runtime. It does not modify downloaded model
directories.

## Health checks

**Verify files** checks the model descriptor, indexed weight shards, broken
links, required Python package, and checksums for small configuration files.

**Verify load** also loads the engine and runs the smallest useful task probe:

| Model type | Probe |
|---|---|
| LLM or VLM | Generate one token |
| Embedding | Embed one short string |
| Reranker | Rank two documents |
| TTS | Synthesize a short sample |
| Image generation | Generate a 256 × 256 PNG with one denoising step |
| STT or speech-to-speech | Load validation; a real audio sample is required for inference |

The result records load time, probe time, observed resident memory, and output
dimensions or bytes where applicable. If verification loaded an otherwise idle
model, it unloads the model afterward.

## Memory plans and collections

In Library, select models and choose **Plan memory** to calculate their additional resident
memory and the exact LRU eviction order under the current memory ceiling. The
plan excludes pinned, loading, selected, and in-use models. The engine pool
performs its normal admission check again during the real load because memory
can change after a dry run.

A collection stores a named set of models. Loading it uses the current plan,
loads each member, and unloads only the newly loaded members if a later member
fails. Models that were already resident stay resident.

**Convert** and **Quantize** are separate operations in one view and use one
source selector. `GET /admin/api/control/prepare-models` merges the serving
registry with the oQ scan, so text, vision, audio, embedding, reranking, and
image models remain visible even when one operation cannot handle them. Each
row reports its model type, modality, format, precision, conversion route,
quantization route, and an exact reason for any unavailable operation.

Conversion changes the runtime format while preserving source precision. It
dispatches LLMs to `mlx-lm`, VLMs to `mlx-vlm`, embedding and reranking models
to `mlx-embeddings`, audio models to `mlx-audio`, and supported image models to
`mflux`. Quantization uses oMLX's oQ implementation and accepts only sources
that oQ identifies as quantizable. A Hugging Face source is converted to MLX
automatically inside the oQ job before the quantized output is written, while
an MLX source proceeds directly to oQ.

Neither operation downloads a source model implicitly. Downloads remain a
separate Add model action, and completed outputs appear in the registry after
discovery refreshes.

## Lifecycle policies

The model drawer maps its policies to the existing persisted model settings:

| Policy | Stored behavior |
|---|---|
| On demand | Not pinned and no model TTL |
| Keep warm | Not pinned, with the selected TTL |
| Always resident | Pinned and loaded immediately |
| Unload after request | Not pinned, with a one-second idle TTL |

The normal TTL polling interval still determines the exact unload time.

## Operations

`GET /admin/api/control/operations` normalizes Hugging Face downloads,
ModelScope downloads, oQ conversions, Hugging Face uploads, health checks,
update checks, and storage moves into `queued`, `running`, `succeeded`, `failed`,
`cancelled`, or `interrupted` states.

Control-center operation history is stored atomically in
`<base_path>/model_control.json`. If the server exits during an operation, the
record becomes `interrupted` on the next start and can be retried. Hugging Face
and ModelScope retries reuse their existing partial downloads.

## Storage and revisions

The storage view reports logical bytes, allocated bytes, unique file bytes,
file count, broken links, and free space for every configured model root. A
deletion preview lists profiles and helper relationships that depend on the
model and flags loaded, pinned, or default models.

Models can move between configured roots. oMLX unloads the model first, moves
the local directory or complete Hugging Face cache entry, refreshes discovery,
and records the operation.

For Hugging Face cache entries, **Check update** compares the active snapshot
with the repository's current commit. **Stage update** downloads the candidate
snapshot without changing the active `main` reference. Activating a cached
revision updates that reference and refreshes discovery, so any older cached
revision is also a rollback target.

## Portable configuration

Export downloads a JSON bundle containing source and capability metadata,
model settings, profiles, policies, and collections. Import always runs a dry
run in the browser first. It applies settings only to model IDs present on the
target server and reports missing models without creating placeholder entries.

All control endpoints require an authenticated admin session. Load access from
the existing bearer-token endpoint remains unchanged.
