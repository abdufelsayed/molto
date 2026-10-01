# Molto

Molto runs and manages local AI models on Apple Silicon. It serves text and
vision language models, embeddings, rerankers, and optional image and audio
models through HTTP APIs. `molto serve` starts the TanStack Start dashboard and
Python inference backend together on one public origin. The dashboard handles
model and server management. Both share this repository. Dashboard source and development instructions are in
[apps/dashboard/](apps/dashboard/README.md).

Requires macOS 15.0 or newer, Apple Silicon, and Python 3.11, 3.12, or 3.13.
The project is licensed under [Apache 2.0](LICENSE).

## Install from source

Install [uv](https://docs.astral.sh/uv/), Node, and pnpm 10.33.0, then install
both workspaces from the repository root:

```bash
git clone https://github.com/abdufelsayed/molto.git
cd molto
uv sync --all-packages --inexact --python 3.11
pnpm install --frozen-lockfile
pnpm --filter molto-dashboard build
```

The six Python workspace members share `uv.lock`; the dashboard and generated
contracts share the root `pnpm-lock.yaml`. `uv sync` includes developer tools.
For a runtime environment omit those with `--no-dev`, retaining
`--all-packages --inexact`. Runtime extras include `mcp`, `audio`, `image`,
`cluster`, `modelscope`, `grammar`, and `paroquant`; enable one with `--extra`,
for example `uv sync --all-packages --inexact --extra image`.

Complete release wheels bundle the dashboard assets and a standalone Node
runtime, so installed-wheel users do not need pnpm or a separate dashboard process.

Native custom kernels for some model families require the full Xcode Metal
toolchain and a source build with `MOLTO_WITH_CUSTOM_KERNEL=1`. A normal install
can use slower fallback paths.

To build a wheel with all five native extensions, first install the optional
build tools from the root workspace:

```sh
uv sync --all-packages --inexact --group native-build
MOLTO_WITH_CUSTOM_KERNEL=1 pnpm build
```

The resulting native wheel targets the Python version used to build it. Install
it with the same Python version.

## Quickstart

Put an MLX-format checkpoint in its own subdirectory under `~/.molto/models`, then
start the foreground server:

```bash
export MOLTO_API_KEY=replace-with-a-secret-key
uv run --all-packages --inexact molto serve
```

Open `http://127.0.0.1:17389` for the dashboard. The same origin serves inference
and the dashboard's session gateway; raw management routes stay on the private
Python loopback listener. If no key
is configured, omit the export above and use the guarded local first-run form
to create one. Initial setup requires a directly connected local browser and
loopback binding. Once a main key exists, the bundled dashboard opens a local
session automatically when visited directly on localhost. Create inference keys
under **Settings → API keys → Create key**. Remote dashboard access requires the
main key. Check startup and discover models with:

```bash
curl http://127.0.0.1:17389/health
curl http://127.0.0.1:17389/v1/models \
  -H "Authorization: Bearer $MOLTO_API_KEY"
```

For a text model, send its ID from `/v1/models` to
`POST /v1/chat/completions`:

```bash
MODEL=your-model-id
curl http://127.0.0.1:17389/v1/chat/completions \
  -H "Authorization: Bearer $MOLTO_API_KEY" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}"
```

Stop the foreground process with Ctrl+C.
The server also supports model directories one level below organization
directories, such as `~/models/mlx-community/model-name/`.

Use `~/.molto/models` for persistent model storage. Managed downloads keep their
`owner/model` directory names, and conversions publish a separate checkpoint
under the configured model root. Keep source weights and prepared checkpoints;
temporary conversion directories are scratch space. The dashboard and CLI use
the same saved `model.model_dirs` setting.

The management API requires the main key even on loopback unless you
explicitly enable the loopback-only `skip_api_key_verification` setting. A
main key is also required when binding to a non-loopback address. Keep
`skip_api_key_verification` disabled for network access. See the
[management API](docs/management-api.md) for control endpoints and key
permissions.

## Install the bundled command

Build and install the complete application from the source checkout:

```sh
pnpm build
uv tool install --python 3.11 --overrides dist/molto-overrides.txt dist/molto-*.whl
```

Run `uv tool update-shell` if uv's executable directory is not already on your
PATH. The installed wheel includes the dashboard
and Node. Molto has not published its first release yet; the source instructions
above are the supported installation path.

## Homebrew service

For an existing Homebrew installation, the CLI retains service lifecycle
commands:

```bash
molto start
brew services info molto
molto stop
```

For default Homebrew installs, `molto start`, `stop`, and `restart` delegate to
`brew services`. Source and pip installs can use the same commands to manage
a background application, or `molto serve` in the foreground. Custom base paths
and startup options select the local application manager. The service
uses the default model directory `~/.molto/models` and port 17389 unless you
configure them. Server settings live under the selected Molto base path,
normally `~/.molto/settings.json`.

The checked-in [formula](Formula/molto.rb) builds the dashboard and bundles Node.
Until Molto publishes a release, build it with
`brew install --HEAD ./Formula/molto.rb` from this checkout.
Node and pnpm are build dependencies, not requirements for the installed runtime.

## APIs and models

The inference server exposes OpenAI-shaped and Anthropic-shaped endpoints.
Compatibility depends on the endpoint, model, and request features; test the
client workflow you need.

| Endpoint                      | Use                                                 |
| ----------------------------- | --------------------------------------------------- |
| `GET /v1/models`              | Discover API model IDs                              |
| `POST /v1/chat/completions`   | Text or vision chat                                 |
| `POST /v1/completions`        | Text completions                                    |
| `POST /v1/messages`           | Anthropic-shaped messages                           |
| `POST /v1/responses`          | Responses                                           |
| `POST /v1/embeddings`         | Embeddings                                          |
| `POST /v1/rerank`             | Reranking                                           |
| `POST /v1/images/generations` | Supported mflux image models with the `image` extra |
| `POST /v1/images/edits`       | Image-to-image, reference editing, and inpainting   |
| `POST /v1/images/operations`  | Explicit diffusion pipeline operations              |
| `GET /v1/images/capabilities` | Pipeline options and checkpoint-specific support    |

Audio routes require the `audio` extra. Image diffusion uses a general pipeline
registry over mflux 0.20; see [image models](docs/image-models.md) for the exact
operation matrix, preparation commands, and real-test coverage. Tool calling
depends on the model's chat template and the parser for its output format.

Model discovery and settings are available through the
[management API](docs/management-api.md). See
[model control](docs/model-control.md) for load, unload, profiles, and cache
behavior. The [architecture](docs/architecture.md) describes
the server boundary and limitations.

## Command line

`molto start`, `stop`, `restart`, and `status` manage the application lifecycle;
`molto serve` runs it in the foreground. `molto init` creates a main key on a
fresh running local server, and `molto open` opens the dashboard.

```sh
molto start
molto status
molto models list
molto keys create --name coding
molto --url https://inference.example --api-key-file ~/.config/molto/main-key models list --json
```

Management groups cover models, keys, settings, jobs, logs, cache, diagnostics,
and monitoring. The CLI uses the same public port as the dashboard through a
main-key bearer gateway. See the [CLI guide](docs/cli.md) for typed settings,
confirmation, credential handling, output, and exit codes.

## Serving controls

`molto serve --help` lists startup options. Common examples:

```bash
molto serve --model-dir ~/models --memory-guard safe
molto serve --model-dir ~/models --max-concurrent-requests 16
molto serve --model-dir ~/models --paged-ssd-cache-dir ~/.molto/cache
molto serve --model-dir ~/models --host 0.0.0.0 --api-key your-secret-key
```

The engine pool loads multiple model types, admits them under a memory limit,
and can evict idle models. Text and vision serving use continuous batching.
The KV cache can reuse prefixes in memory and, when enabled, save cold blocks
to SSD. The management API can inspect state, load or unload a model, and
change supported settings. CLI flags take precedence over saved settings at
startup.

## Migrate from oMLX

Stop oMLX and any cluster workers, then preview and run the migration:

```sh
uv run --all-packages --inexact molto migrate --dry-run
uv run --all-packages --inexact molto migrate --yes
```

Migration moves `~/.omlx` to `~/.molto` on the same filesystem. It preserves
checkpoint files, credentials, settings, caches, and usage history; updates
stored paths and absolute symlinks; and retargets the Hugging Face cache link.
It also moves legacy macOS application support when present. It refuses to
overwrite an existing destination. Original changed settings are backed up
under `~/.molto/migration-backup/`. See [migration](docs/migration.md) for custom
roots, client integrations, and the new package names.

## Dashboard and public API

The macOS menu bar app and bundled web admin UI are no longer part of this
backend. The old `/admin` pages, cookie login, browser chat, and most
`/admin/api/*` routes are gone. Experimental cluster protocol routes remain
under `/admin/api/cluster/*`. Use `molto serve` or the Homebrew service for
process lifecycle, the inference endpoints for model requests, and
the dashboard's `/api/molto/*` session gateway for server controls. Raw
`/management/v1/*` is the private backend contract and requires the main bearer
key; the public proxy blocks it. Old app bundle
build instructions and app-only CLI behavior do not apply. Molto reads its own
`MOLTO_BASE_PATH` environment variable and its migrated base-path pointer.

The dashboard provides model and server management, keys, acquisition and
preparation jobs, monitoring, logs, cache controls, diagnostics, and conditional
cluster management. Browser chat remains outside the dashboard's scope.
The bundled Nitro runtime proxies HTTP inference, streaming responses, and the
realtime transcription WebSocket to private FastAPI. One public bind address is
supported, such as `127.0.0.1` or `0.0.0.0`; comma-separated addresses are rejected.
A backend restart keeps the dashboard and its sessions alive. A restart that
changes the effective public host or port relaunches both processes, respecting
CLI and environment overrides. See [dashboard instructions](apps/dashboard/README.md)
and [architecture](docs/architecture.md).

## Development

Run workspace commands from the repository root:

```bash
uv sync --all-packages --inexact
pnpm install --frozen-lockfile
pnpm check
pnpm test
pnpm build
```

`pnpm check` checks Python dependency boundaries, Ruff, type-aware Oxlint,
Oxfmt, native TypeScript 7, and Python interface/build typechecks. `pnpm test` runs Python and dashboard tests.
`pnpm build` assembles the single distributable Molto wheel, including dashboard
assets and the verified macOS ARM64 Node runtime. It may download the selected
Node archive; inference and model downloads are not part of the build.

For dashboard development, start Vite/Nitro and private FastAPI together:

```bash
uv run --all-packages --inexact molto serve --dashboard-dev --model-dir ~/models
```

This source-only mode uses the configured public port without requiring a
production frontend build. Package unit tests live beside their owning app or
library; cross-package tests live in `tests/integration/`. For example:

```bash
uv run --all-packages --inexact pytest packages/runtime/tests/test_engine_pool.py
```

The default pytest configuration excludes slow and integration-marked tests.
See [architecture](docs/architecture.md), [testing](docs/TESTING.md), and
[contributing](docs/CONTRIBUTING.md). Native kernel build configuration lives
in `packages/runtime/setup.py`; building kernels requires full Xcode.

## Acknowledgments

Molto is a fork of [oMLX](https://github.com/jundot/omlx), which began from
[vllm-mlx v0.1.0](https://github.com/waybarrios/vllm-mlx). Molto adds the bundled
web dashboard, management CLI, and application/library workspace structure.
It builds on [MLX](https://github.com/ml-explore/mlx),
[mlx-lm](https://github.com/ml-explore/mlx-lm),
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm),
[mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings), and
[dflash-mlx](https://github.com/bstnxbt/dflash-mlx). The Lightning MTP
verify-shape kernels use [MTPLX](https://github.com/youssofal/mtplx) by
Youssof Altoukhi. Fused GDN work draws on
[mlx-serve](https://github.com/ddalcu/mlx-serve), and the verify kernels draw
on [Splash](https://github.com/incoai/splash). See source headers and license
files for component-specific credit.
