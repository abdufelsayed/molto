# oMLX

oMLX is an Apple Silicon inference server for local models. It serves text and
vision language models, embeddings, rerankers, and optional image and audio
models through HTTP APIs. `omlx serve` starts the TanStack Start dashboard and
Python inference backend together on one public origin. The dashboard handles
model and server management. Both share this repository. Dashboard source and development instructions are in
[dashboard/](dashboard/README.md).

Requires macOS 15.0 or newer, Apple Silicon, and Python 3.11, 3.12, or 3.13.
The project is licensed under [Apache 2.0](LICENSE).

## Install from source

Install [uv](https://docs.astral.sh/uv/) first, then use the checked-in lockfile:

```bash
git clone https://github.com/jundot/omlx.git
cd omlx
uv sync --locked --python 3.11
```

`uv sync` includes the development dependency group by default. For a runtime
environment without test tools, use `uv sync --locked --no-dev --python 3.11`.
Optional runtime extras include `mcp`, `audio`, `image`, `cluster`,
`modelscope`, `grammar`, and `paroquant`; add one with `--extra`, for example
`uv sync --locked --no-dev --extra image`.

Build the dashboard once before serving from a source checkout. Install Node
and pnpm, then run:

```bash
cd dashboard
pnpm install --frozen-lockfile
pnpm build
cd ..
```

Complete release wheels bundle the dashboard assets and a standalone Node
runtime, so installed-wheel users do not need pnpm or a separate dashboard process.

Native custom kernels for some model families require the full Xcode Metal
toolchain and a source build with `OMLX_WITH_CUSTOM_KERNEL=1`. A normal install
can use slower fallback paths.

## Quickstart

Put an MLX-format checkpoint in its own subdirectory under `~/models`, then
start the foreground server:

```bash
export OMLX_API_KEY=replace-with-a-secret-key
uv run --locked omlx serve --model-dir ~/models
```

Open `http://127.0.0.1:8000` for the dashboard. The same origin serves inference
and the dashboard's session gateway; raw management routes stay on the private
Python loopback listener. If no key
is configured, omit the export above and use the guarded local first-run form
to create one. Initial setup requires a directly connected local browser and
loopback binding. Check startup and discover models with:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $OMLX_API_KEY"
```

For a text model, send its ID from `/v1/models` to
`POST /v1/chat/completions`:

```bash
MODEL=your-model-id
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $OMLX_API_KEY" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}"
```

Stop the foreground process with Ctrl+C.
The server also supports model directories one level below organization
directories, such as `~/models/mlx-community/model-name/`.

The management API requires the main key even on loopback unless you
explicitly enable the loopback-only `skip_api_key_verification` setting. A
main key is also required when binding to a non-loopback address. Keep
`skip_api_key_verification` disabled for network access. See the
[management API](docs/management-api.md) for control endpoints and key
permissions.

## Homebrew service

For an existing Homebrew installation, the CLI retains service lifecycle
commands:

```bash
omlx start
brew services info omlx
omlx stop
```

For default Homebrew installs, `omlx start`, `stop`, and `restart` delegate to
`brew services`. Source and pip installs can use the same commands to manage
a background application, or `omlx serve` in the foreground. Custom base paths
and startup options select the local application manager. The service
uses the default model directory `~/.omlx/models` and port 8000 unless you
configure them. Server settings live under the selected oMLX base path,
normally `~/.omlx/settings.json`.

The checked-in formula builds the dashboard and bundles Node. Its stable URL
still points to a release that predates `dashboard/`; the bundled source install
requires `brew install --HEAD` with this formula until its release URL and
checksum are updated. This repository change has not published a new release.
Node and pnpm are build dependencies, not requirements for the installed runtime.

## APIs and models

The inference server exposes OpenAI-shaped and Anthropic-shaped endpoints.
Compatibility depends on the endpoint, model, and request features; test the
client workflow you need.

| Endpoint | Use |
| --- | --- |
| `GET /v1/models` | Discover API model IDs |
| `POST /v1/chat/completions` | Text or vision chat |
| `POST /v1/completions` | Text completions |
| `POST /v1/messages` | Anthropic-shaped messages |
| `POST /v1/responses` | Responses |
| `POST /v1/embeddings` | Embeddings |
| `POST /v1/rerank` | Reranking |
| `POST /v1/images/generations` | Supported mflux image models with the `image` extra |
| `POST /v1/images/edits` | Image-to-image, reference editing, and inpainting |
| `POST /v1/images/operations` | Explicit diffusion pipeline operations |
| `GET /v1/images/capabilities` | Pipeline options and checkpoint-specific support |

Audio routes require the `audio` extra. Image diffusion uses a general pipeline
registry over mflux 0.20; see [image models](docs/image-models.md) for the exact
operation matrix, preparation commands, and real-test coverage. Tool calling
depends on the model's chat template and the parser for its output format.

Model discovery and settings are available through the
[management API](docs/management-api.md). See
[model control](docs/model-control.md) for load, unload, profiles, and cache
behavior. The [backend architecture](docs/backend-architecture.md) describes
the server boundary and limitations.

## Command line

`omlx start`, `stop`, `restart`, and `status` manage the application lifecycle;
`omlx serve` runs it in the foreground. `omlx init` creates a main key on a
fresh running local server, and `omlx open` opens the dashboard.

```sh
omlx start
omlx status
omlx models list
omlx keys create --name coding
omlx --url https://inference.example --api-key-file ~/.config/omlx/main-key models list --json
```

Management groups cover models, keys, settings, jobs, logs, cache, diagnostics,
and monitoring. The CLI uses the same public port as the dashboard through a
main-key bearer gateway. See the [CLI guide](docs/cli.md) for typed settings,
confirmation, credential handling, output, and exit codes.

## Serving controls

`omlx serve --help` lists startup options. Common examples:

```bash
omlx serve --model-dir ~/models --memory-guard safe
omlx serve --model-dir ~/models --max-concurrent-requests 16
omlx serve --model-dir ~/models --paged-ssd-cache-dir ~/.omlx/cache
omlx serve --model-dir ~/models --host 0.0.0.0 --api-key your-secret-key
```

The engine pool loads multiple model types, admits them under a memory limit,
and can evict idle models. Text and vision serving use continuous batching.
The KV cache can reuse prefixes in memory and, when enabled, save cold blocks
to SSD. The management API can inspect state, load or unload a model, and
change supported settings. CLI flags take precedence over saved settings at
startup.

## Migration from the earlier app and web UI

The macOS menu bar app and bundled web admin UI are no longer part of this
backend. The old `/admin` pages, cookie login, browser chat, and most
`/admin/api/*` routes are gone. Experimental cluster protocol routes remain
under `/admin/api/cluster/*`. Use `omlx serve` or the Homebrew service for
process lifecycle, the inference endpoints for model requests, and
the dashboard's `/api/omlx/*` session gateway for server controls. Raw
`/management/v1/*` is the private backend contract and requires the main bearer
key; the public proxy blocks it. Old app bundle
build instructions and app-only CLI behavior do not apply. Existing Python
base-path resolution still reads the macOS app's saved base-path pointer so
users with an existing data directory can continue to find their settings.

The dashboard provides model and server management, keys, acquisition and
preparation jobs, monitoring, logs, cache controls, diagnostics, and conditional
cluster management. Browser chat remains outside the dashboard's scope.
The bundled Nitro runtime proxies HTTP inference, streaming responses, and the
realtime transcription WebSocket to private FastAPI. One public bind address is
supported, such as `127.0.0.1` or `0.0.0.0`; comma-separated addresses are rejected.
A backend restart keeps the dashboard and its sessions alive. A restart that
changes the effective public host or port relaunches both processes, respecting
CLI and environment overrides. See [dashboard instructions](dashboard/README.md)
and [backend architecture](docs/backend-architecture.md).

## Development

Dashboard development uses the source-only `--dashboard-dev` flag. It starts
Vite/Nitro and private FastAPI together on the configured public port without a
production frontend build:

```bash
pnpm --dir dashboard install --frozen-lockfile
uv run --locked omlx serve --dashboard-dev --model-dir ~/models
```


```bash
uv sync --locked --python 3.11
uv run --locked pytest tests/test_engine_pool.py
uv run --locked pytest
```

The default pytest configuration excludes slow and integration tests. See
[testing](docs/TESTING.md) and [contributing](docs/CONTRIBUTING.md) for scoped
checks and hardware validation. The native kernel build remains in `setup.py`;
building it requires full Xcode, not only Command Line Tools.

## Acknowledgments

oMLX began from [vllm-mlx v0.1.0](https://github.com/waybarrios/vllm-mlx).
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
