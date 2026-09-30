# oMLX

oMLX is an Apple Silicon inference server for local models. It serves text and
vision language models, embeddings, rerankers, and optional image and audio
models through HTTP APIs. A small management API handles model and server
controls. This repository contains the backend; a dashboard can be built as a
separate client.

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

The default address is `http://127.0.0.1:8000`. Check startup and discover
models with:

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

For Homebrew installs, `omlx start`, `stop`, and `restart` delegate to
`brew services`. A source or pip install uses `omlx serve` in the foreground;
those lifecycle commands do not install a background supervisor. The service
uses the default model directory `~/.omlx/models` and port 8000 unless you
configure them. Server settings live under the selected oMLX base path,
normally `~/.omlx/settings.json`.

The checked-in Homebrew formula still points to the upstream release. It has
not been updated or published for this backend change. Use the source checkout
and `uv` commands above to run this version.

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
`/management/v1/*` with the main bearer key for server controls. Old app bundle
build instructions and app-only CLI behavior do not apply. Existing Python
base-path resolution still reads the macOS app's saved base-path pointer so
users with an existing data directory can continue to find their settings.

The new management API is smaller than the old admin feature set. Download
jobs, conversion, quantization, benchmarking, browser chat, cluster setup,
and update workflows do not have equivalent management routes in this
version. A future dashboard should call the backend APIs as a client; it is
not included here.

## Development

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
