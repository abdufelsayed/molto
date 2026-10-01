# Molto command line

Use `molto` to run the application and manage a local or remote server through its public dashboard origin. `molto --help` and each command's `--help` list supported flags. Existing inference integration, offline preparation, and cluster commands remain available.

## Source checkout

Install both workspaces from the repository root before using the CLI:

```sh
uv sync --all-packages --inexact
pnpm install --frozen-lockfile
pnpm --filter molto-dashboard build
uv run --all-packages --inexact molto --help
```

The command implementation and its HTTP client live in `apps/cli/`. Installed
release wheels expose the same `molto` command and include the dashboard runtime.
See [architecture](architecture.md) for package ownership and root checks.

## Start and connect

```sh
molto start
molto status
molto init
molto open
molto models list
```

`start` starts the application in the background and waits for public readiness. `stop` stops the locally managed application; `restart` stops and starts it again. `serve` keeps the application in the foreground until Ctrl+C. `start` and `restart` accept `--model-dir`, `--host`, `--port`, source-only `--dashboard-dev`, and `--no-wait`. Use `status` after `--no-wait` to check readiness.

Lifecycle waits default to 60 seconds. A server preloading models reports `loading`, with `healthy: false`, until inference is ready. A startup timeout leaves the process managed so you can inspect or stop it. A plain restart preserves development mode.

Explicit startup settings such as `--host`, `--port`, and `--model-dir` are saved after successful validation and take priority over saved settings while that application runs. The source-only development flag is not saved to server settings.

Default Homebrew lifecycle commands retain `brew services` behavior. Explicit custom settings or source/pip installations use a managed background application under the selected base path. The CLI checks process identity before stopping it and does not take ownership of an unrelated foreground server. Startup output goes to `<base_path>/logs/application.log`; this local launcher file is separate from the API log tail.

`init` creates a main key through guarded setup on an already running, fresh loopback-bound server. It prompts for the key and confirmation in a terminal. For unattended setup, provide `MOLTO_API_KEY` or `--api-key-file`; init does not generate a key automatically. Setup is unavailable through remote origins or forwarded requests. `open` opens the selected dashboard origin in a browser.

Connection flags work before or after management commands:

```sh
molto --url https://inference.example --api-key-file ~/.config/molto/main-key models list
molto models list --json
```

`--url` overrides `MOLTO_URL`; otherwise the CLI derives the local origin from saved host/port. Supply the main key with `--api-key-file`, `--api-key`, or `MOLTO_API_KEY`. Key files must be private, for example mode `600`. A saved local key is used only for the exact derived loopback origin, never sent automatically to a remote destination. Redirects do not receive credentials. Management always requires a main key, even when local inference authentication is bypassed.

Local lifecycle commands reject another server's `--url` or `MOLTO_URL`. `status` can check remote health; an authenticated remote restart can use `molto api POST /server/restart` when supported.

## Management commands

| Group             | Commands                                                                                                                    |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `models`          | `list`, `show MODEL`, `refresh`, `load MODEL`, `unload MODEL`, `download REPO`, `move MODEL --root-id ROOT`, `remove MODEL` |
| `models settings` | `get MODEL`, `set MODEL FIELD=JSON...`, `reset MODEL`                                                                       |
| `models profiles` | `list MODEL`, `create MODEL NAME`, `apply MODEL NAME`, `delete MODEL NAME`                                                  |
| `keys`            | `list`, `create`, `edit ID`, `revoke ID`, `rotate --main` or `rotate --key-id ID`                                           |
| `settings`        | `get`, `defaults`, `set SECTION.FIELD=JSON...`                                                                              |
| `jobs`            | `list`, `show ID`, `watch ID`, `cancel ID`, `retry ID`                                                                      |
| `logs`            | Read a bounded tail, optionally `--follow`, `--file`, `--level`, or `--lines`                                               |
| `cache`           | `show`, `clear hot`, `clear ssd`                                                                                            |
| `diagnostics`     | `status`, `list`, `show ID`, `results ID`, `start KIND`, `cancel ID`                                                        |
| `monitoring`      | `activity`, `usage`                                                                                                         |
| `api`             | `METHOD management-relative-path`, optionally `--body JSON_OR_@FILE` and repeated `--query NAME=VALUE`                      |

Typed settings preserve JSON types. Shell-quote string assignments so the JSON quotes reach the CLI:

```sh
molto models settings set MODEL temperature=0.7 'model_alias="my-model"'
molto models settings set MODEL temperature=null
molto settings set sampling.temperature=0.7
molto models profiles create MODEL focused --settings '{"temperature":0.2}'
molto models download mlx-community/REPO --provider hf
molto api GET /server/resources
```

Download providers are `hf` and `ms`; `--token-file` supplies a private provider credential. `jobs list` combines operations and diffusion jobs, reporting when optional diffusion history is unavailable. Use `--kind acquisition`, `--kind workspace`, or `--kind diffusion` to select a history. Other job commands default to operation IDs; specify `--kind diffusion` for diffusion jobs. Retry uses a fresh `--token-file` when needed. Diffusion retry is unavailable; start a new preparation job. Backend-advertised state and actions remain authoritative.

```sh
molto jobs watch JOB --interval 1 --wait-timeout 300
molto logs --follow --lines 200 --wait-timeout 60
molto monitoring usage --range 7d --model MODEL --details
molto diagnostics start throughput --options @throughput.json
```

Log following polls overlapping HTTP tail windows and prints new suffixes; it is not a lossless streaming log subscription. Both watches have bounded duration, defaulting to 300 seconds. JSON watches return the final observed object. A job still active at timeout exits with failure; a log follow that reaches its duration completes successfully.

Diagnostics accept `throughput`, `accuracy`, `context`, and `ane`. Local runs can unload models and leave tested engines warm. Review [management API](management-api.md) impacts and diagnostic capabilities before starting. API calls accept only management-relative paths, not another origin or traversal paths.

## Confirmation, secrets, and output

Model deletion prints a deletion plan before confirmation. Moves, profile deletion, key revocation/rotation, cache clearing, and raw DELETE requests require a terminal confirmation or `--yes`. Noninteractive and JSON commands require `--yes` for these operations. Model moves/removal also accept explicit `--drain`; confirmation does not bypass backend guards.

Credentials are masked by default. `keys list --reveal`, `settings ... --reveal`, and `api ... --reveal` explicitly show secrets. Key creation and rotation show the resulting key once, including generated replacements. Store it privately. Main rotation invalidates the old credential; use the replacement for subsequent CLI requests and reconnect browser sessions when required.

Human output uses Rich tables and status displays. `--json` writes the result to stdout and errors to stderr. `--no-color` or `NO_COLOR` disables colors. Deletion previews use stderr in JSON mode so stdout retains one result object.

| Exit code | Meaning                                        |
| --------- | ---------------------------------------------- |
| `0`       | Success                                        |
| `1`       | Operation failure                              |
| `2`       | Invalid usage or required confirmation missing |
| `3`       | Authentication or access denied                |
| `4`       | Connection failure or timeout                  |
| `5`       | Server conflict                                |
| `130`     | Interrupted or cancelled                       |

The CLI uses bearer authentication at `/api/management/v1` on the public dashboard port. The gateway rejects browser Origin-bearing requests and requires the main key. Browser UI sessions continue using the separate `/api/molto` gateway. Raw private `/management/v1` and `/admin` routes remain blocked at the public proxy.
