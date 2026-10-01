# Migrate from oMLX to Molto

Molto uses the `molto` command, `MOLTO_*` environment variables, and `~/.molto`
data directory. It is a command-line application with a web dashboard. It does
not require a macOS app bundle.

## Move existing data

Stop the old server and cluster workers before migrating. From a Molto source
checkout with dependencies installed:

```sh
uv run --all-packages --inexact molto migrate --dry-run
uv run --all-packages --inexact molto migrate --yes
```

The migration renames the data root without copying or downloading checkpoints.
It updates JSON configuration paths, absolute symlinks inside that root, and
the external `~/.cache/huggingface/hub` symlink when it targets the old root.
Relative model-cache symlinks retain their targets. Credentials, model-specific
settings, cache data, cluster identity, and SQLite usage history move together.
Historical logs and benchmark inputs retain their contents.
Molto retains the old cache hash domains and reads old cache metadata so these
blocks remain usable after the move. New writes use Molto metadata names.
Prepared diffusion checkpoints rename `omlx-mflux.json` to `molto-mflux.json`
and update source paths beneath the moved root. Weight files and upstream model
configuration remain unchanged. Conflicting manifest names stop migration
before data moves; failed migrations restore the original names and contents.

If present, `~/Library/Application Support/oMLX` moves to
`~/Library/Application Support/Molto`. Molto writes new command and cluster
interpreter launchers under `~/.molto/bin`. Original changed configurations and
old launchers are backed up under `~/.molto/migration-backup`, with their file
permissions preserved. Migration never merges into or overwrites an existing
destination. Review a destination conflict before moving anything.
It retires stale lifecycle records and public `omlx` symlinks that point to the
old data-root launcher. An independently installed upstream app is left intact.

For a custom data root, pass explicit paths:

```sh
molto migrate --source /path/to/old/data --target /path/to/molto/data --dry-run
molto migrate --source /path/to/old/data --target /path/to/molto/data --yes
export MOLTO_BASE_PATH=/path/to/molto/data
```

The parent of the destination must already exist on the same filesystem.
Configure `MOLTO_MODEL_DIR` or saved `model.model_dirs` for checkpoints kept
outside the data root. Migration changes paths beneath the source root only.

## Update clients and startup commands

Replace the `omlx` command with `molto` in shell aliases, scheduled jobs,
Homebrew service commands, and external integrations. Replace `OMLX_*` with
`MOLTO_*` in their environment. Molto does not read the old environment prefix.
Previously saved credentials remain valid after the data migration.

The public origin remains the configured host and port, normally
`http://127.0.0.1:17389`. Inference endpoints under `/v1` retain their protocol.
The dashboard gateway is now `/api/molto/*`, and its browser session starts
fresh. Raw `/management/v1/*` routes remain private to FastAPI.

Run `molto status`, then `molto start` when ready. Visit the dashboard and check
`molto models list` before running inference. Stop with `molto stop`.

## Source and distribution names

The distribution is `molto`. Python workspace packages are `molto-cli`,
`molto-server`, `molto-config`, `molto-contracts`, `molto-management`, and
`molto-runtime`; imports use the corresponding `molto_*` names. The public
facade is `molto`, and contracts use the `@molto/contracts` pnpm package.
Model files and model IDs keep their existing names.

The repository retains upstream licenses, attribution, historical benchmark
corpora, dependency identities, and oMLX community benchmark URLs and JSON
fields. Those refer to external sources rather than Molto product names.
Release and update checks target `abdufelsayed/molto`.
