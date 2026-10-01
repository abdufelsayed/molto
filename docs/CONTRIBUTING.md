# Contributing to oMLX

oMLX is a bundled inference server, management CLI, and TanStack Start dashboard. Contributions to model
support, serving behavior, performance, tests, and documentation are welcome.
The macOS app and bundled web admin UI are no longer in this repository.

## Set up a checkout

Use an Apple Silicon Mac with macOS 15.0 or newer, Python 3.11 to 3.13, and
[uv](https://docs.astral.sh/uv/). From a checkout:

```bash
uv sync --all-packages --inexact --python 3.11
pnpm install --frozen-lockfile
uv run --all-packages --inexact pytest packages/runtime/tests/test_engine_pool.py
```

The checked-in `uv.lock` is the reproducible dependency source. The `dev`
group is included by default; runtime extras are selected with `--extra`.
Native custom kernels need full Xcode and `OMLX_WITH_CUSTOM_KERNEL=1` when
building from source. See [README](../README.md#install-from-source).

## Make a focused change

- Check existing issues and PRs before a substantial feature or dependency
  change. Describe any API or behavior change in the PR.
- Keep the change within one problem. Update the settings, API, and tests
  needed for that problem without reformatting unrelated files.
- Preserve third-party license notices. Use Apache-2.0 SPDX headers for new
  original code, and follow the repository's Ruff configuration.
- Keep the inference API and management API boundaries clear. Every
  management route uses the main bearer key. Inference subkeys can use the
  retained `/v1/models/{model_id}/load` route. See
  [management API](management-api.md).

## Verify it

For a bug fix, add a regression test that exercises the affected production
path when practical. Run the narrow test first, then the default suite for
code changes:

```bash
uv run --all-packages --inexact pytest packages/runtime/tests/test_engine_pool.py  # Replace with affected tests
pnpm check
pnpm test
```

The default suite excludes slow and integration tests. Run relevant integration
tests or a representative model on hardware when changing inference, cache,
memory, or native kernels. Record the model, quantization, Mac hardware,
request shape, and what you observed. Mock tests do not establish throughput
or numerical behavior on a real checkpoint. See [testing](TESTING.md).

For performance work, compare the branch and base commit on the same Mac,
model, inputs, concurrency, cache state, and sampling settings. Report
throughput, first-token latency, memory, run count, correctness checks, and
any regression. Rebuild native kernels after changing them.

## Submit for review

Describe the problem, approach, changed API behavior, and checks run. State
what remains untested. Update the docs when users need a different command or
request. Keep incomplete work in a draft PR. For suspected vulnerabilities,
follow [SECURITY.md](../SECURITY.md) and report privately.
