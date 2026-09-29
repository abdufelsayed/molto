# Local usage history

The backend records hourly per-model usage in a local SQLite database. The
former web dashboard and macOS app no longer display it, and this version
does not expose history through a REST endpoint. A separate dashboard will
need an authenticated backend API to read it remotely. History starts when
recording is enabled; existing all-time totals cannot be backfilled.

Usage stays on this server. No telemetry is sent. Only the canonical oMLX model
ID, hourly bucket, request/token counts, and accumulated durations are stored.
There are no prompts, responses, messages, token IDs, API keys, headers, client
IPs, upload names, or document contents. Model IDs are the same identifiers used
by the engine pool and existing serving statistics, not display aliases or full
model paths. Renaming an ID starts a new series; replacing weights under the same
ID continues that series. Removed/unloaded models remain in history.

## Accounting

History consumes the existing completed-request serving-statistics hook for
OpenAI completions/chat/responses, Anthropic messages, embeddings, reranking,
and audio. It inherits that hook's coverage: failed/rejected or disconnected
requests that do not reach accounting are not counted. Local benchmark requests
that use these serving endpoints also count. Audio contributes requests with
zero tokens because byte/character counts are not token counts.

- **Prompt tokens**: the engine's full input count, including cached tokens.
- **Output tokens**: the engine's completion count, including generated reasoning
  and tool-call tokens counted by the engine.
- **Total tokens**: prompt + output; cached tokens are already part of prompt.
- **Cached tokens**: the engine-reported prompt tokens actually reused from KV
  prefix cache, after reconstruction, alignment, trimming, and fallback. This
  includes memory/SSD reuse where the engine reports it; it is neither a count of
  cache lookups nor cache writes. Unsupported/no reuse reports zero. Anthropic
  cache-control billing fields do not replace these engine counters.
- **Cache efficiency** (internal query): cached / prompt, a fraction from 0 to 1.
- **Generation speed**: sum(output tokens) / sum(generation seconds), matching
  existing serving-statistics weighting; not the mean of per-request speeds.
- **Prefill speed** (internal query): sum(prompt − cached) / sum(prefill seconds).
- **Request seconds** (internal query): measured serving-handler duration,
  including waits within that measurement, not HTTP middleware/network
  latency. Loading inclusion follows the existing endpoint timer. Unknown
  durations (currently audio) are
  excluded from `timed_requests` and `average_request_seconds`.

Prefill/generation timing follows existing endpoint accounting (time to first
output and subsequent generation; supported diffusion engines use native timing).
Overlapping requests each contribute their own duration. Summed seconds are not
GPU busy time. Missing speed measurements return `null`, not invented throughput.

## Storage and retention

`<base_path>/usage.sqlite3` lives alongside `stats.json` (normally
`~/.omlx/usage.sqlite3`; follows `OMLX_BASE_PATH` and an existing app
base-path pointer). Python's built-in SQLite stores one row per active
model/hour with schema version 1 (`PRAGMA user_version`). Cumulative
`stats.json` counters remain independent. No new dependency is required.

A background thread checks for pending aggregates every 5 seconds and flushes
on normal shutdown. With no pending data, it skips database access unless daily
retention maintenance is due or a storage failure needs retrying. Sudden
termination can lose the unflushed batch. Records are attributed to the local
hour when accounting completes, not split across the hours of a long request.
Epoch bucket keys distinguish repeated
DST hours; calendar queries use server-local dates, including 23/25-hour days and
fractional UTC offsets. The heatmap combines repeated hours. Changing the server's
timezone reinterprets historical bucket timestamps; boundaries then have hourly
precision, not exact request-level precision.

Hourly rows older than 400 days are pruned daily; SQLite reuses freed pages and
incrementally reclaims space. Idle models do not generate rows. At most 4,096
pending model/hour aggregates are retained during storage outages; overflow drops
analytics only. The internal query reports current-process `dropped_requests`
and storage availability. Reads use committed snapshots and never wait for a
flush. Storage failures do not prevent inference; recoverable write failures
retry. Corruption found at startup is moved to one `usage.sqlite3.corrupt`
recovery backup (plus any
SQLite sidecars) and a fresh database is created. Future schema versions are left
untouched. Runtime corruption can require a server restart.

To reset history, stop oMLX and remove `usage.sqlite3`, `usage.sqlite3-wal`, and
`usage.sqlite3-shm` from the configured base directory, if present. Remove the
`.corrupt` backup and its sidecars too if desired. Restart oMLX to begin fresh.
The management `GET /stats` endpoint reads a separate counter store; it does
not erase or return hourly history.

## Disabling

Usage history is on by default. To disable it, stop the server and set
`usage.usage_history` to `false` in `settings.json`, then restart. The
`OMLX_USAGE_HISTORY` environment variable (`0`/`false`/`off` or `1`/`true`/`on`)
overrides the saved value at startup.

Disabling recording does not erase `usage.sqlite3`. A server started with
recording off does not create, open, or repair the database. Normal shutdown
flushes pending aggregates before the process exits. Session and all-time
serving counters are unaffected.

## Reading history

There is no replacement for the old `GET /admin/api/usage` route. The
management `GET /stats` route reports session and all-time counters, not
hourly buckets or model history. For local inspection, open a read-only
SQLite connection to the configured base path. Schema version 1 has the
`model_usage_hourly` table keyed by `timestamp_hour` and canonical `model_id`.
For example, with `DB` set to the actual database path:

```bash
DB="$HOME/.omlx/usage.sqlite3"
python3 - "$DB" <<'PY'
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

uri = Path(sys.argv[1]).resolve().as_uri() + "?mode=ro"
with closing(sqlite3.connect(uri, uri=True)) as db:
    for row in db.execute(
        "SELECT model_id, SUM(requests), SUM(prompt_tokens), "
        "SUM(completion_tokens) FROM model_usage_hourly "
        "GROUP BY model_id ORDER BY SUM(requests) DESC"
    ):
        print(*row)
PY
```

This is a local storage schema, not a stable dashboard API. The internal
`UsageHistory.query` method supports today, yesterday, 7/30/90 days, and the
current month, including model filters and derived rates. A future dashboard
needs an explicit authenticated backend endpoint before it can use those
summaries remotely.
