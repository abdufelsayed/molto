# Model studio

Open **Studio** in the dashboard to experiment with a local Molto model. Each
experiment has its own system prompt, generation controls, tools, conversation,
and virtual files. The system prompt starts empty, generation controls start
unset, and all seven tools start enabled. An empty system prompt adds no
application system message. Unset controls inherit Molto's defaults; an explicit
session value takes precedence over forced sampling settings for that request.

## Running experiments

Choose a local model and enter a prompt. Enter runs it; Shift+Enter inserts a
newline. Cmd/Ctrl+Enter also runs, and Escape stops. Edit the system prompt in the
right sidebar's **Prompt** tab. **Model**, **Tools**, and **Sandbox** hold the other
session controls; virtual files live under **Sandbox**. Both sidebars can collapse.
Controls remain editable during generation and take effect on the next run.

User messages align right; assistant answers align left. Message actions use
icons with tooltips and appear on hover or keyboard focus, while remaining
visible on touch devices. Reasoning appears only when supplied by the model.
The entire reasoning and tool chain expands as one disclosure with individual
steps inside. It opens during generation and collapses after completion unless
you changed its open state manually.

Expand **Observability** below the chat input to inspect a run. Its resizable
drawer has **Trace**, **Request**, **Response**, **Usage**, and **Runs** tabs.
**Usage** includes research notes, recorded settings, and historical file changes.
The inspect icon on a response or activity step opens the matching run and event.

Every run records its requested settings, input conversation, raw model requests
and streamed responses, token usage, timings, starting and ending files, and
tool activity. The inspector also captures inherited global/model configuration
at run start. Those values record saved defaults; model templates can contribute
further defaults. Thinking controls depend on the model's template, and a seed
is best effort rather than a promise of identical output.

Use the **Edit message** or **Rerun live** icon on any user message. Previous responses
remain selectable as run variants. Selecting a variant restores its prompt and
ending files, saves the current conversation as a branch, and starts a continuation
from that turn. Rerunning an earlier turn saves the old
continuation as a branch and starts a new continuation. Restore branches in
**Runs**, or **Fork here** to create an independent experiment with that turn's
settings, conversation, and ending files. **Copy setup** copies settings and
current files; duplicating an experiment copies its complete history.

**Replay tools**, in the message actions menu, reruns the model using recorded tool results and recorded file
changes. Calls must match the recorded tool name, arguments, and order. A
divergence stops replay; it never falls back to live execution. The model still
performs new inference during replay. **Rerun live** executes tools anew.

## Tools and sandbox

The minimal tool set is `read`, `write`, `edit`, `bash`, `web_search`, `fetch_url`,
and `ask_question`. Disable tools individually or set tool choice to none for
plain model experiments. Confirmation mode pauses before execution. Questions
offer suggested answers and always permit free text.

`bash` runs just-bash inside a browser Web Worker, with an isolated in-memory
filesystem. It cannot access host files or execute Python, Node, or installed
programs. Text utilities such as `rg`, `find`, `sed`, and `jq` are available.
Each command starts at the configured working directory and environment.
Read-only mode blocks mutations from every tool, including shell redirection.
Disabling `write` alone does not make bash read-only.

Sandbox network access starts disabled. Enabling it permits `curl` GET through
the URL-fetch broker only while `fetch_url` is also enabled. The broker returns
readable text, not unrestricted HTTP responses. Network activity is recorded.
Timeout, command, loop, filesystem, and output limits are configurable.
Interpreter errors may retain partial file changes; terminating the worker
discards changes since its last completed snapshot and ends the run. A new run
restores that snapshot in a new worker.

The **Files** panel uploads, creates, edits, downloads, deletes, and resets virtual
files. Attachments are stored there, with their paths included in the user
message; image files are also sent as image input. Model image support varies.
Code blocks support syntax highlighting, copying, and saving to the filesystem.
Editing files is locked during a run.

## Web tools

Web execution is TypeScript in the dashboard server. The provider choices are
automatic metasearch (`ddgs`), custom metasearch (`ddgs_custom`), DuckDuckGo,
Brave API, and SearXNG. Metasearch offers Brave, DuckDuckGo, Grokipedia, Mojeek,
Wikipedia, Yahoo, and Yandex adapters. Blocked engines, rate limits, and empty
results produce explicit warnings. Search markup and availability can change.

Brave credentials and the SearXNG address come from server integration settings
and remain on the server. Provider choice, engines, result count, snippet/full
content, and content limits belong to the session and do not modify global
configuration. The integration connectivity test uses the same TypeScript
implementation without saving pending settings.

URL fetch permits public HTTP(S) destinations, checks every redirect, pins the
connection to reviewed DNS addresses, and limits response bytes and duration.
Only the server-configured SearXNG origin may use a private network address.
Sources come from actual tool results. Web outputs receive stable citation IDs;
matching numeric references link to those sources. Citations identify retrieved
sources and do not establish that an answer is supported by them.

## Persistence and ownership

Experiments are saved in the dashboard browser's IndexedDB, with periodic
checkpoints during streaming. Reloaded unfinished runs are marked interrupted.
Storage failures are shown in the workspace. JSON exports/imports include
settings, history, branches, notes, and virtual file snapshots. Persistence is
local to this browser and origin and is not synced between devices.

The agent lives in `apps/dashboard/src/features/studio/agent`. It receives
settings, messages, snapshots, and transport callbacks. It owns no storage and
imports no inference engine code. Dashboard composition supplies the
authenticated model API and web broker, and saves results. Python continues to
own inference; the old Python web executors and `/v1/web/*` routes were removed.
The studio opts out of MCP injection using `include_mcp_tools: false`. No Eve,
Chat SDK, or AI Elements runtime is required. The interface composes current
shadcn Base UI components and custom studio components.

## Verification

Dashboard tests use synthetic OpenAI streams and actual browser workers without
downloading models. Server tests cover sampling precedence. TypeScript tool
tests cover provider parsing, network restrictions, redirects, response limits,
authentication, and connectivity testing. Run `pnpm check` and `pnpm test` from
the workspace root.
