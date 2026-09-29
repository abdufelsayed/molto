# Testing the backend

Use the locked development environment on Apple Silicon. The default pytest
configuration excludes slow and integration tests:

```bash
uv sync --locked --python 3.11
uv run --locked pytest
```

Run the affected test file before the default suite. Add `-m integration` or
`-m slow` only when those tests match the change and the machine has the
required models and hardware. For a server API change, check the actual HTTP
route with an isolated base path and port after the unit tests. See the
[management API](management-api.md) for its authorization contract.

For model management changes, run the retained behavior tests:

```bash
uv run --locked pytest -q \
  tests/test_management_unload.py \
  tests/test_management_reload.py \
  tests/test_management_model_settings.py \
  tests/test_management_hot_cache_clear.py \
  tests/test_management_profiles_api.py
```

## Opt-in real-model backend check

Set `OMLX_TEST_GEMMA_VLM` to a complete local Gemma vision-language
checkpoint, then run:

```bash
OMLX_TEST_GEMMA_VLM=/path/to/complete/local/GemmaVLM \
  uv run --locked pytest -m 'slow and integration' \
  tests/integration/test_backend_serving_real_model.py
```

Without that variable, the test skips. It starts a separate server with a
temporary base path, a symlink to the checkpoint, and an unused local port.
It checks management load, a short text response, a red-square image response,
streamed chat with a final SSE marker, and unload until no model remains
loaded and tracked model memory is zero. This is an end-to-end behavior
check for that checkpoint, not a throughput benchmark or proof for other
model families.

## Cluster test filesystem isolation

The autouse `cluster_home` fixture gives each test a temporary directory for cluster interpreter shims and SSH files. It preserves explicit shim `home` arguments and leaves `HOME` unchanged so model discovery paths still work. The shim unit tests import the original function directly and provide their own temporary paths or patch `HOME` to verify the real default-path behavior.

The cluster GET-route smoke test checks that `/ssh-key` creates its key pair there.

# Text-only VLM loading tests

Run `uv run --locked pytest -q tests/test_vlm_vision_fallback.py` to check strict loading and logits with a small quantized DiffusionGemma checkpoint, unchanged loaders for unreadable shards or retained vision, and patch restoration after loading errors. No model download is required.

# Test timing

CI runs the default tests on Python 3.11 for pull requests and on Python
3.11, 3.12, and 3.13 for pushes to `main`. Use
`uv run --locked pytest --durations=50 --junitxml=test-results.xml` to collect
timing data locally. Compare runner queue time separately from test execution.

The automatic Qwen FP16/BF16 decode route has numerical, cache-state and
fallback tests in `tests/test_qwen35_fp16_decode.py`. Run it with
`tests/test_qwen35_gdn_prework.py` to check that the existing BF16 Qwen4 and
speculative routes remain intact. See
[GDN decode prework](experimental/qwen35_fp16_decode.md) for the hardware, geometry
limits and real-model benchmark requirements.

# First-token burst release

Run `uv run --locked pytest -q tests/test_engine_core.py tests/test_output_collector.py`
to check first-chunk delivery across the executor boundary, late admission,
multi-token chunks, output ordering, later burst limits and request cancellation.
Burst decode releases each request's first generated chunk before continuing
with the configured burst policy. This adds one executor hand-off per request;
it does not shorten prefill or bypass parser, stop-string or stream-interval
buffering. Subsequent chunks still follow the selected Burst Decode setting.

For a real-server comparison, use the same model, prompt, output length and
cache state on main and the branch. Measure client-observed first content and
complete-response time separately from producer-side token timestamps, with
balanced (0.1 s) and aggressive (0.2 s) burst settings. Include short replies,
long replies, a second request admitted during decode, and disconnect/recovery.
Report any custom budgets separately from the stock modes.

Cluster process-group tests use the `mock_cluster_ssh` fixture; remote teardown and serve-marker tests retain their own transport assertions. Mock-model engine tests skip explicit GC, while `test_engine_teardown.py` and `test_per_engine_threads.py` retain teardown and reclamation coverage. GLM5 execution tests reuse the eight-layer KDA/DSA fixture with dense and MoE layers; checkpoint-key tests retain the 45-layer configuration. The SDPA memory test retains the 8K/32K length ratio, head dimension 256, and 6:1 GQA ratio with fewer heads. DeepSeek V4.1 direct and converted engine checks run sequentially in one isolated subprocess with separate checkpoint directories.

# Cache cleanup logging tests

Run `uv run --locked pytest -q tests/test_vision_feature_cache.py tests/test_paged_ssd_cache.py -k "cleanup_unlink_failure or corrupt_block_cleanup_logging"` to check failed-delete warnings and cache cleanup state. These cases use the existing cache fixtures to close writer threads.

# Cache-preserving engine teardown tests

Run `uv run --locked pytest -q tests/test_engine_teardown.py tests/test_engine_core.py tests/test_scheduler.py tests/test_paged_ssd_cache.py tests/test_hot_cache.py tests/test_engine_pool.py tests/test_batched_engine.py tests/test_vlm_engine.py` to check the 60-second teardown budget and one progress-gated extension to 120 seconds. Clock tests cover the deadlines; short-budget subprocesses exercise fatal exits. Real writer-thread cases verify primary/draft flushes and in-flight prefix stores survive a saturated queue and can be reused after reload. Async cases cover event-loop responsiveness, cancellation, and leases during another model's unload.

# ModernBERT embedding tests

Run `uv run --locked pytest -q tests/test_modernbert_attention.py tests/test_embedding.py tests/test_mlx_embeddings_compat.py` to check finite padded attention, single-input equivalence, local-window masking, and embedding integration. The attention regression covers fp16, bf16, and fp32 at lengths around the affected SDPA tile boundaries.

# QSA reservation tests

Run `uv run --locked pytest -q tests/test_qwen4_qsa_reserved_capacity.py` to check QSA capacity reservations.

The integration tests cover restored-prefix lengths with boundary snapshots enabled and disabled, the first allocation after cache restoration, and prefill/decode output equivalence using a small Qwen4 model.

Related regression suites are `test_qwen4_qsa_incremental_cache.py`, `test_qwen4_qsa_decode_gather.py`, and `test_prefill_oom_graceful.py`.

For Qwen4 native sparse-GQA prefill measurements, run `python benchmarks/bench_qwen4_qsa_sparse_gqa.py --key-tokens 24576 --query-tokens 1024 --repetitions 30`. The benchmark reports index scoring, top-k selection, the combined native pipeline, every supported main-attention tile, the portable reference, and maximum error. Production groups native query rows into 4,096-row tiles through 32K keys, 2,048-row tiles through 64K, and 1,024-row tiles above 64K; this bounds the FP32 score sheet while amortizing per-tile dispatch.

# Qwen4 verify attention row tests

Run `uv run --locked pytest -q tests/test_qwen4_verify_attention_rows.py` to check that row-exact Lightning MTP verify windows through Qwen4 attention give every row the bits of the serial one-row decode step and leave the same KV and QSA indexer state. The tests build one attention layer at the real Flash-Next shapes with synthetic 6-bit weights. Masked-arm windows (past the 2,048-token QSA budget, rank-three positions) cover 2 to 8 rows at 2,060, 16,382 and 24,000 cached tokens and compare each row's FP32 block scores and token mask; a rollback case accepts one draft and decodes on. Dense windows below the budget include rows on both sides of MLX's one-pass/two-pass vector SDPA switch at 1,024 keys. `OMLX_QWEN4_QSA_MASKED_VERIFY=0` restores the multi-row masked path.

`test_mlx_vlm_qwen4_exp_compat.py::test_qwen4_mtp_one_row_step_is_the_serial_decode_step` checks that a one-row Lightning MTP window (the activation step and depth-0 cycles) runs the serial decode step: equal logits and cache state, no speculative transaction, and a following verify window that rolls back as usual. `OMLX_QWEN4_MTP_ONE_ROW_DECODE=0` keeps the verify forward for those windows.

# Prefill memory accounting tests

Run `uv run --locked pytest -q tests/test_prefill_transient_tracker.py tests/test_prefill_oom_graceful.py` to check retained versus reclaimed overhead, configured chunk sizes, and abort-cap enforcement. The loop tests run a small initialized MLX model with controlled footprint readings through external and chunked prefill; they do not load a checkpoint.

# Prefix cache completion tests

Run `uv run --locked pytest -q tests/test_scheduler.py tests/test_scheduler_boundary_completion.py tests/test_prefix_cache_gdn_split.py` to check cache-freshness admission and completed boundary recovery. The completion tests use a small initialized Qwen3.5 hybrid model and the real BatchGenerator, then compare restored-prefix logits with a fresh forward pass. They cover embedded snapshots, GDN sidecars, off-boundary completion, and unknown or inconsistent cache positions.

# Cluster join recovery tests

Run `uv run --locked pytest -q tests/test_cluster_pairing_session.py tests/test_cluster_pairing.py` to check joining, cancellation, and approval. Session tests recreate a manager with the same base path to verify that the original code and cancellation proof survive a restart. They also cover offline cancellation, switching peers while cleanup is pending, rejected requests versus lost responses, and storage failures.

For a two-Mac smoke test, start isolated servers with separate base paths. Request a join, restart only the joining server, then cancel and retry; the coordinator must remove the original pending request. Repeat with the coordinator offline: cancellation must return `state: idle` with `cleanup_pending: true`, and joining a different reachable Mac must work. Restore the coordinator and poll the join endpoint to verify cleanup. Keep existing approval/cancel race and token-ownership tests in the run; never relax the coordinator's token check to make a stale request disappear.

# DeepSeek V4.1 offline tests

CED scheduler prefill, short suffix positions, full-logit scoring, and DSpark
continuity: `uv run --locked pytest tests/test_deepseek_v41_ced.py tests/test_deepseek_v41_ssd.py -q`.
Replay is approximate; cache restoration is compared with the same chunk boundaries.

Run `uv run --locked pytest -q tests/test_deepseek_v41.py` for the text/vision port. Synthetic weights and small recorded official outputs cover prefill/decode, DSpark, Engram and vision numerics without a checkpoint download or vendored reference implementation. See `tests/fixtures/deepseek_v41_expected.md` for provenance and reference corrections. PyTorch is optional for the independent FP8/FP4 arithmetic checks. Other cases cover BatchGenerator admission, cache restoration, DSML and VLM engine execution.

FP8 activation boundary and layout checks: `uv run --locked pytest tests/test_deepseek_v41_activation.py -q`. These compare fused FP8 and weighted SwiGLU kernels against reference arithmetic across rounding ties, scale transitions, clipping, intermediate casts and floating-point dtypes.

MoE activation reuse and per-expert reference checks: `uv run --locked pytest tests/test_deepseek_v41_moe.py -q`. These cover sorted prefill, decode, independent routed/shared activation policies, and repeated outputs with small quantized weights.

DeepSeek V4.1 Metal arithmetic and sparse-addressing checks: `uv run --locked pytest tests/test_deepseek_v41_kernels.py -q` (no checkpoint required).

Packed attention rounding: `uv run --locked pytest tests/test_deepseek_v41_attention_rounding.py -q`. An independent MLX oracle checks 64-key online maxima, FP32 denominators, BF16 PV probabilities, masking, sink placement, and growing or strided KV across the threadgroup capacity boundary. This does not execute the official CUDA kernels.

Engram storage and prefetch lifecycle: `uv run --locked pytest tests/test_deepseek_v41_offload.py -q`. These use synthetic fixtures and require no checkpoint download.


### MoE expert residency

`tests/test_deepseek_v41_moe_offload.py` exercises original and converted
V4.1 expert reads, MXFP4/MXFP8 and mixed-bit affine arithmetic, repeated
evictions, sorted routes, load/inference thread separation, Engram coexistence,
draft-weight exclusion, and memory estimates. The load probe rejects whole
expert reads from shared shards and any expert slab read through the Engram
mapping. Further cases check that consumed read buffers are released within the
in-flight byte window and pin the serial LRU order under concurrent reads,
expert-boundary chunking of sorted routes, and the fit-to-budget residency
helper against the admission arithmetic. `tests/test_deepseek_v41_affine_source.py`
covers community `mlx_lm` affine source checkpoints: packed and declared-dense
projections, exact force-dense dequantization, the affine Engram table spec, a
convert round-trip against a direct load, the declared-format resolver, and
offload eligibility. Run alongside `test_deepseek_v41_offload.py`,
`test_moe_expert_offload.py`, and the engine-pool/model-settings suites.

`tests/test_moe_expert_offload.py` also exercises Qwen4-Exp MoE routing with
512 experts, top-k 10, 64 resident slots, shared experts, and repeated
evictions, plus the resident Lightning MTP head (`mtp.*`) and its admission
pricing. `tests/test_moe_offload_compat.py` covers the model-type allowlist,
checkpoint completeness, dense-model exclusion, API/runtime rejection, and
PLE/Engram metadata after expert savings.

# Accuracy benchmark worker tests

Run `uv run --locked pytest -q tests/test_eval_worker_pool.py tests/test_eval.py`. Worker tests cover slot refilling, thinking-mode retries, sequential code scoring without blocking generation, and repeated cancellation during scoring. HumanEval, MBPP, and LiveCodeBench subprocess cases verify normal completion, cancellation draining, and temporary-file cleanup with a controlled engine; no model checkpoint is required.

# Profile API exposure tests

Run `uv run --locked pytest -q tests/test_model_settings_profiles.py` to check profile persistence, exposed model IDs, and name collisions. Management route tests cover the HTTP contract separately.

### Lightning MTP with XTC sampling

Run `uv run --locked pytest tests/test_mtp_xtc_sampling.py -q` for request sampler changes, late-joining mixed batches, row removal, and greedy sampling. These tests use a small MLX model and observe the MTP eligibility boundary; they do not execute a trained MTP head.

### Batched DFlash drafter

Run `uv run --locked pytest tests/test_dflash_batched.py tests/test_mlx_lm_mtp_patch.py -q -k "dflash_batched or block_drafter"`. `test_dflash_batched.py` builds a tiny DFlash2 drafter with random weights and checks that rows drafted together match the same rows drafted alone across ring wrap-around, ragged context segments and cohort changes, plus the prefill seed window slicing and block-size clamping. The `block_drafter` cases in `test_mlx_lm_mtp_patch.py` drive the Lightning MTP verify path with a table drafter on the CountingModel harness and require token parity with standard decoding, one context entry per committed position (including late joins) and release of finished rows. Real drafter acceptance and throughput need a Qwen3.5-family VLM checkpoint with its `z-lab` DFlash draft and are measured against the standard batched engine.

# VLM cache boundary tests

Run `uv run --locked pytest -q tests/test_vlm_cache_boundaries.py tests/test_vlm_engine.py tests/test_prefix_cache.py tests/test_paged_cache.py` to check image-aware prefix keys. Boundary cases cover reasoning-dependent template prefixes, final grid token positions, adjacent images, multiple images per turn, block edges, invalid metadata, and isolation when earlier or later images change. These tests use synthetic processor inputs and KV arrays; checkpoint preprocessing and inference comparisons require local models.


### Profile consistency

Run `uv run --locked pytest tests/test_model_settings_profiles.py -q`. The cases cover profile references, persistence rollback, and saved values. For a manual check, create and apply a model profile through the management API, then verify the effective settings and API-visible model ID.

Global and model profiles may share display names while retaining separate IDs. Applying a global template reads its latest settings; deleting it preserves model copies as independent profiles. Both editors save and apply new profiles, and focus refresh preserves unsaved edits.

Startup reference repair is covered in `tests/test_model_settings_profiles.py`: missing references are cleared without replacing saved values, originals are backed up under `<base_path>/profile-reference-backup-*`, and subsequent loads do not write again. Retries with unchanged originals reuse the same content-hash backup directory and fill missing files. Mismatched backups prevent repair. Invalid storage, unsupported versions, backup failures, and failed writes must not persist inferred repairs. A rollback failure aborts startup and logs the backup path.

## Qwen tool-call recovery

Run `uv run --locked pytest tests/test_tool_calling.py tests/integration/test_e2e_streaming.py -q -m "not slow"` to include the integration cases. Final Qwen parsing preserves unknown tool names for client feedback, recovers complete functions missing only the outer envelope close at normal EOF, and reports unrecoverable siblings without a successful stop. Cases cover all three streaming APIs, chunk boundaries, repeated calls, literal tags in arguments, schema validation and length stops. Other parser families and reasoning-channel promotion keep their existing rules.

For a real-server check, request a small `write(content: string)` call with thinking disabled and greedy sampling. Compare the normal result with a request using `stop: ["</tool_call>"]`: the complete function should still arrive once with identical arguments. Then supply an assistant call to an unknown tool followed by a matching tool-error message naming `write`; verify the next model turn uses `write`. Use an isolated port and base path, and do not execute model-supplied file operations during the check.

# Streamed oQ calibration tests

Run `uv run --locked pytest tests/test_oq.py -k TestStreamedCalibration` for streamed calibration. The small BF16 Qwen4 fixture exercises GDN, sparse attention, mmap PLE and the MTP head. It compares imatrix statistics and fused sensitivity with resident collection, verifies cache reuse with and without MTP, and converts and reloads the artifact with its shared PLE scale intact. A small MiniMax decoder fixture also compares dense and MoE collection. These cases replace the separate streaming test modules and need no external checkpoint.

# Fused routed-expert decode tests

Run `uv run --locked pytest -q tests/test_qwen35_moe_routed_decode.py tests/test_qwen35_moe_router.py tests/test_qwen35_moe_gate_up.py` to check the one-token routed-expert kernels. Real `Qwen3_5MoeSparseMoeBlock` instances laid out like Qwen3.8-Flash-Next oQ (quantized routed experts, 8-bit shared expert and shared-expert gate, bf16 router) must match the served body bit for bit, with the shared expert and its gate folded into the two launches: 5-bit (oQ5e) and 4-bit experts at the Flash-Next shape (hidden 2560, intermediate 640, top-k 10), and 5-bit gs32, 6-bit gs128 and 8-bit experts at smaller shapes. A bf16 shared expert stays composed and must match too. Both launches are also run in FP32 against MLX's FP32 mat-vecs (routed and shared gate+up after SwiGLU, the gate row, every routed and shared down row), because BF16 outputs hide one-ulp FP32 differences (a fast-math `exp` in the SwiGLU sigmoid passes most BF16 cases but fails these). The kernels bind a one-expert view of the stacked weights; routing to experts 500+ of 512 checks that the view still reads the stacked buffer, and replacing the expert or shared-expert arrays must rebuild the cached plan. The other cases check that shapes where MLX would pick a different mat-vec partition, 3-bit experts, top-k 8, prefill and verify rows, float16, blocks without the gate+up fusion and a kernel failure all keep the served body. The one-launch router softmax + top-k must return the indices and scores of the softmax and top-k launches for random logits and engineered near-ties (every logit repeated eight times, logits on adjacent bf16 values, two-valued rows), a block whose router rows repeat eight times must route like the served block, and the softmax runs in FP32 against MLX's FP32 softmax (a fast reciprocal or a precise `exp` still routes identically but fails there). The router gemv must return MLX's `x @ W.T` logits bit for bit at 512x2560, 256x2048 and 128x1024, and its FP32 row sums must equal MLX's FP32 gemv on the same values (a `simd_sum` in place of MLX's shuffle-down tree changes only a few BF16 logits but every FP32 sum); shapes where MLX reduces K differently (K >= 16 N, a guarded K tail) keep `nn.Linear`.
