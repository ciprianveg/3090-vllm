# Fix: DSpark speculative decoding on DeepSeek-V4-Flash-Vision

Status: **DONE** — spec + vision is working, beat the no-spec baseline, k=3 chosen as the winner.

## Symptom (before the fix)

Spec decoding on an image request produced **zero drafts and 0% acceptance**:

```
[MMDBG2] post-sample ... sampled=[[1, 0, 0, 0]]      # drafts [[0,0,0]] -> nothing accepted
```

Root cause was in the patched runner `vllm_patch_dsv4/model_runner.py` (the `_last_scheduler_output` propose path).

## Root cause

The propose gate was:

```python
_step_has_mm = bool(getattr(_last_scheduler_output, "scheduled_mm_req_ids", None))
_step_has_prefill = bool(getattr(input_batch, "has_prefill", False))
if _step_has_mm or _step_has_prefill:
    draft_tokens = torch.zeros_like(...)          # <- zero-filled drafts
```

`scheduled_mm_req_ids` **persists across the whole decode** of a multimodal request
(`sched_output.py:304-306` includes "decode-only steps of such requests"), so the
gate fired on **every** decode step of an image request, permanently zero-filling
its drafts. Result: 0% acceptance on vision (and it also blocked spec on the whole
batch whenever an image request was present).

The intent of the original gate ("mm × draft isolation") was correct, but it was
**too coarse** — it targeted image rows, yet it disabled drafting for all steps of
any MM request.

## The fix (one change in `vllm_patch_dsv4/model_runner.py`)

Key insight: **image placeholder tokens only ever exist in the forward input on a
request's prefill step. On decode steps, all rows are text** (image-conditioned
text hidden states, which are in-distribution for the text-only DSpark draft).

Dropped `_step_has_mm` and kept only the prefill skip:

```python
_step_has_prefill = bool(getattr(input_batch, "has_prefill", False))
if _step_has_prefill:
    # skip propose (stale anchor KV + image rows) -> one unspeculated step
else:
    draft_tokens = self.speculator.propose(...)   # decode steps draft normally
```

This:
- keeps image-feature KV *out* of the draft pool (prefill-step skip), preserving the
  cross-request isolation the old gate was protecting;
- lets **decode steps of vision requests draft** — restoring spec+vision.

### Why the old "2-token death of text-after-image" concern is addressed

The death symptom (NaN in the target sparse-indexer `idxbranch`/`amptopk`, e.g.
`ppL37 amL=nan`) was **reproducible only with the old spec container config**:
`--kv-offloading-size 36 --enforce-eager --no-async-scheduling
--max-num-batched-tokens 1024`. After aligning the spec container to the proven
2×5 baseline config (see below), the death **did not reproduce** across many runs
and request orders (0 NaN traces, all text requests complete 1500/1500 tokens).

## Supporting changes in `dsv4-spec.sh`

1. **GPUs**: `CUDA_VISIBLE_DEVICES` was `0,1,...,9` (GPUs 8/9 belong to qwen38).
   Changed to `0,1,2,3,4,5,6,7,10,11` (line 42).
2. **Align to the 2×5 baseline config** (only real difference from baseline = the
   DSpark spec config + images enabled):
   - `--max-model-len` 500000 → **409600** (later restored to **500000** in the
     repo's `start-dsv4-spec.sh`; 1M is available for 2 users via `start-dsv4-1m.sh`)
   - `--max-num-batched-tokens` 1024 → **2048**
   - removed `--kv-offloading-size 36`, `--enforce-eager`, `-O0`, `--no-async-scheduling`
   - `cudagraph_capture_sizes` → `[1,2,4]`, `max_cudagraph_capture_size` → 4
   - `--gpu-memory-utilization` 0.955 → **0.95**
3. **Multimodal limit** (line 58): `--limit-mm-per-prompt '{"video":0}'` — images
   unlimited (default), video 0. Supports ≥1 image per prompt.
4. **Spec config** (line 58): `--speculative-config
   '{"method":"dspark","num_speculative_tokens":3,
   "draft_sample_method":"probabilistic","enable_adaptive_verification":false}'`.
5. **Log color** (line 82): `logs()` pipes through sed to bold-green the value of
   `Avg generation throughput: X tokens/s`.

## A/B benchmark (same base config: `--max-num-seqs 4`, image on, 2×5 match)

| scenario | per-req tok/s | notes |
|---|---|---|
| **no-spec** text | 25.3 | |
| **DSpark k=3** text | **27.8** | beats the original locked 26.59 baseline |
| no-spec vision 1-conc | ~27.8 | |
| **DSpark k=3** vision 1-conc | **~27.4–28.6** | |
| no-spec vision 4-conc | **6.2–6.7** (wall 20.1 s) | concurrency collapses throughput |
| **DSpark k=3** vision 4-conc | **25.7–27.1** (wall ~9.9 s) | **~4× faster** (and 2× less wall) |
| DSpark k=6 text | 23.8 | **loses** — low acceptance, high verify cost |
| DSpark k=6 vision | ~13–15 | loses |

**Winner: DSpark k=3** (`num_speculative_tokens: 3`).

## Files changed

- `vllm_patch_dsv4/model_runner.py` — propose gate now `if _step_has_prefill:` (only
  skips prefill steps; decode steps of MM requests draft).
- `dsv4-spec.sh` — GPU set, config aligned to 2×5 baseline, spec config, image
  allowed, `Avg generation throughput` highlighted green in `logs()`.
- `dsv4-spec6.sh` — k=6 variant (used for the A/B; not the winner).
- `dsv4-baseline-vision.sh` — no-spec reference used for the A/B.
- `bench.py` — added `vision` mode (multi-threaded, image + question, reports
  per-req/total throughput and tokens-per-req).

## Verified working on the live winner

- Text decode: completes 1500/1500 tokens, ~32–55 tok/s (varies with warmup /
  prefix-cache hit rate).
- Vision (1 image): ~40–44 tok/s.
- Vision (3 images in one prompt): 112 tokens, ~44 tok/s decode.
- 0 NaN, 0 Traceback, 0 CUDA errors across repeated runs.
