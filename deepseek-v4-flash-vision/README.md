# DeepSeek-V4-Flash-Vision-Exp on 12x RTX 3090

**DeepSeek-V4-Flash-Vision-Exp** (285B MoE, FP4 experts + FP8 attention, 185 GB, text + vision)
served by vLLM on consumer Ampere hardware — with **speculative decoding**, **1M-context capability**,
**working tool calls**, and **working image input alongside DSpark**.

First documented run of this model on SM86 (RTX 3090) hardware.

## Highlights (all measured on this hardware)

| Metric | Value |
|---|---:|
| Prefill (long-context, cold) | **~3,500 tok/s** (425K tokens in ~120s) |
| Prefill (3 concurrent 470K requests) | 3,400 tok/s aggregate (~95% scaling) |
| Decode, DSpark speculative (k=3) | **58–60 tok/s** warm, 2.4x vs no-spec (25 tok/s) |
| Spec acceptance (code/text) | 76–87%, mean 3.3–3.5 of 4 tokens/step |
| Max context — no CPU offload | **1,000,000 tokens** (single request) |
| Max context — 36 GB CPU offload | 500K x **2 concurrent** GPU-resident + ~4.16M tokens parked in RAM |
| Vision (image input) | **Working with DSpark (k=3)** — see [Vision](#vision) below |
| Vision decode (1 image) | ~40–44 tok/s |
| Tool calls (OpenAI-compatible) | **Working** — see [the fixes](#the-fixes) below |

## Vision

Image input **works with DSpark speculative decoding enabled** (verified: correct
object/shape/color descriptions, 1–3 images per prompt). This was previously
blocked by an overly-coarse "mm × draft isolation" gate in the model runner that
zero-filled drafts on **every** decode step of a multimodal request (0% acceptance).
See [`SPEC_VISION_FIX.md`](SPEC_VISION_FIX.md) for the full root-cause and fix.

The fix keeps image-feature KV out of the draft pool (prefill-step skip) while
letting **decode steps of vision requests draft normally** — restoring spec on
vision without reintroducing the cross-request contamination. The start scripts
ship with `--limit-mm-per-prompt '{"video":0}'` (images unlimited, video disabled).

## Start scripts

All scripts share the same image and patches; they differ in context/offload
placement and whether DSpark is enabled. They are standalone — `./script start`
is all you need (modify the env defaults at the top if your paths differ).

### 1. `start-dsv4-spec.sh` — DSpark speculation + vision (primary)

```bash
./start-dsv4-spec.sh start
```

- `--max-model-len 409600`, TP=2 PP=5 on 10 GPUs, `--max-num-seqs 4`
- DSpark k=3 (`num_speculative_tokens: 3`), vision enabled
- Aligned to the proven 2x5 baseline config (no `--enforce-eager`/`-O0`/offload)
- **The working configuration**: spec + vision + tool calls together
  (text ~32–55 tok/s, vision 1-image ~40–44 tok/s, 3-image ~44 tok/s)

### 2. `start-dsv4-1m.sh` — 1M context, no CPU offload

```bash
./start-dsv4-1m.sh start
```

- `--max-model-len 1000000`, KV pool **1,010,835 tokens** (2 GiB explicit cap)
- A single request up to 1M tokens fits entirely in GPU memory
- Multiple smaller requests share the same pool (e.g. 2x 450K)

### 3. `start-dsv4-500k-offload.sh` — 500K context x 2 users, 36 GB RAM offload

```bash
./start-dsv4-500k-offload.sh start
```

- `--max-model-len 500000` + `--kv-offloading-size 36`
- **2x 500K requests fully GPU-resident concurrently** — both decode at full
  speculative speed with zero stalls
- ~**4.16M additional tokens** of computed KV parked in pinned RAM: re-querying the
  same long document reloads from RAM instead of re-prefilling (a 500K prefill
  costs ~150s; an offload reload is <1s)
- Verified: 3x 470K concurrent requests (1.41M live KV — 140% of GPU pool) all
  answered correctly, throughput within ~7% of single-stream

**Don't raise the offload size past ~38 GB.** The `cudaHostRegister` pinning wall on
this stack sits at ~4.2 GB per worker rank (10 ranks); 40+ GB fails pinning mid-boot.

## Hardware

| | |
|---|---|
| GPU | 12x NVIDIA RTX 3090 24 GB (Ampere, SM86) — **10 used** (TP=2, PP=5), 2 spare |
| RAM | 251 GB (36 GB pinned for KV offload in variant 3) |
| Disk | 7 TB (model weights: 185 GB) |
| Power | 250 W/GPU limit recommended (stock 220 W is fine) |

## Model & image

- **Weights:** `DeepSeek-V4-Flash-Vision-Exp` (185 GB, 48 shards + fused DSpark MTP
  draft layers, `n_predict=3`)
- **Image:** `ghcr.io/ciprianveg/3090-vllm:dsv4-flash-vision-sm86` — built from
  [wtdcode/vllm-backport PR #58](https://github.com/wtdcode/vllm-backport/pull/58)
  (`pr/vision-sm80`), with `TORCH_CUDA_ARCH_LIST=8.6`, PyTorch 2.11 + cu130
- **Serving stack facts:** Marlin W4A16 FP4→BF16 dequant for MoE experts on Ampere,
  Triton MLA sparse attention with software FP8, TileLang hyperconnections
- **Computed context (from boot logs):** `max_model_len=409600`, GPU KV cache
  **560,058 tokens**, max concurrency for 409,600-token requests = **1.37x**

## The fixes

Runtime patches (bind-mounted, in [`patches/`](patches/)) are **required**:

| Patch | What it does |
|---|---|
| `patches/model.py` | DSpark drafter aliases the target embedding table on the last pipeline rank — this builds `embed_tokens` there (PR #58 companion fix) |
| `patches/model_runner.py` | **Spec + vision fix** (the DSpark propose gate): skip the drafter only on prefill steps (which may carry image rows), and draft normally on decode steps of multimodal requests. Fixes 0% acceptance on vision and the deterministic 2-token death of text-after-image. See `SPEC_VISION_FIX.md` |
| `patches/scheduler.py` | **Transition repair** + **mm × spec row-crossing fix**: grants the missing bonus row for a request finishing chunked prefill, and pads multimodal decode steps to `1+k` rows with auto-rejected `-1` placeholder drafts so ragged mm/text row spans don't leak an EOS tail into the next text request |
| `patches/sched_output.py` | Adds `scheduled_mm_req_ids` to the scheduler output so the speculator can identify multimodal requests |
| `patches/dspark_speculator.py` | Text-only drafting for DSpark (upstream PR #33437 semantics): marks the draft as `supports_mm_inputs=False` so the per-step mm gather never mutates encoder/mm state from the draft path |
| `patches/vision.py` | Routes large-image ViT attention through FlashInfer instead of the O(S²) SDPA math backend (OOM fix for ~4k ViT patches) |
| `patches/flashinfer_sparse.py` | Keys the FlashInfer sparse workspace by `(device, workspace lane)` so the draft (DSpark) and target never share one 128 MB scratch buffer (draft CUDA-graph replay vs target aux-stream race) |
| `patches/block_table.py` | Zeroes the stale tail of a reused block-table row on overwrite, so a new request doesn't inherit the previous occupant's block ids (image-flavored KV after an mm request) |

Without these, DSpark + tool-call grammars is broken upstream
([#49002](https://github.com/vllm-project/vllm/issues/49002),
[#49210](https://github.com/vllm-project/vllm/issues/49210)) and vision + spec is
broken (see `SPEC_VISION_FIX.md`). With them, spec decode + tool calls + vision +
500K context run together, stable through hours of agent traffic.

## Performance notes

- **3,500 tok/s prefill** is the long-context cold number (sparse attention skips
  most of the haystack on filler text). Real-world chunked prefill of a 425K
  context completes in ~2 minutes; prefix-cache hits make re-runs near-instant.
- **58–60 tok/s decode** is the warm single-stream text number with DSpark
  acceptance in the 80%+ range; first request after boot runs ~45 tok/s until
  caches warm.
- **Vision decode** is ~40–44 tok/s (1 image) and ~44 tok/s (3 images in one
  prompt) with DSpark k=3 — the text-only draft drafts the image-conditioned text
  rows, which are in-distribution.
- Decode throughput is memory-bandwidth-bound: aggregate stays ~60 tok/s whether
  1 or 4 users generate (each user gets 1/n of it). Two users is the sweet spot.
- GPU 11 historically flaky on this box — the config pins `CUDA_VISIBLE_DEVICES=0-7,10,11`.

## Quickstart

```bash
# weights
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp --local-dir /mnt/data7tb/models/DeepSeek-V4-Flash-Vision-Exp

# run — primary spec+vision variant (set VLLM_API_KEY for a secured API)
export VLLM_API_KEY="<your-key>"
./start-dsv4-spec.sh start

# alternatives
./start-dsv4-1m.sh start          # 1M ctx, no offload
./start-dsv4-500k-offload.sh start  # 500K ctx x2, 36 GB offload

# manage
./start-dsv4-spec.sh status | logs | stop | restart
```

Boot takes ~9 minutes. Watch for `GPU KV cache size`, and
`Application startup complete` in the logs.

API is OpenAI-compatible:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"dsv4-flash-vision","messages":[{"role":"user","content":"hello"}],"max_tokens":64}'
```

## Tuning knobs that matter on this hardware

- `--kv-cache-memory 2147483648` — explicit 2 GiB cap. Without it the default KV
  pool sizing eats the activation margin and warmup OOMs on 24 GB cards.
- `VLLM_PP_LAYER_PARTITION=10,9,9,9,6` — not layer-balanced but **weight**-balanced:
  the last rank carries the DSpark draft (~5.5 GB) + embed + lm_head.
- `--num-speculative-tokens` must be a multiple of the draft's `n_predict=3` (k=3 or 6).
  k=3 is the measured winner; k=6 loses (low acceptance, high verify cost).
- `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64` caps indexer logits buffers.
- `--max-num-batched-tokens 2048` (spec variant) keeps Triton warmup within margin.
