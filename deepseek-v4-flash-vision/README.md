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
| Decode, DSpark speculative (k=3) | **60+ tok/s** warm (measured at a 240 W power cap; ~70+ expected uncapped) |
| Decode, DSpark (TP4×PP3, 12 GPUs) | **120+ tok/s** — same image, 12-GPU config |
| Spec acceptance (code/text) | 76–87%, mean 3.3–3.5 of 4 tokens/step |
| Max context — no CPU offload | **1,000,000 tokens** (single request) |
| Max context — RAM offload | **4,000,000 tokens** (computed KV parked in pinned RAM) |
| Vision (image input) | **Working with DSpark (k=3)** — see [Vision](#vision) below |
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

Both scripts share the same image, patches, and engine tuning (DSpark k=3 + vision
enabled); they differ only in context size and whether CPU KV offload is used.
They are standalone — `./script start` is all you need (modify the env defaults at
the top if your paths differ).

### 1. `start-dsv4-1m.sh` — 1M context, no CPU offload

```bash
./start-dsv4-1m.sh start
```

- `--max-model-len 1000000`, TP=2 PP=5 on 10 GPUs, `--max-num-seqs 4`
- DSpark k=3 (`num_speculative_tokens: 3`), vision enabled
- A single request up to 1M tokens fits entirely in the GPU KV pool
  (2 GiB explicit cap); multiple smaller requests share the pool (e.g. 2x 450K)
- **The working configuration**: spec + vision + tool calls together
  (text ~60+ tok/s at 240 W cap, ~70+ uncapped)

### 2. `start-dsv4-4m-offload.sh` — 4M context with CPU KV offload

```bash
./start-dsv4-4m-offload.sh start
```

- `--max-model-len 4000000` + `--kv-offloading-size 36` (configurable via
  `KV_OFFLOAD_GB`)
- Computed KV beyond the GPU working set is parked in pinned RAM: re-querying the
  same long document reloads from RAM instead of re-prefilling (a long prefill
  costs ~150s; an offload reload is <1s)
- **Don't raise `KV_OFFLOAD_GB` past ~38 GB.** The `cudaHostRegister` pinning wall
  on this stack sits at ~4.2 GB per worker rank (10 ranks); 40+ GB fails pinning
  mid-boot.

## Hardware

| | |
|---|---|
| GPU | 12x NVIDIA RTX 3090 24 GB (Ampere, SM86) — TP2×PP5 uses **10** (2 spare); TP4×PP3 uses **all 12** |
| RAM | 251 GB (36 GB pinned for KV offload in the 4M variant) |
| Disk | 7 TB (model weights: 185 GB) |
| Power | 250 W/GPU limit recommended (stock 220 W is fine) |

## Model & image

- **Weights:** `DeepSeek-V4-Flash-Vision-Exp` (185 GB, 48 shards + fused DSpark MTP
  draft layers, `n_predict=3`)
- **Image:** `ghcr.io/ciprianveg/3090-vllm:dsv4-flash-vision-sm86` — built from
  [ciprianveg/vllm-backport-3090](https://github.com/ciprianveg/vllm-backport-3090)
  (a pinned snapshot of [wtdcode/vllm-backport PR #58](https://github.com/wtdcode/vllm-backport/pull/58)
  `pr/vision-sm80`), with `TORCH_CUDA_ARCH_LIST=8.6`, PyTorch 2.11 + cu130
- **Serving stack facts:** Marlin W4A16 FP4→BF16 dequant for MoE experts on Ampere,
  Triton MLA sparse attention with software FP8, TileLang hyperconnections
- **Computed context:** the no-offload variant is configured with
  `max_model_len=1000000` (1M, the model's native YaRN-extended limit); the
  offload variant uses `max_model_len=4000000` (4M) with CPU KV offload. The GPU
  KV pool is sized by the explicit 2 GiB `--kv-cache-memory` cap.

## Mods & fixes

**This is not just the upstream PR #58 image.** The image is built from
[ciprianveg/vllm-backport-3090](https://github.com/ciprianveg/vllm-backport-3090)
(a pinned snapshot of [wtdcode/vllm-backport PR #58](https://github.com/wtdcode/vllm-backport/pull/58)
`pr/vision-sm80`), but the working spec + vision + tool-call setup requires our
own runtime patches **and** launch-config mods on top. Everything below is what we
changed beyond the stock upstream PR image.

### Runtime patches (bind-mounted, in [`patches/`](patches/))

All are bind-mounted at container start (`start-dsv4-1m.sh`); none are baked into
the image.

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

### Launch-config mods (`start-dsv4-1m.sh` / `start-dsv4-4m-offload.sh`)

| Mod | Value | Why |
|---|---|---|
| GPUs | `CUDA_VISIBLE_DEVICES=0-7,10,11` | GPUs 8/9 are reserved for another model on this box |
| Parallelism | TP2 × PP5 (10 GPUs) | weight-balanced; last rank carries the DSpark draft (~5.5 GB) + embed + lm_head |
| Layer partition | `VLLM_PP_LAYER_PARTITION=10,9,9,9,6` | not layer-balanced but **weight**-balanced |
| Context | 1M (`start-dsv4-1m.sh`, no offload) / 4M (`start-dsv4-4m-offload.sh`, RAM offload) | model native YaRN limit is 1M; 4M uses offload |
| KV cache | `--kv-cache-memory 2147483648` (2 GiB cap), `--kv-cache-dtype fp8_ds_mla` | explicit cap prevents warmup OOM on 24 GB cards |
| Spec | `--speculative-config '{"method":"dspark","num_speculative_tokens":3,...}'` | DSpark k=3 (measured winner; k=6 loses) |
| Vision | `--limit-mm-per-prompt '{"video":0}'` | images unlimited, video disabled |
| Batch | `--max-num-batched-tokens 2048` | keeps Triton warmup within margin |
| CUDA graphs | `cudagraph_mode=PIECEWISE`, sizes `[1,2,4]`, max 4 | |
| Indexer | `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64` | caps indexer logits buffers |
| FlashInfer | `VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS="trtllm_fp4_block_scale_moe,..."` | skip autotune on unsupported ops |
| NCCL | `NCCL_ALGO=Ring`, `NCCL_PROTO=Simple`, `NCCL_P2P_DISABLE=1`, `NCCL_ASYNC_ERROR_HANDLING=1` | PCIe-only multi-GPU (no NVLink on 3090s) |

Without these, DSpark + tool-call grammars is broken upstream
([#49002](https://github.com/vllm-project/vllm/issues/49002),
[#49210](https://github.com/vllm-project/vllm/issues/49210)) and vision + spec is
broken (see `SPEC_VISION_FIX.md`). With them, spec decode + tool calls + vision +
1M context run together, stable through hours of agent traffic.

## Performance notes

- **3,500 tok/s prefill** is the long-context cold number (sparse attention skips
  most of the haystack on filler text). Real-world chunked prefill of a 425K
  context completes in ~2 minutes; prefix-cache hits make re-runs near-instant.
- **60+ tok/s decode** is the warm single-stream text number with DSpark
  acceptance in the 80%+ range (measured at a 240 W power cap; ~70+ expected
  uncapped); first request after boot runs ~45 tok/s until caches warm.
- **120+ tok/s** on the same image with **TP4×PP3 across all 12 GPUs** (vs the
  TP2×PP5 10-GPU config above).
- Decode throughput is memory-bandwidth-bound: aggregate stays ~60 tok/s whether
  1 or 4 users generate (each user gets 1/n of it). Two users is the sweet spot.
- GPU 11 historically flaky on this box — the config pins `CUDA_VISIBLE_DEVICES=0-7,10,11`.

## Quickstart

```bash
# weights
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp --local-dir /mnt/data7tb/models/DeepSeek-V4-Flash-Vision-Exp

# run (set VLLM_API_KEY for a secured API)
export VLLM_API_KEY="<your-key>"
./start-dsv4-1m.sh start            # 1M ctx, no offload
./start-dsv4-4m-offload.sh start    # 4M ctx, RAM offload

# manage
./start-dsv4-1m.sh status | logs | stop | restart
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
- `--max-num-batched-tokens 2048` (1M/4M variants) keeps Triton warmup within margin.
