# DeepSeek-V4-Flash-Vision-Exp on 12x RTX 3090

**DeepSeek-V4-Flash-Vision-Exp** (285B MoE, FP4 experts + FP8 attention, 185 GB, text + vision)
served by vLLM on consumer Ampere hardware — with **speculative decoding**, **1M-context capability**,
and **working tool calls**.

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
| Vision (image input) | **Only without DSpark** — see [Vision](#vision) below |
| Tool calls (OpenAI-compatible) | **Working** — see [the fixes](#the-fixes) below |

## Vision

Image inputs work **only on configurations without speculative decoding** (verified:
correct object/shape/color descriptions). With DSpark enabled, multimodal +
speculative drafting is broken **upstream in vLLM itself** (open issues
[#38551](https://github.com/vllm-project/vllm/issues/38551),
[#43832](https://github.com/vllm-project/vllm/issues/43832)):
image requests crash the engine with async scheduling, and silently corrupt the
speculation state without it. The start scripts therefore ship with
`--limit-mm-per-prompt '{"image":0,"video":0}'` — image requests get a clean
HTTP 400 instead of killing the server.

For vision workloads, remove the `--speculative-config` argument and set
`{"image":1,"video":0}`; decode speed drops from ~58-60 to ~25 tok/s but image
input works flawlessly. This repo's `patches/dspark_speculator.py` additionally
implements upstream PR #33437 semantics for DSpark (text-only drafting), which
prevents the engine-crash class but does not yet make vision + DSpark stable.

## Two start scripts

Both scripts share the same image, patches, and engine tuning; they differ only in
context/offload placement. They are standalone — `./script start` is all you need
(modify the env defaults at the top if your paths differ).

### 1. `start-dsv4-1m.sh` — 1M context, no CPU offload

```bash
./start-dsv4-1m.sh start
```

- `--max-model-len 1000000`, KV pool **1,010,835 tokens** (2 GiB explicit cap)
- A single request up to 1M tokens fits entirely in GPU memory
- Multiple smaller requests share the same pool (e.g. 2x 450K)

### 2. `start-dsv4-500k-offload.sh` — 500K context x 2 users, 36 GB RAM offload

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
| RAM | 251 GB (36 GB pinned for KV offload in variant 2) |
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

## The fixes

Three runtime patches (bind-mounted, in [`patches/`](patches/)) are **required**:

| Patch | What it does |
|---|---|
| `patches/model.py` | DSpark drafter aliases the target embedding table on the last pipeline rank — this builds `embed_tokens` there (PR #58 companion fix) |
| `patches/structured_output_init.py` | Backport of [vLLM PR #52452](https://github.com/vllm-project/vllm/pull/52452): validates accepted speculative blocks against the grammar bitmask before committing them to request history |
| `patches/scheduler.py` | **Transition repair** (original fix): a request that finishes chunked prefill and enters decode with fresh draft tokens was scheduled `len(drafts)` query rows — its bonus row was never granted — tripping `assert num_scheduled_tokens >= num_logits` in the model runner whenever the grammar had rejected drafts. The patch grants the missing bonus row before `allocate_slots`, keeping KV accounting consistent |

Without the last two, DSpark + tool-call grammars is broken upstream
([#49002](https://github.com/vllm-project/vllm/issues/49002),
[#49210](https://github.com/vllm-project/vllm/issues/49210)): requests with tool
schemas assert or livelock the engine core. With them, spec decode + tool calls +
500K context run together, stable through hours of agent traffic.

## Performance notes

- **3,500 tok/s prefill** is the long-context cold number (sparse attention skips
  most of the haystack on filler text). Real-world chunked prefill of a 425K
  context completes in ~2 minutes; prefix-cache hits make re-runs near-instant.
- **58–60 tok/s decode** is the warm single-stream number with DSpark acceptance
  in the 80%+ range; first request after boot runs ~45 tok/s until caches warm.
- Decode throughput is memory-bandwidth-bound: aggregate stays ~60 tok/s whether
  1 or 4 users generate (each user gets 1/n of it). Two users is the sweet spot.
- GPU 11 historically flaky on this box — the config pins `CUDA_VISIBLE_DEVICES=0-9`.

## Quickstart

```bash
# weights
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp --local-dir /mnt/data7tb/models/DeepSeek-V4-Flash-Vision-Exp

# run (pick a variant)
./start-dsv4-1m.sh start          # 1M ctx, no offload
./start-dsv4-500k-offload.sh start  # 500K ctx x2, 36 GB offload

# manage
./start-dsv4-1m.sh status | logs | stop | restart
```

Boot takes ~9 minutes. Watch for `cudaHostRegister ... pinned`, `GPU KV cache size`,
and `Application startup complete` in the logs.

API is OpenAI-compatible:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer 1791128410062" -H "Content-Type: application/json" \
  -d '{"model":"dsv4-flash-vision","messages":[{"role":"user","content":"hello"}],"max_tokens":64}'
```

## Tuning knobs that matter on this hardware

- `--kv-cache-memory 2147483648` — explicit 2 GiB cap. Without it the default KV
  pool sizing eats the activation margin and warmup OOMs on 24 GB cards.
- `VLLM_PP_LAYER_PARTITION=10,9,9,9,6` — not layer-balanced but **weight**-balanced:
  the last rank carries the DSpark draft (~5.5 GB) + embed + lm_head.
- `--num-speculative-tokens` must be a multiple of the draft's `n_predict=3` (k=3 or 6).
- `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64` caps indexer logits buffers.
- `--max-num-batched-tokens 1024` keeps Triton warmup within margin; higher values
  OOM during `compress_norm_rope_store` warmup on 24 GB cards.
