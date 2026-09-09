# 3090-vllm

vLLM inference solutions for consumer **NVIDIA RTX 3090 (Ampere, SM86)** hardware.

A per-machine vLLM stack with buildable images, deploy scripts, and the
runtime patches that make frontier-class models actually run on the hardware.

## Models index

| Model | Subtree | Highlights |
|-------|---------|------------|
| **DeepSeek-V4-Flash-Vision-Exp** (285B MoE, vision) | [deepseek-v4-flash-vision](deepseek-v4-flash-vision/) | **500K context (spec+vision)** · **1M available for 2 users** (`start-dsv4-1m.sh`) · DSpark speculative decoding 58–60 tok/s (TP2×PP5) / **120+ tok/s (TP4×PP3, 12 GPUs)** · **vision + spec working** (~40–44 tok/s) · ~3,500 tok/s prefill · tool calls fixed and working |

## Hardware target

- **GPU**: 12x NVIDIA RTX 3090 24 GB (Ampere, SM86) — configs use 10 (TP=2, PP=5)
- **RAM**: 251 GB (supports 36 GB of pinned KV offload)
- **Image**: `ghcr.io/ciprianveg/3090-vllm:dsv4-flash-vision-sm86` — built from
  [wtdcode/vllm-backport PR #58](https://github.com/wtdcode/vllm-backport/pull/58)
  with `TORCH_CUDA_ARCH_LIST=8.6`

## Performance summary (DeepSeek-V4-Flash-Vision-Exp)

| Workload | Throughput |
|---|---:|
| Long-context prefill (cold) | ~3,500 tok/s |
| Decode with DSpark speculation | 58–60 tok/s |
| Decode, DSpark (TP4×PP3, 12 GPUs) | 120+ tok/s |
| Decode without speculation | ~25 tok/s |
| Vision decode (image input, with DSpark) | ~40–44 tok/s |
| Tool calls + 500K context + speculation | all simultaneously, stable |

Open a model subtree for the full guide, benchmarks, and tuning notes.
