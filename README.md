# 3090-vllm — DeepSeek-V4-Flash-Vision

**DeepSeek-V4-Flash-Vision-Exp** (285B MoE, FP4 experts + FP8 attention, 185 GB,
text + vision) served by vLLM on consumer **NVIDIA RTX 3090 (Ampere, SM86)** —
with **DSpark speculative decoding**, **working vision + tool calls**, and
**1M / 4M context**.

This repo is focused on this single model: buildable image, deploy scripts, and
the runtime patches that make it actually run on the hardware.

## Highlights

| Metric | Value |
|---|---:|
| Context | **1M (no offload) / 4M (RAM offload)** |
| Decode, DSpark (TP2×PP5) | **60+ tok/s** (240 W cap; ~70+ uncapped) |
| Decode, DSpark (TP4×PP3, 12 GPUs) | **120+ tok/s** |
| Prefill (long-context, cold) | ~3,500 tok/s |
| Vision + spec + tool calls | all working together |

Open [deepseek-v4-flash-vision/](deepseek-v4-flash-vision/) for the full guide,
patches, and tuning notes.

## Hardware target

- **GPU**: 12x NVIDIA RTX 3090 24 GB (Ampere, SM86) — configs use 10 (TP=2, PP=5)
- **RAM**: 251 GB (supports 36 GB of pinned KV offload)
- **Image**: `ghcr.io/ciprianveg/3090-vllm:dsv4-flash-vision-sm86` — built from
  [wtdcode/vllm-backport PR #58](https://github.com/wtdcode/vllm-backport/pull/58)
  with `TORCH_CUDA_ARCH_LIST=8.6`
- **Not just the upstream image**: the working setup adds our own runtime patches
  and launch-config mods on top (see [Mods & fixes](deepseek-v4-flash-vision/README.md#mods--fixes)).
