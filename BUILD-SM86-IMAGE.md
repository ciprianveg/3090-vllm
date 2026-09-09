# Building the SM86 (RTX 3090) vLLM image

The DeepSeek-V4-Flash-Vision image is a stock vLLM build from the
[wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) fork
(PR #58 `pr/vision-sm80`), compiled for Ampere (SM86). The runtime patches in
[`deepseek-v4-flash-vision/patches/`](deepseek-v4-flash-vision/patches/) are
**not baked in** — they are bind-mounted at container start (see
`start-dsv4-spec.sh`). This guide covers building and publishing the base image.

## Prerequisites

- A machine with an NVIDIA GPU (build does not require the target GPU, but
  `TORCH_CUDA_ARCH_LIST` must be set for the target architecture).
- Docker with the buildx/BuildKit frontend.
- The model weights are **not** part of the image; they are mounted at runtime.

## Build

```bash
# 1. Clone the backport fork and check out PR #58
git clone https://github.com/wtdcode/vllm-backport.git
cd vllm-backport
git fetch origin pull/58/head:pr58
git checkout pr58

# 2. Build with the repo's Dockerfile, targeting Ampere (sm86)
docker build \
  -f docker/Dockerfile \
  --build-arg TORCH_CUDA_ARCH_LIST=8.6 \
  -t vllm-backport:pr58-sm86 \
  .
```

Key build facts (from `docker/Dockerfile` defaults and the deployed image):

| Setting | Value |
|---|---|
| Base CUDA | `nvidia/cuda:13.0.3` (Ubuntu 24.04) |
| Python | 3.12 |
| PyTorch | 2.11 (cu130) |
| Target arch | `TORCH_CUDA_ARCH_LIST=8.6` (Ampere / RTX 3090) |
| FlashInfer | 0.6.18 |
| Entrypoint | `vllm serve` |

## Tag & publish to ghcr

```bash
# 2. Tag for the registry
docker tag vllm-backport:pr58-sm86 ghcr.io/<your-org>/3090-vllm:dsv4-flash-vision-sm86

# 3. Log in and push
docker login ghcr.io
docker push ghcr.io/<your-org>/3090-vllm:dsv4-flash-vision-sm86
```

## Notes

- **Architecture matters.** The wheel is built for the target arch. A non-x86_64
  build (e.g. an arm64 server build) cannot run on an x86_64 RTX 3090 box and
  vice-versa — build on the matching platform.
- **Runtime patches are separate.** The model-specific fixes (spec + vision,
  tool-call grammars, vision OOM, etc.) live in
  [`deepseek-v4-flash-vision/patches/`](deepseek-v4-flash-vision/patches/) and are
  mounted by `start-dsv4-spec.sh`. Rebuilding the image does not change them.
- **No secrets in the image.** The image carries no model weights, API keys, or
  host-specific paths; those are supplied at runtime via `docker run`.
