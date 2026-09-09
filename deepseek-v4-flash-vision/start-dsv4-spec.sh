#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-V4-Flash-Vision-Exp — 12x RTX 3090, TP=2 PP=5 (10 GPUs)
# Variant: DSpark speculative decoding + vision, aligned to the proven 2x5
# baseline config. This is the primary working configuration (see
# SPEC_VISION_FIX.md): spec + image input both work, DSpark k=3.
#
# Requires: the vllm-backport PR#58 sm86 image (ghcr.io/ciprianveg/3090-vllm:dsv4-flash-vision-sm86)
# and the model weights locally (see README).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/mnt/data7tb/models/DeepSeek-V4-Flash-Vision-Exp}"
IMAGE="${IMAGE:-ghcr.io/ciprianveg/3090-vllm:dsv4-flash-vision-sm86}"
CONTAINER_NAME="vllm_dsv4_spec"
PORT="${PORT:-8000}"
# Set VLLM_API_KEY in your environment (or edit this default) to secure the API.
API_KEY="${VLLM_API_KEY:-}"
# GPUs 8/9 are reserved for another model on this box; use the rest.
GPUS="${GPUS:-0,1,2,3,4,5,6,7,10,11}"

start() {
    echo "Starting $CONTAINER_NAME ..."
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    # Clean stale KV-offload mmaps from crashed runs (they eat RAM/shm)
    rm -f /dev/shm/vllm_offload_*.mmap 2>/dev/null || true
    docker run -d \
      --name "$CONTAINER_NAME" \
      --gpus all \
      --network host \
      --ipc host \
      --shm-size=16g \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      -v "${MODEL_PATH}:/models/DeepSeek-V4-Flash-Vision-Exp:ro" \
      -v "${SCRIPT_DIR}/patches/model.py:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py:ro" \
      -v "${SCRIPT_DIR}/patches/model_runner.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py:ro" \
      -v "${SCRIPT_DIR}/patches/scheduler.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py:ro" \
      -v "${SCRIPT_DIR}/patches/dspark_speculator.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:ro" \
      -v "${SCRIPT_DIR}/patches/sched_output.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/output.py:ro" \
      -v "${SCRIPT_DIR}/patches/vision.py:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/common/vision.py:ro" \
      -v "${SCRIPT_DIR}/patches/flashinfer_sparse.py:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py:ro" \
      -v "${SCRIPT_DIR}/patches/block_table.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/block_table.py:ro" \
      -e HF_HOME=/tmp/hf_cache \
      -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
      -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
      -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64 \
      -e CUDA_VISIBLE_DEVICES="$GPUS" \
      -e NCCL_DEBUG=WARN \
      -e NCCL_ASYNC_ERROR_HANDLING=1 \
      -e NCCL_ALGO=Ring \
      -e NCCL_PROTO=Simple \
      -e NCCL_P2P_DISABLE=1 \
      -e VLLM_PP_LAYER_PARTITION=10,9,9,9,6 \
      -e VLLM_USE_V2_MODEL_RUNNER=1 \
      -e PYTHONUNBUFFERED=1 \
      -e VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS="trtllm_fp4_block_scale_moe,flashinfer::trtllm_fp4_block_scale_moe" \
      "$IMAGE" \
      --model /models/DeepSeek-V4-Flash-Vision-Exp \
        --served-model-name dsv4-flash-vision \
        --tensor-parallel-size 2 \
        --pipeline-parallel-size 5 \
        --gpu-memory-utilization 0.95 \
        --max-model-len 500000 \
        --kv-cache-memory 2147483648 \
        --max-num-batched-tokens 2048 \
        --limit-mm-per-prompt '{"video":0}' \
        --max-num-seqs 4 \
        --speculative-config '{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic","enable_adaptive_verification":false}' \
        --kv-cache-dtype fp8_ds_mla \
        --disable-custom-all-reduce \
        --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4],"max_cudagraph_capture_size":4}' \
        --block-size 256 \
        --trust-remote-code \
        --tool-call-parser deepseek_v4 \
        --reasoning-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --host 0.0.0.0 \
        --port "$PORT" \
        --api-key "$API_KEY"
    echo "Container started. Tail logs with: $0 logs"
}

stop() {
    echo "Stopping $CONTAINER_NAME ..."
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    rm -f /dev/shm/vllm_offload_*.mmap 2>/dev/null || true
    echo "Stopped."
}

logs() {
    docker logs -f "$CONTAINER_NAME" "$@" 2>&1 | sed -u -E 's/(Avg generation throughput:[[:space:]]*)([0-9.]+[[:space:]]*tokens\/s)/\1\x1b[1;32m\2\x1b[0m/g'
}

status() {
    docker ps -a --format "{{.Names}}\t{{.Status}}" | grep "$CONTAINER_NAME" || echo "no container"
    curl -s -o /dev/null -w "API: HTTP %{http_code}\n" --connect-timeout 3 \
      "http://localhost:${PORT}/v1/models" -H "Authorization: Bearer ${API_KEY}" || true
}

case "${1:-}" in
    start)  start ;;
    stop)   stop ;;
    logs)   logs ;;
    status) status ;;
    restart) stop; start ;;
    *) echo "Usage: $0 {start|stop|logs|status|restart}"; exit 1 ;;
esac
