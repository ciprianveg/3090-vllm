#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-V4-Flash-Vision-Exp — 12x RTX 3090, TP=2 PP=5 (10 GPUs)
# Variant: 500,000-token context with 36 GB CPU KV offload.
#   - 2x 500K requests fully GPU-resident concurrently (2.02x pool)
#   - ~4.16M additional tokens parked in pinned RAM (prefix reuse, extra users)
#   - proven stable with 3x 470K concurrent requests (1.41M live KV)
#
# Requires: the vllm-backport PR#58 sm86 image (ghcr.io/ciprianveg/3090-vllm:dsv4-flash-vision-sm86)
# and the model weights locally (see README).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/mnt/data7tb/models/DeepSeek-V4-Flash-Vision-Exp}"
IMAGE="${IMAGE:-ghcr.io/ciprianveg/3090-vllm:dsv4-flash-vision-sm86}"
CONTAINER_NAME="vllm_dsv4_500k_offload"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-1791128410062}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7,8,9}"
KV_OFFLOAD_GB="${KV_OFFLOAD_GB:-36}"   # pinned-RAM ceiling is ~4.2 GB/rank; keep <= 38

start() {
    echo "Starting $CONTAINER_NAME ..."
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    # Stale offload mmaps from crashed runs trap RAM — always clean first.
    sudo rm -f /dev/shm/vllm_offload_*.mmap 2>/dev/null \
      || rm -f /dev/shm/vllm_offload_*.mmap 2>/dev/null || true
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
      -v "${SCRIPT_DIR}/patches/scheduler.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py:ro" \
      -v "${SCRIPT_DIR}/patches/structured_output_init.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/structured_output/__init__.py:ro" \
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
        --gpu-memory-utilization 0.955 \
        --max-model-len 500000 --kv-cache-memory 2147483648 \
        --kv-offloading-size "$KV_OFFLOAD_GB" \
        --max-num-batched-tokens 1024 --limit-mm-per-prompt '{"image":1,"video":0}' \
        --max-num-seqs 4 \
        --speculative-config '{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic","enable_adaptive_verification":false}' \
        --kv-cache-dtype fp8_ds_mla \
        --disable-custom-all-reduce \
        --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,12,16],"max_cudagraph_capture_size":16}' \
        --block-size 256 \
        --trust-remote-code \
        --tool-call-parser deepseek_v4 \
        --reasoning-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --host 0.0.0.0 \
        --port "$PORT" \
        --api-key "$API_KEY"
    echo "Container started. Tail logs with: $0 logs  (boot takes ~9 minutes)"
}

stop() {
    echo "Stopping $CONTAINER_NAME ..."
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    sudo rm -f /dev/shm/vllm_offload_*.mmap 2>/dev/null \
      || rm -f /dev/shm/vllm_offload_*.mmap 2>/dev/null || true
    echo "Stopped."
}

logs()      { docker logs -f "$CONTAINER_NAME"; }
status()    {
    docker ps -a --format "{{.Names}}\t{{.Status}}" | grep "$CONTAINER_NAME" || echo "no container"
    curl -s -o /dev/null -w "API: HTTP %{http_code}\n" --connect-timeout 3 \
      "http://localhost:${PORT}/v1/models" -H "Authorization: Bearer ${API_KEY}" || true
}
restart()   { stop; start; }

case "${1:-}" in
    start)   start ;;
    stop)    stop ;;
    logs)    logs ;;
    status)  status ;;
    restart) restart ;;
    *) echo "Usage: $0 {start|stop|logs|status|restart}"
       echo "Env: MODEL_PATH, IMAGE, PORT, API_KEY, GPUS, KV_OFFLOAD_GB (<= 38)"
       exit 1 ;;
esac
