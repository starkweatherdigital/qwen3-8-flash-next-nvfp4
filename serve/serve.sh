#!/usr/bin/env bash
# Serve the NVFP4 Flash-Next checkpoint. Every flag here is load-bearing on a
# 121 GB GB10 (DGX Spark); see README.md and docs/REPRODUCE.md for why.
set -euo pipefail

CKPT="${CKPT:?set CKPT to the NVFP4 checkpoint directory}"
PORT="${PORT:-8000}"
CTX="${CTX:-131072}"            # 262144 is the model's native max
SEQS="${SEQS:-16}"
GPU_MEM="${GPU_MEM:-0.78}"      # leave room for OS + the mmap'd table's page cache
MTP="${MTP:-1}"                 # 0 disables speculative decoding
CACHE="${CACHE:-1}"             # 0 disables prefix caching

SPLIT_OPS='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_nvfp4_mmap_lookup"]'

args=(
  --served-model-name qwen3.8-flash-next
  --host 0.0.0.0 --port "$PORT"
  --quantization modelopt_fp4
  --moe-backend marlin                       # the working NVFP4 path on sm120/121
  --gpu-memory-utilization "$GPU_MEM"
  --max-model-len "$CTX"
  --max-num-batched-tokens 8192
  --max-num-seqs "$SEQS"
  --load-format runai_streamer               # 674s -> 198s load on GB10
  --model-loader-extra-config '{"concurrency":16,"memory_limit":4294967296}'
  --no-enable-flashinfer-autotune
  --reasoning-parser qwen3
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --compilation-config "{\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[1,2,4,8,16],\"splitting_ops\":$SPLIT_OPS}"
)

# Prefix caching REQUIRES patches 40 + 41 and both flags, explicitly.
# Read docs/PREFIX-CACHING.md before turning this on.
if [ "$CACHE" = "1" ]; then
  args+=( --enable-prefix-caching --mamba-cache-mode align )
else
  args+=( --no-enable-prefix-caching )
fi

if [ "$MTP" != "0" ]; then
  args+=( --speculative-config "{\"method\":\"qwen3_8_flash_next_mtp\",\"num_speculative_tokens\":$MTP}" )
fi

# VLLM_PLE_NVFP4       selects the 4-bit PLE loader (patch 20)
# VLLM_PLE_NVFP4_MMAP  serves that table from NVMe instead of resident (patch 35)
# PREWARM              streams the table once at boot to fill the page cache
export VLLM_PLE_NVFP4=1
export VLLM_PLE_NVFP4_MMAP="${MMAP:-1}"
export VLLM_PLE_NVFP4_MMAP_PREWARM="${PREWARM:-1}"
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

exec vllm serve "$CKPT" "${args[@]}"
