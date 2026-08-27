# BCN-78: vLLM 0.28 flash-next arm64 + SM121 (GB10) marlin fixes.
# Base = the day-one Flash-Next image, digest-pinned (vllm 0.1.dev20073+g8e685d198,
# torch 2.13.0+cu130, CUDA 13.0.1, marlin MoE kernels already carry sm_120f
# family SASS - verified via cuobjdump; NO kernel rebuild needed).
FROM vllm/vllm-openai@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e

COPY patches/ /opt/bcn78-patches/
RUN set -e; cd /usr/local/lib/python3.12/dist-packages; \
    for p in /opt/bcn78-patches/*.patch; do \
      echo "applying $p"; patch -p1 --forward < "$p"; \
    done; \
    python3 - <<'PY'
import ast, py_compile
f = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py"
src = open(f).read()
ast.parse(src)
py_compile.compile(f, doraise=True)
assert "_sm121_w1_thread_k" in src and src.count("thread_k=_sm121_") == 2
f2 = "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"
src2 = open(f2).read()
ast.parse(src2)
py_compile.compile(f2, doraise=True)
assert "BCN76_PLE_NVFP4" in src2 and "Qwen4ExpPLENvFp4EmbeddingMethod" in src2
assert "BCN79_PLE_GRAPH" in src2 and src2.count("output.copy_(vals)") == 1
print("patch verify OK (marlin + ple + ple-graph)")
PY

# NGC 26.05 serve-env parity (harmless outside GB10; documented in BCN-78):
ENV VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    CUDA_DEVICE_MAX_CONNECTIONS=1

# Trial-window knobs intentionally NOT baked (set per-deploy):
#   VLLM_ATTENTION_BACKEND (FLASHINFER default vs TRITON_ATTN A/B)
#   --moe-backend flashinfer_cutlass_afp8 (BCN-76 untested lead)
