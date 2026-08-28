# vLLM image for Qwen3.8-Flash-Next NVFP4 on sm120/sm121 (GB10 / DGX Spark).
#
# Base = the day-one Flash-Next image, digest-pinned (vllm 0.1.dev20073+g8e685d198,
# torch 2.13.0+cu130, CUDA 13.0.1). Its Marlin MoE kernels already carry sm_120f
# family SASS — verified with cuobjdump — so no kernel rebuild is needed; every
# fix in this repo is a Python-level patch applied over site-packages.
FROM vllm/vllm-openai@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e

COPY patches/ /opt/patches/

# Patches apply in lexical order and each one is asserted afterwards: a build
# that would ship a half-applied patch fails here instead of at inference time.
RUN set -e; cd /usr/local/lib/python3.12/dist-packages; \
    for p in /opt/patches/*.patch; do \
      echo "applying $p"; patch -p1 --forward < "$p"; \
    done; \
    python3 - <<'PY'
import ast, py_compile

PKG = "/usr/local/lib/python3.12/dist-packages/vllm"
CHECKS = [
    # file, [required markers]
    (f"{PKG}/model_executor/layers/fused_moe/experts/marlin_moe.py",
     ["_sm121_w1_thread_k"]),                                   # 10: sm121 thread config
    (f"{PKG}/models/qwen3_8_flash_next/nvidia/ple_layer.py",
     ["BCN76_PLE_NVFP4", "Qwen4ExpPLENvFp4EmbeddingMethod",     # 20: 4-bit PLE loader
      "BCN79_PLE_GRAPH",                                        # 30: graph-safe output buffer
      "BCN82_PLE_NVFP4_MMAP"]),                                 # 35: mmap hook
    (f"{PKG}/models/qwen3_8_flash_next/nvidia/ple_nvfp4_mmap.py",
     ["MmapNvFp4PleTable", "ple_nvfp4_mmap_lookup"]),           # 35: mmap module
    (f"{PKG}/v1/core/single_type_kv_cache_manager.py",
     ["BCN81_EAGLE_DROP"]),                                     # 40: eagle-drop fix
    (f"{PKG}/v1/worker/gpu/model_states/mamba_hybrid.py",
     ["BCN81_STATE_SEED"]),                                     # 41: state-seed fix
]

for path, markers in CHECKS:
    src = open(path).read()
    ast.parse(src)
    py_compile.compile(path, doraise=True)
    for m in markers:
        assert m in src, f"{m} missing from {path}"
    print(f"ok: {path.split('/')[-1]} ({len(markers)} marker(s))")

marlin = open(CHECKS[0][0]).read()
assert marlin.count("thread_k=_sm121_") == 2, "sm121 thread-config not applied twice"
ple = open(CHECKS[1][0]).read()
assert ple.count("output.copy_(vals)") == 1, "graph output-buffer path not applied"
assert ple.count("BCN82_PLE_NVFP4_MMAP") == 1, "mmap hook applied more than once"
print("patch verify OK: all six patches applied and importable")
PY

ENV VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    CUDA_DEVICE_MAX_CONNECTIONS=1

# Runtime knobs are set per-deploy, not baked — see serve/serve.sh:
#   VLLM_PLE_NVFP4=1              select the 4-bit PLE loader (required)
#   VLLM_PLE_NVFP4_MMAP=1         serve the PLE table from NVMe (frees ~27 GB)
#   VLLM_PLE_NVFP4_MMAP_PREWARM=1 stream it once at boot to warm the page cache
