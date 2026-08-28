# Qwen3.8-Flash-Next NVFP4 (109 GB, single-GPU vLLM)

Qwen3.8-Flash-Next is a 335 GB model. This is a 109 GB NVFP4 quantization of it
that runs through vLLM on a single 121 GB GPU (built and tested on an NVIDIA DGX
Spark, GB10/sm121).

This repository contains the quantization recipe, the vLLM patches required to
load the checkpoint, the serving configuration, and the verification scripts.
The weights themselves live on Hugging Face:
[starkweatherdigital/qwen3.8-flash-next-nvfp4](https://huggingface.co/starkweatherdigital/qwen3.8-flash-next-nvfp4).

**Start here: [docs/REPRODUCE.md](docs/REPRODUCE.md)** — the end-to-end guide,
from the official 335 GB release to a served engine with speculative decoding,
CUDA graphs, a 131K window and prefix caching all working, including the traps
that cost us time.

## What was done

Existing server-format quants don't fit a single GPU because they keep the
model's 51.2B-parameter n-gram (PLE) table at FP8 or wider (102.4 GB at BF16).
Two changes close the gap:

1. **The PLE table is quantized to NVFP4 (4-bit)** — 28.80 GB instead of
   102.4 GB — using the same E2M1 + per-16 FP8-block-scale layout as the
   experts. llama.cpp GGUFs already ship ~4-bit n-gram tables, so the quality
   precedent existed; this applies it in ModelOpt format.
2. **A small vLLM loader patch** (`patches/20-ple-4bit-loader.patch`, ~130
   lines, env-gated behind `VLLM_PLE_NVFP4=1`) teaches vLLM to load the packed
   4-bit table. vLLM has no quantized-PLE loading of its own; this is the part
   that makes the checkpoint actually usable.

Sizes for context:

| Checkpoint | Size | Engine |
|---|---|---|
| Official BF16 | 335 GB | any |
| Official FP8 | 173 GB | vLLM |
| Inferact NVFP4 | 182.8 GB | vLLM |
| RadixArk W4A4 | 135 GB | vLLM |
| Unsloth GGUF (dynamic, sub-4-bit) | 74–111 GB | llama.cpp |
| This checkpoint | 109 GB | vLLM |

109 GB is about the floor for the NVFP4 format (~4.5 bits/param over experts +
PLE). Smaller checkpoints (like the 74 GB GGUF) use sub-4-bit formats that
NVFP4 can't express, in a llama.cpp-only format.

## Checkpoint contents

| Component | Format | Size |
|---|---|---|
| Routed experts (48×512 + MTP experts) | NVFP4 | 67.95 GB |
| PLE / n-gram table (51.2B params) | NVFP4 | 28.80 GB |
| MTP layers | mixed | 1.42 GB |
| Attention, embeddings, gates, norms, vision tower | BF16 (unchanged) | 10.98 GB |
| **Total** | | **109.18 GB** |

Quantized in one step from the official BF16 release. `input_scale` values are
calibration-derived from the RadixArk release (×1.25). Verified by byte-parity
against a ModelOpt reference, a sweep over all scale tensors, and dequantization
error checks (~0.094 relative). One caveat: ~0.2 % of bytes (the tails of 3
shards) were regenerated on CPU after a download truncation; they differ from
the originals only in half-ULP rounding ties and verify identically.

## Serving

Needs a vLLM build with Flash-Next (Qwen4Exp) support plus the patches in this
repo. A prebuilt arm64 image with all six patches:

```
docker.io/jstarkg/vllm-gb10-flashnext:0.28-sm121-r6
```

(the Dockerfile here reproduces the patch application.)

```bash
VLLM_PLE_NVFP4=1 VLLM_PLE_NVFP4_MMAP=1 VLLM_PLE_NVFP4_MMAP_PREWARM=1 \
vllm serve /path/to/flashnext-nvfp4 \
  --served-model-name qwen3.8-flash-next \
  --quantization modelopt_fp4 --moe-backend marlin \
  --gpu-memory-utilization 0.78 --max-model-len 131072 \
  --max-num-batched-tokens 8192 --max-num-seqs 16 \
  --load-format runai_streamer \
  --model-loader-extra-config '{"concurrency":16,"memory_limit":4294967296}' \
  --enable-prefix-caching --mamba-cache-mode align \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16],"splitting_ops":["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_nvfp4_mmap_lookup"]}' \
  --speculative-config '{"method":"qwen3_8_flash_next_mtp","num_speculative_tokens":1}'
```

Notes:
- `--moe-backend marlin`: on sm120/121 Marlin is the working NVFP4 path.
  `10-…thread-config.patch` fixes an sm121 thread-config bug (vllm#37030
  class) that otherwise corrupts output. The FlashInfer b12x CUTLASS MoE
  backend faults (Xid 31) at JIT warmup on sm121 with current
  nvidia-cutlass-dsl releases — don't bother until FlashInfer validates it.
- `VLLM_PLE_NVFP4_MMAP=1` (`35-ple-nvfp4-mmap.patch`) serves the 28.8 GB
  4-bit PLE table from NVMe via mmap instead of keeping it resident: a token
  reads 16 rows × 90 B, so the page cache does the work. Frees ~27 GB —
  which is what makes CUDA graphs and the 131K window fit. Costs ~0 at
  decode (the table was never the bottleneck). On unified-memory machines
  this is the only offload that returns real memory; a CPU-offload process
  just moves the allocation within the same pool.
- CUDA graphs: piecewise mode with the splitting-op list shown (the mmap
  lookup op must run outside capture). With the table resident there is no
  capture headroom on a 121 GB card — the mmap patch is the enabler.
- `--load-format runai_streamer`: 674 s → 198 s model load on GB10, whose
  pageable host-to-device copies are the real bottleneck (~50× slower than
  pinned).
- `--kv-cache-dtype fp8` does not work: the model's QSA attention requires a
  BF16 main KV cache and rejects it at startup.
- **Prefix caching crashes this model's day-one tree.** Patches 40 and 41 fix
  it; both flags above are required, explicitly. Read
  [docs/PREFIX-CACHING.md](docs/PREFIX-CACHING.md) before enabling — including
  the testing methodology, because the failure mode passes uniform tests.

## Measured (DGX Spark GB10)

| Config | Decode speed |
|---|---|
| eager | 16.8 tok/s |
| eager + MTP (k=1, 80 % acceptance) | 24.6 tok/s |
| full config above (mmap PLE + piecewise graphs + MTP + prefix caching) | 24.8–25.5 tok/s |

With the full config: boot ~6.5 min (198 s weight load), cached conversation
turns drop from ~10 s to ~2–3 s of prefill, cache hit rate 56–70 % on
conversation-shaped traffic, 131,072-token context (a 46K-token prompt answers
in ~25 s). Reasoning and tool-calling verified through an OpenAI-compatible
gateway. Decode numbers are clock-sensitive — GB10's DVFS parks bandwidth-bound
decode well below max clocks, so log `clocks.sm` alongside any benchmark.

## Contents

- `docs/REPRODUCE.md` — **the end-to-end guide**: quantize, patch, serve,
  verify, plus the optional FP8-periphery calibration and a list of every
  trap that cost us time
- `docs/PREFIX-CACHING.md` — why prefix caching crashes GDN hybrids, the two
  fixes, and how to test cache changes so they can't lie to you
- `quantize/` — the quantization pipeline: `quantlib.py` (NVFP4 packing math),
  `quantdriver.py` (incremental, resumable BF16 → NVFP4 driver),
  `apply_ple_nvfp4_patch.py` (generates the PLE loader patch)
- `patches/` — six vLLM patches (apply with `patch -p1` in vLLM's
  site-packages, lexical order): sm121 Marlin thread config, the 4-bit PLE
  loader, a graph-safe PLE output buffer, PLE-from-NVMe mmap, and two
  prefix-caching fixes ported from unmerged upstream PRs (vllm#48375,
  vllm#53142)
- `serve/` — `serve.sh` (the serving command with every load-bearing flag
  explained), `gates.py` (coherence → tools → speed-with-clocks acceptance),
  `soak.py` (seeded varied-geometry soak with killer-replay), and
  `test_ple_mmap_parity.py` (offline bit-identity proof for the mmap patch)
- `calibrate/` — optional FP8-periphery path: activation-max calibration
  server, its Dockerfile, and the `input_scale` graft
- `Dockerfile` — reproduces the serving image (asserts all six patches applied)
- `upload/` — Hugging Face upload tooling + weights model card
- `LICENSE` — Qwen Community License 1.0 (inherited from the base model)

## License and credit

Weights are a derivative of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
under the Qwen Community License 1.0 (included). Patches and Dockerfile:
Apache-2.0.

Thanks to: Qwen (base model), RadixArk (calibration scales), Unsloth (the
4-bit n-gram precedent), namake-taro/vllm-custom (sm121 thread-config insight),
NVIDIA (sm121 vLLM enablement, upstream in 0.28).
