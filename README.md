# Qwen3.8-Flash-Next NVFP4 (109 GB, single-GPU vLLM)

Qwen3.8-Flash-Next is a 335 GB model. This is a 109 GB NVFP4 quantization of it
that runs through vLLM on a single 121 GB GPU (built and tested on an NVIDIA DGX
Spark, GB10/sm121).

This repository contains the quantization recipe, the vLLM patches required to
load the checkpoint, and the serving configuration. The weights themselves live
on Hugging Face:
[starkweatherdigital/qwen3.8-flash-next-nvfp4](https://huggingface.co/starkweatherdigital/qwen3.8-flash-next-nvfp4).

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
repo. A prebuilt arm64 image:

```
docker.io/jstarkg/vllm-gb10-flashnext:0.28-sm121-r3
```

(base image + the three patches; the Dockerfile here reproduces it.)

```bash
VLLM_PLE_NVFP4=1 vllm serve /path/to/flashnext-nvfp4 \
  --served-model-name qwen3.8-flash-next \
  --quantization modelopt_fp4 --moe-backend marlin --enforce-eager \
  --gpu-memory-utilization 0.92 --max-model-len 32768 \
  --max-num-batched-tokens 8192 --max-num-seqs 16 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"qwen3_8_flash_next_mtp","num_speculative_tokens":1}'
```

Notes:
- `--moe-backend marlin`: on sm120/121 Marlin is the working NVFP4 path.
  `10-…thread-config.patch` fixes an sm121 thread-config bug (vllm#37030
  class) that otherwise corrupts output.
- `--enforce-eager`: CUDA graphs need more free memory than a 109 GB
  checkpoint leaves on a 121 GB card.
- `--max-model-len`: 32768 is what fits alongside the weights on 121 GB.
  Speculative decoding uses the MTP weights already in the checkpoint.

## Measured (DGX Spark GB10)

| Config | Decode speed |
|---|---|
| eager | 16.8 tok/s |
| eager + MTP (k=1, 80 % acceptance) | 24.6 tok/s |

Boot is ~13 min (109 GB from local NVMe). Reasoning and tool-calling verified
through an OpenAI-compatible gateway.

## Contents

- `patches/` — three vLLM patches (apply with `patch -p1` in vLLM's
  site-packages, lexical order)
- `Dockerfile` — reproduces the serving image
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
