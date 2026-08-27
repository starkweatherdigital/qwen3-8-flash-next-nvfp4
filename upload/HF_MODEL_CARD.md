---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- nvfp4
- modelopt
- vllm
- quantized
- dgx-spark
- single-gpu
---

# Qwen3.8-Flash-Next NVFP4 (109 GB, single-GPU vLLM)

A 109 GB NVFP4 quantization of the 335 GB Qwen/Qwen3.8-Flash-Next that runs
through vLLM on a single 121 GB GPU. The 51.2B-parameter n-gram (PLE) table is
quantized to 4-bit; a small vLLM loader patch (linked below) makes it loadable.

Requires the loader patches + serving instructions here:
**https://github.com/starkweatherdigital/qwen3-8-flash-next-nvfp4**
(prebuilt image: `docker.io/jstarkg/vllm-gb10-flashnext:0.28-sm121-r3`, serve with
`VLLM_PLE_NVFP4=1`).

Measured on DGX Spark GB10: 24.6 tok/s single-stream with MTP speculative decoding
(80% acceptance), 16.8 tok/s without. Full recipe, verification, provenance notes,
and benchmarks in the GitHub repo.

License: Qwen Community License 1.0 (inherited; see LICENSE).
