# Reproducing this end to end

How to take the official 335 GB Qwen3.8-Flash-Next release and end up with a
109 GB NVFP4 checkpoint serving through vLLM on a single 121 GB GPU, with
speculative decoding, CUDA graphs, a 131K context window and prefix caching all
working at once.

Everything here was done on a DGX Spark (GB10, sm121, 121 GB unified memory).
Most of it applies to any single-GPU NVFP4 Flash-Next deployment; the parts that
are genuinely GB10-specific are called out.

**Time and cost:** roughly 1 h of rented GPU time for the quantization
(~$3–12 depending on card), plus one image build and a handful of ~7-minute
engine boots. Everything after quantization runs on the target machine.

---

## 0. What you need

- A GPU with ~121 GB of memory to serve on. The checkpoint is 109 GB and the
  runtime footprint after the mmap patch is ~76 GB resident.
- A rented CUDA GPU for the quantization step (any card that fits ~110 GB of
  intermediate work comfortably; an A6000-class card is enough because the work
  is per-tensor and streamed). ~1 h.
- Fast local storage for the checkpoint. The PLE table is served from it at
  runtime, so NVMe, not spinning disk or a network mount.
- Disk for both the 335 GB source and the 109 GB output during conversion.

---

## 1. Quantize (`quantize/`)

The pipeline converts the official BF16 release in a single pass — no lossy
intermediate formats — into three populations:

| Population | Format | Size |
|---|---|---|
| Experts (48 layers × 512 + MTP) | NVFP4 (E2M1 + per-16 FP8 block scales + per-tensor global) | 67.95 GB |
| PLE n-gram table (51.2B params) | NVFP4, same layout, one global scale | 28.80 GB |
| MTP head | NVFP4 | 1.42 GB |
| Everything else (attention, GDN, shared expert, norms, embeddings) | BF16, byte-identical to source | 10.98 GB |

The PLE table is the part nobody else quantizes — every other published quant
keeps it at FP8 or wider, which is exactly why they don't fit on one GPU.
llama.cpp GGUFs already ship ~4-bit n-gram tables, so the quality precedent
exists; this applies it in ModelOpt format and adds the loader vLLM lacks
(step 2).

```bash
# On the rented GPU, with the official weights downloading or downloaded:
export QUANT_HF_CACHE=/work/hf/hub/models--Qwen--Qwen3.8-Flash-Next
export QUANT_OUT=/work/out/ckpt
export QUANT_MANIFEST=/work/out/manifest.json

python3 quantize/quantdriver.py
```

The driver is **incremental and resumable**: it processes input shards as a
snapshot download lands them, and the manifest tracks progress, so killing it
and re-running is safe. `quantize/quantlib.py` holds the packing math
(`nvfp4_quantize`) if you want to reuse the format elsewhere.

**Verify before you trust it.** The pipeline emits a report; check per-tensor
cosine similarity and the relative-error distribution (uniform is good,
outliers are not — for NVFP4 4-bit, dequantization relative error lands around
0.094). Then reconcile every output byte before deleting anything, which brings
us to the trap:

> **Trap: resumable downloads truncate silently.** A killed-and-resumed
> transfer produces files with the right names and plausible sizes. A
> count-and-size check passes; only full byte reconciliation catches it. We lost
> the GPU holding the only copy of three shards while the mirror's audit was
> still running, and had to regenerate their tails on CPU. Sequence it the other
> way: audit completes, *then* release the machine. And note that CPU cannot
> byte-reproduce GPU float math — value-dependent rounding ties differ by an
> ULP — so if byte-parity to a GPU artifact matters, regenerate on a GPU.

---

## 2. Build the vLLM image (`patches/`, `Dockerfile`)

Six patches, applied over vLLM's `site-packages` in lexical order. All are
Python-level; no kernel rebuild.

| Patch | What it fixes |
|---|---|
| `10-sm121-marlin-thread-config` | sm121 Marlin picks a racy thread config for large N and silently corrupts output (vllm#37030 class). Forces a safe config. |
| `20-ple-4bit-loader` | Teaches vLLM to load the packed 4-bit PLE table. Without this the checkpoint from step 1 simply won't load. |
| `30-ple-graph-output-buffer` | Makes the 4-bit PLE path honor the caller's persistent output buffer, so it's CUDA-graph-capture-safe. |
| `35-ple-nvfp4-mmap` | Serves the PLE table from NVMe via mmap instead of keeping it resident. Frees ~27 GB. |
| `40-mamba-eagle-drop` | Prefix-cache crash with speculative decoding (ported from unmerged vllm#48375). |
| `41-mamba-state-seed` | Prefix-cache crash on resume, wrong block-size divisor (from vllm#53142). |

```bash
docker build -t flashnext-vllm:local .
```

The build asserts each patch landed — markers present, files still parse and
compile. A half-applied patch fails the build instead of surfacing as a strange
inference bug later.

### Why the mmap patch matters more than it looks

A token's PLE lookup touches 16 rows of ~90 bytes. There is no reason for 28.8
GB to sit resident for that, and on a unified-memory machine the freed memory is
what makes everything else fit: CUDA graph capture and a 131K context window
both failed with the table resident and both fit afterwards.

> **GB10-specific, and counter-intuitive:** vLLM's built-in
> `VLLM_PLE_CPU_OFFLOAD` does *not* help here. It moves the table to a separate
> process's host memory — which on unified memory is the *same pool* — so
> nothing is freed. Only file-backed mmap (evictable page cache) returns real
> memory. It also depends on cross-process CUDA IPC, which is not confirmed
> working on GB10.

Before deploying the mmap patch, prove it byte-for-byte offline —
`serve/test_ple_mmap_parity.py` gathers rows through the mmap path and compares
against the resident dequantization on the real checkpoint:

```bash
NVFP4_CHECKPOINT=/path/to/ckpt python3 serve/test_ple_mmap_parity.py
```

It must report bit-identical outputs and byte-identical gathers across random
ids, shard boundaries and duplicate indices. That test is why this shipped in a
single boot.

---

## 3. Serve (`serve/serve.sh`)

```bash
CKPT=/path/to/flashnext-nvfp4 ./serve/serve.sh
```

Knobs: `CTX` (default 131072), `SEQS`, `GPU_MEM`, `MTP` (0 disables
speculation), `CACHE` (0 disables prefix caching), `MMAP`, `PREWARM`.

The flags that are load-bearing, and why:

- **`--moe-backend marlin`** — on sm120/121 this is the working NVFP4 path.
  FlashInfer's b12x CUTLASS MoE backend faults with an Xid 31 MMU error during
  JIT warmup on sm121 with current `nvidia-cutlass-dsl` releases; the one
  published GB10 fleet comparison also measured Marlin *faster*. Don't spend a
  day on it like we did.
- **`--gpu-memory-utilization 0.78`** — GB10 reports the whole unified pool
  (including reclaimable page cache) as free, so vLLM's own memory probe
  over-estimates what's available (vllm#35313). Community GB10 recipes land at
  0.78–0.85; going higher risks evicting the mmap'd table or pushing the box
  toward swap, where throughput collapses.
- **`--load-format runai_streamer`** — 674 s → 198 s model load. GB10's
  pageable host-to-device copies are ~50× slower than pinned, and the default
  loader uses the slow path.
- **CUDA graphs, piecewise, with an explicit splitting-op list** — the mmap
  lookup op must run *outside* capture (it's CPU work plus a pageable copy).
  Note that setting `splitting_ops` *replaces* the defaults, so the list has to
  carry the whole attention/GDN/mamba inventory, not just the custom op.
- **`--kv-cache-dtype fp8` does not work.** The model's QSA attention rejects it
  at startup: "QSA requires a BF16 main KV cache". A same-family precedent from
  a model without QSA does not transfer.
- **Prefix caching needs both `--enable-prefix-caching` and
  `--mamba-cache-mode align`**, explicitly, and patches 40 + 41. Read
  [PREFIX-CACHING.md](PREFIX-CACHING.md) first — the default configuration is a
  combination nobody supports, and it crashes on the second turn of every
  conversation.

---

## 4. Verify (`serve/gates.py`, `serve/soak.py`)

Correctness first, always. On this stack a quantization or kernel bug shows up
as empty or garbled content with normal-looking token counts, so a speed number
from an engine that hasn't passed coherence means nothing.

```bash
python3 serve/gates.py --url http://localhost:8000/v1
```

Coherence across varied prompts, a real tool call, single-stream speed **with
GPU clocks logged**, speculative-decode acceptance, cache hit rate.

> **Trap: benchmarks without clock logging are void.** A stuck clock lock held
> our GPU at 1787 MHz for an unknown span and invalidated an entire day of
> measurements. GB10's DVFS also parks bandwidth-bound decode well below max
> clocks on its own. Log `clocks.sm` alongside every number.

Then the soak — mandatory for any prefix-cache or kernel change:

```bash
python3 serve/soak.py --url http://localhost:8000/v1 --rounds 8
```

Eight rounds, each a *different* prompt geometry, multi-turn, every turn
answer-checked. If a round fails it prints the exact command to replay that
seed:

```bash
python3 serve/soak.py --url ... --replay 7919
```

> **The most expensive lesson in this repository:** mamba/GDN prefix-cache bugs
> are geometry-dependent. Uniform test prompts pass while a different mix of
> vocabulary and lengths crashes the engine in a single conversation. We shipped
> two configurations that passed acceptance and died on the first varied prompt.
> Seed the prompts, vary the geometry, check the answers — this bug family
> corrupts output silently before it crashes — and after any fix, replay the
> exact input that broke it. A byte-identical replay of a known killer is the
> sharpest verification signal available.

---

## 5. Optional: FP8 periphery (`calibrate/`)

The ~11 GB of BF16 weights that every token reads (attention, GDN and shared
expert projections) can go to FP8, halving that per-token traffic. On a
bandwidth-bound chip that is a real decode-speed lever.

The trap is that vLLM's FP8 path for these layers is **static-activation W8A8**:
it expects an `input_scale` per layer *from the checkpoint*. Convert the weights
without it and the engine loads fine, runs fast, and produces garbage — we
measured coherence 0/10 with speculative-decode acceptance collapsing to 0.1%.

So calibration is not optional:

```bash
# 1. Record activation maxima for the 300 target layers on a rented GPU.
docker build -t flashnext-calib calibrate/
#    Run it; poll /status; fetch /result when stage == "done".

# 2. Graft the scales into the FP8 checkpoint (input_scale = amax/448 × 1.25).
CALIB_RESULT=calib_result.json FP8P_CHECKPOINT=/path/to/fp8p \
  python3 calibrate/graft_input_scales.py
```

`calib_server.py` runs the official BF16 model with forward hooks on exactly the
layers the FP8 conversion touched, over a mixed prose/code/chat calibration set,
and **fails fast** if the discovered module count doesn't match — checking that
before spending GPU-hours rather than after. The graft audits name-set equality
against the conversion report, float32-exact readback, and index integrity
before writing.

---

## 6. Measured results

DGX Spark GB10, this checkpoint, full configuration:

| | |
|---|---|
| Decode, single stream | 24.8–25.5 tok/s (MTP k=1, ~80% acceptance) |
| Decode without speculation | ~16.8 tok/s |
| Boot | ~6.5 min (198 s weight load) |
| Resident weights | 75.94 GB (from 102.76 GB before the mmap patch) |
| KV pool | 12–14.7 GB, ~350–590K tokens |
| Context | 131,072 (46K-token prompt answers in ~25 s) |
| Cached conversation turns | ~10 s cold → ~2–3 s |
| Cache hit rate | 56–70% on conversation-shaped traffic |

---

## 7. Things that cost us time, in one list

1. `VLLM_PLE_CPU_OFFLOAD` frees nothing on unified memory. Use mmap.
2. FP8 KV cache is rejected by this model's QSA attention.
3. FlashInfer b12x MoE faults (Xid 31) at JIT warmup on sm121.
4. Prefix caching is auto-enabled with an unsupported cache mode by default.
5. Uniform-geometry tests pass while varied geometry crashes.
6. Benchmarks without clock logging are meaningless.
7. `splitting_ops` replaces the defaults rather than adding to them.
8. Resumable downloads truncate silently; audit bytes before releasing a
   machine that holds the only copy.
9. FP8 weight conversion without calibrated activation scales produces a fast,
   confident, wrong model.
