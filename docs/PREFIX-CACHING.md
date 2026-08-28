# Fixing prefix caching on Qwen3.8-Flash-Next (vLLM, GDN hybrids)

Prefix caching on Flash-Next's day-one vLLM tree crashes the engine: turn 1 of
a conversation works, turn 2 (the first cache hit) dies with a CUDA illegal
memory access in the GDN state-restore path
(`qwen_gdn_linear_attn.py: ssm_state[prefill_state_indices] = ...`, corruption
upstream in the Triton `chunk_gated_delta_rule` kernel). This document is the
fix: two small patches ported from unmerged upstream vLLM pull requests, the
required serving configuration, and — just as important — how to test the
result so it doesn't lie to you.

Verified on a DGX Spark (GB10/sm121), but the underlying bugs are
architecture-independent: the same crash cluster is reported on A100, RTX PRO
6000, RTX 5090, and GB200 (vllm#43559 and relatives). This is scheduler-level
state bookkeeping, not a GPU-specific kernel fault.

## The two bugs

**1. Speculative decoding poisons the cache state (vllm#48375, unmerged).**
`MambaManager.find_longest_cache_hit` accepts a `drop_eagle_block` flag and
silently ignores it. With MTP/EAGLE speculative decoding, the final matched
block of a cache hit can contain a recurrent-state snapshot taken over draft
tokens that verification later rejected. Attention layers drop that block;
the mamba/GDN path resumed from it — corrupted state, then a crash or (worse)
silently wrong answers.

`patches/40-mamba-eagle-drop.patch` lowers the cache-hit search ceiling by one
mamba block when `drop_eagle_block` is set. That is arithmetically identical to
upstream's fix for the coarse block search, and additionally bounds the
fine-grained partial-unit search branch that the Flash-Next tree has and the
upstream diff predates. If you rebase onto a different tree, re-check both
branches.

**2. The resume state index uses the wrong block size (vllm#53142).**
`MambaHybridModelState.add_request` seeds the running mamba state column with
`cache_config.block_size` — the attention/CLI block size. On hybrid models the
mamba group's block size is generally different, so a request resuming over a
cached prefix computes an out-of-range block-table column, reads a garbage
block id, and the fused align precopy kernel dies with an illegal memory
access. First requests are unaffected (they seed `-1` regardless), which makes
this look like a spurious second-request crash.

`patches/41-mamba-state-seed.patch` divides by the mamba spec's block size,
with a safe fallback before the spec is populated.

## Install

The patches apply on top of the Flash-Next day-one image tree (after the other
patches in this repo, in lexical order):

```dockerfile
COPY patches/ /opt/patches/
RUN cd /usr/local/lib/python3.12/dist-packages && \
    for p in /opt/patches/*.patch; do patch -p1 --forward < "$p"; done
```

## Serving configuration

```
--enable-prefix-caching
--mamba-cache-mode align
```

Both flags, explicitly. This matters more than it looks:

- The tree auto-enables prefix caching for this model while `mamba_cache_mode`
  defaults to `none` — an unvalidated combination, and the configuration the
  original crashes shipped under. Never rely on the defaults here.
- `align` is the only cache mode the model supports (the model code itself
  rejects `all` with a pointer to `align`).
- Speculative decoding (`--speculative-config` with the MTP method) works
  *with* caching once patch 40 is in. Without it, the combination is the most
  crash-prone configuration reported across every GPU architecture.

## How to test it (do not skip this)

These bugs have a property that defeats normal testing: **uniform test prompts
pass while varied prompts crash.** We had two "fully passing" configurations
die on the first conversation whose filler text had a different shape. Three
rules, learned the expensive way:

1. **Vary the geometry.** Soak with conversations whose vocabulary, sentence
   lengths, paragraph counts, and total length differ every round — different
   shapes land block boundaries differently, and the boundaries are where the
   bugs live. Run several rounds, multiple turns each.
2. **Seed the randomness.** Generate soak prompts from a seeded RNG so any
   crashing conversation is exactly reproducible. A deterministic killer input
   is the sharpest verification tool you can own: after patching, replay it
   byte-for-byte. Both of our original crashers pass on the patched tree.
3. **Check answers, not just liveness.** This bug family corrupts output
   silently before it crashes (vllm#43559 began as an accuracy report). Every
   soak turn should contain a verifiable question (arithmetic, recall of a
   planted number) that gets checked, not eyeballed.

Also verify the cache is actually caching: `vllm:prefix_cache_hits_total` in
`/metrics` must rise above zero. Align mode keeps a single state checkpoint
per request at the last block boundary; with unlucky prompt geometry the hit
rate is silently exactly 0% while everything appears enabled (vllm#45238).
Relatedly, the per-turn speedup varies with where block boundaries fall in
your conversation — some turns get the full win, some partial. That is how
align mode works, not a regression.

## Measured result (DGX Spark, this repo's checkpoint)

- Both previously-crashing conversations replayed verbatim: pass, correct
  answers, no restarts.
- 16 varied-geometry soak rounds (9K–24K-token prompts, 3 checked turns each)
  across caching-only and caching+MTP configurations: zero crashes, zero
  accuracy failures.
- Cached turns: ~10 s cold prefill down to ~2–3 s. Cache hit rate 56–70% on
  conversation-shaped traffic. Cross-conversation prefix reuse works.
- Single-stream decode overhead from caching: ~0.5 tok/s.

## Status of the upstream fixes

Both fixes ported here are, at the time of writing, unmerged upstream
(vllm#48375; the second is a fix posted in the vllm#53142 issue thread). If
they merge, prefer the upstream versions on your next rebase. One known issue
remains outside their scope: a Blackwell-specific Triton autotune corruption in
the same kernel family (fla-org#790) that produces silently wrong output under
particular autotune configurations — one more reason rule 3 above is not
optional.
