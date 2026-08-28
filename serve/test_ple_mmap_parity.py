"""BCN-82 offline gate: mmap-PLE adapter must be BIT-IDENTICAL to the resident path.

Runs on the Mac against the real checkpoint copy (~/bcn76_ckpt_backup), no GPU,
no vllm. The reference implementation below is a verbatim copy of
Qwen4ExpPLENvFp4EmbeddingMethod.embedding (ple_layer.r3.py lines 230-252)
operating on safetensors-loaded tensors via index_select — the exact code the
engine runs today. The adapter path is MmapNvFp4PleTable.gather +
_MmapNvFp4NgramEmbedding.forward. Gate: torch.equal on bf16 outputs AND raw
byte equality of the gathered rows.
"""

import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The module under test ships as patches/35-ple-nvfp4-mmap.patch; for the
# offline parity check, copy it next to this file (or point PYTHONPATH at it).
from ple_nvfp4_mmap import (  # noqa: E402
    MmapNvFp4PleTable,
    _MmapNvFp4NgramEmbedding,
    find_nvfp4_shards,
    read_global_scale,
)

MODEL_PATH = os.environ.get("NVFP4_CHECKPOINT") or sys.exit(
    "set NVFP4_CHECKPOINT to the local NVFP4 checkpoint directory")
LAYER_IDX = 1
HEAD_DIM = 160
EXPECT_SHARDS = 128
EXPECT_ROWS_PER_SHARD = 2_500_012
REFERENCE_SHARDS = [0, 63, 126, 127]  # ends + middles; 127 is the last shard
N_RANDOM = 50_000
SEED = 20260827

failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# --------------------------------------------------------------------------- #
print("== 1. shard discovery ==")
shards, packed_bytes, scale_bytes, gscale_entry = find_nvfp4_shards(MODEL_PATH, LAYER_IDX)
check("128 shards found", len(shards) == EXPECT_SHARDS, f"got {len(shards)}")
check("packed width 80", packed_bytes == HEAD_DIM // 2, f"got {packed_bytes}")
check("scale width 10", scale_bytes == HEAD_DIM // 16, f"got {scale_bytes}")
check("global scale entry present", gscale_entry is not None)
rows_all = [shards[i][4] for i in sorted(shards)]
check(
    "uniform shard rows",
    all(r == EXPECT_ROWS_PER_SHARD for r in rows_all),
    f"min={min(rows_all)} max={max(rows_all)}",
)
total_rows = sum(rows_all)
check("total rows 320,001,536", total_rows == 320_001_536, f"got {total_rows}")

gscale = read_global_scale(gscale_entry)
print(f"  global scale = {gscale.item():.9g}")
check("global scale finite/nonzero", torch.isfinite(gscale).item() and gscale.item() != 0.0)

# --------------------------------------------------------------------------- #
print("== 2. build adapter table + placeholder ==")
table = MmapNvFp4PleTable(
    shards, EXPECT_ROWS_PER_SHARD, packed_bytes, scale_bytes, workers=8, chunk=2048
)
emb = _MmapNvFp4NgramEmbedding(total_rows, HEAD_DIM, params_dtype=torch.bfloat16)
emb.weight_global_scale.copy_(gscale)
emb.table = table

# --------------------------------------------------------------------------- #
print("== 3. load reference shard tensors (safetensors, the engine's own read path) ==")
ref_packed: dict[int, torch.Tensor] = {}
ref_scale: dict[int, torch.Tensor] = {}
t0 = time.time()
prefix = f"model.language_model.layers.{LAYER_IDX}.ple.ple_embedding.ngram_embedding"
# group by file to open each once
by_file: dict[str, list[int]] = {}
for idx in REFERENCE_SHARDS:
    by_file.setdefault(shards[idx][0], []).append(idx)
    by_file.setdefault(shards[idx][2], []).append(idx)
for path, idxs in by_file.items():
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        for idx in set(idxs):
            pk = f"{prefix}.shard_{idx}.weight_packed"
            sk = f"{prefix}.shard_{idx}.weight_scale"
            if pk in keys:
                ref_packed[idx] = f.get_tensor(pk)
            if sk in keys:
                ref_scale[idx] = f.get_tensor(sk)
for idx in REFERENCE_SHARDS:
    assert idx in ref_packed and idx in ref_scale, f"reference shard {idx} incomplete"
    assert ref_packed[idx].dtype == torch.uint8
    assert ref_scale[idx].dtype == torch.float8_e4m3fn
print(f"  loaded {len(REFERENCE_SHARDS)} reference shards in {time.time()-t0:.1f}s")


def reference_embedding(ids: torch.Tensor) -> torch.Tensor:
    """Verbatim Qwen4ExpPLENvFp4EmbeddingMethod.embedding on reference tensors."""
    flat = ids.reshape(-1)
    shard_of = flat // EXPECT_ROWS_PER_SHARD
    local = flat - shard_of * EXPECT_ROWS_PER_SHARD
    packed = torch.empty((flat.shape[0], packed_bytes), dtype=torch.uint8)
    scales_f8 = torch.empty((flat.shape[0], scale_bytes), dtype=torch.float8_e4m3fn)
    for idx in REFERENCE_SHARDS:
        m = shard_of == idx
        if m.any():
            packed[m] = ref_packed[idx].index_select(0, local[m])
            scales_f8[m] = ref_scale[idx].index_select(0, local[m])
    # ---- verbatim method math (ple_layer.r3.py:230-252) ----
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    scales = scales_f8.to(torch.float32)
    lo = packed & 0xF
    hi = packed >> 4
    codes = torch.stack((lo, hi), dim=-1).reshape(flat.shape[0], -1)
    vals = levels[(codes & 7).long()]
    vals = torch.where(codes & 8 > 0, -vals, vals)
    dim = vals.shape[-1]
    eff = scales * gscale
    vals = (vals.view(flat.shape[0], dim // 16, 16) * eff.unsqueeze(-1)).view(
        flat.shape[0], dim
    )
    return vals.to(torch.bfloat16).view(*ids.shape, dim), packed, scales_f8


# --------------------------------------------------------------------------- #
print("== 4. bit-identity: random ids (dups, unsorted) over reference shards ==")
rng = np.random.default_rng(SEED)
parts = []
for idx in REFERENCE_SHARDS:
    lo_g = idx * EXPECT_ROWS_PER_SHARD
    parts.append(rng.integers(lo_g, lo_g + EXPECT_ROWS_PER_SHARD, size=N_RANDOM // len(REFERENCE_SHARDS)))
ids_np = np.concatenate(parts)
rng.shuffle(ids_np)
dup = rng.choice(ids_np, size=N_RANDOM // 10)  # guaranteed duplicates
ids_np = np.concatenate([ids_np, dup])
ids = torch.from_numpy(ids_np.astype(np.int64))

t0 = time.time()
adapter_out = emb.forward(ids)
t_adapter = time.time() - t0
ref_out, ref_packed_rows, ref_scale_rows = reference_embedding(ids)
check("output dtype bf16", adapter_out.dtype == torch.bfloat16)
check("output shape", tuple(adapter_out.shape) == (ids.shape[0], HEAD_DIM))
check("BIT-IDENTICAL output (random)", torch.equal(adapter_out, ref_out))

g_packed, g_scale = table.gather(ids_np.astype(np.int64))
check(
    "gathered packed bytes identical",
    np.array_equal(g_packed, ref_packed_rows.numpy()),
)
check(
    "gathered scale bytes identical",
    np.array_equal(g_scale, ref_scale_rows.view(torch.uint8).numpy()),
)
print(f"  adapter fwd: {len(ids_np):,} lookups in {t_adapter*1000:.0f} ms (cold page cache)")

# --------------------------------------------------------------------------- #
print("== 5. bit-identity: shard boundaries ==")
edge_ids = []
for idx in REFERENCE_SHARDS:
    lo_g = idx * EXPECT_ROWS_PER_SHARD
    edge_ids += [lo_g, lo_g + 1, lo_g + EXPECT_ROWS_PER_SHARD - 2, lo_g + EXPECT_ROWS_PER_SHARD - 1]
# adjacent cross-shard pair 126->127
edge_ids += [127 * EXPECT_ROWS_PER_SHARD - 1, 127 * EXPECT_ROWS_PER_SHARD]
ids_e = torch.tensor(edge_ids, dtype=torch.int64)
a, (r, _, _) = emb.forward(ids_e), reference_embedding(ids_e)
check("BIT-IDENTICAL output (boundaries)", torch.equal(a, r))

print("== 6. 2D id shape (ngram_ids [N,16]) ==")
ids2d = ids[: 4096 - (4096 % 16)].reshape(-1, 16)
a2 = emb.forward(ids2d)
r2, _, _ = reference_embedding(ids2d)
check("2D output shape", tuple(a2.shape) == (ids2d.shape[0], 16, HEAD_DIM))
check("BIT-IDENTICAL output (2D)", torch.equal(a2, r2))

print("== 7. edge cases ==")
e = emb.forward(torch.empty((0,), dtype=torch.int64))
check("empty input -> [0,160]", tuple(e.shape) == (0, HEAD_DIM))
try:
    table.gather(np.array([total_rows], dtype=np.int64))
    check("out-of-range raises", False)
except IndexError:
    check("out-of-range raises", True)
try:
    table.gather(np.array([-1], dtype=np.int64))
    check("negative id raises", False)
except IndexError:
    check("negative id raises", True)

print("== 8. warm-cache latency (decode-shaped: 256 rows) ==")
small = torch.from_numpy(
    rng.choice(ids_np, size=256).astype(np.int64)
)
emb.forward(small)  # warm
t0 = time.time()
for _ in range(50):
    emb.forward(small)
per_step = (time.time() - t0) / 50 * 1000
print(f"  256-row lookup (warm): {per_step:.2f} ms/step")
check("warm decode gather < 5 ms", per_step < 5.0, f"{per_step:.2f} ms")

print()
if failures:
    print(f"RESULT: FAIL ({len(failures)}): {failures}")
    sys.exit(1)
print("RESULT: ALL GATES PASS — adapter is bit-identical to the resident path")
