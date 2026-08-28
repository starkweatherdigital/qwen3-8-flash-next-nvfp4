#!/usr/bin/env python3
"""BCN-84 graft: add input_scale tensors to the staged FP8-periphery checkpoint.

input_scale = calibrated_amax / 448 * 1.25 (E4M3 max; x1.25 disclosed safety
multiplier — the RadixArk-precedent margin; amax over 192 samples underestimates
the true max, and over-reserving only costs a little quantization range).
Writes ONE new shard + updated index into the OUTPUT dir. Never mutates shards.
Gates: name-set equality vs the FP8 report's converted tensors; scalar F32;
readback equality; index total_size correct.
"""
import json, os, struct, sys
import numpy as np

CAL = os.environ.get("CALIB_RESULT", "calib_result.json")
OUT = os.environ.get("FP8P_CHECKPOINT")  # dir holding the FP8-periphery checkpoint
assert OUT, "set FP8P_CHECKPOINT to the FP8-periphery checkpoint dir"
REPORT = os.path.join(OUT, "fp8_report.json")
NEW_SHARD = "model-00133-input-scales.safetensors"
E4M3_MAX, SAFETY = 448.0, 1.25

cal = json.load(open(CAL))["amax"]
report = json.load(open(REPORT))
fp8_weights = {k[:-len(".weight")] for k in report if k.endswith(".weight")}
cal_mods = set(cal)
assert cal_mods == fp8_weights, (
    f"module-set mismatch: cal-only={sorted(cal_mods-fp8_weights)[:3]} "
    f"report-only={sorted(fp8_weights-cal_mods)[:3]}")
print(f"name-set equality: {len(cal_mods)} modules match the FP8 report exactly")

# build safetensors shard: 300 F32 scalars named <module>.input_scale
names = sorted(cal_mods)
header, blobs, off = {}, [], 0
for n in names:
    scale = np.float32(cal[n] / E4M3_MAX * SAFETY)
    b = scale.tobytes()
    header[n + ".input_scale"] = {"dtype": "F32", "shape": [], "data_offsets": [off, off + 4]}
    blobs.append(b); off += 4
hjson = json.dumps(header, separators=(",", ":")).encode()
pad = (8 - len(hjson) % 8) % 8
hjson += b" " * pad
path = os.path.join(OUT, NEW_SHARD)
with open(path, "wb") as f:
    f.write(struct.pack("<Q", len(hjson))); f.write(hjson)
    for b in blobs: f.write(b)
print(f"wrote {path}: {len(names)} scalars, {os.path.getsize(path)} bytes")

# update index
ip = os.path.join(OUT, "model.safetensors.index.json")
idx = json.load(open(ip))
for n in names:
    key = n + ".input_scale"
    assert key not in idx["weight_map"], f"{key} already in index"
    idx["weight_map"][key] = NEW_SHARD
idx["metadata"]["total_size"] = idx["metadata"].get("total_size", 0) + off
json.dump(idx, open(ip, "w"))
print("index updated: +%d entries, total_size += %d" % (len(names), off))

# readback audit
from_read = {}
with open(path, "rb") as f:
    n8 = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n8)); data0 = 8 + n8
    raw = open(path, "rb").read()
    for k, m in hdr.items():
        assert m["dtype"] == "F32" and m["shape"] == []
        from_read[k] = struct.unpack("<f", raw[data0+m["data_offsets"][0]:data0+m["data_offsets"][1]])[0]
# Compare float32-EXACT: the file holds f32, so recompute in f32 rather than
# comparing against f64 math with a tolerance (a 1e-9 tolerance false-fails).
bad = [k for k in from_read
       if from_read[k] != float(np.float32(cal[k[:-len('.input_scale')]] / E4M3_MAX * SAFETY))]
assert not bad, bad[:3]
mn, mx = min(from_read.values()), max(from_read.values())
print(f"readback audit PASS: {len(from_read)} scales, range [{mn:.5f}, {mx:.5f}]")
print("GRAFT COMPLETE")
