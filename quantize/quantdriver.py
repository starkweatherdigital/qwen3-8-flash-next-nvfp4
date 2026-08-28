#!/usr/bin/env python3
"""BCN-76 full quantization driver: official BF16 -> modelopt-NVFP4 experts +
4-bit PLE + BF16 periphery. Incremental (processes input shards as the
snapshot download lands them), resumable via manifest.

Run: python3 quantdriver.py   (needs a CUDA GPU; ~1 h on an A6000-class card)
Outputs to $QUANT_OUT: model-XXXXX.safetensors + index + config + tokenizer files.
Env: QUANT_WORK (default /work), QUANT_HF_CACHE, QUANT_OUT, QUANT_MANIFEST.
Incremental + resumable: safe to kill and rerun; the manifest tracks progress.
"""
import json, os, re, sys, time, gc, glob, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantlib import nvfp4_quantize

# Paths are env-configurable; the defaults are the layout this was built on
# (a rented GPU pod with a /work scratch volume).
WORK = os.environ.get("QUANT_WORK", "/work")
HF = os.environ.get("QUANT_HF_CACHE", WORK + "/hf/hub/models--Qwen--Qwen3.8-Flash-Next")
OUT = os.environ.get("QUANT_OUT", WORK + "/out/ckpt")
MAN = os.environ.get("QUANT_MANIFEST", WORK + "/out/manifest.json")
DEV = "cuda"
LO_FIRST = True                      # set from smoke parity
SHARD_BYTES = 4_200_000_000

RE_EXP_FUSED = re.compile(r"^(model\.language_model\.layers\.\d+|mtp\.layers\.\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")
RE_PLE = re.compile(r"^model\.language_model\.layers\.\d+\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$")

os.makedirs(OUT, exist_ok=True)


def snap_dir():
    d = glob.glob(HF + "/snapshots/*/")
    return d[0] if d else None


def load_manifest():
    if os.path.exists(MAN):
        return json.load(open(MAN))
    return {"done_files": [], "out_shards": [], "weight_map": {}, "ple_amax": None,
            "next_shard": 0}


def save_manifest(m):
    tmp = MAN + ".tmp"
    json.dump(m, open(tmp, "w"))
    os.replace(tmp, MAN)


class Writer:
    def __init__(self, man):
        self.man = man
        self.buf = {}
        self.bytes = 0

    def add(self, name, tensor):
        t = tensor.contiguous().cpu()
        self.buf[name] = t
        self.bytes += t.numel() * t.element_size()
        if self.bytes >= SHARD_BYTES:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        i = self.man["next_shard"]
        fn = f"model-{i:05d}.safetensors"
        save_file(self.buf, os.path.join(OUT, fn))
        for k in self.buf:
            self.man["weight_map"][k] = fn
        self.man["out_shards"].append(fn)
        self.man["next_shard"] = i + 1
        print(f"[writer] wrote {fn} ({self.bytes/1e9:.2f} GB, {len(self.buf)} tensors)", flush=True)
        self.buf = {}
        self.bytes = 0


def quant_expert_stack(writer, base, fused, scales_ref, mtp_fallback):
    """fused: gate_up [E, 2I, H] or down [E, H, I] bf16 CPU. base like
    'model.language_model.layers.7.mlp.experts' + kind."""
    m = RE_EXP_FUSED.match(base)
    prefix, kind = m.group(1), m.group(2)
    E = fused.shape[0]
    H = 2560
    for e in range(E):
        w = fused[e].to(DEV, non_blocking=True)
        if kind == "gate_up_proj":
            if w.shape[1] != H and w.shape[0] == H:
                w = w.T  # -> [2I, H]
            assert w.shape[1] == H, f"gate_up orientation {tuple(w.shape)}"
            inter = w.shape[0] // 2
            parts = {"gate_proj": w[:inter], "up_proj": w[inter:]}
        else:
            if w.shape[0] != H and w.shape[1] == H:
                w = w.T  # -> [H, I]
            assert w.shape[0] == H, f"down orientation {tuple(w.shape)}"
            parts = {"down_proj": w}
        for pname, pw in parts.items():
            tname = f"{prefix}.mlp.experts.{e}.{pname}"
            pk, sc, gs = nvfp4_quantize(pw.contiguous(), lo_first=LO_FIRST)
            writer.add(tname + ".weight", pk)
            writer.add(tname + ".weight_scale", sc)
            writer.add(tname + ".weight_scale_2", gs)
            lm = re.search(r"language_model\.layers\.(\d+)\.", tname)
            if lm is not None:
                isc = scales_ref.get(f"{lm.group(1)}.{pname}") or mtp_fallback[pname]
            else:
                isc = mtp_fallback[pname]
            writer.add(tname + ".input_scale", torch.tensor(float(isc), dtype=torch.float32))
        del w
    torch.cuda.empty_cache()


def main():
    man = load_manifest()
    snap = None
    while snap is None:
        snap = snap_dir()
        if snap is None:
            time.sleep(30)
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    by_file = {}
    for k, f in wm.items():
        by_file.setdefault(f, []).append(k)
    all_files = sorted(by_file)
    ple_files = sorted({f for f, ks in by_file.items() if any(RE_PLE.match(k) for k in ks)})
    nonple_files = [f for f in all_files if f not in ple_files]

    while not os.path.exists("/work/radixark_scales.pt"):
        print("[wait] radixark_scales.pt not ready", flush=True)
        time.sleep(60)
    sr = torch.load("/work/radixark_scales.pt", weights_only=True)
    per_layer = sr["per_layer"]          # {"<L>.<proj>": scale}
    mtp_fb = sr["mtp"]                   # {proj: scale}
    print("mtp input_scales:", mtp_fb, flush=True)

    writer = Writer(man)

    def wait_file(f):
        p = os.path.join(snap, f)
        while not os.path.exists(p):
            print(f"[wait] {f} not downloaded yet", flush=True)
            time.sleep(60)
        return p

    # ---- pass 1: non-PLE files (experts quantized, rest copied) ----
    for f in nonple_files:
        if f in man["done_files"]:
            continue
        p = wait_file(f)
        t0 = time.time()
        with safe_open(p, framework="pt") as sf:
            names = sorted(sf.keys())
            for k in names:
                if RE_EXP_FUSED.match(k):
                    quant_expert_stack(writer, k, sf.get_tensor(k), per_layer, mtp_fb)
                elif RE_PLE.match(k):
                    raise RuntimeError("ple tensor in nonple file: " + k)
                else:
                    writer.add(k, sf.get_tensor(k))
        writer.flush()
        man["done_files"].append(f)
        save_manifest(man)
        print(f"[pass1] {f} done in {time.time()-t0:.0f}s", flush=True)
        gc.collect()

    # ---- pass 2a: PLE global amax ----
    if man["ple_amax"] is None:
        amax = 0.0
        for f in ple_files:
            p = wait_file(f)
            with safe_open(p, framework="pt") as sf:
                for k in sorted(sf.keys()):
                    if RE_PLE.match(k):
                        t = sf.get_tensor(k).to(DEV)
                        amax = max(amax, float(t.abs().amax()))
                        del t
            print(f"[amax] {f} running amax {amax}", flush=True)
        man["ple_amax"] = amax
        save_manifest(man)
        torch.cuda.empty_cache()

    gs_table = man["ple_amax"] / (6.0 * 448.0)

    # ---- pass 2b: PLE quantize ----
    ple_meta_done = "ple_meta" in man["done_files"]
    for f in ple_files:
        if f in man["done_files"]:
            continue
        p = wait_file(f)
        t0 = time.time()
        with safe_open(p, framework="pt") as sf:
            for k in sorted(sf.keys()):
                mm = RE_PLE.match(k)
                if not mm:
                    writer.add(k, sf.get_tensor(k))
                    continue
                t = sf.get_tensor(k).to(DEV)
                pk, sc, _ = nvfp4_quantize(t, global_scale=gs_table, lo_first=LO_FIRST)
                base = k[: -len(".weight")]
                writer.add(base + ".weight_packed", pk)
                writer.add(base + ".weight_scale", sc)
                del t
        writer.flush()
        man["done_files"].append(f)
        save_manifest(man)
        print(f"[pass2] {f} done in {time.time()-t0:.0f}s", flush=True)
        torch.cuda.empty_cache()

    if "ple_global" not in man["done_files"]:
        ple_keys = [k for k in wm if "ngram_embedding" in k]
        tbl_prefix = sorted({k.rsplit(".shard_", 1)[0] for k in ple_keys if ".shard_" in k})
        for tp in tbl_prefix:
            writer.add(tp + ".weight_global_scale", torch.tensor(gs_table, dtype=torch.float32))
        writer.flush()
        man["done_files"].append("ple_global")
        save_manifest(man)

    # ---- finalize: index + config + tokenizer files ----
    writer.flush()
    total = 0
    for s in man["out_shards"]:
        total += os.path.getsize(os.path.join(OUT, s))
    json.dump({"metadata": {"total_size": total}, "weight_map": man["weight_map"]},
              open(os.path.join(OUT, "model.safetensors.index.json"), "w"))

    cfg = json.load(open(os.path.join(snap, "config.json")))
    ignore = [
        "model.embed_tokens", "*.self_attn.*", "*.linear_attn.*", "*.mlp.gate*",
        "*.mlp.shared_expert.*", "*.mlp.shared_expert_gate*", "*hyper_connection*",
        "*.ple.*", "model.visual.*", "model.language_model.embed_tokens", "lm_head",
        "mtp.fc_embedding", "mtp.fc_hidden",
    ]
    cfg["quantization_config"] = {
        "config_groups": {"group_0": {
            "input_activations": {"dynamic": False, "group_size": 16, "num_bits": 4, "type": "float"},
            "targets": ["Linear"],
            "weights": {"dynamic": False, "group_size": 16, "num_bits": 4, "type": "float"},
        }},
        "ignore": ignore,
        "producer": {"name": "modelopt", "version": "0.46.0"},
        "quant_algo": "NVFP4",
        "quant_method": "modelopt",
    }
    cfg["bcn76_ple_quant"] = {
        "format": "nvfp4_packed", "group_size": 16,
        "loader": "BCN76 apply_ple_nvfp4_patch.py + VLLM_PLE_NVFP4=1",
        "source": "Qwen/Qwen3.8-Flash-Next", "pipeline": "quantdriver.py RTN",
    }
    json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=1)
    for extra in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                  "generation_config.json", "preprocessor_config.json",
                  "video_preprocessor_config.json", "chat_template.jinja"):
        src = os.path.join(snap, extra)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(OUT, extra))
    print("TOTAL OUTPUT BYTES:", total, "=", total/1e9, "GB", flush=True)
    print("QUANT_DRIVER_DONE", flush=True)


if __name__ == "__main__":
    main()
