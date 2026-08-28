#!/usr/bin/env python3
"""BCN-76: patch vLLM day-one Qwen3.8-Flash-Next support with a 4-bit (NVFP4-packed)
PLE n-gram embedding storage format.

Adds to vllm/models/qwen4_exp/nvidia/ple_layer.py:
  - Qwen4ExpPLENvFp4EmbeddingMethod: weight_packed u8 [rows, dim/2] +
    weight_scale fp8e4m3 [rows, dim/16] + weight_global_scale fp32 scalar;
    embedding() gathers rows and dequantizes to the model dtype.
  - Selection via env VLLM_PLE_NVFP4=1.
  - load_weights handling for shard_K.weight_packed / shard_K.weight_scale /
    ngram_embedding.weight_global_scale (TP=1 only).
  - forward_impl output_buffer fast path bypassed for packed tables
    (serve with --enforce-eager).

Idempotent: skips if marker present. Usage: python3 apply_ple_nvfp4_patch.py [path-to-ple_layer.py]
"""
import sys, re, shutil, importlib.util

MARKER = "BCN76_PLE_NVFP4"

NEW_CLASS = '''

class Qwen4ExpPLENvFp4EmbeddingMethod(QuantizeMethodBase):
    """%s: 4-bit E2M1 PLE embedding, per-16 FP8 block scales + one global scale."""

    GROUP = 16

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        from vllm.model_executor.utils import set_weight_attrs
        rows = sum(output_partition_sizes)
        dim = input_size_per_partition
        weight_loader = extra_weight_attrs.get("weight_loader")
        wp = nn.Parameter(torch.empty(rows, dim // 2, dtype=torch.uint8),
                          requires_grad=False)
        set_weight_attrs(wp, {"input_dim": 1, "output_dim": 0,
                              "weight_loader": weight_loader})
        layer.register_parameter("weight_packed", wp)
        ws = nn.Parameter(torch.empty(rows, dim // self.GROUP,
                                      dtype=torch.float8_e4m3fn),
                          requires_grad=False)
        set_weight_attrs(ws, {"input_dim": 1, "output_dim": 0,
                              "weight_loader": weight_loader})
        layer.register_parameter("weight_scale", ws)
        gs = nn.Parameter(torch.zeros((), dtype=torch.float32),
                          requires_grad=False)
        layer.register_parameter("weight_global_scale", gs)
        layer._ple_nvfp4_dtype = params_dtype

    def apply(self, layer, x, bias=None):
        raise NotImplementedError("NVFP4 PLE weights only support embedding lookup")

    _levels_cache: dict = {}

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        dev = layer.weight_packed.device
        levels = self._levels_cache.get(dev)
        if levels is None:
            levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                  device=dev)
            self._levels_cache[dev] = levels
        flat = input_.reshape(-1)
        packed = layer.weight_packed.index_select(0, flat)
        scales = layer.weight_scale.index_select(0, flat).to(torch.float32)
        lo = packed & 0xF
        hi = packed >> 4
        codes = torch.stack((lo, hi), dim=-1).reshape(flat.shape[0], -1)
        vals = levels[(codes & 7).long()]
        vals = torch.where(codes & 8 > 0, -vals, vals)
        dim = vals.shape[-1]
        eff = scales * layer.weight_global_scale
        vals = (vals.view(flat.shape[0], dim // self.GROUP, self.GROUP)
                * eff.unsqueeze(-1)).view(flat.shape[0], dim)
        out_dtype = getattr(layer, "_ple_nvfp4_dtype", torch.bfloat16) or torch.bfloat16
        return vals.to(out_dtype).view(*input_.shape, dim)

'''.replace("%s", MARKER)

SELECT_ANCHOR = '''    """Select global-scale FP8 only for quantized PLE checkpoint shards."""

    if not isinstance(quant_config, Fp8Config):'''

SELECT_NEW = '''    """Select global-scale FP8 only for quantized PLE checkpoint shards."""

    import os as _os  # %s
    if _os.environ.get("VLLM_PLE_NVFP4") == "1":
        return Qwen4ExpPLENvFp4EmbeddingMethod()
    if not isinstance(quant_config, Fp8Config):'''.replace("%s", MARKER)

LOAD_ANCHOR = '''            if name.startswith(shard_prefix) and name.endswith(".weight"):'''

LOAD_NEW = '''            if name == "ngram_embedding.weight_global_scale":  # %s
                emb = self.ngram_embedding
                emb.weight_global_scale.data.copy_(loaded_weight.reshape(()))
                loaded.add(name)
                continue
            if name.startswith(shard_prefix) and (
                name.endswith(".weight_packed")
                or (name.endswith(".weight_scale") and "shard_" in name)
            ):
                sfx = ".weight_packed" if name.endswith(".weight_packed") else ".weight_scale"
                shard_text = name[len(shard_prefix): -len(sfx)]
                if shard_text.isdigit():
                    shard_index = int(shard_text)
                    emb = self.ngram_embedding
                    if (emb.shard_indices.org_vocab_start_index != 0
                            or emb.shard_indices.org_vocab_end_index < emb.org_vocab_size):
                        raise NotImplementedError("NVFP4 PLE supports TP=1 only")
                    pname = "weight_packed" if sfx == ".weight_packed" else "weight_scale"
                    param = getattr(emb, pname)
                    shard_size = (emb.org_vocab_size + self.split_ngram_parts - 1
                                  ) // self.split_ngram_parts
                    start = shard_index * shard_size
                    rows = loaded_weight.shape[0]
                    param.data[start:start + rows].copy_(
                        loaded_weight.to(device=param.device))
                    loaded.add("ngram_embedding." + pname)
                    continue
            if name.startswith(shard_prefix) and name.endswith(".weight"):'''.replace("%s", MARKER)

FWD_ANCHOR = '''        if output_buffer is not None:
            output = output_buffer[:num_tokens, : self.embedding_dim]'''

FWD_NEW = '''        if hasattr(self.ngram_embedding, "weight_packed"):  # %s
            return self.ngram_embedding(ngram_ids).flatten(-2)
        if output_buffer is not None:
            output = output_buffer[:num_tokens, : self.embedding_dim]'''.replace("%s", MARKER)

CLASS_ANCHOR = '''def _get_ple_embedding_quant_method('''


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        spec = importlib.util.find_spec("vllm")
        base = spec.submodule_search_locations[0]
        path = base + "/models/qwen4_exp/nvidia/ple_layer.py"
    src = open(path).read()
    if MARKER in src:
        print("already patched:", path)
        return
    for anchor in (SELECT_ANCHOR, LOAD_ANCHOR, FWD_ANCHOR, CLASS_ANCHOR):
        n = src.count(anchor)
        assert n == 1, f"anchor count {n} != 1 for: {anchor[:60]!r}"
    shutil.copy(path, path + ".bcn76.bak")
    src = src.replace(CLASS_ANCHOR, NEW_CLASS + "\n" + CLASS_ANCHOR)
    src = src.replace(SELECT_ANCHOR, SELECT_NEW)
    src = src.replace(LOAD_ANCHOR, LOAD_NEW, 1)
    src = src.replace(FWD_ANCHOR, FWD_NEW)
    open(path, "w").write(src)
    import py_compile
    py_compile.compile(path, doraise=True)
    print("patched OK:", path)


if __name__ == "__main__":
    main()
