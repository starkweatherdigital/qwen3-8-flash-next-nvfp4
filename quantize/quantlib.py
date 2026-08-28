"""BCN-76 NVFP4 packing library (modelopt-compatible) for Qwen3.8-Flash-Next.

Formats produced:
- Experts: modelopt NVFP4 quartet
    <n>.weight          uint8  [out, in//2]   two E2M1 codes per byte (lo nibble = even idx)
    <n>.weight_scale    fp8e4m3 [out, in//16] per-16 block scales (relative to global)
    <n>.weight_scale_2  fp32   scalar         global scale = amax / (6*448)
    <n>.input_scale     fp32   scalar         copied from calibrated reference
- PLE 4-bit (our extension, loaded by ple_layer patch):
    <shard>.weight_packed uint8   [rows, dim//2]
    <shard>.weight_scale  fp8e4m3 [rows, dim//16]
    table-level weight_global_scale fp32 scalar
"""
import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
BOUND = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
F8MAX = 448.0
F4MAX = 6.0


def _codes(x: torch.Tensor) -> torch.Tensor:
    """x: normalized values (|x| ideally <= 6). Returns uint8 codes 0..15."""
    ax = x.abs().clamp(max=F4MAX)
    idx = torch.bucketize(ax, BOUND.to(x.device))
    codes = idx.to(torch.uint8) | ((x < 0).to(torch.uint8) << 3)
    return codes


def _pack(codes: torch.Tensor, lo_first: bool = True) -> torch.Tensor:
    lo = codes[..., 0::2]
    hi = codes[..., 1::2]
    if not lo_first:
        lo, hi = hi, lo
    return lo | (hi << 4)


def unpack(packed: torch.Tensor, lo_first: bool = True) -> torch.Tensor:
    lo = packed & 0xF
    hi = packed >> 4
    if not lo_first:
        lo, hi = hi, lo
    out = torch.stack([lo, hi], dim=-1)
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def code_values(codes: torch.Tensor) -> torch.Tensor:
    mag = E2M1.to(codes.device)[(codes & 7).long()]
    sign = torch.where(codes & 8 > 0, -1.0, 1.0)
    return mag * sign


def nvfp4_quantize(w: torch.Tensor, group: int = 16, global_scale: float | None = None,
                   lo_first: bool = True):
    """w: [out, in] bf16/fp32 on GPU. Returns (packed u8, scale fp8, gs fp32 tensor)."""
    w = w.to(torch.float32)
    out_dim, in_dim = w.shape
    assert in_dim % group == 0 and in_dim % 2 == 0
    gs = float(w.abs().amax().item()) / (F4MAX * F8MAX) if global_scale is None else float(global_scale)
    if gs <= 0:
        gs = 1e-12
    blocks = w.view(out_dim, in_dim // group, group)
    bmax = blocks.abs().amax(dim=-1)
    scale = (bmax / F4MAX) / gs
    scale_f8 = scale.clamp(max=F8MAX).to(torch.float8_e4m3fn)
    eff = scale_f8.to(torch.float32) * gs
    eff = torch.where(eff == 0, torch.ones_like(eff), eff)
    xn = blocks / eff.unsqueeze(-1)
    codes = _codes(xn).view(out_dim, in_dim)
    packed = _pack(codes, lo_first=lo_first)
    return packed.contiguous(), scale_f8.contiguous(), torch.tensor(gs, dtype=torch.float32)


def nvfp4_dequant(packed: torch.Tensor, scale_f8: torch.Tensor, gs, group: int = 16,
                  lo_first: bool = True) -> torch.Tensor:
    codes = unpack(packed, lo_first=lo_first)
    vals = code_values(codes)
    out_dim, in_dim = vals.shape
    eff = scale_f8.to(torch.float32) * float(gs)
    return (vals.view(out_dim, in_dim // group, group) * eff.unsqueeze(-1)).view(out_dim, in_dim)


import re
RE_EXPERT = re.compile(r"^(model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj))\.weight$")
RE_PLE_SHARD = re.compile(r"^(model\.language_model\.layers\.\d+\.ple\.ple_embedding\.ngram_embedding\.shard_\d+)\.weight$")


def classify(name: str) -> str:
    if RE_EXPERT.match(name):
        return "expert"
    if RE_PLE_SHARD.match(name):
        return "ple"
    return "copy"
