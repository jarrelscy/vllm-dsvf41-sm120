# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference-exact BF16 output projection for DeepseekV41 attention.

The released checkpoint's ``inference/model.py`` computes o_proj in bf16
(Attention.forward): inverse RoPE on the attention output's rope tail, then a
block-diagonal grouped bf16 einsum with ``wo_a``, then the ``wo_b`` row
projection. ``wo_a`` ships as block-fp8 (E4M3 weight + 32x32 E8M0 block scales);
the reference ``convert.py`` dequantizes it to bf16 at load. We reproduce that
dequant exactly and run the reference einsum.

The fork's fp8 DeepGEMM einsum (``deep_gemm_fp8_o_proj``) is a memory/latency
optimization whose fused inv-rope quant kernel is hardcoded to a 128-element
quant group; the checkpoint's 32-element attention-block granularity would need
a fused-kernel rewrite to feed it. This bf16 path is the lossless reference used
for bring-up and as the numerical gold for that later optimization.

Numerics are byte-for-byte the reference:
  * dequant ``(w_fp8.f32 * upcast_e8m0(scale)).bf16`` block-expanded 32x32,
  * inverse (conjugate) interleaved-pair RoPE on the last ``rope_dim`` per head
    (``y0 = x0*cos + x1*sin``, ``y1 = x1*cos - x0*sin``, cache ``[cos||sin]``),
    matching ``fused_inv_rope_fp8_quant`` exactly,
  * grouped einsum ``tgd,grd->tgr`` == reference ``bsgd,grd->bsgr``.
"""

import torch
import torch.nn as nn

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_to_fp32,
)


def dequantize_block_fp8_to_bf16(
    weight: torch.Tensor,  # [N, K] float8_e4m3fn
    scale: torch.Tensor,  # [N//block_n, K//block_k] E8M0 or fp32
    block_n: int,
    block_k: int,
) -> torch.Tensor:
    """Block-scaled fp8 -> bf16, matching the reference convert.py dequant.

    ``scale`` is a per-(block_n x block_k) tile scale. E8M0 (ue8m0) scales are
    upcast to their exact fp32 power-of-two before expansion; fp32 scales pass
    through. Multiply in fp32, round once to bf16.
    """
    if scale.dtype in (torch.float8_e8m0fnu, torch.uint8, torch.int8):
        scale_f32 = _upcast_e8m0_to_fp32(scale)
    else:
        scale_f32 = scale.to(torch.float32)
    n, k = weight.shape
    sf = scale_f32.repeat_interleave(block_n, dim=0).repeat_interleave(block_k, dim=1)
    sf = sf[:n, :k]
    return (weight.to(torch.float32) * sf).to(torch.bfloat16)


def bf16_inv_rope_o_proj(
    o: torch.Tensor,  # [T, n_local_heads, head_dim] bf16
    positions: torch.Tensor,  # [T] int
    cos_sin_cache: torch.Tensor,  # [max_pos, rope_dim] fp32 (cos||sin halves)
    wo_a_bf16: torch.Tensor,  # [n_groups, o_lora_rank, heads_per_group*head_dim] bf16
    wo_b: nn.Module,  # RowParallelLinear (return_bias=False)
    *,
    n_groups: int,
    heads_per_group: int,
    head_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    T, H, D = o.shape
    rd = rope_dim
    half = rd // 2

    # cos/sin for each token; cache stores [cos(half) || sin(half)], indexed by
    # interleaved-pair index (same as the fused fp8 kernel: cs_idx = local >> 1).
    cs = cos_sin_cache.index_select(0, positions)  # [T, rd]
    cos = cs[:, :half].unsqueeze(1)  # [T, 1, half]
    sin = cs[:, half:].unsqueeze(1)  # [T, 1, half]

    tail = o[..., -rd:].to(torch.float32)  # [T, H, rd]
    pair = tail.unflatten(-1, (half, 2))  # [T, H, half, 2]
    x0 = pair[..., 0]  # even positions
    x1 = pair[..., 1]  # odd positions
    # Inverse (conjugate) rotation, matching apply_rotary_emb(..., inverse=True)
    # and the fused fp8 inv-rope kernel.
    y0 = x0 * cos + x1 * sin
    y1 = x1 * cos - x0 * sin
    rot_tail = torch.stack((y0, y1), dim=-1).flatten(-2).to(o.dtype)  # [T, H, rd]
    o_rot = torch.cat((o[..., :-rd], rot_tail), dim=-1)  # [T, H, D]

    # heads[0..hpg-1] -> group 0, [hpg..2*hpg-1] -> group 1, head-major within
    # group (matches the fused kernel's g = head // heads_per_group layout).
    og = o_rot.reshape(T, n_groups, heads_per_group * head_dim)  # [T, g, d]
    z = torch.einsum("tgd,grd->tgr", og, wo_a_bf16)  # [T, g, o_lora_rank]
    return wo_b(z.flatten(1))


def o_proj_bf16(
    o: torch.Tensor,  # [T, n_local_heads, head_dim] bf16
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,  # block-fp8 ColumnParallelLinear (weight + weight_scale_inv)
    wo_b: nn.Module,  # RowParallelLinear (return_bias=False)
    *,
    n_groups: int,
    heads_per_group: int,
    head_dim: int,
    rope_dim: int,
    o_lora_rank: int,
) -> torch.Tensor:
    """BF16 reference-exact o_proj, shared by all CUDA backends.

    ``wo_a`` is dequantized to bf16 once and cached on the module as
    ``_bf16_dequant``. The first forward (profile_run / warmup) runs eagerly
    before any cudagraph capture, so the cached tensor is populated by the time
    a captured replay reads it and the ``is None`` branch is never traced.
    """
    wo_a_bf16 = getattr(wo_a, "_bf16_dequant", None)
    if wo_a_bf16 is None:
        w = wo_a.weight
        s = getattr(wo_a, "weight_scale_inv", None)
        if s is None:
            s = wo_a.weight_scale
        block_n = w.shape[0] // s.shape[0]
        block_k = w.shape[1] // s.shape[1]
        w_bf16 = dequantize_block_fp8_to_bf16(w, s, block_n, block_k)
        wo_a_bf16 = w_bf16.view(n_groups, o_lora_rank, heads_per_group * head_dim)
        # Plain tensor attribute (not a Parameter/buffer), persists on the module.
        wo_a._bf16_dequant = wo_a_bf16
    return bf16_inv_rope_o_proj(
        o,
        positions,
        cos_sin_cache,
        wo_a_bf16,
        wo_b,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        rope_dim=rope_dim,
    )
