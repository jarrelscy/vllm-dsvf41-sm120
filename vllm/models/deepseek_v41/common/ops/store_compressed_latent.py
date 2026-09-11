# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Store-only compress-KV paged writer for DeepSeek-V4.1-Flash.

This is the store body of deepseek_v4's
``_fused_kv_compress_norm_rope_insert_sparse_attn``
(fused_compress_quant_cache.py:219-296) transcribed VERBATIM, with the
state-gather + softmax pool + RMSNorm prologue REMOVED and replaced by a load
of a precomputed post-norm, PRE-RoPE latent (``latent[token, :HEAD_SIZE]``).

The v41 producer (Seam B) pools the compressor with the already-validated
``DeepseekV41RefCompressor`` in torch, then hands the resulting fp32 latent to
this kernel. Because the quant/RoPE/store math is byte-for-byte the v4 store
body, the emitted fp8_ds_mla pages are identical to what the flashinfer
sparse-MLA consumer reads back — zero reimplementation risk.

Cache page layout (``block_size`` tokens per block, uint8 flat):
  [0, bs*576):       token data (448 fp8 nope + 64 bf16 rope = 128 bytes each)
  [bs*576, +bs*8):   uint8 UE8M0 scales (7 real + 1 pad each)
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _store_compressed_latent_kernel(
    # ── precomputed latent (post-norm, PRE-RoPE) [num_tokens, HEAD_SIZE] fp32 ──
    latent_ptr,
    latent_stride0,
    # ── metadata ──
    positions_ptr,
    # ── RoPE ──
    cos_sin_cache_ptr,
    cos_sin_stride,
    # ── KV cache output ──
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    # ── constexprs ──
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,  # 448.0
    QUANT_BLOCK: tl.constexpr,  # 64 for DeepseekV4
    TOKEN_STRIDE: tl.constexpr,  # 576 for DeepseekV4
    SCALE_DIM: tl.constexpr,  # 8 for DeepseekV4 (7 real + 1 pad)
    KV_BLOCK_STRIDE: tl.constexpr,
):
    """One program per token; early-exits for non-boundary (slot<0) tokens."""
    token_idx = tl.program_id(0)

    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return

    position = tl.load(positions_ptr + token_idx)

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE

    # ── Load precomputed post-norm, PRE-RoPE latent (replaces pool+norm) ──
    normed = tl.load(
        latent_ptr + token_idx * latent_stride0 + block,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    # ── KV cache pointers ────────────────────────────────────────────────
    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size

    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    fp8_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM  # 448
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2  # 32

    # FP8 UE8M0 quant: cast fp32 → bf16 → fp32 before quant to match reference.
    N_QUANT_BLOCKS: tl.constexpr = TRITON_BLOCK_SIZE // QUANT_BLOCK
    N_NOPE_BLOCKS: tl.constexpr = NOPE_HEAD_DIM // QUANT_BLOCK  # 7
    INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX

    quant_input = normed.to(tl.bfloat16).to(tl.float32)
    quant_2d = tl.reshape(quant_input, (N_QUANT_BLOCKS, QUANT_BLOCK))
    abs_2d = tl.abs(quant_2d)
    block_absmax = tl.max(abs_2d, axis=1)  # [N_QUANT_BLOCKS] fp32
    block_absmax = tl.maximum(block_absmax, 1e-4)

    raw_scales = block_absmax * INV_FP8_MAX
    exponents = tl.ceil(tl.log2(raw_scales))
    inv_scales = tl.exp2(-exponents)
    inv_scales_col = tl.reshape(inv_scales, (N_QUANT_BLOCKS, 1))
    x_scaled = quant_2d * inv_scales_col
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
    x_uint8_flat = tl.reshape(x_uint8, (TRITON_BLOCK_SIZE,))

    nope_mask = block < NOPE_HEAD_DIM
    tl.store(fp8_ptr + block, x_uint8_flat, mask=nope_mask)

    scale_idx = tl.arange(0, N_QUANT_BLOCKS)
    encoded = exponents + 127.0
    encoded = tl.maximum(tl.minimum(encoded, 255.0), 0.0)
    tl.store(
        scale_ptr + scale_idx,
        encoded.to(tl.uint8),
        mask=scale_idx < N_NOPE_BLOCKS,
    )
    tl.store(scale_ptr + N_NOPE_BLOCKS, tl.zeros((), dtype=tl.uint8))

    # Register-based GPT-J RoPE in fp32.
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    pair_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)  # each [NUM_PAIRS] fp32

    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)

    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    result = tl.interleave(new_even, new_odd)  # [TRITON_BLOCK_SIZE] fp32

    # Store rotated rope portion as bf16 into the cache's bf16 area.
    bf16_ptr = (fp8_ptr + NOPE_HEAD_DIM).to(tl.pointer_type(tl.bfloat16))
    rope_local = block - NOPE_HEAD_DIM
    is_rope = (block >= NOPE_HEAD_DIM) & mask
    tl.store(bf16_ptr + rope_local, result.to(tl.bfloat16), mask=is_rope)


def store_compressed_latent(
    latent: torch.Tensor,
    positions: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    compress_ratio: int,
    *,
    head_dim: int = 512,
    rope_head_dim: int = 64,
    quant_block: int = 64,
    token_stride: int = 576,
    scale_dim: int = 8,
) -> None:
    """Write a precomputed post-norm PRE-RoPE latent into the fp8_ds_mla
    compress-KV paged cache.

    Args:
        latent: ``[num_tokens, head_dim]`` fp32/bf16 post-norm PRE-RoPE latent.
            Rows whose ``kv_slot_mapping`` entry is <0 are ignored.
        positions: ``[num_tokens]`` int — absolute token positions (for RoPE).
        kv_slot_mapping: ``[num_tokens]`` int — compressed paged slot per token
            (``pos//ratio`` at group boundaries, ``-1`` otherwise).
        kv_cache: the compress-KV paged cache, shaped so ``stride(0)`` is the
            per-block byte stride (``storage_block_size*584``) and ``shape[1]``
            is ``storage_block_size`` (tokens/block). uint8 view.
        cos_sin_cache: ``[max_pos, rope_head_dim]`` compress-RoPE table
            (theta 160000 YaRN), fp32: cos[:32] ‖ sin[:32].
        compress_ratio: 1 or 2.
    """
    num_tokens = latent.shape[0]
    if num_tokens == 0:
        return
    latent = latent.contiguous()
    _store_compressed_latent_kernel[(num_tokens,)](
        latent,
        latent.stride(0),
        positions,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        kv_cache,
        kv_slot_mapping,
        kv_cache.shape[1],
        HEAD_SIZE=head_dim,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_dim),
        COMPRESS_RATIO=compress_ratio,
        ROPE_HEAD_DIM=rope_head_dim,
        FP8_MAX=448.0,
        QUANT_BLOCK=quant_block,
        TOKEN_STRIDE=token_stride,
        SCALE_DIM=scale_dim,
        KV_BLOCK_STRIDE=kv_cache.stride(0),
        num_warps=4,
    )
