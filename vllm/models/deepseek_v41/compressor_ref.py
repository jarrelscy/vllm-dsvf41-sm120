# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference-faithful, pure-torch DeepSeek-V4.1-Flash KV compressor (ratios 0/1/2).

This is the CPU golden oracle for the compressor pooling math, decoupled from the paged
fused-kernel serving path (``compressor.py``). It mirrors ``inference/model.py::Compressor``
exactly so it can be (a) validated bit-exact against the reference source on CPU and (b) used
as the reference in the eventual GPU golden-diff of the paged kernel.

Pooling per ratio (from the released config's ``compress_ratios``):
  * ratio 0 -- pure sliding-window layer, no compressor runs (nothing to pool).
  * ratio 1 -- one token per group: a plain bf16 projection + norm, no gate, no fp32.
  * ratio 2 -- softmax-gated non-overlapping pool-of-2 in fp32, with ``kv_state``/``score_state``
    carrying a partial trailing group across decode steps.

The latent is returned BEFORE RoPE (the indexer needs the unrotated form; attention rotates
afterwards), or ``None`` while a decode group is still filling. No APE, no overlap -- those are
v4 concepts the v4.1 compressor drops.
"""

import torch
import torch.nn.functional as F
from torch import nn


class _RMSNorm(nn.Module):
    """Matches inference/model.py::RMSNorm exactly (fp32 reduction, weight applied in fp32)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


class CompressorRef(nn.Module):
    """Pools ``compress_ratio`` consecutive tokens into one KV latent with a learned softmax gate.

    Weight names (``norm``, ``wkv``, ``wgate``) match the reference so a reference ``state_dict``
    loads directly. bf16 for ratio 1's plain projection; fp32 for ratio 2's softmax pooling.
    """

    def __init__(self, dim: int, head_dim: int, compress_ratio: int,
                 max_batch_size: int, norm_eps: float = 1e-6):
        super().__init__()
        assert compress_ratio in (1, 2), "ratio 0 runs no compressor; only 1/2 pool"
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.norm = _RMSNorm(head_dim, norm_eps)
        # ratio 1 stays bf16 (plain projection); ratio 2's fp32 pooling promotes its weights to fp32
        wkv_dtype = torch.float32 if compress_ratio > 1 else torch.bfloat16
        self.wkv = nn.Linear(dim, head_dim, bias=False, dtype=wkv_dtype)
        if compress_ratio > 1:
            self.wgate = nn.Linear(dim, head_dim, bias=False, dtype=torch.float32)
            state_shape = (max_batch_size, compress_ratio, head_dim)
            self.register_buffer("kv_state", torch.zeros(state_shape, dtype=torch.float32),
                                 persistent=False)
            self.register_buffer("score_state",
                                 torch.full(state_shape, -torch.inf, dtype=torch.float32),
                                 persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        bsz, seqlen, _ = x.size()
        ratio, dtype = self.compress_ratio, x.dtype
        if ratio == 1:  # one token per group: nothing to pool, so no gate and no fp32
            return self.norm(self.wkv(x))

        x = x.float()
        kv, score = self.wkv(x), self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder:  # trailing partial group waits in the state
                kv, self.kv_state[:bsz, :remainder] = kv.split([cutoff, remainder], dim=1)
                score, self.score_state[:bsz, :remainder] = score.split([cutoff, remainder], dim=1)
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:  # one token per step: fill a slot, pool only when the group just completed
            should_compress = (start_pos + 1) % ratio == 0
            slot = start_pos % ratio
            self.kv_state[:bsz, slot] = kv.squeeze(1)
            self.score_state[:bsz, slot] = score.squeeze(1)
            if should_compress:
                kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(
                    dim=1, keepdim=True)
        if not should_compress:
            return None
        return self.norm(kv.to(dtype))
