# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native vision encoder for DeepSeek-V4.1-Flash: ViT (2D-RoPE, bidirectional) + Aligner.

Faithful translation of the reference ``inference/vision.py``. No MoonViT graft — this is the
model's own 32-layer ViT (dim 1024, 16 heads, patch 14) followed by an Aligner that downsamples
3x3 patch blocks and MLP-projects into the LM hidden size. All bf16 ``nn.Linear`` (no quant), so
the encoder is bit-exact against the reference given identical weights.

Submodule attribute names are kept identical to the reference (``patch_embed``, ``blocks``,
``attn.wqkv``/``wo``, ``mlp.w1``/``w2``, ``norm1``/``norm2``/``norm``, Aligner ``w1``/``w2``) so a
reference ``state_dict`` loads directly — used by the bit-exact validation harness and by the
checkpoint loader (the released weights use these names under the ``vision.``/``aligner.`` prefixes).
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn


@lru_cache(8)
def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class VisionRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class PatchEmbed(nn.Module):
    def __init__(self, dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Linear(3 * patch_size**2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class VisionAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wqkv = nn.Linear(dim, 3 * dim)
        self.wo = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (t.view(n, self.n_heads, self.head_dim)
                   for t in self.wqkv(x).chunk(3, dim=-1))
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        return self.wo(o.transpose(0, 1).reshape(n, -1))


class VisionMLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, 2 * inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class VisionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, inter_dim: int):
        super().__init__()
        self.norm1 = VisionRMSNorm(dim)
        self.attn = VisionAttention(dim, n_heads)
        self.norm2 = VisionRMSNorm(dim)
        self.mlp = VisionMLP(dim, inter_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """DeepSeek ViT: full bidirectional attention over one image with 2D RoPE."""

    def __init__(self, dim: int, n_heads: int, inter_dim: int, n_layers: int,
                 patch_size: int, rope_theta: float):
        super().__init__()
        self.rope_dim = dim // n_heads // 2
        self.rope_theta = rope_theta
        self.patch_embed = PatchEmbed(dim, patch_size)
        self.blocks = nn.ModuleList(
            [VisionBlock(dim, n_heads, inter_dim) for _ in range(n_layers)])
        self.norm = VisionRMSNorm(dim)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        # cos/sin are built (and lru-cached) on CPU; move them to the patch device
        # so GPU inference works. Pure device move -- numerically identical.
        cos, sin = cos.to(x.device), sin.to(x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class Aligner(nn.Module):
    def __init__(self, vision_dim: int, out_dim: int, downsample_ratio: int):
        super().__init__()
        self.downsample_ratio = downsample_ratio
        in_dim = vision_dim * downsample_ratio**2
        self.w1 = nn.Linear(in_dim, out_dim)
        self.w2 = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))


# token-type codes (mirror image_processor); duplicated here to avoid a runtime import cycle
_IMAGE_START, _IMAGE, _IMAGE_NEW_LINE, _IMAGE_END = range(4)


def merge_image_span(
    encoder: "DeepseekV41VisionEncoder",
    image_start: torch.Tensor,
    image_end: torch.Tensor,
    image_newline: torch.Tensor,
    patches: torch.Tensor,
    n_vit_h: int,
    n_vit_w: int,
    types: torch.Tensor,
) -> torch.Tensor:
    """Build one image's [span_len, dim] embedding block, matching the reference
    ``merge_image_embeddings`` fill order exactly: IMAGE slots take aligner rows in reading order,
    the three delimiter kinds take their learned embeddings. ``span_len == types.numel()``.

    The reference overwrites a slice of ``h`` in place; here we materialise the same block so the
    caller can scatter it into the LM's placeholder positions (all carry ``image_token_id``)."""
    dim = image_start.numel()
    dtype = image_start.dtype
    device = types.device
    span = torch.empty(types.numel(), dim, dtype=dtype, device=device)
    span[types == _IMAGE_START] = image_start.to(dtype)
    span[types == _IMAGE_END] = image_end.to(dtype)
    span[types == _IMAGE_NEW_LINE] = image_newline.to(dtype)
    embeds = encoder(patches.to(device), n_vit_h, n_vit_w)
    span[types == _IMAGE] = embeds.to(dtype)
    return span


class DeepseekV41VisionEncoder(nn.Module):
    """ViT + Aligner. Runs replicated (weights are tiny); returns [n_downsampled_tokens, out_dim]."""

    def __init__(self, cfg):
        super().__init__()
        ra = cfg.reference_args if hasattr(cfg, "reference_args") else cfg
        self.vit = ViT(
            dim=ra["vision_dim"], n_heads=ra["vision_n_heads"],
            inter_dim=ra["vision_inter_dim"], n_layers=ra["vision_n_layers"],
            patch_size=ra["vision_patch_size"], rope_theta=ra["vision_rope_theta"])
        self.aligner = Aligner(
            vision_dim=ra["vision_dim"], out_dim=ra["dim"],
            downsample_ratio=ra["vision_downsample_ratio"])

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        return self.aligner(self.vit(patches, n_h, n_w), n_h, n_w)
