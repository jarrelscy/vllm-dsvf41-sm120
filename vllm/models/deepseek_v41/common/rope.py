# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepseekV41 rotary embedding initialization."""

import torch

from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding


def build_deepseek_v4_rope(
    config,
    *,
    head_dim: int,
    rope_head_dim: int,
    max_position_embeddings: int,
    compress_ratio: int,
) -> RotaryEmbedding:
    # Copy so each layer's build does not mutate the config's shared dict for
    # the next layer (compress_ratio differs per layer in v41).
    rope_parameters = dict(config.rope_parameters)
    # v41: any layer that compresses (ratio 1 or 2) uses the long-context
    # compress_rope_theta + YaRN; only pure sliding-window layers (ratio 0)
    # fall back to base rope_theta with YaRN disabled.
    if compress_ratio > 0:
        rope_parameters["rope_theta"] = config.compress_rope_theta
        if rope_parameters["rope_type"] != "default":
            rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
    else:
        rope_parameters["rope_theta"] = config.rope_theta
        rope_parameters["rope_type"] = "default"  # YaRN off for pure SWA
    rope_parameters["mscale"] = 0  # Disable mscale
    rope_parameters["mscale_all_dim"] = 0  # Disable mscale
    rope_parameters["is_deepseek_v41"] = True
    rope_parameters["rope_dim"] = rope_head_dim
    # The v41 fused attention kernels (inv-RoPE fp8 o_proj, qnorm/rope KV
    # inserts) read cos_sin_cache directly and require float32. The compressed
    # layers already get float32 via the deepseek_yarn subclass, but the
    # pure sliding-window layers (compress_ratio == 0) fall to the "default"
    # rope, which downcasts the float32 cache to the compute dtype (bf16).
    # Pin float32 so every layer's cache satisfies the kernel contract; the
    # cache is computed in float32 upstream, so this is lossless.
    return get_rope(
        head_dim,
        max_position=max_position_embeddings,
        rope_parameters=rope_parameters,
        is_neox_style=False,
        dtype=torch.float32,
    )
