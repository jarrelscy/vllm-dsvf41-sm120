# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference-faithful, pure-torch DeepSeek-V4.1-Flash MoE gate (CPU oracle).

Mirrors ``inference/model.py::Gate`` exactly so it can be validated bit-exact on CPU and used as
the reference in the eventual GPU golden-diff. The released config routes with
``scoring_func='sqrtsoftplus'`` (``F.softplus(scores).sqrt()``), 384 routed experts, top-6, plain
GLOBAL top-k (no group-limited routing), correction bias steering selection only, and
``routed_scaling_factor`` (== ``route_scale``) 1.5.

Two subtleties the fused vLLM path must preserve:
  * the correction bias picks experts but does NOT scale them -- the routing weights are gathered
    from the UNBIASED scores, then optionally renormalized (``+1e-20``, not ``norm_eps``) and scaled.
  * image-span tokens use a separate ``bias_vl`` (training ``noaux_tc_for_vl``); non-image tokens
    use ``bias``.
"""

import torch
import torch.nn.functional as F
from torch import nn


class GateRef(nn.Module):
    def __init__(self, dim: int, n_routed_experts: int, topk: int,
                 score_func: str = "sqrtsoftplus", gate_temp: float = 1.0,
                 norm_topk_prob: bool = True, route_scale: float = 1.0,
                 vision_enabled: bool = True):
        super().__init__()
        self.dim = dim
        self.topk = topk
        self.score_func = score_func
        self.gate_temp = gate_temp
        self.norm_topk_prob = norm_topk_prob
        self.route_scale = route_scale
        self.weight = nn.Parameter(torch.empty(n_routed_experts, dim))
        self.bias = nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32))
        self.bias_vl = (nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32))
                        if vision_enabled else None)

    def forward(self, x: torch.Tensor,
                image_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float()) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        bias = self.bias
        if image_mask is not None and self.bias_vl is not None:
            bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
        # the bias picks experts but does not scale them: weights come from the raw scores
        indices = (scores + bias).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights /= weights.sum(dim=-1, keepdim=True) + 1e-20  # not norm_eps, matches training
        weights *= self.route_scale
        return weights, indices
