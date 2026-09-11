# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Level-one block selection for DeepSeek-V4.1-Flash's two-level sparse index.

The candidate source layer (``candidate_source_layer_id`` == 20) scores every compressed
position with its indexer, groups those positions into fixed ``candidate_block_size`` blocks,
scores each block by its best (amax) position, and keeps the ``candidate_topk_blocks`` highest
blocks per query as a bool mask. Later index-source layers (24/28/32/36) then run their own
indexer but ``masked_fill(~candidates, -inf)`` so their level-two top-k only ever picks positions
inside those surviving blocks.

This is a verbatim port of the reference ``inference/model.py::select_candidate_blocks``; it is a
pure function of its tensors (no module state, no paging), so it is validated bit-exact against the
reference on CPU independent of the paged runtime.
"""

import torch
import torch.nn.functional as F


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Keep the ``topk_blocks`` highest-scoring blocks per query.

    ``logits`` is ``[..., n_positions]`` with positions the query cannot reach already at ``-inf``,
    so a block score of ``-inf`` means "not reachable yet". ``compress_lens`` is a plain int during
    decode, or broadcasts against ``logits``' leading dims during prefill. Returns a bool mask shaped
    like ``logits``.
    """
    width = logits.size(-1)
    # score each block by its best position; -inf pads the last one out to block_size
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    # the block holding this query's newest position is only partly filled, so pin it in: it holds
    # the most recent tokens but could otherwise be outscored by an older, full block
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last, torch.inf
    )

    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    # fewer reachable blocks than topk_blocks means leftover picks came back -inf: drop them
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]
