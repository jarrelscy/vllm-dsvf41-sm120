# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Level-one block selection for DeepSeek-V4.1-Flash's two-level sparse index.

The candidate source layer (``candidate_source_layer_id`` == 20) scores every compressed
position with its indexer, groups those positions into fixed ``candidate_block_size`` blocks,
scores each block by its best (amax) position, and keeps the ``candidate_topk_blocks`` highest
blocks per query as a bool mask. Later index-source layers (24/28/32/36) then run their own
indexer but ``masked_fill(~candidates, -inf)`` so their level-two top-k only ever picks positions
inside those surviving blocks.

``select_candidate_blocks`` is a verbatim port of the reference
``inference/model.py::select_candidate_blocks``; it is a pure function of its tensors (no module
state, no paging), so it is validated bit-exact against the reference on CPU independent of the
paged runtime.

The rest of this module is the paged-runtime wiring used by
``vllm.model_executor.layers.sparse_attn_indexer.sparse_attn_indexer`` (the op that owns the
fp32 indexer logits in both the prefill and decode paths):

* the SOURCE layer's op call computes per-request block-keep masks from its logits
  (``build_prefill_chunk_candidates`` / ``build_decode_candidates``) and publishes them into the
  process-wide ``CANDIDATE_STORE``;
* CONSUMER layers' op calls re-read the store (same forward, layers run in order, the op is an
  eager cudagraph break) and ``masked_fill`` their own logits with ``-inf`` outside the kept
  blocks (``apply_prefill_chunk_candidates`` / ``apply_decode_candidates``) BEFORE their level-2
  top-k kernels run.

Masks are stored at BLOCK granularity (``[rows, num_blocks]`` bool) and expanded lazily through
strided views at apply time, so no ``[tokens, n_positions]`` bool mask is ever materialized.

TP note: the port's indexer is fully replicated (``ReplicatedLinear`` wq_b / weights_proj over
all 64 index heads, replicated q-lora input), unlike the reference which shards index heads and
``all_reduce``s the score. Every rank therefore computes bit-identical logits and derives the
identical candidate mask locally -- no communication is needed and ranks cannot diverge. DCP
(sharded KV) WOULD shard the logit columns, so candidates are explicitly rejected under
``dcp_world_size > 1``.

Kill switch: ``VLLM_DSV41_CANDIDATES=0`` disables both publish and consume (read per call, so it
can be flipped for A/B without code changes; default on).
"""

import os
import weakref

import torch
import torch.nn.functional as F

# Roles threaded through the sparse_attn_indexer custom op (ints, so they are
# schema-friendly op arguments).
CANDIDATE_ROLE_NONE = 0
CANDIDATE_ROLE_SOURCE = 1
CANDIDATE_ROLE_CONSUMER = 2


def candidates_enabled() -> bool:
    """Runtime kill switch (default ON)."""
    return os.environ.get("VLLM_DSV41_CANDIDATES", "1") not in ("0", "false", "False")


def full_decode_static_width() -> bool:
    """Decode-only FULL cudagraphs (VLLM_DSV41_FULL_DECODE=1): score decode
    candidates over the static full logits width so no python-int seqlen bakes a
    stale mask shape into the captured graph. Read per op call (capture-time
    under FULL); constant across a run so it never diverges capture vs replay."""
    return os.environ.get("VLLM_DSV41_FULL_DECODE", "0") == "1"


def candidate_keep_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Block-granular level-one selection: bool keep mask ``[..., num_blocks]``.

    Exactly the block-scoring half of the reference ``select_candidate_blocks``:
    score each ``block_size`` block of positions by its best (amax) logit (with
    ``-inf`` meaning "not reachable yet"), pin the block holding the newest
    position, and keep the ``topk_blocks`` best per query.
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
    return keep


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
    keep = candidate_keep_blocks(logits, compress_lens, topk_blocks, block_size)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


# --------------------------------------------------------------------------
# Paged-runtime plumbing (used by sparse_attn_indexer).
# --------------------------------------------------------------------------

# One prefill-chunk group: (row_start, row_end, kv_col_start, keep_blocks) with
# keep_blocks ``[row_end - row_start, num_blocks]`` bool (None when the group has
# no reachable KV in this chunk). Rows are the op's token-major logit rows;
# kv_col_start is the request's cu_seqlen_ks (constant per request), so block b
# of the mask covers packed logit columns
# [kv_col_start + b*block_size, kv_col_start + (b+1)*block_size).
PrefillGroup = tuple[int, int, int, torch.Tensor | None]


class CandidateMaskStore:
    """Per-forward candidate masks published by the source layer's op call.

    The source (layer 20) runs strictly before every consumer (24/28/32/36)
    within one model forward, and sparse_attn_indexer executes eagerly (it is a
    cudagraph break), so a process-global slot that is re-begun by the source
    each forward and read by consumers in the same forward is race-free. The
    store is tagged with the indexer metadata object (all five layers share the
    SAME ``DeepseekV32IndexerMetadata`` instance -- their k_cache resolves to
    layer 20's cache prefix), and consumers hard-fail on a tag mismatch instead
    of silently using a stale mask.
    """

    def __init__(self) -> None:
        self._meta_ref: weakref.ref | None = None
        self.prefill: dict[int, list[PrefillGroup]] = {}
        self.decode: torch.Tensor | None = None

    def begin(self, metadata: object) -> None:
        """Source layer: start a fresh forward keyed to this step's metadata."""
        self._meta_ref = weakref.ref(metadata)
        self.prefill = {}
        self.decode = None

    def matches(self, metadata: object) -> bool:
        return self._meta_ref is not None and self._meta_ref() is metadata


CANDIDATE_STORE = CandidateMaskStore()


def _mask_unreachable(
    logits_slice: torch.Tensor, lens: torch.Tensor
) -> torch.Tensor:
    """Return a copy of ``logits_slice`` [rows, width] with per-row columns
    ``>= lens[row]`` set to -inf. The logits kernels run with
    ``clean_logits=False``, so out-of-range columns hold garbage and MUST be
    neutralized before block amax scoring (reference logits arrive with -inf
    already in place; the paged kernels' don't)."""
    width = logits_slice.size(-1)
    visible = (
        torch.arange(width, device=logits_slice.device).unsqueeze(0)
        < lens.unsqueeze(-1)
    )
    return logits_slice.masked_fill(~visible, -torch.inf)


def build_prefill_chunk_candidates(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> list[PrefillGroup]:
    """Source layer, one prefill chunk: block-keep masks per request group.

    ``logits`` is the chunk's ``[rows, packed_kv_width]`` fp32 score matrix;
    row i's visible columns are ``[cu_seqlen_ks[i], cu_seqlen_ke[i])`` with
    ``cu_seqlen_ks`` constant per request (start of that request's gathered KV,
    i.e. its position 0) and ``cu_seqlen_ke - cu_seqlen_ks`` the per-query-token
    visible compressed length ((pos+1)//ratio -- chunked-prefill continuations
    included, the metadata kernel bakes in start_pos). Rows of one request are
    contiguous, so grouping by ks gives per-request views whose local column 0
    is the request's position 0 -- the reference block alignment.
    """
    ks_cpu = cu_seqlen_ks.tolist()
    ke_cpu = cu_seqlen_ke.tolist()
    rows = len(ks_cpu)
    groups: list[PrefillGroup] = []
    r0 = 0
    for r in range(1, rows + 1):
        if r < rows and ks_cpu[r] == ks_cpu[r0]:
            continue
        ks = ks_cpu[r0]
        # ke is nondecreasing within a request (later query tokens see more),
        # but take the max to stay robust to any row ordering.
        width = max(ke_cpu[r0:r]) - ks
        if width <= 0:
            groups.append((r0, r, ks, None))
        else:
            lens = cu_seqlen_ke[r0:r] - ks  # [g] on-device
            masked = _mask_unreachable(logits[r0:r, ks : ks + width], lens)
            keep = candidate_keep_blocks(
                masked, lens.unsqueeze(-1), topk_blocks, block_size
            )
            groups.append((r0, r, ks, keep))
        r0 = r
    return groups


def _apply_keep_blocks(
    logits_rows: torch.Tensor,
    col_start: int,
    keep: torch.Tensor,
    block_size: int,
) -> None:
    """masked_fill_(-inf) columns of ``logits_rows`` [rows, W] outside kept
    blocks. Block b covers columns [col_start + b*bs, col_start + (b+1)*bs);
    the expansion is a strided (unflatten) view, no full-width bool mask is
    materialized. Columns past the last (possibly partial) block are left
    untouched -- they are beyond every row's reachable length and the top-k
    kernels never read them."""
    num_blocks = keep.size(-1)
    width = logits_rows.size(-1)
    avail = min(num_blocks * block_size, width - col_start)
    if avail <= 0:
        return
    nb_full = avail // block_size
    if nb_full > 0:
        view = logits_rows[:, col_start : col_start + nb_full * block_size].unflatten(
            -1, (nb_full, block_size)
        )
        view.masked_fill_(~keep[:, :nb_full].unsqueeze(-1), -torch.inf)
    tail = avail - nb_full * block_size
    if tail > 0:
        logits_rows[
            :, col_start + nb_full * block_size : col_start + avail
        ].masked_fill_(~keep[:, nb_full : nb_full + 1], -torch.inf)


def apply_prefill_chunk_candidates(
    logits: torch.Tensor,
    groups: list[PrefillGroup],
    block_size: int,
) -> None:
    """Consumer layer, one prefill chunk: -inf outside candidate blocks, in
    place on this layer's logits, before its level-2 top-k."""
    for r0, r1, ks, keep in groups:
        if keep is None:
            continue
        assert keep.size(0) == r1 - r0
        _apply_keep_blocks(logits[r0:r1], ks, keep, block_size)


def build_decode_candidates(
    logits: torch.Tensor,
    lens_row: torch.Tensor,
    max_seq_len: int,
    topk_blocks: int,
    block_size: int,
    static_width: bool = False,
) -> torch.Tensor:
    """Source layer, decode/verify batch: block-keep masks ``[rows, nb]``.

    ``logits`` is ``[rows, max_model_len]`` from the paged-MQA kernel; column j
    IS request-local compressed position j (per-row block table), so block
    alignment needs no offset. ``lens_row`` is the per-row visible length
    (``seq_lens`` flattened token-major: (B, next_n) rows map to row b*next_n+j,
    covering plain decode, native spec-decode verify, and the flattened
    multi-token path alike). Only the first round_up(max_seq_len) columns are
    scored -- everything past that is invisible to every row.

    ``static_width`` (decode-only FULL cudagraphs, VLLM_DSV41_FULL_DECODE=1):
    score the FULL ``logits.size(-1)`` width instead of trimming to this step's
    ``max_seq_len``. ``max_seq_len`` is a python int that grows every decode
    step, so trimming would bake a stale mask shape into the captured graph.
    The result is bit-identical -- ``_mask_unreachable`` sets every column
    ``>= lens_row`` to -inf using the DEVICE ``lens_row`` tensor, so the extra
    scored blocks come back -inf and are dropped by the ``> -inf`` topk filter.
    Costs more compute at short context (why it is opt-in), free at long.
    """
    rows = logits.size(0)
    assert lens_row.numel() == rows, (
        f"candidate lens rows {lens_row.numel()} != logit rows {rows}"
    )
    if static_width:
        width = logits.size(-1)
    else:
        width = min(
            -(-max(max_seq_len, 1) // block_size) * block_size, logits.size(-1)
        )
    masked = _mask_unreachable(logits[:, :width], lens_row.reshape(-1))
    return candidate_keep_blocks(
        masked, lens_row.reshape(-1, 1), topk_blocks, block_size
    )


def apply_decode_candidates(
    logits: torch.Tensor,
    keep: torch.Tensor,
    block_size: int,
) -> None:
    """Consumer layer, decode/verify batch: -inf outside candidate blocks, in
    place, before the level-2 top-k kernels."""
    assert keep.size(0) == logits.size(0), (
        f"candidate mask rows {keep.size(0)} != logit rows {logits.size(0)}"
    )
    _apply_keep_blocks(logits, 0, keep, block_size)
