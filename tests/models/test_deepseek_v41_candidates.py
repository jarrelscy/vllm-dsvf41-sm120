# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the DeepSeek-V4.1-Flash two-level candidate masking.

Covers:
1. bit-exact parity of the ported ``select_candidate_blocks`` against the
   reference implementation (extracted verbatim from the reference
   ``inference/model.py`` when available, with a vendored copy as fallback);
2. the no-op guarantee below candidate_topk_blocks * candidate_block_size
   (= 2048 * 8 = 16384) visible positions: masked level-2 top-k selects the
   same index SET as unmasked;
3. the paged-runtime transforms inserted into sparse_attn_indexer
   (build/apply for prefill chunks and decode batches): per-request block
   alignment via cu_seqlen_ks, per-query-token visible lens via cu_seqlen_ke,
   cross-request isolation, garbage (clean_logits=False) robustness,
   newest-block pin, spec-decode style multi-row requests, and >16K divergence
   with containment inside candidate blocks.

Runs on CPU with only torch installed (candidates.py is imported by file path
if the vllm package is not importable).
"""

import importlib.util
import os
import re
import sys

import torch
import torch.nn.functional as F  # noqa: F401  (used by exec'd reference code)

# ---------------------------------------------------------------------------
# Import the module under test without requiring the full vllm package.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_CAND_PATH = os.path.join(
    _HERE, "..", "..", "vllm", "models", "deepseek_v41", "candidates.py"
)

try:
    from vllm.models.deepseek_v41 import candidates as C  # type: ignore
except Exception:
    _spec = importlib.util.spec_from_file_location(
        "dsv41_candidates", os.path.normpath(_CAND_PATH)
    )
    C = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(C)

REFERENCE_MODEL_PY = "/data/huggingface/dsv41-work/inference/model.py"

# Vendored copy of the reference select_candidate_blocks (inference/model.py),
# used only if the reference tree is not present on the test machine.
_VENDORED_REFERENCE = '''
def select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last, torch.inf
    )
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]
'''


def _load_reference_select():
    """Extract the reference select_candidate_blocks verbatim from the
    reference model.py (preferred), else use the vendored copy."""
    src = None
    if os.path.exists(REFERENCE_MODEL_PY):
        text = open(REFERENCE_MODEL_PY).read()
        m = re.search(
            r"^def select_candidate_blocks\(.*?(?=^(?:class |def |@))",
            text,
            re.M | re.S,
        )
        if m:
            src = m.group(0)
    if src is None:
        src = _VENDORED_REFERENCE
        print("  [reference model.py not found -> vendored oracle]")
    else:
        print("  [oracle extracted from reference model.py]")
    ns = {"torch": torch, "F": F}
    exec(src, ns)
    return ns["select_candidate_blocks"]


REF_SELECT = _load_reference_select()

TOPK_BLOCKS = 2048
BLOCK_SIZE = 8
INDEX_TOPK = 512
THRESHOLD = TOPK_BLOCKS * BLOCK_SIZE  # 16384


def _prefill_style_logits(bsz, seqlen, n, ratio=1, seed=0):
    """Reference-prefill-shaped logits [bsz, seqlen, n] with unreachable
    positions already at -inf, plus the per-query compress_lens tensor."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(bsz, seqlen, n, generator=g)
    compress_lens = (torch.arange(1, seqlen + 1) // ratio).unsqueeze(-1)
    invisible = torch.arange(n) >= compress_lens.clamp(max=n)
    logits = logits.masked_fill(invisible.unsqueeze(0), -torch.inf)
    return logits, compress_lens


def test_reference_parity():
    """Port vs reference select_candidate_blocks, bit-exact (torch.equal)."""
    cases = []
    # decode style: [1, n] logits, int compress_lens == n
    for n in [1, 7, 8, 9, 100, 16384, 16385, 16391, 16392, 20000, 40001]:
        cases.append(("decode", n))
    for tag, n in cases:
        g = torch.Generator().manual_seed(n)
        logits = torch.randn(1, n, generator=g)
        got = C.select_candidate_blocks(logits, n, TOPK_BLOCKS, BLOCK_SIZE)
        want = REF_SELECT(logits, n, TOPK_BLOCKS, BLOCK_SIZE)
        assert torch.equal(got, want), f"{tag} n={n}"
        assert got.shape == logits.shape
    # prefill style: [bsz, seqlen, n] with tensor compress_lens
    for seqlen, n in [(5, 5), (64, 64), (300, 300), (17000, 17000)]:
        logits, lens = _prefill_style_logits(2, seqlen, n, seed=seqlen)
        got = C.select_candidate_blocks(logits, lens, TOPK_BLOCKS, BLOCK_SIZE)
        want = REF_SELECT(logits, lens, TOPK_BLOCKS, BLOCK_SIZE)
        assert torch.equal(got, want), f"prefill seqlen={seqlen}"
    # small topk_blocks / odd block sizes to hit block-boundary edges
    for tk, bs, n in [(3, 8, 65), (3, 8, 64), (1, 8, 17), (4, 16, 130)]:
        g = torch.Generator().manual_seed(tk * 1000 + n)
        logits = torch.randn(2, n, generator=g)
        got = C.select_candidate_blocks(logits, n, tk, bs)
        want = REF_SELECT(logits, n, tk, bs)
        assert torch.equal(got, want), f"tk={tk} bs={bs} n={n}"
    print("test_reference_parity PASS")


def test_newest_block_pin():
    """The block holding the newest position is kept even when it scores
    worst (torch.inf pin)."""
    n = 20000
    logits = torch.full((1, n), 5.0)
    lens = 16389  # newest block = block 2048 (positions 16384..16391), partial
    last_block = (lens - 1) // BLOCK_SIZE
    # newest block scores far below everything else
    logits[0, last_block * BLOCK_SIZE : lens] = -1000.0
    logits[0, lens:] = -torch.inf
    mask = C.select_candidate_blocks(logits, lens, TOPK_BLOCKS, BLOCK_SIZE)
    assert mask[0, last_block * BLOCK_SIZE : (last_block + 1) * BLOCK_SIZE].all()
    # and keeps exactly topk_blocks reachable blocks
    keep_blocks = mask[0].unflatten(-1, (-1, BLOCK_SIZE)).any(-1)
    assert keep_blocks.sum().item() == TOPK_BLOCKS
    print("test_newest_block_pin PASS")


def _topk_set(row, k):
    return set(torch.topk(row, min(k, row.numel())).indices.tolist())


def test_mask_noop_below_threshold():
    """ctx <= 16384: masked level-2 top-512 == unmasked top-512 (index sets)."""
    for lens in [1, 100, 511, 512, 513, 8192, 16383, 16384]:
        g = torch.Generator().manual_seed(lens)
        logits = torch.randn(1, lens, generator=g)
        mask = C.select_candidate_blocks(logits, lens, TOPK_BLOCKS, BLOCK_SIZE)
        assert mask.all(), f"lens={lens}: mask must be all-true at/below 16384"
        masked = logits.masked_fill(~mask, -torch.inf)
        assert _topk_set(masked[0], INDEX_TOPK) == _topk_set(logits[0], INDEX_TOPK)
    print("test_mask_noop_below_threshold PASS")


def test_mask_diverges_above_threshold():
    """ctx > 16384: the level-1 mask comes from the SOURCE layer's logits and
    is applied to a CONSUMER layer's independent logits (that is the two-level
    scheme); the consumer's masked top-512 then differs from its unmasked
    top-512, and every selected index lies inside a candidate block.

    (Self-masking a single layer's logits is a near-no-op for iid scores --
    blocks holding the global top positions have top block-amax -- which is
    why the served bug only corrupts once consumer layers rank positions
    differently from layer 20.)"""
    lens = THRESHOLD + 4096
    g = torch.Generator().manual_seed(7)
    src_logits = torch.randn(1, lens, generator=g)
    mask = C.select_candidate_blocks(src_logits, lens, TOPK_BLOCKS, BLOCK_SIZE)
    assert not mask.all() and mask.any()
    # kept = topk_blocks blocks exactly (all reachable here)
    assert int(mask[0].sum()) == THRESHOLD

    con_logits = torch.randn(1, lens, generator=g)  # independent consumer scores
    masked = con_logits.masked_fill(~mask, -torch.inf)
    sel_masked = _topk_set(masked[0], INDEX_TOPK)
    sel_unmasked = _topk_set(con_logits[0], INDEX_TOPK)
    assert all(mask[0, i] for i in sel_masked)
    # ~20% of blocks are dropped; 512 independent picks make divergence certain
    assert sel_masked != sel_unmasked
    # self-masking sanity: the source's own top-512 is preserved by its mask
    self_masked = src_logits.masked_fill(~mask, -torch.inf)
    assert _topk_set(self_masked[0], INDEX_TOPK) == _topk_set(
        src_logits[0], INDEX_TOPK
    )
    print("test_mask_diverges_above_threshold PASS")


# ---------------------------------------------------------------------------
# Paged-runtime transforms (the exact tensor code inserted in
# sparse_attn_indexer around the fp32 logits).
# ---------------------------------------------------------------------------


def _garbage_fill(shape, gen):
    """clean_logits=False leaves garbage outside each row's [ks, ke): include
    NaN and huge values to prove the visibility masking neutralizes them."""
    x = torch.randn(*shape, generator=gen) * 1e30
    x[..., ::13] = torch.nan
    x[..., 5::17] = torch.inf
    return x


def _reference_row_mask(row_logits, length, topk_blocks, block_size):
    """Oracle for one query row over its request-local width: reference
    select_candidate_blocks on [-inf beyond length] logits."""
    width = row_logits.numel()
    vis = row_logits.masked_fill(torch.arange(width) >= length, -torch.inf)
    return REF_SELECT(vis.unsqueeze(0), int(length), topk_blocks, block_size)[0]


def test_prefill_build_apply():
    """Synthetic packed prefill chunk: 3 requests (short, >16K chunked-prefill
    continuation, mid), garbage outside visible ranges, cross-request
    isolation, oracle parity per row, containment + divergence at >16K."""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    tk, bs = 64, 8  # scaled-down topk_blocks to keep the test fast
    threshold = tk * bs  # 512

    # (num_query_rows, per-request total kv len, first query token's ctx len)
    reqs = [
        (3, 40, 38),  # short request, fully below threshold
        (4, 900, 897),  # chunked-prefill continuation crossing blocks, >16K-analog
        (2, 300, 299),  # mid request
    ]
    ks_list, ke_rows, ks_rows = [], [], []
    off = 0
    for q, kv, ctx0 in reqs:
        ks_list.append(off)
        for i in range(q):
            ks_rows.append(off)
            ke_rows.append(off + ctx0 + i)
        off += kv
    W = off
    rows = len(ks_rows)
    cu_ks = torch.tensor(ks_rows, dtype=torch.int32)
    cu_ke = torch.tensor(ke_rows, dtype=torch.int32)

    src_logits = _garbage_fill((rows, W), g)
    # visible regions get sane values
    for r in range(rows):
        ks, ke = ks_rows[r], ke_rows[r]
        src_logits[r, ks:ke] = torch.randn(ke - ks, generator=g)

    groups = C.build_prefill_chunk_candidates(src_logits, cu_ks, cu_ke, tk, bs)
    assert len(groups) == len(reqs)
    assert [gr[2] for gr in groups] == ks_list

    # oracle parity per row (keep expanded == reference mask on visible logits)
    for (r0, r1, ks, keep), (q, kv, ctx0) in zip(groups, reqs):
        assert keep is not None and keep.shape[0] == r1 - r0
        gw = keep.shape[1] * bs
        for i, r in enumerate(range(r0, r1)):
            length = ke_rows[r] - ks
            row = src_logits[r, ks : ks + gw]
            row = F.pad(row, (0, gw - row.numel()), value=-torch.inf)
            want = _reference_row_mask(row, length, tk, bs)
            got = keep[i].repeat_interleave(bs)[: want.numel()]
            assert torch.equal(got, want), f"req rows {r0}:{r1} row {r}"

    # consumer apply: different logits, same layout
    con_logits = _garbage_fill((rows, W), g)
    for r in range(rows):
        ks, ke = ks_rows[r], ke_rows[r]
        con_logits[r, ks:ke] = torch.randn(ke - ks, generator=g)
    before = con_logits.clone()
    C.apply_prefill_chunk_candidates(con_logits, groups, bs)

    # NaN-aware change detection (NaN != NaN would be a false positive)
    changed = (con_logits != before) & ~(
        torch.isnan(con_logits) & torch.isnan(before)
    )
    for (r0, r1, ks, keep), (q, kv, ctx0) in zip(groups, reqs):
        span = keep.shape[1] * bs
        # cross-request isolation: nothing outside [r0:r1) x [ks, ks+span) moved
        outside = changed.clone()
        outside[r0:r1, ks : ks + span] = False
        assert not outside[r0:r1].any(), "columns outside own request changed"
        # the mask's last block may be padded past the packed buffer's edge
        span_eff = min(span, W - ks)
        for i, r in enumerate(range(r0, r1)):
            length = ke_rows[r] - ks
            kept = keep[i].repeat_interleave(bs)[:span_eff]
            seg = con_logits[r, ks : ks + span_eff]
            # everything outside kept blocks is -inf; kept visible cols untouched
            assert (seg[~kept] == -torch.inf).all()
            vis_kept = kept.clone()
            vis_kept[length:] = False
            assert torch.equal(
                seg[vis_kept], before[r, ks : ks + span_eff][vis_kept]
            )
            # level-2 selection: sets equal below threshold, contained+different above
            vis = seg[:length]
            vis_before = before[r, ks : ks + length]
            sel_m = _topk_set(vis, 32)
            sel_u = _topk_set(vis_before, 32)
            if length <= threshold:
                assert sel_m == sel_u, f"row {r}: <=thresh must be identical"
            else:
                assert all(kept[j] for j in sel_m)
                assert sel_m != sel_u, f"row {r}: >thresh should diverge"
    # rows outside every group span but inside other requests' columns
    all_rows_changed = changed.clone()
    for r0, r1, ks, keep in groups:
        all_rows_changed[r0:r1, ks : ks + keep.shape[1] * bs] = False
    assert not all_rows_changed.any()
    print("test_prefill_build_apply PASS")


def test_decode_build_apply():
    """Decode/verify batch: (B, next_n) lens incl. rows above/below threshold
    and a zero-len padded row; oracle parity; containment + divergence."""
    g = torch.Generator().manual_seed(3)
    tk, bs = 64, 8
    threshold = tk * bs
    max_model_len = 2048
    # rows: [b0 spec j0, b0 spec j1, b1 plain, b2 padded]
    lens_row = torch.tensor([900, 901, 300, 0], dtype=torch.int32)
    max_seq_len = 901
    rows = lens_row.numel()
    logits = _garbage_fill((rows, max_model_len), g)
    for r in range(rows):
        n = int(lens_row[r])
        logits[r, :n] = torch.randn(n, generator=g)

    keep = C.build_decode_candidates(logits, lens_row, max_seq_len, tk, bs)
    width = keep.shape[1] * bs
    assert width >= max_seq_len and width <= max_model_len

    for r in range(rows):
        n = int(lens_row[r])
        if n == 0:
            assert not keep[r].any(), "padded row must keep nothing"
            continue
        want = _reference_row_mask(logits[r, :width], n, tk, bs)
        got = keep[r].repeat_interleave(bs)[:width]
        assert torch.equal(got, want), f"decode row {r}"

    con = _garbage_fill((rows, max_model_len), g)
    for r in range(rows):
        n = int(lens_row[r])
        con[r, :n] = torch.randn(n, generator=g)
    before = con.clone()
    C.apply_decode_candidates(con, keep, bs)

    for r in range(rows):
        n = int(lens_row[r])
        kept = keep[r].repeat_interleave(bs)[:width]
        assert (con[r, :width][~kept] == -torch.inf).all()
        if n == 0:
            continue
        vis_kept = kept.clone()
        vis_kept[n:] = False
        assert torch.equal(con[r, :width][vis_kept], before[r, :width][vis_kept])
        sel_m = _topk_set(con[r, :n], 32)
        sel_u = _topk_set(before[r, :n], 32)
        if n <= threshold:
            assert sel_m == sel_u
        else:
            assert all(kept[j] for j in sel_m)
            assert sel_m != sel_u
    # columns beyond the scored width are untouched (invisible to top-k)
    assert torch.equal(con[:, width:], before[:, width:]) or bool(
        (torch.isnan(con[:, width:]) == torch.isnan(before[:, width:])).all()
        and (con[:, width:][~torch.isnan(con[:, width:])]
             == before[:, width:][~torch.isnan(before[:, width:])]).all()
    )
    print("test_decode_build_apply PASS")


def test_store_and_kill_switch():
    """CandidateMaskStore tags per metadata object; env kill switch works."""

    class Meta:  # stand-in for DeepseekV32IndexerMetadata
        pass

    store = C.CandidateMaskStore()
    m1, m2 = Meta(), Meta()
    assert not store.matches(m1)
    store.begin(m1)
    store.prefill[0] = []
    assert store.matches(m1) and not store.matches(m2)
    store.begin(m2)
    assert store.matches(m2) and store.prefill == {} and store.decode is None

    old = os.environ.get("VLLM_DSV41_CANDIDATES")
    try:
        os.environ.pop("VLLM_DSV41_CANDIDATES", None)
        assert C.candidates_enabled()
        os.environ["VLLM_DSV41_CANDIDATES"] = "1"
        assert C.candidates_enabled()
        os.environ["VLLM_DSV41_CANDIDATES"] = "0"
        assert not C.candidates_enabled()
    finally:
        if old is None:
            os.environ.pop("VLLM_DSV41_CANDIDATES", None)
        else:
            os.environ["VLLM_DSV41_CANDIDATES"] = old
    print("test_store_and_kill_switch PASS")


def test_full_scale_decode_16k_boundary():
    """Full-scale (2048x8) decode boundary: 16384 -> all-true, 16385+ -> mask
    engages; masked top-512 stays identical at the boundary."""
    g = torch.Generator().manual_seed(11)
    max_model_len = 24576
    lens_row = torch.tensor([16384, 16385, 20000], dtype=torch.int32)
    logits = _garbage_fill((3, max_model_len), g)
    for r in range(3):
        n = int(lens_row[r])
        logits[r, :n] = torch.randn(n, generator=g)
    keep = C.build_decode_candidates(
        logits, lens_row, 20000, TOPK_BLOCKS, BLOCK_SIZE
    )
    width = keep.shape[1] * BLOCK_SIZE
    # row 0: every reachable block kept
    kept0 = keep[0].repeat_interleave(BLOCK_SIZE)[:width]
    assert kept0[:16384].all()
    con = logits.clone()
    C.apply_decode_candidates(con, keep, BLOCK_SIZE)
    assert _topk_set(con[0, :16384], INDEX_TOPK) == _topk_set(
        logits[0, :16384], INDEX_TOPK
    )
    # rows 1, 2: mask engages but level-2 top-512 always has candidates
    for r in (1, 2):
        n = int(lens_row[r])
        kr = keep[r].repeat_interleave(BLOCK_SIZE)[:width]
        assert not kr[:n].all()
        assert int(kr[:n].sum()) >= INDEX_TOPK
        sel = _topk_set(con[r, :n], INDEX_TOPK)
        assert all(kr[j] for j in sel)
    print("test_full_scale_decode_16k_boundary PASS")


if __name__ == "__main__":
    test_reference_parity()
    test_newest_block_pin()
    test_mask_noop_below_threshold()
    test_mask_diverges_above_threshold()
    test_prefill_build_apply()
    test_decode_build_apply()
    test_store_and_kill_switch()
    test_full_scale_decode_16k_boundary()
    print("ALL CANDIDATE TESTS PASS")
    sys.exit(0)
