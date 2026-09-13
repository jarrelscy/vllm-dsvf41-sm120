# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram: n-gram hash memory for DeepSeek-V4.1-Flash.

The engram adds two n-gram-hash-indexed embedding lookups into the residual stream at
layers ``engram_layer_ids`` (``[1, 14]``). Each table has ~384M rows of ``head_dim=256``
fp8 values; at ~98 GB each they dominate the checkpoint (~196 GB, larger than the KV or
any weight group). They are accessed sparsely by hash id, so the fit lever at TP4 is to
keep the tables in **host DRAM** and move only the gathered ``[.., n_hash_cols, head_dim]``
rows across PCIe per token (``VLLM_DSV41_ENGRAM_DEVICE=cpu``, default). This mirrors the
lossless engram-on-CPU edit proven in Phase 1 with the reference impl.

Math is a faithful translation of the reference ``engram.py`` + ``model.py`` Engram module:
- ``NgramHashState`` computes, per position, the hash ids of the ``max_ngram_size-1`` n-grams
  ending there (int64 XOR-rolling hash into disjoint prime-sized buckets). Pure integer ops,
  no quantization — validated bit-exact against the reference.
- ``ParallelEngramEmbedding`` gathers + dequantizes fp8 rows (row-sharded over TP).
- ``Engram`` projects the gathered rows into a key/value and gates them against the residual
  stream (signed-sqrt → sigmoid dot), writing into all ``hc_mult`` residual copies.

Serving note: the reference ``NgramHashState`` keeps a per-batch ring cache keyed by
``start_pos``. Under vLLM continuous batching the hash ids are a pure function of a request's
token history, so the serving runner supplies per-request token windows; the hash arithmetic
below is unchanged. Because the CPU gather + H2D copy is a host sync, the engram runs in the
eager / piecewise region, never inside a captured FULL cudagraph (same discipline as the v4
indexer's python-metadata path).

Cudagraph mode (``VLLM_DSV41_ENGRAM_ALLOW_CUDAGRAPH=1``, V2 runner only): ALL host-dependent
work (hash + CPU mmap row gather + dequant + TP all-reduce) is hoisted out of the model
forward into the runner's ``prepare_inputs`` (see :func:`precompute_engram_gathered_rows`),
which writes the gathered bf16 rows into a persistent GPU buffer. The captured forward then
consumes the prefilled buffer via :meth:`Engram.forward_gathered` -- pure tensor ops (fp8 wkv
projection + gate), all capturable under PIECEWISE graphs. The math is IDENTICAL to the
in-forward path: ``Engram.forward`` is literally ``forward_gathered(x, embed(ids).flatten(-2))``,
and the precompute calls the very same ``hasher``/``embed`` modules on the very same windows.
"""

import json
import mmap
import os
import struct
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)


def dsv41_engram_runtime_enabled(config) -> bool:
    """Whether the serving engram is on: the checkpoint has engram layers AND the
    ``VLLM_DSV41_LOAD_ENGRAM`` kill-switch (default ON) has not been set to 0.

    Used by BOTH module construction and the weight-loader skip list so they can
    never disagree (a module with skipped tensors would serve uninitialized
    garbage; loaded tensors without a module would crash the loader)."""
    return bool(getattr(config, "engram_layer_ids", None)) and (
        os.environ.get("VLLM_DSV41_LOAD_ENGRAM", "1") == "1"
    )


def dsv41_engram_cudagraph_enabled(config) -> bool:
    """Whether the cudagraph-compatible engram path is on: the engram itself is
    on AND ``VLLM_DSV41_ENGRAM_ALLOW_CUDAGRAPH=1`` (default OFF). When enabled,
    the runner precomputes the gathered table rows into a persistent GPU buffer
    before the captured region (V2 model runner only) and the forward consumes
    the buffer with pure tensor ops, so eager is no longer forced."""
    return dsv41_engram_runtime_enabled(config) and (
        os.environ.get("VLLM_DSV41_ENGRAM_ALLOW_CUDAGRAPH", "0") == "1"
    )


def dsv41_full_decode_enabled(config) -> bool:
    """Whether decode-only FULL cudagraphs are on: the cudagraph-compatible
    engram path is on AND ``VLLM_DSV41_FULL_DECODE=1`` (default OFF).

    FULL_DECODE_ONLY captures the whole decoder forward in one graph for
    uniform decode/verify batches (prefill stays eager, NONE), reclaiming the
    host-side eager-break stitching idle that PIECEWISE leaves on the table.
    Requires the engram consumer to read its persistent precomputed buffer
    (guaranteed by the cudagraph engram path) and the decode candidate masks to
    score a STATIC width (see build_decode_candidates) so no python-int seqlen
    bakes a stale mask shape into the captured graph."""
    return dsv41_engram_cudagraph_enabled(config) and (
        os.environ.get("VLLM_DSV41_FULL_DECODE", "0") == "1"
    )


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """The smallest prime above ``start`` not already handed out."""
    from sympy import isprime

    candidate = start + 1
    while not isprime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize alike collapse.

    N-grams are hashed over these compressed ids, so " The", "the" and "THE" all hash the same
    way. Returns the lookup plus the compressed vocab size — that size matters beyond bounds
    checking, because every hash multiplier is derived from it.
    """
    from tokenizers import Regex, normalizers

    # a private-use char, so a token that is exactly one space survives Strip() instead of
    # collapsing to the empty string and merging with unrelated tokens
    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            # a partial UTF-8 byte token: nothing to normalize, so key it by its raw form
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, tokenizer_vocab_size: int
) -> torch.Tensor:
    """One multiplier per (layer, lookback), from a per-layer RNG so layers hash differently.

    Kept odd, and bounded so that ``token_id * multiplier`` cannot overflow int64.
    """
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout of the n-gram hash tables.

    A position is hashed as ``max_ngram_size - 1`` n-grams (2-gram .. max_ngram_size-gram), each
    split over ``n_heads`` heads. Every (n-gram size, head) pair owns its own prime-sized bucket
    range in the layer's table; primes are drawn in order and never reused, keeping ranges disjoint.
    """

    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]  # table rows, per engram layer
    primes: tuple[tuple[tuple[int, ...], ...], ...]  # [layer][n-gram size][head] bucket modulus
    n_heads: int
    head_dim: int

    @classmethod
    def from_config(cls, cfg) -> "EngramLayout | None":
        """Build from a ``DeepseekV41Config`` (uses ``reference_args`` field names)."""
        ra = cfg.reference_args if hasattr(cfg, "reference_args") else cfg
        layer_ids = tuple(ra["engram_layer_ids"])
        if not layer_ids:
            return None
        max_ngram_size, n_heads = ra["engram_max_ngram_size"], ra["engram_n_heads"]
        vocab = ra["engram_vocab_size"]
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], vocab - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(ra["engram_num_embeddings"]),
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=ra["engram_head_dim"],
        )


def build_hash_buffers(cfg, layout: "EngramLayout", tokenizer) -> dict:
    """Shared constructor for the hash buffers (token map, multipliers, primes,
    offsets, pad id). One implementation feeds both the reference-shaped
    :class:`NgramHashState` (validation harness) and the serving
    :class:`EngramServingHasher`, so they can never drift."""
    ra = cfg.reference_args if hasattr(cfg, "reference_args") else cfg
    # every hash multiplier derives from the compressed vocab size, so a mismatch
    # there would silently rehash the whole table
    token_map, vocab_size = build_compressed_token_map(tokenizer)
    assert vocab_size == ra["engram_compressed_vocab_size"], (
        vocab_size, ra["engram_compressed_vocab_size"])
    multipliers = compute_hash_multipliers(
        layout.layer_ids, layout.max_ngram_size, vocab_size)
    flat = [[p for per_ngram in layer for p in per_ngram] for layer in layout.primes]
    offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
    return {
        "pad_id": token_map[ra["engram_pad_id"]],
        "primes": torch.tensor(layout.primes),
        "offsets": torch.tensor(np.array(offsets)),
        "multipliers": multipliers,
        "token_map": torch.tensor(token_map),
    }


def hash_ngram_windows(
    tokens: torch.Tensor,
    blocked: torch.Tensor,
    pad_id: int,
    token_map: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Hash per-token n-gram windows: ``tokens`` [T, n] carries raw token ids
    (column 0 = the token itself, column s = its s-th predecessor); ``blocked``
    [T, n] marks look-back that ran off the sequence start or crossed a dead
    (image) token. Returns [T, n_engram_layers, (n-1)*n_heads] table row ids.

    Bit-identical to :meth:`NgramHashState.forward` given the same windows: the
    same compressed-map lookup, pad substitution, XOR-rolling multiply and
    prime bucketing, just over stateless per-token windows instead of the
    reference per-batch ring cache.
    """
    compressed = torch.where(blocked, pad_id, token_map[tokens])
    products = compressed.unsqueeze(-2) * multipliers  # [T, n_layers, n]
    # XOR the multiplied ids together one lookback at a time, so the running
    # value after step i is the hash of the (i+1)-gram
    rolling, hashes = products[..., 0], []
    for i in range(1, tokens.shape[-1]):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1])
    return torch.cat(hashes, dim=-1) + offsets


class EngramServingHasher(nn.Module):
    """Serving-side hasher over per-token trailing windows.

    The vLLM runner supplies, for every scheduled token, the window of raw ids
    ``[token, pred_1, .., pred_{n-1}]`` gathered from the persistent batch's
    ``token_ids_cpu`` (which always holds each request's FULL token history --
    prompt tokens skipped by a prefix-cache hit, accepted spec tokens and
    preemption-restored tokens included). Hashes are therefore a pure,
    recomputed-every-step function of the sequence state: chunked prefill,
    decode, prefix-cache hits and spec-decode rollback all fall out for free.

    ``forward`` also returns the engram token mask (False on image-span
    positions, whose gate the reference closes), or None for text-only configs.
    """

    def __init__(self, cfg, layout: EngramLayout, tokenizer,
                 image_token_id: int | None = None):
        super().__init__()
        self.layout = layout
        self.image_token_id = image_token_id
        bufs = build_hash_buffers(cfg, layout, tokenizer)
        self.pad_id = bufs["pad_id"]
        self.register_buffer("primes", bufs["primes"], persistent=False)
        self.register_buffer("offsets", bufs["offsets"], persistent=False)
        self.register_buffer("multipliers", bufs["multipliers"], persistent=False)
        self.register_buffer("token_map", bufs["token_map"], persistent=False)

    @torch.inference_mode()
    def forward(
        self, windows: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """windows: [T, max_ngram_size] raw token ids (col 0 = current token);
        positions: [T]. Returns ([T, n_layers, n_hash_cols], token_mask or None)."""
        n = self.layout.max_ngram_size
        # Clamp defensively: -1 spec-scheduler placeholders can only appear under
        # async scheduling, which the engram wiring refuses at boot; a stray
        # negative id must not crash the token_map gather.
        tokens = windows[:, :n].to(torch.int64).clamp_min_(0)
        shifts = torch.arange(n, device=tokens.device)
        blocked = positions.to(torch.int64).unsqueeze(-1) < shifts
        token_mask = None
        if self.image_token_id is not None:
            dead = tokens == self.image_token_id
            token_mask = ~dead[:, 0]
            # reference semantics: once a lookback hits a dead (image) token,
            # every older predecessor is blocked too (sticky over shifts)
            blocked = (blocked | dead).to(torch.int32).cummax(-1).values.bool()
        hash_ids = hash_ngram_windows(
            tokens, blocked, self.pad_id, self.token_map,
            self.multipliers, self.primes, self.offsets)
        return hash_ids, token_mask


@torch.inference_mode()
def gather_token_windows_gpu(
    windows: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    all_token_ids: torch.Tensor,
    num_tokens: int,
    num_reqs: int,
) -> None:
    """Fill ``windows[:num_tokens]`` with each token's trailing n-gram window,
    entirely on GPU (V2 model runner).

    Predecessor at shift s of flat token t (position p, request row r):
      * another token of the SAME batch run when it is at least s tokens into
        its run -- read from ``input_ids`` (this covers chunk-internal
        predecessors AND speculative draft tokens, which exist only in the
        batch buffer at this point);
      * otherwise a COMMITTED token -- read from ``all_token_ids[r, p - s]``,
        which the V2 runner keeps current on-GPU in stream order (accepted
        sampled tokens are written by post_update before the next step's
        prepare kernels run, so async scheduling and spec-decode rollback are
        both correct by construction; prompt rows hold the full prompt, so
        prefix-cache hits are too).

    ``windows`` is a persistent [max_num_tokens, n] int32 buffer; rows past
    ``num_tokens`` are left stale (they are only ever read for cudagraph
    padding rows, whose outputs are discarded -- the hash of any int32 id is a
    valid table row).
    """
    device = windows.device
    n = windows.shape[1]
    t = torch.arange(num_tokens, device=device)
    # batch row of each token: count of run boundaries <= t
    bounds = query_start_loc[1 : num_reqs + 1].to(torch.int64)
    batch_idx = torch.searchsorted(bounds, t, right=True).clamp_max(num_reqs - 1)
    local = t - query_start_loc[batch_idx].to(torch.int64)
    rows = idx_mapping[batch_idx].to(torch.int64)
    pos = positions[:num_tokens].to(torch.int64)
    max_len = all_token_ids.shape[1]
    windows[:num_tokens, 0] = input_ids[:num_tokens]
    for s in range(1, n):
        from_batch = local >= s
        in_batch = input_ids[(t - s).clamp_min(0)]
        hist_pos = (pos - s).clamp(min=0, max=max_len - 1)
        from_hist = all_token_ids[rows, hist_pos]
        windows[:num_tokens, s] = torch.where(from_batch, in_batch, from_hist)


class NgramHashState(nn.Module):
    """Maps each position to the hash ids of the n-grams ending there.

    Ids go through the compressed table, then each position is hashed with the
    ``max_ngram_size - 1`` tokens before it. Look-back stops at the start of the sequence and at
    any dead token (an image span, cached as DEAD), so an n-gram never spans one.

    ``max_batch_size``/``max_seq_len`` size the reference-style ring cache used by the standalone
    (non-paged) path and by the bit-exact validation harness. Serving uses the stateless
    :class:`EngramServingHasher` over runner-supplied per-token windows instead.
    """

    DEAD = -1

    def __init__(self, cfg, layout: EngramLayout, tokenizer, max_batch_size: int = 4,
                 max_seq_len: int = 4096):
        super().__init__()
        self.layout = layout
        bufs = build_hash_buffers(cfg, layout, tokenizer)
        self.pad_id = bufs["pad_id"]
        self.register_buffer("primes", bufs["primes"], persistent=False)
        self.register_buffer("offsets", bufs["offsets"], persistent=False)
        self.register_buffer("multipliers", bufs["multipliers"], persistent=False)
        self.register_buffer("token_map", bufs["token_map"], persistent=False)
        self.register_buffer(
            "cache", torch.empty(max_batch_size, max_seq_len, dtype=torch.int64), persistent=False
        )

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int,
                token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """token_mask: [B, L], False for tokens that take no part in an n-gram (image spans).
        Returns the hash ids, shaped [B, L, n_engram_layers, n_hash_cols]."""
        batch, seqlen = input_ids.shape
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, self.DEAD)
        self.cache[:batch, start_pos : start_pos + seqlen] = compressed

        positions = torch.arange(
            start_pos, start_pos + seqlen, device=input_ids.device).expand(batch, seqlen)
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.layout.max_ngram_size):
            source = self.cache[:batch].gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, self.pad_id, source))
        tokens = torch.stack(tokens, dim=-1)  # [B, L, max_ngram_size]

        # XOR the multiplied ids together one lookback at a time, so the running value after
        # step i is the hash of the (i+1)-gram; each lands in its own prime-sized bucket range
        products = tokens.unsqueeze(2) * self.multipliers  # [B, L, n_layers, max_ngram_size]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


_SAFETENSORS_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
}


def _mmap_checkpoint_tensor(
    ckpt_dir: str, name: str, keep_alive: list
) -> tuple[torch.Tensor, list[int]]:
    """A zero-copy CPU view of one tensor inside a .safetensors shard.

    The file is mapped MAP_PRIVATE (copy-on-write; nothing ever writes, so no
    private pages materialize) and the tensor is a dtype view straight into the
    page cache -- the fit lever for the two ~98 GB engram tables: no anonymous
    host RAM is committed, the kernel pages rows in on demand and can evict them
    under pressure. The mmap object is appended to ``keep_alive`` so the mapping
    outlives this call. Returns (full_tensor, shape).
    """
    index = json.load(
        open(os.path.join(ckpt_dir, "model.safetensors.index.json"))
    )["weight_map"]
    path = os.path.join(ckpt_dir, index[name])
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        info = header[name]
        data_start = 8 + header_len
        mm = mmap.mmap(f.fileno(), 0, flags=mmap.MAP_PRIVATE,
                       prot=mmap.PROT_READ | mmap.PROT_WRITE)
    keep_alive.append(mm)
    lo, hi = info["data_offsets"]
    flat = torch.frombuffer(mm, dtype=torch.uint8, count=hi - lo,
                            offset=data_start + lo)
    dtype = _SAFETENSORS_DTYPES[info["dtype"]]
    return flat.view(dtype).view(info["shape"]), info["shape"]


class ParallelEngramEmbedding(nn.Module):
    """The n-gram hash table, sharded over its rows. Stays fp8: rows are dequantized on lookup.

    ``VLLM_DSV41_ENGRAM_DEVICE=cpu`` (default) keeps the ~49 GB/rank table in host DRAM; only the
    small gathered ``[.., n_hash_cols, head_dim]`` rows cross PCIe per token. Set to ``cuda`` to
    keep it resident (needs the 614 GB config; will not fit 4x96 GB).

    Host-table backing (``table_source``):
      * ``"mmap"`` (serving default) -- the weight/scale are dtype views straight
        into the checkpoint shard's page cache (see :func:`_mmap_checkpoint_tensor`).
        All ranks share one physical copy; the weight loader only VALIDATES the
        checkpoint tensors (no 98 GB copy). Requires ``ckpt_dir``.
      * ``"ram"`` -- each rank copies its row shard into (optionally pinned,
        ``VLLM_DSV41_ENGRAM_PIN=1``) anonymous host memory at load time.
        ~51 GB/rank for the two v41 tables; 4 ranks need ~203 GB host RAM.
    The gather math and the TP all-reduce are identical in both modes.
    """

    def __init__(self, num_embeddings: int, dim: int, table_device: str = "cpu",
                 block_size: int = 32, table_source: str = "ram",
                 ckpt_dir: str | None = None, ckpt_prefix: str | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim = dim
        world_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        self.part_num_embeddings = (num_embeddings + world_size - 1) // world_size
        self.vocab_start_idx = rank * self.part_num_embeddings
        self.vocab_end_idx = self.vocab_start_idx + self.part_num_embeddings
        # The ceil split can overrun the table on the last rank; allocate/map only
        # the real rows. Indices are always < num_embeddings, so the mask math
        # against vocab_end_idx never reaches the missing tail rows.
        self.part_rows = max(
            0, min(self.vocab_end_idx, num_embeddings) - self.vocab_start_idx
        )
        self.world_size = world_size
        self.block_size = block_size
        self.table_device = table_device
        self.table_source = table_source
        self._mmaps: list = []
        if table_source == "mmap":
            assert table_device == "cpu", "mmap tables are host-resident"
            assert ckpt_dir is not None and ckpt_prefix is not None
            full_w, w_shape = _mmap_checkpoint_tensor(
                ckpt_dir, f"{ckpt_prefix}.weight", self._mmaps)
            full_s, s_shape = _mmap_checkpoint_tensor(
                ckpt_dir, f"{ckpt_prefix}.scale", self._mmaps)
            assert w_shape == [num_embeddings, dim], (w_shape, num_embeddings, dim)
            assert s_shape == [num_embeddings, dim // block_size], s_shape
            rows = slice(self.vocab_start_idx, self.vocab_start_idx + self.part_rows)
            self.weight = nn.Parameter(full_w[rows], requires_grad=False)
            self.scale = nn.Parameter(full_s[rows], requires_grad=False)
            # The tables are already backed by the checkpoint bytes: the weight
            # loader must neither copy 98 GB nor let anything move them to GPU.
            self.weight.weight_loader = self._validate_only_loader
            self.scale.weight_loader = self._validate_only_loader
        else:
            pin = (table_device == "cpu"
                   and os.environ.get("VLLM_DSV41_ENGRAM_PIN", "0") == "1")
            self.weight = nn.Parameter(
                torch.empty(self.part_rows, dim,
                            dtype=torch.float8_e4m3fn, device=table_device,
                            pin_memory=pin if table_device == "cpu" else False),
                requires_grad=False,
            )
            self.scale = nn.Parameter(
                torch.empty(self.part_rows, dim // block_size,
                            dtype=torch.float8_e8m0fnu, device=table_device,
                            pin_memory=pin if table_device == "cpu" else False),
                requires_grad=False,
            )
            self.weight.weight_loader = self._row_shard_loader
            self.scale.weight_loader = self._row_shard_loader

    def _validate_only_loader(self, param: nn.Parameter,
                              loaded_weight: torch.Tensor) -> None:
        """mmap mode: the parameter already aliases the checkpoint bytes."""
        expect = (self.num_embeddings, param.shape[-1])
        if tuple(loaded_weight.shape) != expect:
            raise ValueError(
                f"engram table shape mismatch: checkpoint "
                f"{tuple(loaded_weight.shape)} vs expected {expect}")

    def _row_shard_loader(self, param: nn.Parameter,
                          loaded_weight: torch.Tensor) -> None:
        """ram mode: copy only this rank's rows (source stays mmap-backed)."""
        if loaded_weight.shape[0] != self.num_embeddings:
            raise ValueError(
                f"engram table row mismatch: checkpoint {loaded_weight.shape[0]} "
                f"vs config {self.num_embeddings}")
        rows = slice(self.vocab_start_idx, self.vocab_start_idx + self.part_rows)
        param.data.copy_(loaded_weight[rows])

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        compute_device = indices.device
        mask = (indices < self.vocab_start_idx) | (indices >= self.vocab_end_idx)
        local_indices = indices - self.vocab_start_idx
        local_indices = local_indices.masked_fill(mask, 0)

        # gather on the table's device (host for CPU-offload), then move the small dequantized
        # result to the compute device. F.embedding is a pure gather, so it runs on fp8 CPU rows
        # unchanged; the [.., n_hash_cols, head_dim] output is a few KB per token.
        gather_indices = local_indices.to(self.weight.device)
        values = F.embedding(gather_indices, self.weight)
        scales = F.embedding(gather_indices, self.scale)
        values = values.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)
        values = values.flatten(-2).to(torch.bfloat16).to(compute_device)
        values = values.masked_fill(mask.unsqueeze(-1), 0)

        if self.world_size > 1:
            values = tensor_model_parallel_all_reduce(values)
        return values


def engram_gate(x: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                q_weight: torch.Tensor, k_weight: torch.Tensor, dim: int, eps: float,
                clamp_value: float = 1e-6,
                token_mask: torch.Tensor | None = None) -> torch.Tensor:
    """The engram write into the residual stream, gated by stream-vs-key match.

    x: [B, L, hc_mult, dim]; key: [B, L, hc_mult, dim]; value: [B, L, dim].
    Pure fp32 arithmetic (no quant) — signed-sqrt before sigmoid, matching the training kernel.
    Factored out of :class:`Engram.forward` so it can be validated bit-exact in isolation.

    Long prefill chunks are tiled over the token dim: the fp32 transients here
    (x.float() alone is 640 MiB at 8192 tokens) are the documented OOM at high
    gpu-mem-utilization. All math is per-token, so tiling is bit-exact. Decode
    batches (<= a few hundred tokens) take the untiled path unchanged.
    """
    _TILE = 2048
    if x.dim() == 3 and x.shape[0] > _TILE:
        out = torch.empty_like(x)
        for s in range(0, x.shape[0], _TILE):
            e = min(s + _TILE, x.shape[0])
            out[s:e] = engram_gate(
                x[s:e], key[s:e], value[s:e], q_weight, k_weight, dim, eps,
                clamp_value,
                None if token_mask is None else token_mask[s:e],
            )
        return out
    weight = q_weight.float() * k_weight.float()  # only ever used as a product
    h = x.float()
    # normalized per (token, hc copy) over `dim`, NOT jointly over the copies
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (h * weight * key).sum(-1) * rstd * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


class Engram(nn.Module):
    """Writes an n-gram lookup into the residual stream, gated by how well it matches the stream.

    The hash ids fetch ``n_hash_cols`` rows; ``wkv`` turns them into one key per hc copy plus a
    shared value. The gate is a normalized dot product of stream against key. ``wkv`` is injected
    (the assembled model passes its fp8 quant Linear; a plain Linear is used standalone).
    """

    def __init__(self, cfg, layer_id: int, layout: EngramLayout, wkv: nn.Module,
                 table_device: str = "cpu", table_source: str = "ram",
                 ckpt_dir: str | None = None, ckpt_prefix: str | None = None):
        super().__init__()
        ra = cfg.reference_args if hasattr(cfg, "reference_args") else cfg
        self.layer_id = layer_id
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.dim = ra["dim"]
        self.hc_mult = ra["hc_mult"]
        self.eps = ra["norm_eps"]
        self.clamp_value = 1e-6

        self.embed = ParallelEngramEmbedding(
            layout.num_embeddings[self.layer_hash_index], layout.head_dim,
            table_device=table_device, table_source=table_source,
            ckpt_dir=ckpt_dir,
            ckpt_prefix=f"{ckpt_prefix}.embed" if ckpt_prefix else None)
        self.wkv = wkv  # Linear(n_hash_cols*head_dim -> dim*(hc_mult+1))
        self.q_weight = nn.Parameter(torch.ones(ra["hc_mult"], ra["dim"]), requires_grad=False)
        self.k_weight = nn.Parameter(torch.ones(ra["hc_mult"], ra["dim"]), requires_grad=False)

    def forward_gathered(self, x: torch.Tensor, gathered_rows: torch.Tensor,
                         token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Consume ALREADY-GATHERED (dequantized bf16, TP all-reduced) table
        rows: pure tensor ops from here (fp8 wkv GEMM + fp32 gate), so this half
        is safe inside a captured CUDA graph. ``gathered_rows``:
        [.., n_hash_cols * head_dim], i.e. ``embed(hash_ids).flatten(-2)``.
        All-zero rows (cudagraph padding slots / unregistered windows) are a
        no-op on the stream: wkv(0) = 0 so value = 0 and the gated write adds
        nothing (the gate itself is finite: sigmoid of +-sqrt(clamp))."""
        kv = self.wkv(gathered_rows)
        if isinstance(kv, tuple):  # vLLM Linear layers return (out, bias)
            kv = kv[0]
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        return engram_gate(x, key, value, self.q_weight, self.k_weight,
                           self.dim, self.eps, self.clamp_value, token_mask)

    def forward(self, x: torch.Tensor, hash_ids: torch.Tensor,
                token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, L, hc_mult, dim] or [T, hc_mult, dim]; hash_ids: [.., n_hash_cols];
        token_mask: [B, L] / [T] (True = text token whose gate may open)."""
        return self.forward_gathered(
            x, self.embed(hash_ids).flatten(-2), token_mask)


@torch.inference_mode()
def precompute_engram_gathered_rows(
    hasher: EngramServingHasher,
    engrams: list[Engram],
    windows: torch.Tensor,
    positions: torch.Tensor,
    num_tokens: int,
    num_tokens_padded: int,
    rows_buf: torch.Tensor,
    mask_buf: torch.Tensor | None,
) -> None:
    """The HOST-DEPENDENT half of the engram, run by the model runner in
    prepare_inputs, BEFORE the captured region (VLLM_DSV41_ENGRAM_ALLOW_
    CUDAGRAPH=1). Hashes this step's per-token windows and gathers +
    dequantizes the table rows into the persistent ``rows_buf``, so the
    captured forward only reads the (stable-address) buffer.

    Bit-losslessness: this calls the EXACT modules the in-forward path calls
    (``hasher`` -> :class:`EngramServingHasher`, ``eng.embed`` ->
    :class:`ParallelEngramEmbedding` incl. its CPU fp32 dequant, single bf16
    round and TP all-reduce, in the same order on the same [num_tokens, ..]
    shapes), then copies bf16 -> bf16 (exact). The consumer
    (:meth:`Engram.forward_gathered`) is the same code the in-forward path
    runs after its own gather.

    * ``windows``: the persistent [max_num_tokens, max_ngram_size] window
      buffer already filled by :func:`gather_token_windows_gpu` for THIS
      step's token set -- chunked prefill, prefix-cache hits and spec-decode
      rollback are therefore keyed identically to the eager path.
    * ``engrams``: this PP rank's local Engram modules; each fills its own
      ``layer_hash_index`` slot (each TP rank gathers its own row shard,
      the all-reduce inside ``embed`` completes the rows, exactly as today).
    * ``rows_buf``: persistent [n_engram_layers, max_num_tokens,
      n_hash_cols * head_dim] bf16 GPU buffer (slot-major so each layer's
      [T, D] slice stays contiguous for the fp8 wkv GEMM).
    * padding: rows (and mask) for cudagraph padding slots
      [num_tokens:num_tokens_padded] are zeroed (/False) -- deterministic,
      and a no-op on the stream in ``forward_gathered``.
    """
    hash_ids, token_mask = hasher(windows[:num_tokens], positions[:num_tokens])
    pad_end = min(int(num_tokens_padded), rows_buf.shape[1])
    for eng in engrams:
        slot = eng.layer_hash_index
        rows_buf[slot, :num_tokens] = eng.embed(
            hash_ids[:, slot]).flatten(-2)
        if pad_end > num_tokens:
            rows_buf[slot, num_tokens:pad_end].zero_()
    if mask_buf is not None:
        if token_mask is not None:
            mask_buf[:num_tokens] = token_mask
        else:
            mask_buf[:num_tokens] = True
        if pad_end > num_tokens:
            mask_buf[num_tokens:pad_end] = False
