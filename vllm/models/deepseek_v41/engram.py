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
"""

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


class NgramHashState(nn.Module):
    """Maps each position to the hash ids of the n-grams ending there.

    Ids go through the compressed table, then each position is hashed with the
    ``max_ngram_size - 1`` tokens before it. Look-back stops at the start of the sequence and at
    any dead token (an image span, cached as DEAD), so an n-gram never spans one.

    ``max_batch_size``/``max_seq_len`` size the reference-style ring cache used by the standalone
    (non-paged) path and by the bit-exact validation harness. The serving runner may instead pass
    complete per-request token windows to :meth:`hash_ids_for_window`.
    """

    DEAD = -1

    def __init__(self, cfg, layout: EngramLayout, tokenizer, max_batch_size: int = 4,
                 max_seq_len: int = 4096):
        super().__init__()
        self.layout = layout
        ra = cfg.reference_args if hasattr(cfg, "reference_args") else cfg
        # every hash multiplier derives from the compressed vocab size, so a mismatch there
        # would silently rehash the whole table
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        assert vocab_size == ra["engram_compressed_vocab_size"], (
            vocab_size, ra["engram_compressed_vocab_size"])
        self.pad_id = token_map[ra["engram_pad_id"]]
        multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab_size)
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in layout.primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        self.register_buffer("primes", torch.tensor(layout.primes), persistent=False)
        self.register_buffer("offsets", torch.tensor(np.array(offsets)), persistent=False)
        self.register_buffer("multipliers", multipliers, persistent=False)
        self.register_buffer("token_map", torch.tensor(token_map), persistent=False)
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


class ParallelEngramEmbedding(nn.Module):
    """The n-gram hash table, sharded over its rows. Stays fp8: rows are dequantized on lookup.

    ``VLLM_DSV41_ENGRAM_DEVICE=cpu`` (default) keeps the ~49 GB/rank table in host DRAM; only the
    small gathered ``[.., n_hash_cols, head_dim]`` rows cross PCIe per token. Set to ``cuda`` to
    keep it resident (needs the 614 GB config; will not fit 4x96 GB).
    """

    def __init__(self, num_embeddings: int, dim: int, table_device: str = "cpu",
                 block_size: int = 32):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim = dim
        world_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        self.part_num_embeddings = (num_embeddings + world_size - 1) // world_size
        self.vocab_start_idx = rank * self.part_num_embeddings
        self.vocab_end_idx = self.vocab_start_idx + self.part_num_embeddings
        self.world_size = world_size
        self.block_size = block_size
        self.table_device = table_device
        self.weight = nn.Parameter(
            torch.empty(self.part_num_embeddings, dim,
                        dtype=torch.float8_e4m3fn, device=table_device),
            requires_grad=False,
        )
        self.scale = nn.Parameter(
            torch.empty(self.part_num_embeddings, dim // block_size,
                        dtype=torch.float8_e8m0fnu, device=table_device),
            requires_grad=False,
        )

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
    """
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
                 table_device: str = "cpu"):
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
            table_device=table_device)
        self.wkv = wkv  # Linear(n_hash_cols*head_dim -> dim*(hc_mult+1))
        self.q_weight = nn.Parameter(torch.ones(ra["hc_mult"], ra["dim"]), requires_grad=False)
        self.k_weight = nn.Parameter(torch.ones(ra["hc_mult"], ra["dim"]), requires_grad=False)

    def forward(self, x: torch.Tensor, hash_ids: torch.Tensor,
                token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, L, hc_mult, dim]; hash_ids: [B, L, n_hash_cols]; token_mask: [B, L]."""
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        return engram_gate(x, key, value, self.q_weight, self.k_weight,
                           self.dim, self.eps, self.clamp_value, token_mask)
