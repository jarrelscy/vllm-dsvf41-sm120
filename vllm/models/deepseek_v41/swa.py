# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepseekV41-specific sliding-window / compressed-KV metadata plumbing.

v41 uses a DIFFERENT compress_ratio taxonomy than v4. v4 layers carry ratios in
{1, 4, 128} where ratio 1 means *pure* sliding-window (after its ``max(1, ...)``
clamp). v41 backbone layers carry ratios in {0, 1, 2}:

  * ratio 0  -> pure sliding-window, NO compressed-KV stream (layers 0, 1 + MTP).
  * ratio 1  -> plain bf16 1:1 projection; a genuine compressed-KV stream whose
                length equals the sequence length (layers 20..39; kv_source 20).
  * ratio 2  -> fp32 softmax pool-of-2; compressed-KV length = seq // 2
                (layers 2..19; kv_source 2, 8, 14).

The shared ``DeepseekSparseSWAMetadataBuilder`` raises on ratio 2 and misroutes
ratio 1 to ``swaonly`` (its compressed KV would be skipped). Rather than mutate
the shared v4 builder (which is productionized), v41 SUBCLASSES it and swaps in a
v41 taxonomy via the ``_classify_layer_type`` seam. This mirrors the existing
AMD subclass pattern (``DeepseekV4ROCMAiterSparseSWAMetadataBuilder``).

Wiring: ``DeepseekV41SWACache.get_attn_backend`` -> ``DeepseekV41SparseSWABackend``
-> ``get_builder_cls`` -> ``DeepseekV41SparseSWAMetadataBuilder``. The attention
layer instantiates ``DeepseekV41SWACache`` (not the v4 cache), so every v41 SWA
cache resolves the v41 builder and the v41 tile-scheduler taxonomy.
"""

from typing import ClassVar

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.sparse_swa import (
    _LAYER_TYPE_SWAONLY,
    DeepseekSparseSWABackend,
    DeepseekSparseSWAMetadataBuilder,
    DeepseekV4SWACache,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowMLASpec

# v41 compressed-KV layer types, keyed by compress_ratio. Each needs a distinct
# FlashMLA tile-scheduler plan (topk / compressed-prefix width differ), so they
# cannot share a plan with each other or with swaonly.
#   ratio 0 -> swaonly (reuse the shared constant; identical scheduling).
#   ratio 1 -> c1a: 1:1 compressed KV, compressed-prefix width = seq_len.
#   ratio 2 -> c2a: pool-of-2 compressed KV, compressed-prefix width = seq_len//2.
_LAYER_TYPE_C1A = "c1a"
_LAYER_TYPE_C2A = "c2a"


def v41_layer_type_for(compress_ratio: int) -> str:
    """v41 taxonomy: admit ratios {0, 1, 2}; reject v4's {4, 128}."""
    if compress_ratio == 0:
        return _LAYER_TYPE_SWAONLY
    if compress_ratio == 1:
        return _LAYER_TYPE_C1A
    if compress_ratio == 2:
        return _LAYER_TYPE_C2A
    raise ValueError(
        f"Unsupported DeepseekV41 compress_ratio={compress_ratio}; "
        "expected 0, 1, or 2."
    )


class DeepseekV41SparseSWAMetadataBuilder(DeepseekSparseSWAMetadataBuilder):
    """v4 SWA builder with the v41 {0,1,2} compress-ratio taxonomy.

    Only ``_classify_layer_type`` changes the base ``__init__`` path (so ratio-2
    layers stop raising and ratio-1 layers are recognised as a compressed type
    rather than pure SWA). The tile-scheduler plans for ``c1a`` / ``c2a`` are
    built in ``build`` exactly like the v4 c4a/c128a plans; that extension lands
    with the ratio-{1,2} forward path.
    """

    @staticmethod
    def _classify_layer_type(compress_ratio: int) -> str:
        return v41_layer_type_for(compress_ratio)


class DeepseekV41SparseSWABackend(DeepseekSparseSWABackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V41_SPARSE_SWA"

    @staticmethod
    def get_builder_cls() -> type[DeepseekSparseSWAMetadataBuilder]:
        if current_platform.is_rocm():
            # ROCm v41 aiter builder is a separate seam; fall back to the v41
            # CUDA builder taxonomy until it exists so metadata build is correct.
            return DeepseekV41SparseSWAMetadataBuilder
        return DeepseekV41SparseSWAMetadataBuilder


class DeepseekV41SWACache(DeepseekV4SWACache):
    """v41 window-KV cache: identical paged layout to v4, but routed to the v41
    SWA backend/builder so metadata build uses the v41 taxonomy."""

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # The packed fp8_ds_mla layout is driven by the RESOLVED KV-cache tensor
        # dtype (uint8), not by cache_config.cache_dtype. Under DSpark the drafter
        # builds its own SWA caches whose cache_config string can lag behind the
        # target's fp8_ds_mla resolution; keying off self.dtype keeps the drafter
        # and target SWA specs identical (both 584B packed) so the KV-cache group
        # stays uniform. Otherwise a 512-wide draft spec becomes the group
        # representative and the target's own forward reads a 512 cache the SM120
        # sparse-MLA kernel rejects.
        import torch as _torch
        uses_fp8_ds_mla_layout = (
            self.dtype == _torch.uint8
            or self.cache_config.cache_dtype == "fp8_ds_mla"
        )
        cache_dtype_str = (
            "fp8_ds_mla" if uses_fp8_ds_mla_layout else self.cache_config.cache_dtype
        )
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            sliding_window=self.window_size,
            cache_dtype_str=cache_dtype_str,
            alignment=576 if uses_fp8_ds_mla_layout else None,
            model_version="deepseek_v41",
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV41SparseSWABackend
