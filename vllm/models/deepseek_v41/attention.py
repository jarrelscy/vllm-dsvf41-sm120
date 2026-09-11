# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DeepseekV41 MLA Attention Layer
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DeepseekV2Config, DeepseekV3Config

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.models.deepseek_v41.common.ops import (
    fused_indexer_q_rope_quant,
    fused_q_kv_rmsnorm,
)
from vllm.models.deepseek_v41.common.ops.store_compressed_latent import (
    store_compressed_latent,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )

from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v41.common.rope import build_deepseek_v4_rope
from vllm.models.deepseek_v41.candidates import select_candidate_blocks
from vllm.models.deepseek_v41.compressor import (
    DeepseekCompressor,
    DeepseekV41RefCompressor,
)
from vllm.utils.multi_stream_utils import (
    execute_in_parallel,
    maybe_execute_in_parallel,
)
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV41IndexerBackend,
    DeepseekV4IndexerBackend,
    get_max_prefill_buffer_size,
)
from vllm.models.deepseek_v41.swa import DeepseekV41SWACache
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

logger = init_logger(__name__)


def _resolve_dsv4_kv_cache_dtype(
    use_fp8_ds_mla_layout: bool,
    kv_cache_dtype: str,
    cache_config: CacheConfig | None,
) -> tuple[str, torch.dtype]:
    """Map ``(layout, --kv-cache-dtype)`` to ``(cache_dtype_str, torch_dtype)``.

    Both layouts are paged; they differ in the per-token block format. The
    ``fp8_ds_mla`` format is UE8M0 block-scaled fp8 packed as ``uint8`` (the
    canonical ``fp8_ds_mla`` string is written back onto ``cache_config`` so the
    page-size specs pick the 576B per-token slot). Plain-row backends store each
    token's KV row in its element dtype: bf16 or per-tensor FP8 E4M3.
    """
    if use_fp8_ds_mla_layout:
        # fp8_ds_mla block format: UE8M0 block-scaled fp8 packed as uint8.
        assert kv_cache_dtype.startswith("fp8"), (
            f"DeepseekV41 fp8_ds_mla layout only supports fp8 kv-cache, "
            f"got {kv_cache_dtype}"
        )
        if kv_cache_dtype != "fp8_ds_mla":
            if cache_config is not None:
                cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"
            logger.info_once("Using DeepSeek's fp8_ds_mla KV cache format.")
        return kv_cache_dtype, torch.uint8

    # Plain bf16 / per-tensor fp8 KV row (FlashInfer).
    if kv_cache_dtype.startswith("fp8"):
        return kv_cache_dtype, torch.float8_e4m3fn
    # auto / bfloat16 -> plain bf16 KV row.
    return kv_cache_dtype, torch.bfloat16


class DeepseekV41Attention(nn.Module, AttentionLayerBase, ABC):
    """DeepseekV41 MLA attention layer.

    The platform-specific sparse-MLA forward (``forward_mqa`` /
    ``get_padded_num_q_heads`` / ``_o_proj`` / ``backend_cls``) is provided by a
    subclass — ``DeepseekV41FlashMLAAttention`` /
    ``DeepseekV41FlashInferSM120Attention`` /
    ``DeepseekV41FlashInferMLAAttention`` (CUDA) or
    ``DeepseekV41ROCMAiterMLAAttention`` (ROCm) — selected by the platform-specific
    deepseek_v41 model module. The base is never instantiated directly.
    """

    # Provided by the platform subclass.
    backend_cls: ClassVar[type[AttentionBackend]]
    # KV-cache per-token block format (both layouts are paged). True (default)
    # = fp8_ds_mla (UE8M0 block-scaled fp8 packed as uint8); False = plain
    # bf16 / per-tensor fp8 KV row. Backends can override the instance hook when
    # a single attention class dispatches across arch-specific layouts.
    use_fp8_ds_mla_layout: ClassVar[bool] = True
    # Prefill is processed in fixed-size chunks; this bounds the bf16 kv-gather
    # workspace allocated in _forward_prefill and is also read by the dummy-run
    # path to pre-reserve that workspace.
    PREFILL_CHUNK_SIZE: ClassVar[int] = 4

    @classmethod
    @abstractmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        """Q head count the q/output buffers are allocated at.

        The layer allocates the q/output buffers at
        ``[N, get_padded_num_q_heads(n_local_heads), head_dim]``. Must satisfy
        ``result >= num_heads``. Backends with no padding constraint return
        ``num_heads``.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Platform-specific sparse MLA forward; writes attention into ``output``."""
        raise NotImplementedError

    @abstractmethod
    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Inverse-RoPE + wo_a + wo_b output projection (platform-specific)."""
        raise NotImplementedError

    def _uses_fp8_ds_mla_layout(self) -> bool:
        """Return whether this instance stores fp8 KV in fp8_ds_mla layout."""
        return self.use_fp8_ds_mla_layout

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        tp_size = get_tensor_model_parallel_world_size()
        layer_id = extract_layer_index(prefix)

        self.prefix = prefix  # Alias for compatibility with compressor
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        assert self.n_heads % tp_size == 0
        self.n_local_heads = self.n_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = config.sliding_window
        self.layer_id = layer_id
        self.num_hidden_layers = config.num_hidden_layers
        # v41 CED (cross-layer entangled decode) semantics.
        # compress_ratios has length num_hidden_layers + num_mtp and carries the
        # TRUE per-layer ratio: 0 = pure sliding-window (no compressor), 1 = plain
        # bf16 projection (one token / group), 2 = fp32 softmax pool-of-2. Unlike
        # v4 we do NOT clamp ratio 0 -> 1: a ratio-0 layer must stay pure-SWA and
        # (via build_deepseek_v4_rope) use base rope_theta with YaRN OFF, while
        # ratios 1/2 use compress_rope_theta + YaRN. Clamping would misroute rope
        # on layers 0/1 (and the MTP layers).
        compress_ratios = list(config.compress_ratios)
        if layer_id < len(compress_ratios):
            self.compress_ratio = compress_ratios[layer_id]
        else:
            # Defensive: layer index beyond the published ratio list -> pure SWA.
            self.compress_ratio = 0

        # --- CED membership (backbone layers only; MTP layers are pure SWA). ---
        is_backbone = layer_id < config.num_hidden_layers
        kv_source_layers = list(config.kv_source_layer_ids or ())
        index_source_layers = list(config.index_source_layer_ids or ())
        candidate_source_layer = config.candidate_source_layer_id
        self.is_kv_source = is_backbone and layer_id in kv_source_layers
        self.is_index_source = is_backbone and layer_id in index_source_layers
        self.is_candidate_source = (
            is_backbone and layer_id == candidate_source_layer
        )
        self.uses_candidates = (
            is_backbone and 0 <= candidate_source_layer < layer_id
        )
        # A layer that owns index keys is exactly a kv_source layer (index keys
        # are derived from that layer's compressor latent). Every kv_source layer
        # is also an index_source layer in the released config.
        self.owns_index_k = self.is_kv_source
        # Does this layer read any compressed KV at all? Only compressing layers
        # (ratio > 0) concatenate compressed KV + compress-topk onto the window
        # KV for the single sparse_attn call. Ratio-0 layers are pure SWA.
        self.uses_compress = is_backbone and self.compress_ratio > 0

        # --- Static "nearest preceding source" prefix resolution. ---
        # Replaces the reference module-global ``shared_attn`` singleton (which
        # relies on strictly in-order layer execution and is NOT cudagraph/batch
        # safe). Each consumer resolves its source module/caches at forward time
        # via ``static_forward_context[source_prefix]`` / ``attn_metadata[
        # source_prefix]`` -- the exact cross-module read pattern v4 already uses.
        def _nearest(sources: list[int]) -> int | None:
            cands = [s for s in sources if s <= layer_id]
            return max(cands) if cands else None

        def _sibling_prefix(src: int | None) -> str | None:
            if src is None or not prefix:
                return None
            # prefix looks like "<...>.layers.<layer_id>.attn"; retarget the layer.
            return prefix.replace(f".layers.{layer_id}.", f".layers.{src}.")

        # compress_kv cache and index_k cache both come from the nearest kv_source
        # (<= layer_id); topk_idxs come from the nearest index_source; candidates
        # always come from candidate_source_layer.
        self._kv_source_layer = _nearest(kv_source_layers) if is_backbone else None
        self._index_topk_source_layer = (
            _nearest(index_source_layers) if is_backbone else None
        )
        self._candidate_source_layer = (
            candidate_source_layer
            if (is_backbone and 0 <= candidate_source_layer <= layer_id)
            else None
        )
        # Attention-layer prefix of the compress-KV owner (its self.kv_cache is
        # the compressed-KV paged cache, exactly as v4 wires k_cache_prefix).
        self._kv_source_prefix = _sibling_prefix(self._kv_source_layer)
        # Index-k owner's indexer k-cache prefix.
        self._index_k_source_prefix = (
            f"{self._kv_source_prefix}.indexer.k_cache"
            if self._kv_source_prefix is not None
            else None
        )
        self._index_topk_source_prefix = _sibling_prefix(
            self._index_topk_source_layer
        )
        self._candidate_source_prefix = _sibling_prefix(
            self._candidate_source_layer
        )

        self.eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5

        # Padded Q head count is dictated by the platform subclass.
        self.padded_heads = self.get_padded_num_q_heads(self.n_local_heads)
        # Sink padded to the same head count, initialized to -inf (no sink
        # effect). Weight loading fills the first n_local_heads slots.
        self.attn_sink = nn.Parameter(
            torch.full((self.padded_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )

        self.fused_wqa_wkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_wqa_wkv",
            disable_tp=True,  # fused ReplicatedLinear
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )

        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )

        # Initialize rotary embedding before the indexer/compressor consume it.
        self.rotary_emb = build_deepseek_v4_rope(
            config,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=self.compress_ratio,
        )
        self.indexer_rotary_emb = self.rotary_emb
        self.topk_indices_buffer = topk_indices_buffer

        self.indexer = None
        if self.is_index_source:
            # v41: an indexer lives on every index_source layer (2,8,14,20,24,28,
            # 32,36), NOT on "compress_ratio == 4" (v4). Only kv_source layers
            # (owns_index_k) derive + publish index keys from the MAIN compressor
            # latent; the other index_source layers READ the nearest kv_source's
            # published index_k. See DeepseekV41Indexer for the owns_k branch.
            # aux_stream_list[2] is free here (outer GEMMs joined) for the inner
            # overlap of wq_b+fused_indexer_q_rope_quant vs the index-k derive.
            # None on ROCm, where aux_stream_list is None.
            indexer_aux_stream = (
                aux_stream_list[2] if aux_stream_list is not None else None
            )
            self.indexer = DeepseekV41Indexer(
                vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                q_lora_rank=self.q_lora_rank,
                quant_config=quant_config,
                cache_config=cache_config,
                topk_indices_buffer=topk_indices_buffer,
                compress_ratio=self.compress_ratio,
                prefix=f"{prefix}.indexer",
                aux_stream=indexer_aux_stream,
                owns_k=self.owns_index_k,
                index_k_source_prefix=self._index_k_source_prefix,
                is_candidate_source=self.is_candidate_source,
                uses_candidates=self.uses_candidates,
                candidate_source_prefix=self._candidate_source_prefix,
                candidate_topk_blocks=config.candidate_topk_blocks,
                candidate_block_size=config.candidate_block_size,
            )

        # Will be None on ROCm for now.
        self.aux_stream_list = aux_stream_list
        # [0]: GEMM start / post-GEMM event0. [1..3]: GEMM done events;
        # [1] doubles as post-GEMM event1. Reuse is safe: GEMM fully joins
        # before post-GEMM starts.
        self.ln_events = [torch.Event() for _ in range(4)]

        assert cache_config is not None, "DeepseekV41 attention requires cache_config"
        # ---- Attention / KV-cache setup ----
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len

        # Resolve the kv-cache dtype from this backend's block format. The same
        # resolution drives the SWA cache tensor dtype below.
        self.kv_cache_dtype, self.kv_cache_torch_dtype = _resolve_dsv4_kv_cache_dtype(
            self._uses_fp8_ds_mla_layout(), cache_config.cache_dtype, cache_config
        )

        self.swa_cache_layer = DeepseekV41SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=self.kv_cache_torch_dtype,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        # Register with compilation context for metadata lookup.
        compilation_config = vllm_config.compilation_config
        if prefix and prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        if prefix:
            compilation_config.static_forward_context[prefix] = self
        self.kv_cache = torch.tensor([])

        # Module that owns the compressed KV cache this layer reads at forward
        # time. kv_source layers (2/8/14/20) own their own compress-KV cache;
        # non-owner compressed layers (3-19, 21-39) borrow the nearest kv_source's
        # -- the metadata dict is keyed only by cache-owning layers, so a
        # non-owner gets no flashmla metadata / cache under its own prefix and
        # must resolve the source's via _kv_source_prefix. The module handle is
        # stable (constructed earlier, layer_id < this one); its ``.kv_cache`` is
        # rebound after KV allocation and read at forward time. Mirrors the
        # indexer's _source_k_cache resolution (see DeepseekV41Indexer.__init__).
        # Held in a 1-element list so nn.Module.__setattr__ does NOT register the
        # handle as a submodule -- a direct module attribute here would put self
        # (kv_source) or an ancestor kv_source into the module tree, creating a
        # cycle that makes .train()/.eval() recurse forever. The list keeps it a
        # plain attribute; read via the _compress_source_module property below.
        self._compress_source_holder: list[nn.Module] = []
        if self.uses_compress:
            if self.is_kv_source:
                self._compress_source_holder = [self]
            elif self._kv_source_prefix is not None:
                src = compilation_config.static_forward_context.get(
                    self._kv_source_prefix
                )
                assert src is not None, (
                    f"compressed layer {prefix} could not resolve kv_source "
                    f"{self._kv_source_prefix} in static_forward_context"
                )
                self._compress_source_holder = [src]

        # Create the compressor ONLY on kv_source layers (v4 gated on
        # compress_ratio > 1; v41 gates on CED membership). The kv_source layers
        # 2/8/14/20 carry ratios 2/2/2/1 respectively, so the compressor must
        # express BOTH ratio 2 (fp32 softmax pool-of-2 with a partial-group state)
        # AND ratio 1 (plain bf16 projection, one token/group, no gate/state).
        # The v4 fused DeepseekCompressor assumes {4,128}+APE+overlap and cannot
        # express {1,2}; per the port plan we take the correctness-first path and
        # route pooling through the validated reference math
        # (DeepseekV41RefCompressor / compressor_ref.py), which produces the
        # pre-RoPE latent that BOTH the main compress-KV cache write AND the
        # indexer's index-k derivation consume.
        # TODO(gpu-gate / kernel-fusion): restore a fused compress->norm->rope->
        # quant->paged-store fast path for ratios {1,2} (no APE, no overlap) once
        # the reference path is golden-diff validated. The paged store of the
        # precomputed latent (incl. fp8_ds_mla UE8M0 packing) is the top kernel
        # seam -- see attention_ced_adaptation_report.md risk list.
        self.compressor = None
        if self.is_kv_source:
            self.compressor = DeepseekV41RefCompressor(
                vllm_config=vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.prefix,
            )

    @property
    def _compress_source_module(self) -> "nn.Module | None":
        # Module owning the compress-KV cache this layer reads. Backed by a list
        # to avoid submodule registration (see __init__). Empty for layers that
        # do not use the compressed path.
        return self._compress_source_holder[0] if self._compress_source_holder else None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-allocate attention output with FlashMLA-padded head count.
        # The op writes into `o_padded`; we slice to n_local_heads after.
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Metadata-independent input GEMMs + RMSNorm stay in the captured
        # graph; the metadata-dependent rest (q up-proj + kv-insert, indexer,
        # compressor, MLA attention) runs in the eager break.
        qr_kv, indexer_weights = self.attn_gemm_parallel_execute(hidden_states)
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr, kv = fused_q_kv_rmsnorm(
            qr,
            kv,
            self.q_norm.weight.data,
            self.kv_norm.weight.data,
            self.eps,
        )

        # attention_impl is wrapped with @eager_break_during_capture: this is
        # where the breakable cudagraph capture breaks (the attention op runs
        # eagerly between captured graph segments).
        self.attention_impl(
            hidden_states,
            qr,
            kv,
            indexer_weights,
            positions,
            o_padded,
        )
        o = o_padded[:, : self.n_local_heads, :]

        # Inverse-RoPE + wo_a + wo_b output projection (platform-specific).
        return self._o_proj(o, positions)

    def attn_gemm_parallel_execute(self, hidden_states) -> tuple[Any, ...]:
        # v41: the two compressor kv/score GEMMs the v4 prologue prefetched are
        # gone. The MAIN compressor (kv_source layers only) now runs the
        # reference pooling directly from hidden_states inside attention_impl,
        # and index keys are derived from that shared latent -- there is no
        # private indexer compressor. Only indexer.weights_proj is prefetched.
        aux_streams = self.aux_stream_list
        if aux_streams is not None:
            assert len(aux_streams) >= 3
            aux_streams = aux_streams[:3]

        aux_fns: list[Callable[[], Any] | None] = [None, None, None]

        if self.indexer is not None:
            indexer = self.indexer

            def indexer_weights_proj() -> torch.Tensor:
                # ReplicatedLinear returns (output, bias); bias is None.
                weights, _ = indexer.weights_proj(hidden_states)
                return weights

            aux_fns[0] = indexer_weights_proj

        def fused_wqa_wkv() -> torch.Tensor:
            # MergedColumnParallelLinear returns (output, bias); bias is None.
            qr_kv, _ = self.fused_wqa_wkv(hidden_states)
            return qr_kv

        qr_kv, (indexer_weights, _unused1, _unused2) = execute_in_parallel(
            fused_wqa_wkv,
            aux_fns,
            self.ln_events[0],
            self.ln_events[1:4],
            aux_streams,
            enable=hidden_states.shape[0]
            <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
        )

        return qr_kv, indexer_weights

    def _run_compressor(self, hidden_states: torch.Tensor, positions: torch.Tensor):
        """kv_source layers only: produce the pre-RoPE compressed latent via the
        reference ragged pooling, and publish this layer's compress-KV cache.

        Returns the per-input-token pre-RoPE latent ``[num_tokens, head_dim]``
        (boundary rows filled, others zero) that the co-located indexer consumes
        for index-k derivation, and side-effects a RoPE'd + UE8M0-quantized copy
        into this layer's fp8_ds_mla compress-KV paged cache at the compressed
        slots (``flashmla_metadata.slot_mapping``; -1 for non-boundary tokens).
        """
        assert self.compressor is not None
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict):
            # Profile / dummy run: no paged store; indexer tolerates latent=None.
            return None
        flashmla_metadata = attn_metadata.get(self.prefix)
        swa_metadata = attn_metadata.get(self.swa_cache_layer.prefix)
        if flashmla_metadata is None or swa_metadata is None:
            return None
        latent = self.compressor.pool_ragged(
            hidden_states,
            positions,
            swa_metadata.query_start_loc_cpu,
            flashmla_metadata.num_reqs,
        )
        # Paged store: RoPE tail + UE8M0 quant of the boundary latents into the
        # compress-KV cache at the metadata's compressed slots (byte-identical to
        # the v4 fp8_ds_mla store the flashinfer sparse-MLA consumer reads back).
        store_compressed_latent(
            latent,
            positions,
            flashmla_metadata.slot_mapping,
            self.kv_cache.view(torch.uint8),
            self.rotary_emb.cos_sin_cache,
            self.compress_ratio,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
        )
        return latent

    @eager_break_during_capture
    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,  # [num_tokens, padded_heads, head_dim], written in place
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        # The window/SWA path is unchanged from v4: wq_b + fused qnorm/RoPE/KV
        # insert, then forward_mqa reads the paged window KV (+ compressed KV via
        # metadata). This stays intact per the CED adaptation scope.
        q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
        q = self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)

        # ---- v41 CED compressed-KV / index path (correctness-first) ----
        # kv_source layers run the MAIN compressor to get the shared pre-RoPE
        # latent, which the co-located indexer consumes for index-k derivation.
        # Non-kv-source index_source layers pass latent=None and their indexer
        # reads the nearest kv_source's published index_k.
        latent = None
        if self.is_kv_source:
            # Seam B: ragged pooling + paged compress-KV store; returns the
            # per-input-token pre-RoPE latent the co-located indexer consumes.
            latent = self._run_compressor(hidden_states, positions)

        if self.indexer is not None:
            # Produces this index_source layer's topk_idxs (level-2 top-k). On the
            # candidate source (layer 20) it also publishes the candidate mask;
            # uses_candidates layers (24/28/32/36) apply masked_fill(~candidates)
            # before their top-k. Both are wired structurally on the indexer; the
            # numeric candidate masking is a GPU-gated seam (see indexer.forward).
            self.indexer(
                hidden_states,
                qr,
                latent,
                indexer_weights,
                positions,
                self.indexer_rotary_emb,
            )
        # TODO(gpu-gate): publish/consume of per-forward topk_idxs (and
        # candidates) via address-stable buffers keyed by source
        # (_index_topk_source_prefix / _candidate_source_prefix), and the
        # backend read of the nearest kv_source's compressed-KV cache
        # (_kv_source_prefix) inside forward_mqa. flashmla.py:forward_mqa still
        # hardcodes compress_ratio in {4,128}; ratio {1,2} + shared-source-cache
        # reads are the dominant backend seam.

        # MLA attention writes into the pre-allocated `out` buffer
        # ([num_tokens, padded_heads, head_dim]).
        self.forward_mqa(q, kv, positions, out)

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None
        ),
    ) -> torch.Tensor:
        if not isinstance(attn_metadata, dict):
            # Profile run: kernel doesn't fire; produce a padded tensor so
            # downstream FlashMLA gets the right shape.
            if self.n_local_heads < self.padded_heads:
                return F.pad(
                    q,
                    (0, 0, 0, self.padded_heads - self.n_local_heads),
                    value=0.0,
                )
            return q

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_kv_cache = self.swa_cache_layer.kv_cache
        # The fused insert ops require int64 position_ids; the runner's positions
        # buffer is already int64, so no cast is needed.
        assert positions.dtype == torch.int64
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        cache_dtype = swa_kv_cache.dtype

        # kv is unchanged; attention reads kv solely via swa_kv_cache.
        if cache_dtype == torch.uint8:
            # fp8_ds_mla UE8M0 paged path. Horizontally fused:
            #   Q side:  per-head RMSNorm (no weight) + GPT-J RoPE, zero-filling
            #            the padding head slots; the kernel allocates and returns
            #            the padded q tensor.
            #   KV side: GPT-J RoPE + UE8M0 FP8 quant + paged cache insert.
            swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)
            # DSV4.1-Flash reference applies NO per-head q RMSNorm: only q_norm on
            # the 1280-dim q-lora (already done upstream), then RoPE. The shared V4
            # kernel normalizes each q head, which is wrong for this arch. Cancel it
            # exactly: RMSNorm is a per-row scalar and RoPE is linear, so the kernel
            # returns rope(q)/rms(q); multiplying the result back by rms(q) recovers
            # rope(q). Gated so the (broken) native behaviour stays A/B-testable.
            import os as _os
            _undo_qnorm = _os.environ.get("V41_UNDO_QNORM", "1") == "1"
            if _undo_qnorm:
                _q_rms = (
                    q.float().square().mean(dim=-1, keepdim=True)
                    .add_(self.eps).sqrt_()
                )  # [T, n_local_heads, 1]
            q_out = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                q,
                kv,
                swa_kv_cache_2d,
                swa_metadata.slot_mapping,
                positions,
                cos_sin_cache,
                self.padded_heads,
                self.eps,
                swa_metadata.block_size,
            )
            if _undo_qnorm:
                _n = self.n_local_heads
                q_out[:, :_n] = (q_out[:, :_n].float() * _q_rms).to(q_out.dtype)
            return q_out

        # Plain-row path: the [num_blocks, block_size, 512] cache stores the KV
        # row in its element dtype (no Q padding). bf16 rewrites q in place;
        # per-tensor fp8 writes a separately-allocated fp8 q and quantizes the
        # KV row.
        block_size = swa_metadata.block_size
        swa_kv_cache_3d = swa_kv_cache.view(-1, block_size, self.head_dim)
        if cache_dtype == torch.bfloat16:
            torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
                q,
                kv,
                swa_kv_cache_3d,
                swa_metadata.slot_mapping,
                positions,
                cos_sin_cache,
                self.eps,
                block_size,
            )
            return q

        # per-tensor fp8 (torch.float8_e4m3fn)
        q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
            q,
            kv,
            q_fp8,
            swa_kv_cache_3d,
            swa_metadata.slot_mapping,
            positions,
            cos_sin_cache,
            self._flashinfer_fp8_kv_scale,
            self._flashinfer_fp8_q_scale_inv,
            self.eps,
            block_size,
        )
        return q_fp8

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend_cls

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        # v41: the compressed-KV paged cache is owned by kv_source layers only
        # (v4 gated on compress_ratio > 1). Non-source layers -- including
        # ratio-1 non-source layers 21..39 and ratio-2 non-source layers 3..19 --
        # read the nearest kv_source's cache via _kv_source_prefix and allocate no
        # compress cache of their own. Pure-SWA layers (ratio 0) also return None;
        # their window KV lives in the separately-allocated DeepseekV4SWACache.
        if not self.is_kv_source:
            return None
        # fp8_ds_mla is a UE8M0 block-scaled uint8 layout and needs 576B
        # alignment; plain bf16 / per-tensor fp8 rows use natural element-size
        # pages.
        uses_fp8_ds_mla_layout = self.kv_cache_dtype == "fp8_ds_mla"
        # v41 SM120: the FlashInfer sparse-MLA decode kernel derives the
        # compress-KV page block size from the cache tensor's storage dim (dim1)
        # and is only instantiated for page_block_size 64
        # (_DECODE_DSV4_PAGE_BLOCK_SIZE in flashinfer's _sparse_mla_sm120.py).
        # storage_block_size = block_size // compress_ratio, so a ratio-2 layer at
        # logical block 64 would store only 32 rows/page and the compiled kernel
        # would mis-address every compressed gather -- it converts the flat
        # compressed indices to (block, offset) with its hard-coded 64, reading
        # the wrong physical block. Scale the LOGICAL block by the ratio -- exactly
        # as the indexer does (DeepseekV41IndexerCache.get_kv_cache_spec, block =
        # 64 * ratio) -- so storage stays 64 for both ratios:
        #   ratio-1 (20,24,...): block 64  -> 64  // 1 = 64 storage
        #   ratio-2 (2,8,14):    block 128 -> 128 // 2 = 64 storage
        # Paired with the sparse-MLA backend advertising [64, 128] so each group
        # negotiates its own logical size directly (select_common_block_size
        # Case 1, no virtual-block split that would re-pool a ratio-2 page back
        # down to storage 32). Both ratios then share one page size
        # (64 * row_bytes), so no page_size_padded unification hack is needed.
        spec = MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size * self.compress_ratio,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576 if uses_fp8_ds_mla_layout else None,
            model_version="deepseek_v41",
        )
        return spec


class DeepseekV41IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        self.compress_ratio = compress_ratio
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # head_dim already carries the fp8 scale padding
        # compress_ratio=1 for V3.2, >1 for DeepseekV41; both use the same cache layout.
        #
        # The DeepGEMM paged-MQA-logits kernel (._C, compiled) HARD-asserts
        # block_kv == 64 exactly (smxx_fp8_fp4_paged_mqa_logits.hpp:236), NOT
        # {32,64} as an earlier note assumed. block_kv is the pooled STORAGE dim
        # per page = block_size // compress_ratio (MLAAttentionSpec.storage_block_size).
        # So the indexer's LOGICAL block must scale with the ratio to keep storage 64:
        #   ratio-1 (layers 20,24,28,32,36): block 64  -> 64 // 1 = 64 == block_kv OK
        #   ratio-2 (layers 2,8,14):         block 128 -> 128 // 2 = 64 == block_kv OK
        # This mirrors v4, whose ratio-4 indexer ran logical block 256 -> storage 64.
        # Both indexer groups end up with IDENTICAL page bytes (storage 64 * head_dim),
        # so they share the block pool cleanly. Paired with the v41 indexer backend's
        # get_supported_kernel_block_sizes -> [64, 128] so each group negotiates its
        # own logical size directly (Case 1, no virtual-block split that would re-pool
        # a ratio-2 page back down to storage 32).
        indexer_block_size = 64 * self.compress_ratio
        return MLAAttentionSpec(
            block_size=indexer_block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
            # DeepseekV41 aligns indexer pages to FlashMLA's 576B so they can pack with
            # the indexer's compressor state cache. V3.2 keeps the legacy layout.
            alignment=576,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV41IndexerBackend


class DeepseekV41Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        compress_ratio: int = 1,
        prefix: str = "",
        aux_stream: torch.cuda.Stream | None = None,
        owns_k: bool = True,
        index_k_source_prefix: str | None = None,
        is_candidate_source: bool = False,
        uses_candidates: bool = False,
        candidate_source_prefix: str | None = None,
        candidate_topk_blocks: int = 2048,
        candidate_block_size: int = 8,
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        self.compress_ratio = compress_ratio
        # v41 CED: index keys are derived from the MAIN compressor's pre-RoPE
        # latent and only a kv_source (owns_k) layer produces them; every other
        # index_source layer READS the nearest kv_source's published index_k.
        self.owns_k = owns_k
        self.index_k_source_prefix = index_k_source_prefix
        # Two-level candidate blocks: layer 20 publishes; 24/28/32/36 consume.
        self.is_candidate_source = is_candidate_source
        self.uses_candidates = uses_candidates
        self.candidate_source_prefix = candidate_source_prefix
        self.candidate_topk_blocks = candidate_topk_blocks
        self.candidate_block_size = candidate_block_size
        self.use_fp4_kv = self.vllm_config.attention_config.use_fp4_indexer_cache
        logger.info_once(
            "Using %s indexer cache for Lightning Indexer.",
            "MXFP4" if self.use_fp4_kv else "FP8",
        )

        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.n_head,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128  # TODO: get from config
        self.topk_indices_buffer = topk_indices_buffer

        self.max_model_len = (
            vllm_config.model_config.max_model_len // self.compress_ratio
        )
        self.prefix = prefix

        self.max_total_seq_len = (
            get_max_prefill_buffer_size(vllm_config) // self.compress_ratio
        )

        assert cache_config is not None, "Deepseek V4 indexer requires cache_config"
        # NOTE(yifan): FP8 indxer cache use the same layout as V3.2:
        # head_dim bytes = 128 fp8 + 4 fp32 scale = 132.
        # For FP4 indexer cache, we still allocate the same amount of memory as FP8,
        # but only use the first half of the memory.
        k_cache_head_dim = self.head_dim + self.head_dim // self.quant_block_size * 4

        # v41: DELETE the indexer's private compressor. Only an owns_k (kv_source)
        # layer allocates its own index-k cache and derives index keys FROM the
        # MAIN compressor's pre-RoPE latent (k_norm(wk(latent)), RoPE tail, fp4
        # E8M0 quant in groups of 32); it then publishes that cache. Non-owner
        # index_source layers own NO index-k cache and read the nearest kv_source's
        # published cache, resolved at forward via static_forward_context.
        self.compressor = None  # removed (was DeepseekCompressor); see above.
        if self.owns_k:
            self.k_cache = DeepseekV41IndexerCache(
                head_dim=k_cache_head_dim,
                dtype=torch.uint8,
                prefix=f"{prefix}.k_cache",
                cache_config=cache_config,
                compress_ratio=self.compress_ratio,
            )
            # Reference Indexer.wk / k_norm: derive index keys from the shared
            # latent (head_dim -> index_head_dim), bf16, matching checkpoint names.
            self.wk = ReplicatedLinear(
                config.head_dim,
                self.head_dim,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.wk",
            )
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
            self._source_k_cache = None
        else:
            # Read-only: resolve the nearest kv_source's index-k cache module now
            # (it is constructed earlier -- kv_source layer_id < this layer_id --
            # so it is already registered in static_forward_context).
            self.k_cache = None
            self.wk = None
            self.k_norm = None
            assert index_k_source_prefix is not None, (
                "non-owner index_source layer requires an index_k source"
            )
            src = vllm_config.compilation_config.static_forward_context.get(
                index_k_source_prefix
            )
            assert src is not None, (
                f"index_k source {index_k_source_prefix} not yet registered"
            )
            self._source_k_cache = src

        # The SparseAttnIndexer wraps whichever index-k cache this layer reads:
        # its own (owns_k) or the shared source cache (non-owner). skip_k_cache_
        # insert=True because v41 writes index-k via the owns_k derive path above,
        # NOT inside the op.
        indexer_cache = self.k_cache if self.owns_k else self._source_k_cache
        self.indexer_op = SparseAttnIndexer(
            indexer_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            # owns_k layers derive + store index-k here (via derive_index_k RoPE +
            # the op's indexer_k_quant_and_cache); non-owner layers read the shared
            # source cache and must NOT insert.
            skip_k_cache_insert=not self.owns_k,
            use_fp4_cache=self.use_fp4_kv,
        )

        # None on ROCm — maybe_execute_in_parallel falls back to sequential.
        self.aux_stream = aux_stream
        self.ln_events: list[torch.Event] = [torch.Event(), torch.Event()]

    def derive_index_k(
        self,
        latent: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        """owns_k only: turn the MAIN compressor's pre-RoPE latent into index
        keys and write this layer's index-k cache (published as the shared
        index_k). Reference ``Indexer.forward`` owns_k branch (model.py 553-564):
        ``k = k_norm(wk(latent)); RoPE(k[..., -rope_dim:]); fp4_act_quant(k, 32,
        E8M0); k_cache[...] = k``.

        TODO(gpu-gate): the fp4 (group=32, E8M0) quant of the derived key and the
        paged write into the UE8M0 index-k cache must reuse the indexer's quant +
        paged-store kernels (the v4 path fused these inside the private-compressor
        store, which no longer exists). This returns the pre-quant bf16 key; the
        quant+store kernel wiring is the seam validated at the GPU golden-diff.
        """
        assert self.owns_k and self.wk is not None and self.k_norm is not None
        k, _ = self.wk(latent)  # [T, index_head_dim] bf16
        k = self.k_norm(k)
        # RoPE the last rope_dim of the key at the GROUP-LEAD (compressed) position
        # = (pos // ratio) * ratio, matching the reference index-k freqs and the
        # compress-KV store's rope. Interleaved (GPT-J) pairs, fp32 -> bf16, using
        # the same compress-rope table as the index-q path (q/k rope parity).
        ratio = self.compress_ratio
        half = self.rope_dim // 2
        comp_pos = torch.div(positions, ratio, rounding_mode="floor") * ratio
        cs = rotary_emb.cos_sin_cache.index_select(0, comp_pos)  # [T, rope_dim]
        cos = cs[:, :half].float()
        sin = cs[:, half : 2 * half].float()
        tail = k[:, -self.rope_dim :].float().unflatten(-1, (half, 2))
        even = tail[..., 0]
        odd = tail[..., 1]
        rotated = torch.stack([even * cos - odd * sin, odd * cos + even * sin], dim=-1)
        k = k.clone()
        k[:, -self.rope_dim :] = rotated.flatten(-2).to(k.dtype)
        return k

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        latent: torch.Tensor | None,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        """v41 indexer forward.

        ``latent`` is this kv_source layer's pre-RoPE compressed latent (from the
        MAIN compressor), or ``None`` for a non-owner index_source layer (which
        reads the shared index_k cache instead). This REPLACES v4's private
        ``compressor(compressed_kv_score, ...)`` -> index-k path.
        """

        def wq_b_and_q_quant():
            # ReplicatedLinear returns (output, bias); bias is None.
            q, _ = self.wq_b(qr)
            q = q.view(-1, self.n_head, self.head_dim)
            return fused_indexer_q_rope_quant(
                positions,
                q,
                rotary_emb.cos_sin_cache,
                indexer_weights,
                self.softmax_scale,
                self.n_head**-0.5,
                use_fp4=self.use_fp4_kv,
            )

        if self.owns_k and latent is not None:
            # Derive index-k from the shared latent (parallel with the q path);
            # the join orders the index-k write before indexer_op.
            (q_quant, weights), k = maybe_execute_in_parallel(
                wq_b_and_q_quant,
                lambda: self.derive_index_k(latent, positions, rotary_emb),
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )
        elif self.owns_k:
            # owns_k but the MAIN latent is not wired yet (ragged compressor
            # production is GPU-gated -- see attention_impl._run_compressor).
            # TODO(gpu-gate): feed the real latent so index-k is derived + stored.
            q_quant, weights = wq_b_and_q_quant()
            k = None
        else:
            # Non-owner: no index-k derive; the op reads the shared source cache.
            q_quant, weights = wq_b_and_q_quant()
            k = None

        # TODO(gpu-gate): two-level candidates. The candidate mask
        # (masked_fill(~candidates, -inf) before this op's level-2 top-k) and the
        # candidate publish (select_candidate_blocks on the level-1 index_score at
        # the candidate source) both need to happen around the index_score the
        # SparseAttnIndexer computes internally. This requires either op-level
        # candidate support or a torch index_score path here (reference
        # Indexer.forward 571-596). Wired structurally via
        # self.is_candidate_source / self.uses_candidates / candidate buffers;
        # the numeric masking is the GPU-gated seam.
        return self.indexer_op(hidden_states, q_quant, k, weights)
