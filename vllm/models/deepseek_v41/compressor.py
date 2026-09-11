# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch
from torch import nn

import torch.nn.functional as F

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_triton,
)
from vllm.models.deepseek_v41.common.ops.fused_indexer_q import MXFP4_BLOCK_SIZE
from vllm.models.deepseek_v41.common.ops.save_partial_states import (
    save_partial_states,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)


class CompressorBackend(AttentionBackend):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_name() -> str:
        return "CompressorBackend"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512, 1024]

    @staticmethod
    def get_builder_cls() -> type["CompressorMetadataBuilder"]:
        return CompressorMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


@dataclass
class CompressorMetadata:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int

    token_to_req_indices: torch.Tensor | None = None  # [num_tokens]


class CompressorMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.kv_cache_spec, SlidingWindowMLASpec | MLAAttentionSpec)
        mla_spec = cast(SlidingWindowMLASpec | MLAAttentionSpec, self.kv_cache_spec)
        self.block_size = mla_spec.block_size

        self.token_to_req_indices = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CompressorMetadata:
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        num_reqs = common_attn_metadata.num_reqs
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        x = torch.repeat_interleave(torch.arange(num_reqs), query_lens).pin_memory()
        token_to_req_indices = self.token_to_req_indices[: x.shape[0]]
        token_to_req_indices.copy_(x, non_blocking=True)
        return CompressorMetadata(
            block_table=common_attn_metadata.block_table_tensor.clamp_(min=0),
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=self.block_size,
            token_to_req_indices=token_to_req_indices,
        )


class CompressorStateCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        assert self.dtype == torch.float32
        assert compress_ratio in [4, 128]
        coff = 1 + (compress_ratio == 4)
        self.sliding_window = coff * compress_ratio
        # Block size is constrained by tensor sharing between compressor states
        # and KV blocks. Since compressor states share the same physical tensor
        # as KV blocks, they must use the same page size.
        # The KV block shape [256//4, head_dim] = [64, 584] determines:
        # - C4 compressor block shape [4, 2*512*2*4] -> block_size = 4
        # - C128 compressor block shape [8, 512*2*4] -> block_size = 8
        # TODO(yifan): make block size automatically determined and configurable.
        if compress_ratio == 4:
            self.block_size = 4
        elif compress_ratio == 128:
            self.block_size = 8
        else:
            raise ValueError(f"Invalid compress ratio: {compress_ratio}")

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # fp8_ds_mla is the UE8M0 paged layout and needs 576B alignment. Plain
        # full-cache rows share state pages with contiguous KV pages, so padding
        # would break page matching.
        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        return SlidingWindowMLASpec(  # only has one vector instead of K + V
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=576 if uses_fp8_ds_mla_layout else None,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return CompressorBackend


class DeepseekCompressor(nn.Module):
    """DeepSeek V4 KV/score compressor.

    Owns the linear / norm / state-cache / ape state and the shared forward
    prologue (kv/score split, save_partial_states launch). The
    compress → norm → RoPE → store step is dispatched to a triton kernel
    (``compress_norm_rope_store_triton``) by default, except for the NVIDIA
    head_dim=128 indexer path which uses the cutedsl kernel
    (``compress_norm_rope_store_cutedsl``) for better performance.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        k_cache_prefix="",
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rotate = rotate
        self.prefix = prefix
        self.k_cache_prefix = k_cache_prefix
        self.use_fp4_cache = use_fp4_cache

        config = vllm_config.model_config.hf_config
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.device = current_platform.device_type
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len

        self.overlap = compress_ratio == 4
        self.coff = 1 + self.overlap

        state_dtype = torch.float32
        self.ape = nn.Parameter(
            torch.empty(
                (compress_ratio, self.coff * self.head_dim),
                dtype=state_dtype,
                device=self.device,
            ),
            requires_grad=False,
        )

        self.fused_wkv_wgate = MergedColumnParallelLinear(
            self.hidden_size,
            [self.coff * self.head_dim, self.coff * self.head_dim],
            bias=False,
            return_bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
        )
        self.norm = RMSNorm(self.head_dim, self.rms_norm_eps)

        self.state_cache = CompressorStateCache(
            state_dim=2 * self.coff * self.head_dim,  # kv_state + score_state
            dtype=state_dtype,
            compress_ratio=compress_ratio,
            prefix=f"{prefix}.state_cache",
        )

        # Save reference to static_forward_context for forward-time KV cache lookup.
        # get_current_vllm_config() is only available during __init__, not forward.
        self._static_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )

        if self.head_dim == 512:
            assert not use_fp4_cache, (
                "MXFP4 cache is only supported for indexer (head=128)"
            )
            self._quant_block = 64
            self._token_stride = self.nope_head_dim + self.rope_head_dim * 2
            self._scale_dim = self.nope_head_dim // 64 + 1  # 7 real + 1 pad
        elif self.head_dim == 128:
            if use_fp4_cache:
                self._quant_block = MXFP4_BLOCK_SIZE
                self._token_stride = self.head_dim // 2
                self._scale_dim = self.head_dim // MXFP4_BLOCK_SIZE
            else:
                self._quant_block = 128
                self._token_stride = self.head_dim
                self._scale_dim = 4  # single float32 scale
        else:
            raise ValueError(
                f"Unsupported head_dim for fused quant+cache: {self.head_dim}"
            )

    def forward(
        self,
        # [num_tokens, 2 * self.coff * self.head_dim]
        kv_score: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        # Each of shape [num_tokens, coff * self.head_dim]
        # input bf16, output are fp32
        kv, score = kv_score.split(
            [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
        )

        # Get the metadata and handle dummy profiling run.
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return

        state_metadata = cast(
            CompressorMetadata, attn_metadata[self.state_cache.prefix]
        )
        token_to_req_indices = state_metadata.token_to_req_indices
        slot_mapping = state_metadata.slot_mapping
        num_actual = slot_mapping.shape[0]
        block_table = state_metadata.block_table
        block_size = state_metadata.block_size

        # [num_blocks, block_size, kv_dim+score_dim], where kv_dim == score_dim
        state_cache = self.state_cache.kv_cache
        # kv_state stored in first half, score_state stored in second half
        state_width = state_cache.shape[-1] // 2
        pdl_kwargs = (
            {}
            if current_platform.is_rocm() or current_platform.is_xpu()
            else {"launch_pdl": False}
        )

        # Store the KV and score (with fused APE addition) in the state.
        # NOTE: PDL is disabled — both this kernel and the compress kernels
        # below depend on preceding kernel outputs (kv/score from the cublas
        # GEMM; state_cache from this kernel) but neither emits/waits on PDL
        # grid dependency primitives, so launch_pdl=True caused a
        # read-after-write race and non-deterministic output.
        save_partial_states(
            kv=kv,
            score=score,
            ape=self.ape,
            positions=positions,
            state_cache=state_cache,
            slot_mapping=slot_mapping,
            block_size=block_size,
            state_width=state_width,
            compress_ratio=self.compress_ratio,
            pdl_kwargs=pdl_kwargs,
        )

        # Fused: compress → RMSNorm → RoPE → FP8 quant → KV cache write.
        # RoPE requirements (kernel applies forward GPT-J style rotation):
        # - is_neox_style=False (interleaved pairs, NOT split-half)
        # - cos_sin_cache layout: [max_pos, rope_head_dim] with first half cos,
        #   second half sin (per-pair, length rope_head_dim // 2 each)
        # - applied to LAST rope_head_dim elements of head_dim
        # - position used: (positions // compress_ratio) * compress_ratio
        cos_sin_cache = rotary_emb.cos_sin_cache
        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        k_cache_layer = self._static_forward_context[self.k_cache_prefix]
        kv_cache = k_cache_layer.kv_cache

        # Plain-row V4 reads a contiguous bf16 / per-tensor fp8 cache row; the
        # fp8_ds_mla path uses the UE8M0 paged uint8 layout.
        store_full_kv = self.head_dim == 512 and kv_cache.dtype != torch.uint8
        store_full_fp8 = kv_cache.dtype == torch.float8_e4m3fn
        fp8_scale = (
            getattr(k_cache_layer, "_flashinfer_fp8_kv_scale", None)
            if store_full_fp8
            else None
        )

        # cutedsl (head=512) accepts the full-cache flags; triton (indexer/AMD)
        # does not, so the two callables have different signatures.
        compress_norm_rope_store_fn: Any
        if current_platform.is_cuda() and self.head_dim == 512:
            from .nvidia.ops.sparse_attn_compress_cutedsl import (
                compress_norm_rope_store_cutedsl,
            )

            # head=512 on CUDA always uses cutedsl, for both the fp8_ds_mla
            # layout and the plain full-cache layout. The full-cache flags
            # are consumed only here.
            compress_norm_rope_store_fn = compress_norm_rope_store_cutedsl
            extra_kwargs: dict[str, Any] = dict(
                store_full_kv=store_full_kv,
                store_full_fp8=store_full_fp8,
                fp8_scale=fp8_scale,
            )
        else:
            # Indexer path (head_dim == 128) or non-CUDA GPUs (AMD, XPU, etc.).
            compress_norm_rope_store_fn = compress_norm_rope_store_triton
            extra_kwargs = {}

        compress_norm_rope_store_fn(
            state_cache=state_cache,
            num_actual=num_actual,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            block_size=block_size,
            state_width=state_width,
            cos_sin_cache=cos_sin_cache,
            kv_cache=kv_cache,
            k_cache_metadata=k_cache_metadata,
            pdl_kwargs=pdl_kwargs,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            compress_ratio=self.compress_ratio,
            overlap=self.overlap,
            use_fp4_cache=self.use_fp4_cache,
            rms_norm_weight=self.norm.weight,
            rms_norm_eps=self.rms_norm_eps,
            quant_block=self._quant_block,
            token_stride=self._token_stride,
            scale_dim=self._scale_dim,
            **extra_kwargs,
        )


class DeepseekV41RefCompressor(nn.Module):
    """Reference-faithful (correctness-first) DeepSeek-V4.1-Flash KV compressor.

    Only kv_source layers own one of these. It expresses the released config's
    ratios {1, 2} that the v4 fused ``DeepseekCompressor`` cannot (v4 assumes
    {4, 128} + APE + overlap):

      * ratio 1 -- one token per group: a plain bf16 projection + RMSNorm, no
        gate, no fp32 state.
      * ratio 2 -- fp32 softmax-gated non-overlapping pool-of-2, carrying a
        partial trailing group across decode steps in ``kv_state`` /
        ``score_state``.

    The weight names ``norm`` / ``wkv`` / ``wgate`` match the reference
    ``inference/model.py::Compressor`` (and the CPU-validated
    ``compressor_ref.py``) so a reference/checkpoint state_dict loads directly.

    ``forward`` returns the latent BEFORE RoPE (the indexer derives index keys
    from the unrotated latent; attention rotates + quants + stores afterwards),
    or ``None`` while a decode group is still filling.

    TODO(gpu-gate / kernel-fusion): this class carries the correct pooling MATH.
    Two integration seams are GPU-gated and NOT yet fused (see
    attention_ced_adaptation_report.md):
      1. ragged continuous-batch pooling: the reference contract below is the
         per-request ``[bsz, seqlen]`` view. In the paged runtime, tokens are
         flattened with a ``slot_mapping`` / ``token_to_req_indices`` and the
         partial-group carry must be keyed per request. The fused v4 path does
         this with ``save_partial_states`` + ``CompressorStateCache``; the
         ratio-{1,2} equivalent still needs to be written/validated on GPU.
      2. paged store of the precomputed latent: RoPE(tail) + quant (fp8_ds_mla
         UE8M0 for head=512 / mxfp4 for the indexer head=128) + scatter into the
         paged compress-KV cache. The v4 cutedsl/triton store kernels FUSE the
         pooling with the store and assume APE/overlap, so a store-only kernel
         (or a validated torch scatter matching the exact packed layout) is
         required. Until then the compressed serving path is structurally wired
         but not numerically executable end-to-end.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        rope_head_dim: int,
        prefix: str = "",
        k_cache_prefix: str = "",
    ):
        super().__init__()
        assert compress_ratio in (1, 2), (
            f"kv_source compressor supports ratios 1/2, got {compress_ratio}"
        )
        config = vllm_config.model_config.hf_config
        self.compress_ratio = compress_ratio
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.prefix = prefix
        self.k_cache_prefix = k_cache_prefix
        self.rms_norm_eps = config.rms_norm_eps
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs

        # ratio 1 stays bf16 (plain projection); ratio 2's fp32 pooling promotes
        # its projection weights to fp32 -- matches the reference exactly.
        wkv_dtype = torch.float32 if compress_ratio > 1 else torch.bfloat16
        self.norm = RMSNorm(head_dim, self.rms_norm_eps)
        # Raw parameters (not quantized Linear) so the pooling math is bit-exact
        # against compressor_ref.py and the reference state_dict names line up.
        self.wkv = nn.Parameter(
            torch.empty(head_dim, hidden_size, dtype=wkv_dtype),
            requires_grad=False,
        )
        if compress_ratio > 1:
            self.wgate = nn.Parameter(
                torch.empty(head_dim, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            # Per-request partial trailing group, carried across decode steps.
            # (Retained only for the reference ``forward`` below, which the paged
            # runtime never calls; the paged pooling uses the ring buffers.)
            state_shape = (self.max_num_reqs, compress_ratio, head_dim)
            self.register_buffer(
                "kv_state",
                torch.zeros(state_shape, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "score_state",
                torch.full(state_shape, -torch.inf, dtype=torch.float32),
                persistent=False,
            )
            # Spec-decode-safe pool-of-2 carry. The single-slot kv_state above
            # assumes MONOTONIC single-pass processing: each decode step advances
            # the carry from the batch's trailing token. Under DSpark spec-verify
            # a step processes nspec+1 SPECULATIVE tokens and the trailing token
            # may be REJECTED -- with no rollback the carry (hence every pooled
            # compressed-KV latent) is corrupted for all subsequent steps
            # (progressive garbling of ~half the backbone's KV). Instead of one
            # carry slot we keep a per-request RING of recent pool-OPENER
            # (even-position) projections keyed by absolute pool index
            # k = pos // ratio. A pool closing at odd position o = ratio*k + 1
            # reads opener ratio*k from ring[k % ring_size]. Reprocessing after a
            # rejection overwrites the same ring slot with the corrected value
            # BEFORE it is read (exactly like the paged compress-KV store
            # self-heals), so no rollback hook into the spec machinery is needed.
            # ring_size > nspec + 1 guarantees an opener is never clobbered by a
            # newer pool before its next-step read. Bit-identical to the
            # single-slot carry for non-spec decode/prefill (one live opener).
            self.ring_size = 64
            ring_shape = (self.max_num_reqs, self.ring_size, head_dim)
            self.register_buffer(
                "even_kv_ring",
                torch.zeros(ring_shape, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "even_score_ring",
                torch.zeros(ring_shape, dtype=torch.float32),
                persistent=False,
            )

        self._static_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )

    def forward(
        self, x: torch.Tensor, start_pos: int
    ) -> torch.Tensor | None:
        """Reference contract (``[bsz, seqlen, hidden]``), mirrors
        ``inference/model.py::Compressor.forward`` and the CPU-validated
        ``compressor_ref.py``. Returns the pre-RoPE latent or ``None``.

        The paged runtime does NOT call this directly on a ragged batch -- see the
        class-level TODO. It is kept as the numerically-authoritative reference
        that the fused ratio-{1,2} kernel must match at the GPU golden-diff gate.
        """
        bsz, seqlen, _ = x.size()
        ratio, dtype = self.compress_ratio, x.dtype
        if ratio == 1:  # one token per group: no pooling, no gate, no fp32
            return self.norm(F.linear(x, self.wkv))

        x = x.float()
        kv, score = F.linear(x, self.wkv), F.linear(x, self.wgate)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder:  # trailing partial group waits in the state
                kv, self.kv_state[:bsz, :remainder] = kv.split(
                    [cutoff, remainder], dim=1
                )
                score, self.score_state[:bsz, :remainder] = score.split(
                    [cutoff, remainder], dim=1
                )
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:  # decode: fill a slot, pool only when the group just completed
            should_compress = (start_pos + 1) % ratio == 0
            slot = start_pos % ratio
            self.kv_state[:bsz, slot] = kv.squeeze(1)
            self.score_state[:bsz, slot] = score.squeeze(1)
            if should_compress:
                kv = (
                    self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)
                ).sum(dim=1, keepdim=True)
        if not should_compress:
            return None
        return self.norm(kv.to(dtype))

    def pool_ragged(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        """Ragged paged adapter over the reference pooling math.

        Produces a per-input-token latent ``[num_tokens, head_dim]`` (model
        dtype) where each group-boundary row holds the pooled + normed pre-RoPE
        latent and every other row is left zero (its compressed slot is -1, so
        the paged store skips it). Reuses the exact ratio-{1,2} arithmetic of
        ``forward`` but indexes the per-request ``kv_state`` / ``score_state`` by
        request slot, so single-shot prefill, chunked prefill, and multi-step
        decode all carry the trailing partial group correctly.

        State is keyed by the request's position in the (reordered) batch, which
        is stable across steps for a running single sequence. Batch churn that
        moves a request between slots mid-generation is the milestone-2 paged
        state case; single-stream and stable batches are exact here.
        """
        ratio = self.compress_ratio
        hd = self.head_dim
        if ratio == 1:
            # One token per group: plain bf16 projection + norm, all boundaries.
            return self.norm(F.linear(hidden_states, self.wkv))

        # ratio == 2: fp32 softmax pool-of-2 with per-request trailing carry.
        num_tokens = hidden_states.shape[0]
        latent = hidden_states.new_zeros(num_tokens, hd)
        if num_tokens == 0:
            return latent
        kv_all = F.linear(hidden_states.float(), self.wkv)  # [T, hd] fp32
        score_all = F.linear(hidden_states.float(), self.wgate)  # [T, hd] fp32
        device = hidden_states.device

        # ---- Uniform-width-Q fast path (cudagraph-capturable, sync-free) ----
        # Every request contributes exactly Q query tokens laid out request-major
        # (starts == arange * Q), so flat index for (req r, column j) is r*Q + j.
        #   Q == 1        -> base pure-decode (the only shape captured pre-spec).
        #   Q == 1+nspec  -> DSpark spec-verify block (the shape captured WITH
        #                    speculation).
        # CUDAGRAPH PADDING: hidden_states/positions are padded to the capture
        # size, so ``hidden_states.shape[0]`` OVER-counts. Padding is token-level
        # (the verify block is not always a multiple of a capture bucket, e.g.
        # 1 req * 6 tok pads to an 8-tok bucket), so the padding columns carry
        # STALE positions -- and a stale even position collides with a real
        # ring slot, silently overwriting a live opener (France-sensitive,
        # Primes-robust, diverges mid-stream not at token 0). Drive Q from the
        # REAL query-token count instead: query_start_loc_cpu is a CPU tensor,
        # so int(...) is a capture-safe constant read (host syncs on GPU tensors
        # -- .item()/.tolist() -- are the thing to avoid), and for uniform
        # decode/verify the real count num_reqs*(1+nspec) is fixed across steps,
        # so baking it at capture is correct. ``latent`` keeps the padded row
        # count; padding rows stay zero (their compressed slot is -1 -> skipped).
        real_tokens = int(query_start_loc_cpu[num_reqs]) if num_reqs > 0 else 0
        uniform_q = num_reqs > 0 and real_tokens % num_reqs == 0
        Q = real_tokens // num_reqs if uniform_q else 0
        if uniform_q and 1 <= Q <= 16:
            R = num_reqs
            rows = torch.arange(R, device=device)
            for j in range(Q):
                flat = rows * Q + j  # [R] request-major column j
                pos = positions[flat]  # [R] int, gpu
                kv = kv_all[flat]  # [R, hd]
                score = score_all[flat]  # [R, hd]
                is_close = (pos % ratio).bool().unsqueeze(-1)  # [R,1] odd => close
                # Opener (even ratio*k) and closer (odd ratio*k+1) share pool index
                # k, so a closer reads its opener from ring[k]. Position-keyed =>
                # spec-rollback-safe. All-tensor indexing keeps this capture-safe.
                k = torch.div(pos, ratio, rounding_mode="floor")  # [R]
                ring_idx = k % self.ring_size  # [R]
                prev_kv = self.even_kv_ring[rows, ring_idx]  # [R, hd] opener
                prev_sc = self.even_score_ring[rows, ring_idx]
                g_kv = torch.stack([prev_kv, kv], dim=1)  # [R, 2, hd]
                g_sc = torch.stack([prev_sc, score], dim=1)  # [R, 2, hd]
                pooled = (g_kv * g_sc.softmax(dim=1)).sum(dim=1)  # [R, hd]
                normed = self.norm(pooled.to(hidden_states.dtype))
                # Emit pooled latent on close steps; leave zero otherwise (its
                # compressed slot is -1, so the paged store skips it).
                latent[flat] = torch.where(is_close, normed, latent[flat])
                # On open steps (pos even) publish this token as the group opener;
                # on close steps keep the stored opener untouched.
                self.even_kv_ring[rows, ring_idx] = torch.where(
                    is_close, prev_kv, kv
                )
                self.even_score_ring[rows, ring_idx] = torch.where(
                    is_close, prev_sc, score
                )
            return latent

        # ---- General ragged path (prefill / chunked / mixed / spec-verify) --
        # Spec-verify (qlen = nspec+1 for one req) lands here. Correctness under
        # speculative rollback comes from the position-keyed opener ring (see
        # __init__): the lead reads opener (start_pos-1) from the ring rather
        # than a single carry slot that a rejected trailing token would poison.
        starts = query_start_loc_cpu.tolist()
        RING = self.ring_size
        for r in range(num_reqs):
            s, e = int(starts[r]), int(starts[r + 1])
            L = e - s
            if L <= 0:
                continue
            start_pos = int(positions[s].item())
            kv = kv_all[s:e]  # [L, hd]
            score = score_all[s:e]  # [L, hd]
            # Publish this window's openers (even absolute positions) to the ring
            # keyed by pool index BEFORE pooling, so a future step whose lead
            # closes one of them (after a rejection lands there) reads the right
            # opener. The lead below reads pool (start_pos-1)//ratio, which is a
            # SMALLER k written in a prior step, so it is never clobbered here.
            local_pos = start_pos + torch.arange(L, device=device)
            even_local = (local_pos % ratio == 0).nonzero(as_tuple=True)[0]
            if even_local.numel():
                eidx = torch.div(
                    local_pos[even_local], ratio, rounding_mode="floor"
                ) % RING
                self.even_kv_ring[r, eidx] = kv[even_local]
                self.even_score_ring[r, eidx] = score[even_local]
            lead = start_pos % ratio  # 1 -> first token closes a carried group
            out_idx: list[torch.Tensor] = []
            out_rows: list[torch.Tensor] = []
            if lead:
                # First token closes the group opened by absolute position
                # start_pos-1 (an already-confirmed even token); read its opener
                # projection from the ring keyed by pool index
                # (start_pos-1)//ratio.
                k0 = (start_pos - 1) // ratio % RING
                g_kv = torch.stack(
                    [self.even_kv_ring[r, k0], kv[0]], dim=0
                )  # [2, hd]
                g_sc = torch.stack(
                    [self.even_score_ring[r, k0], score[0]], dim=0
                )
                pooled0 = (g_kv * g_sc.softmax(dim=0)).sum(dim=0)  # [hd]
                out_idx.append(torch.tensor([s], device=device))
                out_rows.append(pooled0.unsqueeze(0))
            n_body = (L - lead) // ratio
            if n_body:
                b0 = lead
                body_kv = kv[b0 : b0 + ratio * n_body].reshape(n_body, ratio, hd)
                body_sc = score[b0 : b0 + ratio * n_body].reshape(n_body, ratio, hd)
                pooled = (body_kv * body_sc.softmax(dim=1)).sum(dim=1)  # [n_body, hd]
                # Boundary local indices: b0 + (ratio-1), b0 + ratio + (ratio-1), ...
                bidx = (
                    b0 + torch.arange(n_body, device=device) * ratio + (ratio - 1)
                )
                out_idx.append(s + bidx)
                out_rows.append(pooled)
            # Trailing partial group: its opener was already published to the ring
            # above. Re-publish it LAST (position-keyed) so it wins any ring
            # aliasing from a long (> ring_size*ratio) prefill window, where the
            # trailing opener is the only one whose read is deferred to next step.
            if (L - lead) % ratio:
                ti = lead + ratio * n_body
                k_tail = (start_pos + ti) // ratio % RING
                self.even_kv_ring[r, k_tail] = kv[ti]
                self.even_score_ring[r, k_tail] = score[ti]
            if out_rows:
                rows = torch.cat(out_rows, dim=0)  # [M, hd] fp32
                idx = torch.cat(out_idx, dim=0)  # [M]
                latent[idx] = self.norm(rows.to(hidden_states.dtype))
        return latent
