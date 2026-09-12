# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import typing
from collections.abc import Callable, Iterable
from itertools import islice

import regex as re
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_fused_post_pre_tilelang,
    mhc_post_tilelang,
    mhc_pre_tilelang,
)
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.router.base_router import (
    eplb_map_to_physical_and_record,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v41.attention import DeepseekV41Attention
from vllm.models.deepseek_v41.engram import (
    Engram,
    EngramLayout,
    EngramServingHasher,
    dsv41_engram_runtime_enabled,
)
from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
    DeepseekV41FlashInferMLAAttention,
    DeepseekV41FlashInferSM120Attention,
)
from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV41FlashMLAAttention
from vllm.models.deepseek_v41.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs
from vllm.models.deepseek_v41.nvidia.image_processor import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
)
from vllm.models.deepseek_v41.nvidia.processing import (
    DeepseekV41DummyInputsBuilder as _DeepseekV41DummyInputsBuilder,
)
from vllm.models.deepseek_v41.nvidia.processing import (
    DeepseekV41MultiModalProcessor as _DeepseekV41MultiModalProcessor,
)
from vllm.models.deepseek_v41.nvidia.processing import (
    DeepseekV41ProcessingInfo as _DeepseekV41ProcessingInfo,
)
from vllm.models.deepseek_v41.nvidia.vision import Aligner, ViT
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.ubatching import dbo_current_ubatch_id


class DeepseekV41MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        #
        # Block-FP8 shards in whole 128-blocks; cdiv rounds the per-rank block
        # count up so the linear's even TP split stays block-aligned, with the
        # trailing ranks zero-filled by load_weights.
        block_size = getattr(quant_config, "weight_block_size", None)
        if block_size is not None and not is_sequence_parallel:
            tp_size = get_tensor_model_parallel_world_size()
            n_local = cdiv(intermediate_size // block_size[0], tp_size)
            intermediate_size = n_local * block_size[0] * tp_size
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


def make_deepseek_v4_expert_params_mapping(
    num_experts: int,
) -> list[tuple[str, str, int, str]]:
    return [
        (
            "experts.w13_" if shard_id in ("w1", "w3") else "experts.w2_",
            f"experts.{expert_id}.{weight_name}.",
            expert_id,
            shard_id,
        )
        for expert_id in range(num_experts)
        for shard_id, weight_name in [
            ("w1", "w1"),
            ("w2", "w2"),
            ("w3", "w3"),
        ]
    ]


class DeepseekV41MegaMoEExperts(nn.Module):
    _symm_buffer_cache: dict[tuple[int, int, int, int, int, int, int], object] = {}

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        num_experts: int,
        num_local_experts: int,
        experts_start_idx: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        prefix: str = "",
        num_logical_experts: int | None = None,
    ):
        super().__init__()
        self.prefix = prefix
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_start_idx = experts_start_idx
        self.experts_end_idx = experts_start_idx + num_local_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        self.num_logical_experts = (
            num_logical_experts if num_logical_experts is not None else num_experts
        )

        self.eplb_state = EplbLayerState()

        weight_attrs = {"weight_loader": self.weight_loader}
        self.w13_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_size,
                hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight, weight_attrs)

        self.w13_weight_scale = nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_size,
                hidden_size // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight_scale, weight_attrs)
        self.w13_weight_scale.quant_method = "block"

        self.w2_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                hidden_size,
                intermediate_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight, weight_attrs)

        self.w2_weight_scale = nn.Parameter(
            torch.zeros(
                num_local_experts,
                hidden_size,
                intermediate_size // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight_scale, weight_attrs)
        self.w2_weight_scale.quant_method = "block"

        self._transformed_l1_weights: tuple[torch.Tensor, torch.Tensor] | None = None
        self._transformed_l2_weights: tuple[torch.Tensor, torch.Tensor] | None = None

        # Register in the static forward context so the custom-op wrapper
        # can look up this module by name from within a torch.compile graph.
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def _map_global_expert_id(self, expert_id: int) -> list[int]:
        """Return local (per-rank) slot offsets where logical expert
        `expert_id` should land on this rank.
        """
        physical_ids: list[int] = []
        for p in range(self.experts_start_idx, self.experts_end_idx):
            if p % self.num_logical_experts == expert_id:
                physical_ids.append(p - self.experts_start_idx)
        return physical_ids

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        local_expert_ids = self._map_global_expert_id(expert_id)
        if not local_expert_ids:
            return False if return_success else None

        loaded_any = False
        for local_expert_id in local_expert_ids:
            expert_data = param.data[local_expert_id]
            if shard_id in ("w1", "w3"):
                if "w13_" not in weight_name:
                    continue
                shard_offset = 0 if shard_id == "w1" else self.intermediate_size
                expert_data = expert_data.narrow(
                    0, shard_offset, self.intermediate_size
                )
            elif shard_id == "w2":
                if "w2_" not in weight_name:
                    continue
            else:
                raise ValueError(f"Unsupported expert shard id: {shard_id}")

            if expert_data.shape != loaded_weight.shape:
                raise ValueError(
                    f"DeepSeek V4 MegaMoE expert weight shape mismatch for "
                    f"{weight_name}: parameter shard {tuple(expert_data.shape)} "
                    f"vs checkpoint {tuple(loaded_weight.shape)}"
                )
            expert_data.copy_(loaded_weight)
            loaded_any = True

        if return_success:
            return loaded_any
        return None

    @staticmethod
    def _ue8m0_uint8_to_float(sf: torch.Tensor) -> torch.Tensor:
        return (sf.to(torch.int32) << 23).view(torch.float32)

    def _check_runtime_supported(self) -> None:
        device = self.w13_weight.device
        if torch.cuda.get_device_capability(device)[0] != 10:
            raise NotImplementedError("DeepGEMM MegaMoE requires SM100 GPUs.")
        if self.hidden_size % 128 != 0 or self.intermediate_size % 128 != 0:
            raise ValueError(
                "DeepGEMM MegaMoE requires hidden and intermediate sizes "
                "to be multiples of 128."
            )

    def finalize_weights(self) -> None:
        if self._transformed_l1_weights is not None:
            return

        self._check_runtime_supported()
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()

        w13_scale = deep_gemm.transform_sf_into_required_layout(
            self._ue8m0_uint8_to_float(self.w13_weight_scale.data).contiguous(),
            2 * self.intermediate_size,
            self.hidden_size,
            (1, 32),
            self.num_local_experts,
        )
        w2_scale = deep_gemm.transform_sf_into_required_layout(
            self._ue8m0_uint8_to_float(self.w2_weight_scale.data).contiguous(),
            self.hidden_size,
            self.intermediate_size,
            (1, 32),
            self.num_local_experts,
        )
        self._transformed_l1_weights, self._transformed_l2_weights = (
            deep_gemm.transform_weights_for_mega_moe(
                (self.w13_weight.data.view(torch.int8).contiguous(), w13_scale),
                (self.w2_weight.data.view(torch.int8).contiguous(), w2_scale),
            )
        )
        # Drop the original loader-side parameters: the MegaMoE kernels only
        # consume the transformed views above. transform_weights_for_mega_moe
        # allocates a fresh tensor for the L1 weight (see _interleave_l1_weights)
        # and fresh SF tensors for L1/L2; the L2 weight is the only tensor that
        # aliases the original storage, and _transformed_l2_weights still holds
        # it, so the storage stays live after we drop the Parameter.
        self.w13_weight = None
        self.w13_weight_scale = None
        self.w2_weight = None
        self.w2_weight_scale = None

    def get_symm_buffer(self):
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()

        group = get_ep_group().device_group
        device = torch.accelerator.current_device_index()
        key = (
            id(group),
            device,
            self.num_experts,
            self.max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
        )
        symm_buffer = self._symm_buffer_cache.get(key)
        if symm_buffer is None:
            symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
                group,
                self.num_experts,
                self.max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
            )
            self._symm_buffer_cache[key] = symm_buffer
        return symm_buffer

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        self.eplb_state.set_layer_state(
            moe_layer_idx,
            expert_load_view,
            logical_to_physical_map,
            logical_replica_count,
        )

    def get_expert_weights(self) -> list[torch.Tensor]:
        self.finalize_weights()
        assert self._transformed_l1_weights is not None
        assert self._transformed_l2_weights is not None

        def _to_eplb_view(name: str, t: torch.Tensor) -> torch.Tensor:
            """Return a (num_local_experts, -1) view with contiguous memory layout."""
            assert t.shape[0] == self.num_local_experts
            if t.is_contiguous():
                return t.view(self.num_local_experts, -1)
            elif t.dim() == 3 and t.stride(1) == 1 and t.stride(2) == t.shape[1]:
                # scales have shape (E, M, N) with memory layout (E, N, M)
                back = torch.transpose(t, 1, 2)
                assert back.is_contiguous()
                return back.view(self.num_local_experts, -1)

            raise AssertionError(
                f"DSv4 EPLB {name}: non-contiguous expert tensor with "
                f"unexpected layout shape={tuple(t.shape)} "
                f"stride={tuple(t.stride())} dtype={t.dtype}"
            )

        return [
            _to_eplb_view("l1_packed", self._transformed_l1_weights[0]),
            _to_eplb_view("l1_scale", self._transformed_l1_weights[1]),
            _to_eplb_view("l2_weight", self._transformed_l2_weights[0]),
            _to_eplb_view("l2_scale", self._transformed_l2_weights[1]),
        ]

    def update_expert_map(self) -> None:
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        activation_clamp: float | None,
        fast_math: bool = True,
    ) -> torch.Tensor:
        if hidden_states.shape[0] > self.max_num_tokens:
            raise ValueError(
                f"DeepSeek V4 MegaMoE got {hidden_states.shape[0]} tokens, "
                f"but the symmetric buffer was sized for {self.max_num_tokens}."
            )
        y = torch.empty_like(hidden_states, dtype=torch.bfloat16)

        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()

        symm_buffer = self.get_symm_buffer()
        num_tokens = hidden_states.shape[0]
        is_padding = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            is_padding = get_forward_context().is_padding
            if is_padding is not None:
                is_padding = is_padding[:num_tokens]

        # EPLB: map logical expert IDs to physical replicas and record load.
        eplb_state = self.eplb_state
        if eplb_state.logical_to_physical_map is not None:
            assert eplb_state.expert_load_view is not None
            assert eplb_state.logical_replica_count is not None
            assert eplb_state.should_record_tensor is not None
            if is_padding is not None:
                topk_ids = torch.where(is_padding.unsqueeze(1), -1, topk_ids)
            topk_ids = eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                expert_load_view=eplb_state.expert_load_view,
                logical_to_physical_map=eplb_state.logical_to_physical_map,
                logical_replica_count=eplb_state.logical_replica_count,
                record_enabled=eplb_state.should_record_tensor,
                num_unpadded_tokens=eplb_state.num_unpadded_tokens_tensors[
                    dbo_current_ubatch_id()
                ]
                if eplb_state.num_unpadded_tokens_tensors is not None
                else None,
            )

        prepare_megamoe_inputs(
            hidden_states,
            topk_weights,
            topk_ids,
            symm_buffer.x[:num_tokens],
            symm_buffer.x_sf[:num_tokens],
            symm_buffer.topk_idx[:num_tokens],
            symm_buffer.topk_weights[:num_tokens],
            is_padding=is_padding,
        )

        # This method must have been already called during the weight loading phase.
        # We call it again here to cover the dummy weight loading case.
        self.finalize_weights()

        assert self._transformed_l1_weights is not None
        assert self._transformed_l2_weights is not None
        deep_gemm.fp8_fp4_mega_moe(
            y,
            self._transformed_l1_weights,
            self._transformed_l2_weights,
            symm_buffer,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )
        return y


DeepseekV41MegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]


class DeepseekV41MoE(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.prefix = prefix
        self.use_mega_moe = (
            vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
        )
        if self.use_mega_moe and not vllm_config.parallel_config.enable_expert_parallel:
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE currently requires expert parallel. "
                "Enable it with --enable-expert-parallel, or pick a different "
                "moe backend."
            )

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.hidden_size = config.hidden_size

        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.renormalize = config.norm_topk_prob
        self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")
        if self.use_mega_moe and self.scoring_func != "sqrtsoftplus":
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE currently supports sqrtsoftplus routing only."
            )
        if self.use_mega_moe and getattr(config, "expert_dtype", "fp4") != "fp4":
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE only supports fp4 experts; got expert_dtype="
                f"{config.expert_dtype!r}. Drop --kernel-config moe_backend="
                "deep_gemm_mega_moe for this checkpoint."
            )

        self.gate = GateLinear(
            input_size=config.hidden_size,
            output_size=config.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        self.gate.e_score_correction_bias = None
        self.gate.tid2eid = None
        # v41 has no token-id hash-MoE gate layers (a v4-only concept: the
        # checkpoint carries zero tid2eid tensors and the reference Gate does
        # plain topk routing). Its early-layer memory augmentation is the engram
        # n-gram table attached at ``engram_layer_ids``, handled separately.
        # Default num_hash_layers to 0 so is_hash_moe never fires for v41.
        is_hash_moe = extract_layer_index(prefix) < getattr(
            config, "num_hash_layers", 0
        )
        self.hash_indices_dtype = torch.int64 if self.use_mega_moe else torch.int32
        if is_hash_moe:
            # hash MoE doesn't use e_score_correction_bias
            # Use randint instead of empty to avoid garbage values causing
            # invalid memory access in dummy mode (--load-format="dummy")
            self.gate.tid2eid = nn.Parameter(
                torch.randint(
                    0,
                    config.n_routed_experts,
                    (config.vocab_size, config.num_experts_per_tok),
                    dtype=self.hash_indices_dtype,
                ),
                requires_grad=False,
            )
        elif getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        # Vision (VL) routing correction bias. Image-span tokens route with a
        # separate ``bias_vl`` (training ``noaux_tc_for_vl``); it steers expert
        # SELECTION only, exactly like ``bias``. The checkpoint stores it at
        # ``layers.N.ffn.gate.bias_vl`` (43 tensors), so it must live on the gate.
        self.gate.bias_vl = None
        self.image_token_id = getattr(config, "image_token_id", None)
        if (
            getattr(config, "vision_enabled", False)
            and getattr(config, "topk_method", None) == "noaux_tc"
        ):
            self.gate.bias_vl = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = DeepseekV41MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=quant_config,
                reduce_results=self.use_mega_moe,
                prefix=f"{prefix}.shared_experts",
            )

        if self.use_mega_moe:
            self._init_mega_moe_experts(vllm_config, config, prefix)
        else:
            self._init_fused_moe_experts(vllm_config, config, quant_config, prefix)

    def _init_mega_moe_experts(
        self,
        vllm_config: VllmConfig,
        config,
        prefix: str,
    ) -> None:
        self.ep_group = get_ep_group()
        self.ep_size = self.ep_group.world_size
        self.ep_rank = self.ep_group.rank_in_group

        eplb_config = vllm_config.parallel_config.eplb_config
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts or 0
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        assert self.n_physical_experts % self.ep_size == 0, (
            f"n_physical_experts={self.n_physical_experts} must be divisible by "
            f"ep_size={self.ep_size}. Adjust num_redundant_experts."
        )
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.n_local_experts = self.n_local_physical_experts
        self.experts_start_idx = self.physical_expert_start
        self.experts_end_idx = self.physical_expert_end

        self.experts = DeepseekV41MegaMoEExperts(
            vllm_config,
            num_experts=self.n_physical_experts,
            num_local_experts=self.n_local_physical_experts,
            experts_start_idx=self.physical_expert_start,
            num_logical_experts=self.n_logical_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            prefix=f"{prefix}.experts",
        )

    def _init_fused_moe_experts(
        self,
        vllm_config: VllmConfig,
        config,
        quant_config,
        prefix: str,
    ) -> None:
        parallel_config = vllm_config.parallel_config
        self.tp_rank = get_tensor_model_parallel_rank()

        eplb_config = parallel_config.eplb_config
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_shared_experts = config.n_shared_experts or 0
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        assert self.n_physical_experts % self.tp_size == 0, (
            f"n_physical_experts={self.n_physical_experts} must be divisible by "
            f"tp_size={self.tp_size}. Adjust num_redundant_experts."
        )
        self.n_local_physical_experts = self.n_physical_experts // self.tp_size
        self.n_local_experts = self.n_local_physical_experts
        self.experts_start_idx = self.tp_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.physical_expert_start = self.experts_start_idx
        self.physical_expert_end = self.experts_end_idx

        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=self.gate.tid2eid,
            swiglu_limit=self.swiglu_limit,
            router_logits_dtype=torch.float32,
            enable_eplb=parallel_config.enable_eplb,
            num_redundant_experts=eplb_config.num_redundant_experts,
        )

    def _vl_image_mask(self, input_ids: torch.Tensor | None) -> torch.Tensor | None:
        """[n] bool mask of image-span tokens, or None when there are none.

        Every image-span position (delimiters included) carries ``image_token_id``
        in ``input_ids`` -- this recovers exactly the reference ``token_types >= 0``
        image mask without threading token-type arrays through the runtime.
        Returns None (cheap) whenever the vision bias is absent or no image token
        is present, so text decode/prefill never leaves the fast path."""
        if self.gate.bias_vl is None or input_ids is None:
            return None
        # The presence check below does a host sync (``bool(.any())``), which is
        # illegal inside a CUDA graph capture. Captured graphs are only ever text
        # decode / text prefill -- image prefills run eager (see forward comments)
        # -- so there are never image-span tokens under capture. Returning None
        # keeps capture on the fused fast path, losslessly. Under DSpark the
        # runner threads input_ids into the captured region (the drafter needs
        # them), so unlike the base model this method is now reached during
        # capture; without this guard the .any() sync aborts graph capture.
        if torch.cuda.is_current_stream_capturing():
            return None
        image_mask = input_ids == self.image_token_id
        if not bool(image_mask.any()):
            return None
        return image_mask.flatten()

    def _ref_gate_topk(
        self, router_logits: torch.Tensor, image_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference-faithful gate (mirrors gate_ref.GateRef.forward) with per-token
        bias/bias_vl selection. ``router_logits`` is the raw gate projection output
        (== ``F.linear(x, weight)``); the correction bias steers selection only, the
        routing weights come from the unbiased scores."""
        scores = router_logits.float()
        if self.scoring_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.scoring_func == "sigmoid":
            scores = scores.sigmoid()
        else:  # sqrtsoftplus
            scores = torch.nn.functional.softplus(scores).sqrt()
        bias = self.gate.e_score_correction_bias.float()
        bias = torch.where(image_mask.unsqueeze(-1), self.gate.bias_vl.float(), bias)
        indices = (scores + bias).topk(self.n_activated_experts, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.renormalize and self.n_activated_experts > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.routed_scaling_factor
        return weights.to(torch.float32), indices.to(self.hash_indices_dtype)

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.gate.tid2eid is not None and input_ids is None:
            raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")

        if not self.use_mega_moe:
            return self._forward_fused_moe(hidden_states, input_ids)

        org_shape = hidden_states.shape
        router_logits, _ = self.gate(hidden_states)
        # Image-span tokens (input_ids == image_token_id) route with bias_vl. The
        # fused kernel takes a single global correction bias, so when an image span
        # is present we fall back to the reference-faithful torch gate that mixes
        # bias/bias_vl per token. This branch only fires on (non-captured) image
        # prefills; every decode step and text-only prefill keeps the fused path.
        image_mask = self._vl_image_mask(input_ids)
        if image_mask is not None:
            topk_weights, topk_ids = self._ref_gate_topk(router_logits, image_mask)
        else:
            topk_weights, topk_ids = fused_topk_bias(
                hidden_states=hidden_states,
                gating_output=router_logits,
                scoring_func=self.scoring_func,
                e_score_correction_bias=self.gate.e_score_correction_bias.data
                if self.gate.e_score_correction_bias is not None
                else None,
                topk=self.n_activated_experts,
                renormalize=self.renormalize,
                indices_type=self.hash_indices_dtype,
                input_tokens=input_ids,
                hash_indices_table=self.gate.tid2eid,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        activation_clamp = (
            float(self.swiglu_limit) if self.swiglu_limit is not None else None
        )
        final_hidden_states = self.experts(
            hidden_states,
            topk_weights,
            topk_ids,
            activation_clamp=activation_clamp,
        )

        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
            final_hidden_states += shared_output

        return final_hidden_states.view(org_shape)

    def _forward_fused_moe(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        org_shape = hidden_states.shape
        # Image-span tokens (input_ids == image_token_id) must route with
        # bias_vl. The internal-router FusedMoE runs its router with a single
        # global correction bias, so when an image span is present we precompute
        # per-token top-k with the reference gate (text rows come out identical
        # to the fused router; image rows use bias_vl) and hand it to the runner
        # via the router's precomputed-topk override. This keeps every downstream
        # step -- padding, shared experts, and all-reduce -- bit-identical to the
        # text path. Only fires on (non-captured) image prefills; text
        # decode/prefill never leaves the fast path below.
        image_mask = self._vl_image_mask(input_ids)
        if image_mask is not None:
            if self.experts.is_monolithic:
                # Monolithic kernels route inside the fused kernel and never
                # consult select_experts, so the precomputed-topk override
                # cannot reach them. Refuse rather than be silently lossy; the
                # fp4 production path is the deep_gemm_mega_moe branch above.
                raise NotImplementedError(
                    "DeepSeek-V4.1 image-span routing (bias_vl) on the generic "
                    "FusedMoE path requires a modular MoE backend (e.g. marlin); "
                    "the selected monolithic backend routes in-kernel."
                )
            router_logits, _ = self.gate(hidden_states)
            topk_weights, topk_ids = self._ref_gate_topk(router_logits, image_mask)
            router = self.experts.router
            router._precomputed_topk = (topk_weights, topk_ids)
            try:
                final_hidden_states = self.experts(
                    hidden_states=hidden_states,
                    router_logits=hidden_states,
                    input_ids=input_ids,
                )
            finally:
                router._precomputed_topk = None
            return final_hidden_states.view(org_shape)

        if self.experts.is_internal_router:
            # In this case, the gate/router runs inside the FusedMoE class
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states,
                input_ids=input_ids,
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )

        return final_hidden_states.view(org_shape)

    def finalize_mega_moe_weights(self) -> None:
        if self.use_mega_moe:
            self.experts.finalize_weights()


def _select_dsv4_attn_cls(vllm_config: VllmConfig) -> type[DeepseekV41Attention]:
    """Pick the CUDA sparse-MLA attention class for the configured backend.

    The generic CUDA backend selector does not instantiate DSv4 layers directly,
    so map generic sparse-MLA choices to the DSv4-specialized attention class.
    Without an explicit backend, SM12 defaults to FlashInfer while the other
    CUDA arches keep the FlashMLA path.
    """
    backend = vllm_config.attention_config.backend
    device_capability = current_platform.get_device_capability()
    if backend in (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
    ):
        raise ValueError(
            f"{backend.name} is not a DeepSeek V4 attention backend. "
            "Use FLASHINFER_MLA_SPARSE_DSV4 for DeepSeek V4 FlashInfer "
            "sparse MLA."
        )
    if backend == AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4:
        if device_capability is not None and device_capability.major == 12:
            return DeepseekV41FlashInferSM120Attention
        return DeepseekV41FlashInferMLAAttention
    if backend in (
        AttentionBackendEnum.FLASHMLA_SPARSE,
        AttentionBackendEnum.FLASHMLA_SPARSE_DSV4,
    ):
        return DeepseekV41FlashMLAAttention

    if device_capability is not None and device_capability.major == 12:
        return DeepseekV41FlashInferSM120Attention
    return DeepseekV41FlashMLAAttention


class DeepseekV41DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
        engram_layout: EngramLayout | None = None,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.layer_idx = extract_layer_index(prefix)

        self.rms_norm_eps = config.rms_norm_eps
        self.attn = _select_dsv4_attn_cls(vllm_config)(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        )
        self.ffn = DeepseekV41MoE(vllm_config, prefix=f"{prefix}.ffn")

        # Engram n-gram hash memory (layers ``engram_layer_ids`` = 1, 14). The
        # two fp8 row tables (~98 GB each) stay HOST-resident (mmap of the
        # checkpoint shard by default; VLLM_DSV41_ENGRAM_TABLE=ram copies each
        # rank's row shard into anonymous host memory) -- only the gathered
        # [T, 24, 256] rows cross PCIe. The small tensors (fp8 wkv projection,
        # bf16 q/k gate weights) live on GPU: wkv is REPLICATED across TP ranks
        # exactly like the reference/sglang (the TP split of the engram is the
        # row-sharded table + all-reduce inside ParallelEngramEmbedding; the
        # projection and gate then run identically on every rank).
        self.engram = None
        if engram_layout is not None and self.layer_idx in engram_layout.layer_ids:
            n_hash_cols = (engram_layout.max_ngram_size - 1) * engram_layout.n_heads
            wkv = ReplicatedLinear(
                n_hash_cols * engram_layout.head_dim,
                config.hidden_size * (config.hc_mult + 1),
                bias=False,
                quant_config=vllm_config.quant_config,
                return_bias=False,
                prefix=f"{prefix}.engram.wkv",
            )
            self.engram = Engram(
                config,
                self.layer_idx,
                engram_layout,
                wkv,
                table_device=os.environ.get("VLLM_DSV41_ENGRAM_DEVICE", "cpu"),
                table_source=os.environ.get("VLLM_DSV41_ENGRAM_TABLE", "mmap"),
                ckpt_dir=vllm_config.model_config.model,
                ckpt_prefix=f"layers.{self.layer_idx}.engram",
            )

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def _hc_mixes3(self, R, fn, scale, base):
        """Faithful port of gold hc_mixes + hc_split_sinkhorn (model.py / kernel.py).

        R: [tok, hc_mult, hidden] -> (pre[tok,hc], post[tok,hc,1], comb[tok,hc,hc]),
        all fp32. `pre` collapses the hc copies (deferred one sub-block); `post`/`comb`
        expand the sublayer output back through hc_post."""
        tok = R.shape[0]
        hc = self.hc_mult
        x = R.reshape(tok, hc * self.hidden_size).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.rms_norm_eps)
        mixes = torch.matmul(x, fn.t()) * rsqrt
        pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + self.hc_eps
        post = torch.sigmoid(
            mixes[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc]
        ) * self.hc_post_alpha
        comb = mixes[:, 2 * hc :].view(tok, hc, hc) * scale[2] + base[2 * hc :].view(
            1, hc, hc
        )
        comb = torch.softmax(comb, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        return pre, post.unsqueeze(-1), comb

    def _hc_collapse(self, R, pre):
        """Collapse hc copies into one, weighted by `pre`. [tok,hc,d] x [tok,hc]->[tok,d]."""
        return (pre.unsqueeze(-1) * R.float()).sum(dim=1).to(R.dtype)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        pre_mix: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        # Hyper-Connections dataflow, faithful to gold Block.forward: the residual
        # stream is carried as hc_mult parallel copies (`residual`); `pre_mix` (the
        # collapse coefficient) is DEFERRED one sub-block -- attention collapses with
        # the previous FFN's pre, the FFN collapses with this attention's pre. `x` is
        # kept single-copy across the layer boundary and the previous sublayer's
        # hc_post (expand) is applied lazily at the start of the next sub-block.
        from vllm.model_executor.kernels.mhc.torch import mhc_post_torch

        # ---- attention sub-block ----
        if residual is None:
            # First layer: x is the expanded embedding [tok, hc, d]. Layer-0
            # attention collapses with the identity pre_mix (copy 0), matching gold
            # make_identity_pre_mix.
            R = x
            if pre_mix is None:
                pre_mix = R.new_zeros(R.shape[0], self.hc_mult, dtype=torch.float32)
                pre_mix[:, 0] = 1.0
        else:
            # Expand the previous layer's FFN output into the residual stream.
            R = mhc_post_torch(x, residual, post_mix, res_mix)
        residual = R
        attn_pre, post_mix, res_mix = self._hc_mixes3(
            R, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self._hc_collapse(R, pre_mix)
        x = self.attn_norm(x)
        x = self.attn(positions, x, None)
        pre_mix = attn_pre  # defer attention's pre for the FFN collapse

        # ---- FFN sub-block ----
        R = mhc_post_torch(x, residual, post_mix, res_mix)  # expand attn output
        residual = R
        ffn_pre, post_mix, res_mix = self._hc_mixes3(
            R, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self._hc_collapse(R, pre_mix)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        pre_mix = ffn_pre  # defer FFN's pre for the next layer's attention collapse

        return x, residual, post_mix, res_mix, pre_mix


class DeepseekV41Model(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.parallel_config = vllm_config.parallel_config
        self.use_mega_moe = (
            vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
        )
        if self.use_mega_moe and not vllm_config.parallel_config.enable_expert_parallel:
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE currently requires expert parallel. "
                "Enable it with --enable-expert-parallel, or pick a different "
                "moe backend."
            )
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        # Three aux streams: one per non-default input GEMM in
        # DeepseekV41Attention.attn_gemm_parallel_execute
        # (compressor kv_score, indexer.weights_proj, indexer.compressor
        # kv_score). fused_wqa_wkv stays on the default stream.
        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]

        # Reserved topk indices buffer for all Indexer layers to reuse.
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Engram layout (None disables the engram everywhere: no modules are
        # built AND the loader skips the checkpoint tensors -- the same
        # dsv41_engram_runtime_enabled() gates both, so they cannot disagree).
        self.engram_layout = (
            EngramLayout.from_config(config)
            if dsv41_engram_runtime_enabled(config)
            else None
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV41DecoderLayer(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
                engram_layout=self.engram_layout,
            ),
            prefix=f"{prefix}.layers",
        )

        # Serving-side n-gram hasher, built only when a LOCAL layer carries an
        # engram (PP ranks without one need neither the tokenizer scan nor the
        # per-token windows).
        self.engram_hasher: EngramServingHasher | None = None
        self._engram_token_windows: torch.Tensor | None = None
        if self.engram_layout is not None and any(
            getattr(layer, "engram", None) is not None
            for layer in self.layers
            if not isinstance(layer, PPMissingLayer)
        ):
            # The engram row gather is a host-side op (CPU tables + H2D of the
            # gathered rows): it cannot run inside a captured CUDA graph. Fail
            # loudly at boot instead of failing capture (or silently corrupting
            # output with VLLM_DSV41_LOAD_ENGRAM=0). Speed work (host-fn gather
            # inside graphs, as sglang does) comes after the correctness gate.
            if (
                not vllm_config.model_config.enforce_eager
                and os.environ.get("VLLM_DSV41_ENGRAM_ALLOW_CUDAGRAPH", "0") != "1"
            ):
                raise ValueError(
                    "DeepSeek-V4.1-Flash engram is enabled (VLLM_DSV41_LOAD_"
                    "ENGRAM=1, the default) but the engine is not running in "
                    "eager mode. The engram's CPU-resident table gather cannot "
                    "be captured in a CUDA graph yet: pass --enforce-eager "
                    "(serve_v41.sh: DSV41_EAGER=1), or set "
                    "VLLM_DSV41_LOAD_ENGRAM=0 to serve WITHOUT the engram "
                    "(known token-level copy corruption)."
                )
            # Engram hashes recompute per-request trailing windows every step.
            # The V2 model runner (forced for DSpark/MTP spec, i.e. the
            # production config) gathers them ON-GPU from state kept current in
            # stream order, so async scheduling is safe there. The V1 runner
            # gathers them on the CPU from token_ids_cpu, which async
            # scheduling leaves one step stale -- refuse that combination.
            if (
                vllm_config.scheduler_config.async_scheduling
                and not vllm_config.use_v2_model_runner
            ):
                raise ValueError(
                    "DeepSeek-V4.1-Flash engram on the V1 GPU model runner "
                    "does not support async scheduling (which this build "
                    "enables BY DEFAULT): the previous step's sampled tokens "
                    "are not in token_ids_cpu yet when the next step's n-gram "
                    "windows are gathered. Pass --no-async-scheduling, or set "
                    "VLLM_DSV41_LOAD_ENGRAM=0 to serve WITHOUT the engram "
                    "(known token-level copy corruption)."
                )
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                vllm_config.model_config.tokenizer,
                trust_remote_code=vllm_config.model_config.trust_remote_code,
            )
            self.engram_hasher = EngramServingHasher(
                config,
                self.engram_layout,
                tokenizer,
                image_token_id=(
                    config.image_token_id if config.vision_enabled else None
                ),
            )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.hc_head_fn = nn.Parameter(
            torch.empty(
                self.hc_mult,
                self.hc_dim,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(
                self.hc_mult,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        # Pre-hc_head residual stream buffer for the MTP draft. Stable
        # address (outside the cudagraph pool) so the copy_ in forward()
        # refreshes it correctly across captured shapes.
        # refreshes it correctly across captured shapes. Only allocated on
        # the last PP rank — that's where MTP target hidden states are
        # produced.
        if get_pp_group().is_last_rank:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        # PP intermediate tensors carry the multi-stream hidden_states
        # of shape (num_tokens, hc_mult, hidden_size) — V4 expands the
        # token embedding to hc_mult streams before the first decoder
        # layer and keeps that shape until hc_head() collapses it.
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.hc_mult, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                ),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        if self.use_mega_moe:
            input_ids = input_ids.to(torch.int64)

        # Engram hash ids for every token of this batch, from the per-token
        # trailing n-gram windows the runner gathered out of the persistent
        # batch's full token history (NOT from input_ids: with multimodal
        # support enabled the runner hands the model inputs_embeds and
        # input_ids may be None; the windows also carry the predecessors of
        # chunk-leading tokens that a prefix-cache hit skipped).
        engram_hash_ids = engram_token_mask = None
        if self.engram_hasher is not None:
            num_tokens = positions.shape[0]
            windows = self._engram_token_windows
            if windows is None or windows.shape[0] < num_tokens:
                # Profiling / dummy runs before the runner registered its
                # window buffer: hash all-zero windows (valid table rows, so
                # memory/comms are exercised; the outputs are discarded).
                windows = torch.zeros(
                    num_tokens,
                    self.engram_hasher.layout.max_ngram_size,
                    dtype=torch.int32,
                    device=positions.device,
                )
            engram_hash_ids, engram_token_mask = self.engram_hasher(
                windows[:num_tokens], positions
            )

        residual, post_mix, res_mix, pre_mix = None, None, None, None
        aux_hidden_states: list[torch.Tensor] = []
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if layer.engram is not None and engram_hash_ids is not None:
                # Reference insertion point (inference/model.py:1288): the
                # engram is applied to the multi-stream HC state h entering
                # layer i, BEFORE the layer. The port DEFERS each sublayer's
                # hc_post expansion: at this loop boundary the reference h
                # exists only implicitly as the 4-tuple (hidden_states,
                # residual, post_mix, res_mix) with
                #   h == mhc_post(hidden_states, residual, post_mix, res_mix).
                # The engram write is NONLINEAR in h (its gate is a sigmoid of
                # a normalized <h, key> dot per hc copy) and adds the gated
                # value into every copy, so the deferred form cannot be kept:
                # materialize h with mhc_post_torch -- the exact same kernel
                # (same FP reduction order) the layer itself would have used
                # one call later, so the collapsed tensor is bit-identical to
                # the tensor the reference engram sees -- apply the engram,
                # and re-enter the layer through its residual=None path with
                # the DEFERRED pre_mix carried over unchanged (the layer's
                # attention still collapses with the previous FFN's pre, as
                # in the reference).
                from vllm.model_executor.kernels.mhc.torch import mhc_post_torch

                if residual is None:
                    engram_h = hidden_states
                else:
                    engram_h = mhc_post_torch(
                        hidden_states, residual, post_mix, res_mix
                    )
                engram_h = layer.engram(
                    engram_h,
                    engram_hash_ids[:, layer.engram.layer_hash_index],
                    engram_token_mask,
                )
                if idx in self.aux_hidden_state_layers and aux_hidden_states:
                    # The DSpark aux capture for THIS layer ran at the end of
                    # the previous iteration, i.e. pre-engram. The reference
                    # captures the target layer's attention input AFTER the
                    # engram write (model.py:1288-1292) -- replace it. (Draft
                    # numerics only affect acceptance rate, but stay faithful.)
                    aux_hidden_states[-1] = engram_h.mean(dim=1)
                hidden_states, residual, post_mix, res_mix = (
                    engram_h, None, None, None
                )
            hidden_states, residual, post_mix, res_mix, pre_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
                pre_mix,
            )
            if idx + 1 in self.aux_hidden_state_layers:
                # Reconstruct the aux hidden state for draft models
                aux_recon = mhc_post_tilelang(
                    hidden_states, residual, post_mix, res_mix
                )
                aux_hidden_states.append(aux_recon.mean(dim=1))
        if layer is not None:
            # LOSSLESSNESS: the main residual stream MUST use mhc_post_torch --
            # the exact kernel the no-spec path uses (and the one the needle /
            # coherence gates validated). Do NOT reuse the ``mhc_post_tilelang``
            # aux recon computed above for the main hidden state: it is
            # algebraically the same collapse but a different FP reduction order,
            # so it is not bit-identical to torch. Substituting it shifts the
            # final logits sub-ulp and flips greedy argmax on near-tie tokens
            # (confident prompts unaffected, uncertain ones diverge, some at
            # token 0). The tilelang aux recon still feeds the drafter via
            # ``aux_hidden_states`` above -- draft numerics never affect
            # spec-decode losslessness, only acceptance rate. The duplicate
            # collapse this reintroduces is one kernel per full forward (draft
            # steps are Q-blocks), negligible vs correctness.
            from vllm.model_executor.kernels.mhc.torch import mhc_post_torch
            hidden_states = mhc_post_torch(
                hidden_states, residual, post_mix, res_mix
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        # Stash the fully-expanded final residual stream for the MTP draft.
        num_tokens = hidden_states.shape[0]
        self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        # Final collapse: gold collapses the hc copies with the LAST layer's deferred
        # ffn_pre (Transformer.forward: h = layer.hc_pre(h, pre_mix)). The checkpoint
        # has no hc_head_* tensors, so there is NO learned head projection here.
        hidden_states = (
            pre_mix.unsqueeze(-1) * hidden_states.float()
        ).sum(dim=1).to(hidden_states.dtype)
        hidden_states = self.norm(hidden_states)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            # NOTE: the kv_source compressor is the reference-faithful
            # ``DeepseekV41RefCompressor`` (separate bf16/fp32 ``wkv``/``wgate``
            # nn.Parameters for ratios {1,2}), NOT the v4 fused
            # ``DeepseekCompressor``. So there is no ``fused_wkv_wgate`` to stack
            # into -- ``wkv``/``wgate`` load straight through (suffix-stripped
            # below). Restore a fused entry here only if the fast-path compressor
            # is reinstated.
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # TP for attention
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        # Pre-compute expert mapping ONCE.
        expert_mapping = self.get_expert_mapping()

        # Block-FP8 shared experts: pad the intermediate up to the TP-uniform
        # block count so the standard loaders below slice it evenly (trailing
        # ranks land on the zero pad). SP / unquantized ones need no padding.
        pad_shared_expert = (
            getattr(self.quant_config, "weight_block_size", None) is not None
            and not self.parallel_config.use_sequence_parallel_moe
        )

        for name, loaded_weight in weights:
            # RefCompressor stores ``wkv``/``wgate`` as bare nn.Parameters
            # (``compressor.wkv`` / ``compressor.wgate``), but the checkpoint
            # names them Linear-style (``compressor.wkv.weight``). Strip the
            # trailing ``.weight`` so the generic single-tensor loader below
            # finds the parameter. ``compressor.norm.weight`` is a real RMSNorm
            # weight and is left untouched.
            if name.endswith(".compressor.wkv.weight") or name.endswith(
                ".compressor.wgate.weight"
            ):
                name = name[: -len(".weight")]
            # The global ``.scale`` -> ``.weight_scale_inv`` mapper rule is meant
            # for the fp8 Linear layers (and fits engram.wkv.scale exactly), but
            # it also renames the engram TABLE's e8m0 row scales. Those live on
            # ParallelEngramEmbedding under their checkpoint name ``scale`` (with
            # a row-shard / validate-only weight_loader) -- undo the rename.
            if ".engram.embed." in name and name.endswith(".weight_scale_inv"):
                name = name[: -len(".weight_scale_inv")] + ".scale"
            # Same collision for the ``embed.weight`` -> ``embed_tokens.weight``
            # suffix rule (meant for the main token embedding): it renames the
            # engram table rows too. The parameter is ``engram.embed.weight``.
            if name.endswith(".engram.embed_tokens.weight"):
                name = name[: -len(".embed_tokens.weight")] + ".embed.weight"
            # Hybrid per-expert tensors (nvfp4_aqlm_hybrid) are stored at the
            # checkpoint-stable path ffn.experts.*; the runtime parameters live
            # on the MoERunner's RoutedExperts submodule. Remap and load them
            # through the generic single-tensor loader, bypassing both the
            # stacked-param loop and the per-expert expert_mapping loop (whose
            # w1/w2/w3 ckpt names can't match these stacked tensors).
            if any(
                f".ffn.experts.{p}" in name
                for p in ("w13_", "w2_", "w2m_", "w2c_", "nvfp4_", "hyb_")
            ):
                name = name.replace(
                    ".ffn.experts.", ".ffn.experts.routed_experts."
                )
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
                continue
            if pad_shared_expert and ".shared_experts." in name:
                loaded_weight = self._pad_shared_expert_weight(name, loaded_weight)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, self):
                    break
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    # E8M0 scales are stored as float8_e8m0fnu in
                    # checkpoints but the MoE param is uint8. copy_()
                    # would do a numeric conversion (e.g. 2^-7 → 0),
                    # destroying the raw exponent bytes.
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name_mapped, self):
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or not
                        # here since otherwise we may skip experts with other
                        # available replicas.
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            break
                    loaded_params.add(name_mapped)
                    continue
                elif "attn_sink" in name:
                    if is_pp_missing_parameter(name, self):
                        continue
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                else:
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue

        return loaded_params

    def _pad_shared_expert_weight(
        self, name: str, loaded_weight: torch.Tensor
    ) -> torch.Tensor:
        """Zero-pad a block-FP8 shared-expert weight/scale on its intermediate
        axis so the standard TP loaders split it into even, block-aligned shards
        (trailing ranks get the zero pad). gate (w1)/up (w3) [I, H] pad dim 0;
        down (w2 -> down_proj) [H, I] pads dim 1.
        """
        block_size = getattr(self.quant_config, "weight_block_size", None)
        assert block_size is not None
        # Round the intermediate axis up to a whole number of TP shards. The axis
        # is in elements for weights (step = block) and in blocks for scales.
        step = 1 if name.endswith("weight_scale_inv") else block_size[0]
        dim = 1 if ".down_proj." in name else 0
        mult = get_tensor_model_parallel_world_size() * step
        pad = cdiv(loaded_weight.shape[dim], mult) * mult - loaded_weight.shape[dim]
        if pad == 0:
            return loaded_weight
        pad_shape = list(loaded_weight.shape)
        pad_shape[dim] = pad
        return torch.cat([loaded_weight, loaded_weight.new_zeros(pad_shape)], dim=dim)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        first_layer = next(iter(islice(self.layers, self.start_layer, self.end_layer)))
        if first_layer.ffn.use_mega_moe:
            return make_deepseek_v4_expert_params_mapping(self.config.n_routed_experts)
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
        )

    def finalize_mega_moe_weights(self) -> None:
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            layer.ffn.finalize_mega_moe_weights()


def _make_deepseek_v4_weights_mapper(expert_dtype: str) -> WeightsMapper:
    if expert_dtype == "fp4":
        # MXFP4 experts use Mxfp4MoEMethod, which registers scales as
        # ``w{1,2,3}_weight_scale`` (no _inv suffix). FP8 linear and
        # shared experts use Fp8LinearMethod's block scales, which
        # register as ``weight_scale_inv``.
        scale_regex = {
            re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
            re.compile(r"\.scale$"): ".weight_scale_inv",
        }
    else:
        # FP8 experts use Fp8MoEMethod (block_quant=True), which registers
        # scales as ``w{13,2}_weight_scale_inv``. Map all ``.scale`` keys
        # there.
        scale_regex = {
            re.compile(r"\.scale$"): ".weight_scale_inv",
        }
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "hc_head": "model.hc_head",
            "mtp.": "model.mtp.",
        },
        orig_to_new_regex=scale_regex,
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".shared_experts.w2": ".shared_experts.down_proj",
        },
    )


class DeepseekV41MixtureOfExperts(MixtureOfExperts):
    moe_mlp_layers: list["DeepseekV41MoE"]

    def extract_moe_parameters(self, example_moe: "DeepseekV41MoE | None") -> None:
        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
            return
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


@MULTIMODAL_REGISTRY.register_processor(
    _DeepseekV41MultiModalProcessor,
    info=_DeepseekV41ProcessingInfo,
    dummy_inputs=_DeepseekV41DummyInputsBuilder,
)
class DeepseekV41ForCausalLM(
    nn.Module,
    SupportsPP,
    SupportsEagle3,
    SupportsMultiModal,
    DeepseekV41MixtureOfExperts,
):
    model_cls = DeepseekV41Model

    # MM kwargs arrive as already-collated fields (see nvidia/processing.py).
    merge_by_field_config = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<｜deepseek_image｜>"
        raise ValueError(f"Unsupported modality: {modality}")

    # Default mapper assumes the original FP4-expert checkpoint layout.
    # Overridden per-instance in __init__ when expert_dtype != "fp4".
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        expert_dtype = getattr(config, "expert_dtype", "fp4")
        if expert_dtype != "fp4":
            self.hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper(expert_dtype)

        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)

        # Native vision tower (ViT + Aligner) and the three learned span
        # delimiters. Mounted directly on the top-level module (MOUNT OPTION B)
        # so the checkpoint's ``vision.*`` / ``aligner.*`` / ``image_*`` tensors
        # resolve with no name mapping. All bf16 (under the model dtype context);
        # the vision RMSNorms keep their own fp32 params.
        self.vision_enabled = config.vision_enabled
        if self.vision_enabled:
            self.image_token_id = config.image_token_id
            ra = config.reference_args
            self.vision = ViT(
                dim=ra["vision_dim"],
                n_heads=ra["vision_n_heads"],
                inter_dim=ra["vision_inter_dim"],
                n_layers=ra["vision_n_layers"],
                patch_size=ra["vision_patch_size"],
                rope_theta=ra["vision_rope_theta"],
            )
            self.aligner = Aligner(
                vision_dim=ra["vision_dim"],
                out_dim=ra["dim"],
                downsample_ratio=ra["vision_downsample_ratio"],
            )
            self.image_start = nn.Parameter(torch.empty(config.hidden_size))
            self.image_end = nn.Parameter(torch.empty(config.hidden_size))
            self.image_newline = nn.Parameter(torch.empty(config.hidden_size))

        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )

        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.num_moe_layers = self.config.num_hidden_layers
        self.moe_layers: list[nn.Module] = []
        self.moe_mlp_layers: list[DeepseekV41MoE] = []
        example_moe: DeepseekV41MoE | None = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if not isinstance(layer, DeepseekV41DecoderLayer):
                continue
            if isinstance(layer.ffn, DeepseekV41MoE):
                example_moe = layer.ffn
                self.moe_mlp_layers.append(layer.ffn)
                self.moe_layers.append(layer.ffn.experts)

        self.num_moe_layers = len(self.moe_layers)
        self.extract_moe_parameters(example_moe)

    # NOTE: ``embed_input_ids`` is intentionally NOT overridden here -- the
    # ``SupportsMultiModal`` base implementation embeds text via
    # ``get_language_model().embed_input_ids`` (auto-discovered as ``self.model``)
    # and scatters the vision embeddings into the image-span positions.

    def _process_image_input(
        self,
        patches: torch.Tensor,
        n_vit_h: int,
        n_vit_w: int,
        types: torch.Tensor,
    ) -> torch.Tensor:
        """Build one image's ``[span_len, hidden]`` embedding block. IMAGE slots
        take aligner rows in reading order; the three delimiter kinds take their
        learned embeddings -- matching the reference ``merge_image_embeddings``."""
        feats = self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)
        hidden = self.image_start.numel()
        span = torch.empty(
            types.numel(), hidden, dtype=self.image_start.dtype, device=feats.device
        )
        types = types.to(feats.device)
        span[types == IMAGE_START] = self.image_start.to(span.dtype)
        span[types == IMAGE_END] = self.image_end.to(span.dtype)
        span[types == IMAGE_NEW_LINE] = self.image_newline.to(span.dtype)
        span[types == IMAGE] = feats.to(span.dtype)
        return span

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Return one ``[span_len, hidden]`` block per image, in prompt order.

        With ``merge_by_field_config=True`` the ragged ``patches``/``types`` fields
        arrive concatenated across images; ``n_vit`` / ``num_tokens`` (kept on CPU)
        give the per-image split, and ``n_vit_h`` / ``n_vit_w`` the ViT grid."""
        patches = kwargs.pop("patches", None)
        if patches is None:
            return []
        types = kwargs.pop("types")
        n_vit_h = kwargs.pop("n_vit_h")
        n_vit_w = kwargs.pop("n_vit_w")
        n_vit = kwargs.pop("n_vit")
        num_tokens = kwargs.pop("num_tokens")

        def _as_list(x):
            return x.tolist() if isinstance(x, torch.Tensor) else list(x)

        n_vit_list = _as_list(n_vit)
        num_tokens_list = _as_list(num_tokens)
        n_vit_h_list = _as_list(n_vit_h)
        n_vit_w_list = _as_list(n_vit_w)

        param_dtype = self.image_start.dtype
        blocks: list[torch.Tensor] = []
        p_off = t_off = 0
        for i in range(len(n_vit_list)):
            np_i = int(n_vit_list[i])
            nt_i = int(num_tokens_list[i])
            patches_i = patches[p_off : p_off + np_i].to(param_dtype)
            types_i = types[t_off : t_off + nt_i]
            blocks.append(
                self._process_image_input(
                    patches_i, int(n_vit_h_list[i]), int(n_vit_w_list[i]), types_i
                )
            )
            p_off += np_i
            t_off += nt_i
        return blocks

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        # The v41 backbone (DeepseekV41Model) IS the EagleModelMixin and is
        # auto-discovered directly as ``self.model`` by
        # ``SupportsMultiModal.get_language_model`` (via ``embed_input_ids``).
        # The generic SupportsEagle3 path assumes one extra wrapping level
        # (``get_language_model().model``); v41 has none, so drive the backbone
        # directly. This wires DSpark/DFlash/EAGLE3 aux-hidden-state capture.
        self.model._set_aux_hidden_state_layers(layers)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        ids = getattr(self.config, "dspark_target_layer_ids", None)
        if ids:
            return tuple(i + 1 for i in ids)
        n = self.config.num_hidden_layers
        return (2, n // 2, n - 3)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Pre-hc_head residual stream buffer (max_num_batched_tokens,
        hc_mult * hidden_size) for the MTP draft model. Populated by
        forward(); valid after each target step."""
        return getattr(self.model, "_mtp_hidden_buffer", None)

    @property
    def engram_window_size(self) -> int:
        """Non-zero when the engram is wired: the runner must supply, per
        scheduled token, a window of this many raw token ids ([token, pred_1,
        .., pred_{n-1}]) gathered from the persistent batch's full per-request
        token history (prefix-cache-skipped prompt tokens included)."""
        hasher = getattr(self.model, "engram_hasher", None)
        return hasher.layout.max_ngram_size if hasher is not None else 0

    def set_engram_token_windows(self, windows_gpu: torch.Tensor) -> None:
        """Register the runner's persistent [max_num_tokens, window] GPU buffer
        of per-token n-gram windows (stable address; refreshed every step,
        sliced to the batch inside forward)."""
        self.model._engram_token_windows = windows_gpu

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # ``mtp.`` is skipped whenever speculative decoding is off (no draft
        # model built).
        skip = ["mtp."]
        # Engram tensors load by DEFAULT into the layer-1/14 Engram modules
        # (host-resident tables + GPU wkv/gates). VLLM_DSV41_LOAD_ENGRAM=0 is
        # the kill-switch: it disables module construction (see
        # dsv41_engram_runtime_enabled) and must skip the tensors here too.
        if not dsv41_engram_runtime_enabled(self.config):
            skip += [".engram."]
        # The vision stack (``vision.`` tower, ``aligner.`` projector, ``image_*``
        # span delimiters, and the per-layer ``gate.bias_vl`` VL routing bias) now
        # has a home (see __init__ / DeepseekV41MoE). Skip it only when the
        # checkpoint has no vision tower (``vision_enabled`` is False). The MTP
        # ``gate.bias_vl`` tensors ride under ``mtp.`` and are skipped with it.
        if not self.config.vision_enabled:
            skip += ["vision.", "aligner.", "image_start", "image_end",
                     "image_newline", ".bias_vl"]
        loader = AutoWeightsLoader(self, skip_substrs=skip)
        loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self.model.finalize_mega_moe_weights()
        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
