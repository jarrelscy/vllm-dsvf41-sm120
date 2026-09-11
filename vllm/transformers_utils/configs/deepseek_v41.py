# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config for DeepSeek-V4.1-Flash (``deepseek_v41``).

The released HF ``config.json`` is nested: a ``text_config`` (the 552B backbone
with hyper-connections, CED/CSA2 sharing, engram memory, MoE, MTP + DSpark) and a
``vision_config`` (native 32-layer ViT + Aligner). vLLM's core reads HW-standard
attribute names (``hidden_size``, ``num_hidden_layers``, ``num_attention_heads``,
``num_key_value_heads``) off the top-level config for KV/graph sizing, so we promote
``text_config`` to the top level here, keep ``vision_config`` as a sub-config, and
carry every v41-specific key verbatim. A ``.reference_args`` property maps these onto
the flat ``ModelArgs`` field names the reference impl (and our port) use internally,
so the port never has to translate names twice.
"""

from typing import Any

from transformers import PretrainedConfig


class DeepseekV41VisionConfig(PretrainedConfig):
    model_type = "deepseek_v41_vision"

    def __init__(
        self,
        num_hidden_layers: int = 32,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        intermediate_size: int = 2816,
        patch_size: int = 14,
        rope_theta: float = 10000.0,
        downsample_ratio: int = 3,
        max_image_tokens: int = 1024,
        min_pixels: int = 295936,
        max_wh_ratio: int | None = None,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.patch_size = patch_size
        self.rope_theta = rope_theta
        self.downsample_ratio = downsample_ratio
        self.max_image_tokens = max_image_tokens
        self.min_pixels = min_pixels
        self.max_wh_ratio = max_wh_ratio
        super().__init__(**kwargs)


class DeepseekV41Config(PretrainedConfig):
    """Flattened text config for DeepSeek-V4.1-Flash.

    Accepts either the released nested form (``text_config``/``vision_config`` present)
    or an already-flat dict. Unknown keys are preserved by ``PretrainedConfig``.
    """

    model_type = "deepseek_v41"

    def __init__(
        self,
        # ---- promoted from text_config (HF-standard names) ----
        vocab_size: int = 129280,
        hidden_size: int = 5120,
        moe_intermediate_size: int = 2304,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 1,
        head_dim: int = 512,
        qk_rope_head_dim: int = 64,
        q_lora_rank: int = 1280,
        o_lora_rank: int = 1024,
        o_groups: int = 8,
        hidden_act: str = "silu",
        swiglu_limit: float = 10.0,
        rms_norm_eps: float = 1e-20,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        max_position_embeddings: int = 1048576,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, Any] | None = None,
        # ---- MoE ----
        n_routed_experts: int = 384,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 6,
        scoring_func: str = "sqrtsoftplus",
        topk_method: str = "noaux_tc",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.5,
        # ---- CSA2 / CED sparse attention ----
        sliding_window: int = 128,
        compress_ratios: list[int] | None = None,
        compress_rope_theta: float = 160000.0,
        kv_source_layer_ids: list[int] | None = None,
        index_source_layer_ids: list[int] | None = None,
        index_n_heads: int = 32,
        index_head_dim: int = 128,
        index_topk: int = 512,
        candidate_source_layer_id: int = 20,
        candidate_topk_blocks: int = 2048,
        candidate_block_size: int = 8,
        # ---- hyper-connections ----
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        # ---- engram ----
        engram_layer_ids: list[int] | None = None,
        engram_num_embeddings: list[int] | None = None,
        engram_max_ngram_size: int = 4,
        engram_vocab_size: int = 16000000,
        engram_n_heads: int = 8,
        engram_head_dim: int = 256,
        engram_pad_token_id: int = 2,
        engram_compressed_vocab_size: int = 99092,
        # ---- spec decode ----
        num_nextn_predict_layers: int = 3,
        dspark_block_size: int = 5,
        dspark_noise_token_id: int = 128799,
        dspark_target_layer_ids: list[int] | None = None,
        dspark_markov_rank: int = 256,
        dspark_n_routed_experts: int = 128,
        dspark_num_experts_per_tok: int = 3,
        # ---- vision / multimodal ----
        vision_config: dict[str, Any] | None = None,
        image_token_id: int = 129264,
        # ---- quant ----
        quantization_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        # If handed the released nested form, promote text_config before anything
        # else so the explicit kwargs above are overridden by the real values.
        text_config = kwargs.pop("text_config", None)
        if text_config is not None:
            # Re-enter with the flattened dict; nested vision/quant ride along.
            merged = dict(text_config)
            merged.pop("model_type", None)
            merged["vision_config"] = vision_config
            merged["image_token_id"] = kwargs.pop("image_token_id", image_token_id)
            merged["quantization_config"] = quantization_config
            self.__init__(**merged, **kwargs)
            return

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.hidden_act = hidden_act
        self.swiglu_limit = swiglu_limit
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor

        self.sliding_window = sliding_window
        self.compress_ratios = compress_ratios
        self.compress_rope_theta = compress_rope_theta
        self.kv_source_layer_ids = kv_source_layer_ids
        self.index_source_layer_ids = index_source_layer_ids
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.candidate_source_layer_id = candidate_source_layer_id
        self.candidate_topk_blocks = candidate_topk_blocks
        self.candidate_block_size = candidate_block_size

        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps

        self.engram_layer_ids = engram_layer_ids
        self.engram_num_embeddings = engram_num_embeddings
        self.engram_max_ngram_size = engram_max_ngram_size
        self.engram_vocab_size = engram_vocab_size
        self.engram_n_heads = engram_n_heads
        self.engram_head_dim = engram_head_dim
        self.engram_pad_token_id = engram_pad_token_id
        self.engram_compressed_vocab_size = engram_compressed_vocab_size

        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.dspark_block_size = dspark_block_size
        self.dspark_noise_token_id = dspark_noise_token_id
        self.dspark_target_layer_ids = dspark_target_layer_ids
        self.dspark_markov_rank = dspark_markov_rank
        self.dspark_n_routed_experts = dspark_n_routed_experts
        self.dspark_num_experts_per_tok = dspark_num_experts_per_tok

        self.image_token_id = image_token_id
        self.quantization_config = quantization_config

        if vision_config is None:
            self.vision_config = DeepseekV41VisionConfig()
        elif isinstance(vision_config, dict):
            vc = dict(vision_config)
            vc.pop("model_type", None)
            self.vision_config = DeepseekV41VisionConfig(**vc)
        else:
            self.vision_config = vision_config

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def vision_enabled(self) -> bool:
        return getattr(self.vision_config, "num_hidden_layers", 0) > 0

    @property
    def reference_args(self) -> dict[str, Any]:
        """Map the flattened HF names onto the reference ``ModelArgs`` field names.

        The port's compute modules were translated from the reference impl and use
        its names (``dim``, ``n_layers``, ``rope_factor`` …). Building the reference
        arg dict from one place keeps the port and the golden oracle in lock-step.
        """
        rs = self.rope_scaling or {}
        vc = self.vision_config
        return {
            "vocab_size": self.vocab_size,
            "dim": self.hidden_size,
            "moe_inter_dim": self.moe_intermediate_size,
            "n_layers": self.num_hidden_layers,
            "n_mtp_layers": self.num_nextn_predict_layers,
            "n_heads": self.num_attention_heads,
            "head_dim": self.head_dim,
            "rope_head_dim": self.qk_rope_head_dim,
            "q_lora_rank": self.q_lora_rank,
            "o_lora_rank": self.o_lora_rank,
            "o_groups": self.o_groups,
            "norm_eps": self.rms_norm_eps,
            "swiglu_limit": self.swiglu_limit,
            # moe
            "n_routed_experts": self.n_routed_experts,
            "n_shared_experts": self.n_shared_experts,
            "n_activated_experts": self.num_experts_per_tok,
            "score_func": self.scoring_func,
            "norm_topk_prob": self.norm_topk_prob,
            "route_scale": self.routed_scaling_factor,
            # attention / sparse
            "window_size": self.sliding_window,
            "compress_ratios": tuple(self.compress_ratios or ()),
            "kv_source_layers": tuple(self.kv_source_layer_ids or ()),
            "index_source_layers": tuple(self.index_source_layer_ids or ()),
            "compress_rope_theta": self.compress_rope_theta,
            "rope_theta": self.rope_theta,
            "rope_factor": rs.get("factor", 16),
            "original_seq_len": rs.get("original_max_position_embeddings", 0),
            "beta_fast": rs.get("beta_fast", 32),
            "beta_slow": rs.get("beta_slow", 1),
            "index_n_heads": self.index_n_heads,
            "index_head_dim": self.index_head_dim,
            "index_topk": self.index_topk,
            "candidate_source_layer": self.candidate_source_layer_id,
            "candidate_topk_blocks": self.candidate_topk_blocks,
            "candidate_block_size": self.candidate_block_size,
            # hyper-connections
            "hc_mult": self.hc_mult,
            "hc_sinkhorn_iters": self.hc_sinkhorn_iters,
            "hc_eps": self.hc_eps,
            # engram
            "engram_layer_ids": tuple(self.engram_layer_ids or ()),
            "engram_num_embeddings": tuple(self.engram_num_embeddings or ()),
            "engram_max_ngram_size": self.engram_max_ngram_size,
            "engram_vocab_size": self.engram_vocab_size,
            "engram_n_heads": self.engram_n_heads,
            "engram_head_dim": self.engram_head_dim,
            "engram_pad_id": self.engram_pad_token_id,
            "engram_compressed_vocab_size": self.engram_compressed_vocab_size,
            # vision
            "vision_n_layers": getattr(vc, "num_hidden_layers", 0),
            "vision_dim": getattr(vc, "hidden_size", 1024),
            "vision_n_heads": getattr(vc, "num_attention_heads", 16),
            "vision_inter_dim": getattr(vc, "intermediate_size", 2816),
            "vision_patch_size": getattr(vc, "patch_size", 14),
            "vision_rope_theta": getattr(vc, "rope_theta", 10000.0),
            "vision_downsample_ratio": getattr(vc, "downsample_ratio", 3),
            "vision_max_n_token": getattr(vc, "max_image_tokens", 1024),
            "vision_min_pixels": getattr(vc, "min_pixels", 544 * 544),
            "vision_max_wh_ratio": getattr(vc, "max_wh_ratio", None),
            "image_token_id": self.image_token_id,
            # dspark
            "dspark_block_size": self.dspark_block_size,
            "dspark_noise_token_id": self.dspark_noise_token_id,
            "dspark_target_layer_ids": tuple(self.dspark_target_layer_ids or ()),
            "dspark_markov_rank": self.dspark_markov_rank,
            "dspark_n_routed_experts": self.dspark_n_routed_experts,
            "dspark_n_activated_experts": self.dspark_num_experts_per_tok,
        }
