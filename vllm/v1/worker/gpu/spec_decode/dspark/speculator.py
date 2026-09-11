# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator: semi-autoregressive parallel drafting.

DSpark drafts a block of ``num_speculative_tokens`` tokens in one parallel pass
(reusing the DFlash machinery: context-KV precompute + a query-block forward),
then injects intra-block dependency with a lightweight sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly ``N =
    num_speculative_tokens`` query tokens (anchor + N-1 noise), NOT ``1 + N``.
    Every query position is a prediction (the anchor predicts the first draft
    token), so we sample at all N positions and ``sample_pos = query_pos + 1``
    (standard next-token), whereas DFlash's masks sit AT the predicted position.
    This is the ``sample_from_anchor`` path in the shared prepare-inputs kernel.
    Speculators-format checkpoints instead use the DFlash ``1 + N`` fill-in
    layout (anchor is the bonus token).
  * Sequential Markov sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step.

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.
"""

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # LOSSLESSNESS: default to the standard 1+N bonus layout, NOT
        # anchor-as-first. Anchor-as-first (N query slots, where the anchor
        # doubles as the first prediction) corrupts the KV cache: when a draft
        # position is rejected, the KV written there holds the rejected draft
        # token, and when the next block's anchor lands on that same position
        # its self-KV is never rewritten. The anchor then attends to stale KV
        # and its greedy argmax silently flips on borderline tokens (proven on
        # SM120: dspark nspec=3 France/Primes diverge from the no-spec base at
        # temp 0, always at the anchor's own position-0 logit, never a draft
        # position). The 1+N bonus layout redecodes rejected positions cleanly
        # via vLLM's known-lossless standard path, and is bit-exact with base.
        # Anchor-as-first stays available for perf experiments via the env, but
        # only when the checkpoint does not itself request the bonus layout.
        import os as _os
        self.sample_from_anchor = (
            _os.environ.get("VLLM_DSPARK_ANCHOR_AS_FIRST") == "1"
            and not getattr(
                self.draft_model_config.hf_config, "dspark_bonus_anchor", False
            )
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps

        # A/B knob: when set, source draft base logits from the KV layout's own
        # sample_indices (noise-aligned in the bonus layout). Default OFF -> use
        # the reference anchor-first draft geometry (see _sample_sequential).
        self._noise_aligned_sample = (
            _os.environ.get("VLLM_DSPARK_NOISE_ALIGNED_SAMPLE") == "1"
        )

        # DSpark consumes mean-pooled target aux hidden states at the target
        # layers, combined to hidden_size via main_proj. Store that combined
        # main_x (hidden_size wide). DSpark does not use the same pre-allocated buffer
        # that DeepSeek-V4's MTP uses.
        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )

        self.dflash_causal = False

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        # Reduced-vocab probabilistic drafting only; set in load_draft_model.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter logits into target vocab before sampling.
        if self.draft_logits is not None and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            # -inf once; the per-step scatter overwrites the draft->target
            # columns. Kept separate from draft_logits to avoid aliasing.
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        return model

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        # DRAFT GEOMETRY (decoupled from the KV verify layout). DSpark is
        # semi-autoregressive: block position k's backbone hidden predicts the
        # token at position k+1 (next-token within the block), STARTING AT THE
        # ANCHOR. So draft token 0 is sourced from the anchor position (query
        # offset 0), draft token i from query offset i -- reference:
        #   output_ids[:, i+1] = sample(logits[:, i] + markov_bias(output_ids[:, i]))
        # with logits[:, 0] the ANCHOR's backbone hidden.
        # The bonus KV layout's ``sample_indices`` instead point at the NOISE
        # positions (query offsets 1..n_spec), which sources every draft token
        # one position too late and collapses acceptance. Base logits must come
        # from anchor-first offsets 0..n_spec-1; this is INTERNAL to the drafter
        # and does not touch the target's (lossless) bonus verify. For the
        # anchor-as-first KV layout ``_anchor_idx`` strides by n_spec, so these
        # indices coincide with ``sample_indices`` there.
        if self._noise_aligned_sample:  # legacy A/B path
            base_idx = self.sample_indices[:num_sample]
        else:
            base_idx = (
                self._anchor_idx[:num_reqs, None] + self._step_cols[None, :]
            ).reshape(-1)
        # Per-(req, position) head hidden, ordered (req, step).
        sample_hidden = head_hidden[base_idx]
        # Draft-vocab logits; sampled ids are remapped to target vocab below.
        base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)

        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias
            if self.draft_logits is not None:
                # Probabilistic: sample in target vocab (a reduced draft vocab is
                # scattered into its target columns; full vocab is already there).
                if self._d2t_scatter_index is not None:
                    assert self._draft_scatter_buf is not None
                    buf = self._draft_scatter_buf[:num_reqs]
                    buf.index_copy_(1, self._d2t_scatter_index, logits_i.to(buf.dtype))
                    logits_i = buf
                # sample_pos is the predicted token's position Q; the target
                # verifies it with the predecessor's Gumbel key (Q-1). Pass Q-1.
                draft_sampled_i = gumbel_sample(
                    logits_i,
                    idx_map[:, i],
                    self.temperature,
                    self.seeds,
                    sample_pos[:, i] - 1,
                    apply_temperature=True,
                    output_processed_logits=self.draft_logits,
                    output_processed_logits_col=self._step_cols[i],
                    use_fp64=self.use_fp64_gumbel,
                )
            else:
                draft_sampled_i = self.model.map_draft_to_target(
                    logits_i.argmax(dim=-1)
                )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        # Full draft step (captured under CUDA graph): parallel backbone forward
        # then sequential Markov sampling over its hidden state outputs.
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._sample_sequential(num_reqs, head_hidden)
