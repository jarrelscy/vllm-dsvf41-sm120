# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-modal processing for DeepSeek-V4.1-Flash native vision.

The released checkpoint ships **no** HF ``processor``/``preprocessor_config.json`` --
image handling lives entirely in the reference ``inference/image_processor.py``. So the
vLLM ``_call_hf_processor`` here is fully custom: it tokenizes the prompt itself and runs
the ported geometry (``image_to_patches`` + ``image_token_types``) to emit ViT patches and
the per-position token-type layout.

Prompt layout mirrors the reference exactly. The prompt text carries one
``<｜deepseek_image｜>`` placeholder (a single ``image_token_id`` == 129264) per image; each
placeholder is expanded by a ``PromptReplacement`` into ``num_image_tokens`` copies of
``image_token_id`` -- the flattened

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

span. Every span position carries ``image_token_id`` in ``input_ids``; the accompanying
``types`` field (kept out of ``input_ids``) is what distinguishes delimiters from IMAGE
slots at embed time (see ``model.py::_process_image_input``).

Field plumbing (consumed by ``DeepseekV41ForCausalLM.embed_multimodal``):
  * ``patches``  -- ViT patches for every image, concatenated on dim 0
                    (``flat_from_sizes`` split by ``n_vit``).
  * ``types``    -- token-type codes for every span, concatenated on dim 0
                    (``flat_from_sizes`` split by ``num_tokens``), kept on CPU.
  * ``n_vit_h`` / ``n_vit_w`` -- per-image ViT grid (batched, CPU scalars).
  * ``n_vit``    -- per-image patch count (== n_vit_h * n_vit_w); doubles as the
                    ``patches`` split size (batched, CPU scalar).
  * ``num_tokens`` -- per-image span length (== num_image_tokens); doubles as the
                    ``types`` split size (batched, CPU scalar).
"""

from collections.abc import Mapping, Sequence

import torch
from transformers import BatchFeature

from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing.dummy_inputs import BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

from .image_processor import (
    image_to_patches,
    image_token_types,
    num_image_tokens,
    plan_image_grid,
)

# The training-time spelling of the image placeholder; a single token whose id is
# ``config.image_token_id`` (129264). Every image-span position also carries this id.
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"


class DeepseekV41ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> DeepseekV41Config:
        return self.ctx.get_hf_config(DeepseekV41Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def _vision_args(self) -> dict:
        return self.get_hf_config().reference_args

    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int:
        """Number of ``image_token_id`` positions the given image expands into.

        Pure function of pixels + config, identical to the reference geometry:
        ``num_image_tokens(n_llm_h, n_llm_w)`` where the grid comes from
        ``plan_image_grid``. Equals the assembled span length exactly."""
        ra = self._vision_args()
        n_llm_h, n_llm_w, _, _ = plan_image_grid(
            image_width,
            image_height,
            patch_size=ra["vision_patch_size"],
            downsample_ratio=ra["vision_downsample_ratio"],
            max_n_token=ra["vision_max_n_token"],
            min_pixels=ra["vision_min_pixels"],
            max_wh_ratio=ra["vision_max_wh_ratio"],
        )
        return num_image_tokens(n_llm_h, n_llm_w)

    def get_image_size_with_most_features(self) -> ImageSize:
        """A square input whose token count sits at the ``max_n_token`` cap.

        ``plan_image_grid`` clamps any larger image down to this, so this is the
        worst case for buffer sizing. For a square, ``num_image_tokens`` is
        ``g*(g+1)+2`` with ``g`` the per-side LLM-grid width; solve for the largest
        ``g`` under the cap and map back to pixels (``g * downsample * patch``)."""
        ra = self._vision_args()
        cap = ra["vision_max_n_token"]
        cell = ra["vision_patch_size"] * ra["vision_downsample_ratio"]
        # largest g with g*(g+1)+2 <= cap
        g = int((((4 * cap - 7) ** 0.5) - 1) // 2)
        g = max(g, 1)
        side = g * cell
        return ImageSize(width=side, height=side)


class DeepseekV41DummyInputsBuilder(
    BaseDummyInputsBuilder[DeepseekV41ProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return IMAGE_PLACEHOLDER * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        max_image_size = self.info.get_image_size_with_most_features()
        image_overrides = mm_options.get("image") if mm_options else None
        return {
            "image": self._get_dummy_images(
                width=max_image_size.width,
                height=max_image_size.height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class DeepseekV41MultiModalProcessor(
    BaseMultiModalProcessor[DeepseekV41ProcessingInfo]
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Fully custom: no HF processor exists for this model.

        Tokenize the prompt ourselves (the placeholder token survives as a single
        ``image_token_id``), and run the reference geometry on each image."""
        tokenizer = self.info.get_tokenizer()
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        images = mm_data.get("images", []) if mm_data else []
        if not images:
            return BatchFeature({"input_ids": [prompt_ids]})

        ra = self.info.get_hf_config().reference_args
        patches_list: list[torch.Tensor] = []
        types_list: list[torch.Tensor] = []
        n_vit_h_list: list[int] = []
        n_vit_w_list: list[int] = []
        n_vit_list: list[int] = []
        num_tokens_list: list[int] = []

        for image in images:
            patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = image_to_patches(
                image,
                patch_size=ra["vision_patch_size"],
                downsample_ratio=ra["vision_downsample_ratio"],
                max_n_token=ra["vision_max_n_token"],
                min_pixels=ra["vision_min_pixels"],
                max_wh_ratio=ra["vision_max_wh_ratio"],
            )
            types = image_token_types(n_llm_h, n_llm_w)
            patches_list.append(patches)
            types_list.append(types)
            n_vit_h_list.append(n_vit_h)
            n_vit_w_list.append(n_vit_w)
            n_vit_list.append(int(patches.shape[0]))
            num_tokens_list.append(int(types.numel()))

        return BatchFeature(
            {
                "input_ids": [prompt_ids],
                "patches": torch.cat(patches_list, dim=0),
                "types": torch.cat(types_list, dim=0),
                "n_vit_h": torch.tensor(n_vit_h_list, dtype=torch.long),
                "n_vit_w": torch.tensor(n_vit_w_list, dtype=torch.long),
                "n_vit": torch.tensor(n_vit_list, dtype=torch.long),
                "num_tokens": torch.tensor(num_tokens_list, dtype=torch.long),
            }
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # Ragged per-image fields are stored concatenated; split them back with the
        # per-image sizes. Scalars ride along as batched CPU tensors.
        n_vit = hf_inputs.get("n_vit", torch.empty(0, dtype=torch.long))
        num_tokens = hf_inputs.get("num_tokens", torch.empty(0, dtype=torch.long))
        return dict(
            patches=MultiModalFieldConfig.flat_from_sizes("image", n_vit),
            types=MultiModalFieldConfig.flat_from_sizes(
                "image", num_tokens, keep_on_cpu=True
            ),
            n_vit_h=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            n_vit_w=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            n_vit=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            num_tokens=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        image_token_id = self.info.get_hf_config().image_token_id

        def get_replacement(item_idx: int) -> list[int]:
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)
            n = self.info.get_num_image_tokens(
                image_width=image_size.width,
                image_height=image_size.height,
            )
            return [image_token_id] * n

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            )
        ]
