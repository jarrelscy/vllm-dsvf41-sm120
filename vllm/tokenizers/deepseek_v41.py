# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 tokenizer mode.

Thin wrapper around :mod:`vllm.tokenizers.deepseek_v41_encoding` (a
self-contained transcription of the checkpoint's reference encoder) that
mirrors the ``deepseek_v4`` tokenizer-mode interface.

V4.1 differences from the V4 dialect that this mode fixes:

1. ``{"role": "system"}`` messages are rendered behind the ``<|System|>``
   special token (leading the conversation, and again for mid-conversation
   system messages). The V4 encoder rendered system content as bare text.
2. Thinking mode always renders a numeric reasoning-effort prefix behind the
   system token: ``Reasoning Effort: {N} (range 1-100, ...)`` with
   low/high/xhigh/max mapped to 25/50/75/100, default "high" (50), and
   integer efforts 1-100 accepted. The V4 encoder emitted nothing unless
   effort was the string "max" (and then a different V4 prose block);
   integers were silently dropped.
3. DSML tool tags use leading-space names (``<|DSML| calls>`` /
   ``<|DSML| invoke>`` / ``<|DSML| parameter>``) instead of V4's
   ``<|DSML|tool_calls>`` / ``<|DSML|invoke>`` / ``<|DSML|parameter>``.
"""

import copy
from typing import Any

from transformers import PreTrainedTokenizerFast

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

from .deepseek_v41_encoding import REASONING_EFFORT_MAPPINGS, encode_messages
from .hf import HfTokenizer, get_cached_tokenizer
from .protocol import TokenizerLike

# Extra OpenAI-style tier names tolerated at the API boundary and mapped onto
# the closest V4.1 tier. The reference encoder itself only accepts
# low/high/xhigh/max or an int in [1, 100].
_EXTRA_EFFORT_ALIASES = {
    "minimal": "low",
    "medium": "high",
}


def _resolve_reasoning_effort(effort: Any) -> tuple[Any, bool]:
    """Map an API ``reasoning_effort`` onto what the V4.1 encoder accepts.

    Returns ``(effort, force_chat)``. ``effort`` is either ``None`` (use the
    encoder default, "high" = 50), an int in [1, 100], or one of the tier
    strings low/high/xhigh/max. ``force_chat`` disables thinking mode
    (``reasoning_effort="none"``).

    Invalid or unrecognized values fall back to the default rather than
    raising, so a bad request-level value cannot 500 the server.
    """
    if effort is None:
        return None, False
    if isinstance(effort, bool):
        return None, False
    if isinstance(effort, int):
        return (effort, False) if 1 <= effort <= 100 else (None, False)
    if isinstance(effort, float):
        # OpenAI-style fractional effort in [0, 1]; mirror sglang's mapping.
        return min(100, max(1, round(effort * 100))), False
    if isinstance(effort, str):
        value = effort.strip().lower()
        if value == "none":
            return None, True
        value = _EXTRA_EFFORT_ALIASES.get(value, value)
        if value in REASONING_EFFORT_MAPPINGS:
            return value, False
        if value.isdigit() and 1 <= int(value) <= 100:
            return int(value), False
    return None, False


def get_deepseek_v41_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """
    Wraps a tokenizer to use the custom DeepSeek V4.1 chat template encoding.
    """
    dsv41_tokenizer = copy.copy(tokenizer)

    added_vocab = tokenizer.get_added_vocab()
    added_vocab_size = len(added_vocab)
    tokenizer_vocab_size = tokenizer.vocab_size

    class _DeepseekV41Tokenizer(tokenizer.__class__):  # type: ignore
        def apply_chat_template(
            self,
            messages: list["ChatCompletionMessageParam"],
            tools: list[dict[str, Any]] | None = None,
            **kwargs,
        ) -> str | list[int]:
            thinking = kwargs.get("thinking", False)
            enable_thinking = kwargs.get("enable_thinking", False)
            thinking = thinking or enable_thinking
            thinking_mode = "thinking" if thinking else "chat"

            conversation = kwargs.get("conversation", messages)
            messages = list(conversation)
            if tools is not None and len(tools) > 0:
                if messages and messages[0].get("role") == "system":
                    # Attach tools to the existing leading system message so
                    # its content and the tools block render together behind
                    # a single <|System|> token, as in the reference encoder.
                    first = copy.copy(messages[0])
                    first["tools"] = tools  # type: ignore[typeddict-unknown-key]
                    messages[0] = first
                else:
                    messages.insert(0, {"role": "system"})
                    messages[0]["tools"] = tools  # type: ignore[typeddict-unknown-key]

            reasoning_effort, force_chat = _resolve_reasoning_effort(
                kwargs.get("reasoning_effort")
            )
            if force_chat:
                thinking_mode = "chat"

            encode_config = dict(
                thinking_mode=thinking_mode,
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=reasoning_effort,
            )

            prompt_str = encode_messages(messages, **encode_config)  # type: ignore

            if kwargs.get("tokenize", True):
                tokenizer_kwargs = {
                    k: kwargs[k] for k in ("truncation", "max_length") if k in kwargs
                }
                return self.encode(
                    prompt_str,
                    add_special_tokens=False,
                    **tokenizer_kwargs,
                )

            return prompt_str

        def num_special_tokens_to_add(self) -> int:
            return len(self.encode(""))

        def __len__(self) -> int:
            return tokenizer_vocab_size + added_vocab_size

        def get_added_vocab(self) -> dict[str, int]:
            return added_vocab.copy()

        def __reduce__(self):
            return get_deepseek_v41_tokenizer, (tokenizer,)

    _DeepseekV41Tokenizer.__name__ = f"DSV41{tokenizer.__class__.__name__}"

    dsv41_tokenizer.__class__ = _DeepseekV41Tokenizer
    return dsv41_tokenizer


class DeepseekV41Tokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> HfTokenizer:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(*args, **kwargs)
        return get_cached_tokenizer(get_deepseek_v41_tokenizer(tokenizer))
