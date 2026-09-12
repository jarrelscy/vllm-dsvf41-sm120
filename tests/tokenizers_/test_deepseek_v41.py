# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the DeepSeek-V4.1 tokenizer mode.

The encoding module is validated byte-for-byte against the checkpoint's
golden input/output pairs (copied into fixtures/deepseek_v41), plus ports of
the relevant cases from the checkpoint's own test_encoding.py. If the
reference encoder directory is present on this machine, the encoder is
additionally cross-checked against it directly.
"""

import importlib.util
import json
from pathlib import Path

import pytest

import vllm.tokenizers.deepseek_v41_encoding as enc
from vllm.tokenizers.deepseek_v41 import (
    _resolve_reasoning_effort,
    get_deepseek_v41_tokenizer,
)
from vllm.tokenizers.deepseek_v41_encoding import (
    SYSTEM_SP_TOKEN,
    encode_messages,
    merge_tool_messages,
    parse_message_from_completion_text,
    render_message,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "deepseek_v41"

# Optional cross-check against the checkpoint's reference encoder. The shipped
# module must never depend on this path; only these tests peek at it.
REFERENCE_ENCODING_PY = Path(
    "/data/huggingface/dsv41-serve/encoding/encoding.py"
)

REASONING_EFFORT_TEMPLATE = (
    SYSTEM_SP_TOKEN + "Reasoning Effort: {budget} "
    "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
)

V41_TOOL_CALL_OUTPUT = (
    "  reason  </think>summary\n\n"
    "<｜DSML｜ calls>\n"
    '<｜DSML｜ invoke name="lookup">\n'
    '<｜DSML｜ parameter name="query" string="true">value'
    "</｜DSML｜ parameter>\n"
    '<｜DSML｜ parameter name="limit" string="false">2'
    "</｜DSML｜ parameter>\n"
    "</｜DSML｜ invoke>\n"
    "</｜DSML｜ calls><｜end▁of▁sentence｜>"
)


class FakeHfTokenizer:
    vocab_size = 100

    def get_added_vocab(self) -> dict[str, int]:
        return {"</think>": 100}

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        **kwargs,
    ) -> list[int]:
        self.last_encode = (text, add_special_tokens, kwargs)
        return [len(text)]


def _tokenizer():
    return get_deepseek_v41_tokenizer(FakeHfTokenizer())


def _load_reference_module():
    spec = importlib.util.spec_from_file_location(
        "dsv41_reference_encoding", REFERENCE_ENCODING_PY
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================
# Registration
# ============================================================


def test_deepseek_v41_tokenizer_registered():
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers.registry import TokenizerRegistry

    assert TokenizerRegistry.load_tokenizer_cls("deepseek_v41").__name__ == (
        "DeepseekV41Tokenizer"
    )
    # V4.1 reuses the V4 renderer; the dialect lives in the tokenizer wrapper.
    assert RENDERER_REGISTRY.load_renderer_cls("deepseek_v41").__name__ == (
        "DeepseekV4Renderer"
    )


def test_deepseek_v41_parsers_registered():
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    assert (
        ReasoningParserManager.get_reasoning_parser("deepseek_v41").__name__
        == "DeepSeekV41ParserReasoningAdapter"
    )
    assert (
        ToolParserManager.get_tool_parser("deepseek_v41").__name__
        == "DeepSeekV41EngineToolParser"
    )


# ============================================================
# Golden fixtures from the checkpoint (byte equality)
# ============================================================


@pytest.mark.parametrize("case_id", [1, 2, 3, 4, 5])
def test_deepseek_v41_matches_checkpoint_golden_fixtures(case_id):
    """Each checkpoint golden input must encode byte-for-byte."""
    case = enc.load_cases(str(FIXTURES_DIR / f"test_input_{case_id}.json"))[0]
    prompt, _ = enc.encode_case(case, thinking_mode="chat")

    expected = (FIXTURES_DIR / f"test_output_{case_id}.txt").read_text()
    assert prompt == expected


def test_deepseek_v41_goldens_use_v41_format_markers():
    """Sanity-check the goldens actually exercise V4.1-specific format."""
    out1 = (FIXTURES_DIR / "test_output_1.txt").read_text()
    assert "<｜DSML｜ calls>" in out1
    assert '<｜DSML｜ invoke name="get_weather">' in out1
    assert "<｜DSML｜tool_calls>" not in out1

    out5 = (FIXTURES_DIR / "test_output_5.txt").read_text()
    assert out5.startswith(
        "<｜begin▁of▁sentence｜>" + REASONING_EFFORT_TEMPLATE.format(budget=100)
    )


# ============================================================
# Probe conditions (expected strings from the reference module)
# ============================================================

_SYS_MSGS = [
    {"role": "system", "content": "You are an AI assistant."},
    {"role": "user", "content": "Hello!"},
]
_NOSYS_MSGS = [{"role": "user", "content": "Hello!"}]

_PROBE_SYS_CHAT = (
    "<｜begin▁of▁sentence｜><｜System｜>You are an AI assistant."
    "<｜User｜>Hello!<｜Assistant｜></think>"
)
_PROBE_SYS_THINKING_100 = (
    "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 100 "
    "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
    "You are an AI assistant.<｜User｜>Hello!<｜Assistant｜><think>"
)
_PROBE_NOSYS_THINKING_100 = (
    "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 100 "
    "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
    "<｜User｜>Hello!<｜Assistant｜><think>"
)


def test_probe_sys_chat():
    assert encode_messages(_SYS_MSGS, thinking_mode="chat") == _PROBE_SYS_CHAT


def test_probe_sys_thinking_effort_100():
    assert (
        encode_messages(_SYS_MSGS, thinking_mode="thinking", reasoning_effort=100)
        == _PROBE_SYS_THINKING_100
    )


def test_probe_nosys_thinking_effort_100():
    assert (
        encode_messages(_NOSYS_MSGS, thinking_mode="thinking", reasoning_effort=100)
        == _PROBE_NOSYS_THINKING_100
    )


def test_probes_through_tokenizer_wrapper():
    tok = _tokenizer()
    assert (
        tok.apply_chat_template(_SYS_MSGS, tokenize=False) == _PROBE_SYS_CHAT
    )
    assert (
        tok.apply_chat_template(
            _SYS_MSGS, tokenize=False, enable_thinking=True, reasoning_effort=100
        )
        == _PROBE_SYS_THINKING_100
    )
    assert (
        tok.apply_chat_template(
            _NOSYS_MSGS, tokenize=False, enable_thinking=True, reasoning_effort=100
        )
        == _PROBE_NOSYS_THINKING_100
    )


@pytest.mark.skipif(
    not REFERENCE_ENCODING_PY.exists(),
    reason="checkpoint reference encoder not present on this machine",
)
def test_probes_and_goldens_match_reference_module():
    ref = _load_reference_module()
    assert ref.encode_messages(_SYS_MSGS, thinking_mode="chat") == _PROBE_SYS_CHAT
    assert (
        ref.encode_messages(_SYS_MSGS, thinking_mode="thinking", reasoning_effort=100)
        == _PROBE_SYS_THINKING_100
    )
    assert (
        ref.encode_messages(
            _NOSYS_MSGS, thinking_mode="thinking", reasoning_effort=100
        )
        == _PROBE_NOSYS_THINKING_100
    )
    for case_id in [1, 2, 3, 4, 5]:
        case = ref.load_cases(str(FIXTURES_DIR / f"test_input_{case_id}.json"))[0]
        ref_prompt, _ = ref.encode_case(case, thinking_mode="chat")
        ours = enc.load_cases(str(FIXTURES_DIR / f"test_input_{case_id}.json"))[0]
        our_prompt, _ = enc.encode_case(ours, thinking_mode="chat")
        assert our_prompt == ref_prompt


# ============================================================
# Ported from the checkpoint's test_encoding.py
# ============================================================


@pytest.mark.parametrize(
    ("effort", "budget"),
    [
        (None, 50),
        ("low", 25),
        ("high", 50),
        ("xhigh", 75),
        ("max", 100),
        (1, 1),
        (42, 42),
        (100, 100),
    ],
)
def test_v41_maps_reasoning_effort_to_1_100_budget(effort, budget):
    prompt = encode_messages(
        [{"role": "user", "content": "question"}],
        thinking_mode="thinking",
        reasoning_effort=effort,
    )

    assert prompt == (
        "<｜begin▁of▁sentence｜>"
        f"{REASONING_EFFORT_TEMPLATE.format(budget=budget)}"
        "<｜User｜>question<｜Assistant｜><think>"
    )


def test_v41_only_adds_reasoning_effort_to_first_thinking_message():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
    ]

    later_message = render_message(
        1, messages, thinking_mode="thinking", reasoning_effort=100
    )
    chat_message = render_message(
        0, messages, thinking_mode="chat", reasoning_effort=100
    )

    assert "Reasoning Effort:" not in later_message
    assert "Reasoning Effort:" not in chat_message


def test_v41_chat_mode_has_no_reasoning_effort_or_system_token():
    prompt = encode_messages(
        [{"role": "user", "content": "hello"}],
        thinking_mode="chat",
        reasoning_effort="max",
    )
    assert prompt == "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>"


@pytest.mark.parametrize("effort", [-1, 0, 101, "medium"])
def test_v41_encoder_rejects_out_of_range_or_unknown_reasoning_effort(effort):
    with pytest.raises(AssertionError, match=r"int within \[1,100\]"):
        encode_messages(
            [{"role": "user", "content": "question"}],
            thinking_mode="thinking",
            reasoning_effort=effort,
        )


def test_v41_leading_system_message_uses_system_token():
    prompt = encode_messages(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"},
        ],
        thinking_mode="chat",
    )
    assert prompt == (
        "<｜begin▁of▁sentence｜><｜System｜>You are a helpful assistant."
        "<｜User｜>hello<｜Assistant｜></think>"
    )


def test_v41_mid_conversation_system_message():
    prompt = encode_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
            {"role": "system", "content": "mid sys"},
        ],
        thinking_mode="thinking",
        reasoning_effort=88,
    )
    # Mid-conversation system gets its own <｜System｜> token and triggers
    # the assistant generation header afterwards.
    assert prompt == (
        "<｜begin▁of▁sentence｜>"
        f"{REASONING_EFFORT_TEMPLATE.format(budget=88)}"
        "sys<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>"
        "<｜System｜>mid sys<｜Assistant｜><think>"
    )


def test_v41_tool_instructions_use_spaced_dsml_tags():
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
        },
    }
    prompt = encode_messages(
        [
            {"role": "system", "content": "system", "tools": [tool]},
            {"role": "user", "content": "question"},
        ],
        thinking_mode="chat",
    )

    assert (
        "<｜DSML｜ calls>\n"
        '<｜DSML｜ invoke name="$TOOL_NAME">\n'
        '<｜DSML｜ parameter name="$PARAMETER_NAME" '
        'string="true|false">$PARAMETER_VALUE</｜DSML｜ parameter>\n'
        "...\n"
        "</｜DSML｜ invoke>"
    ) in prompt
    assert "<｜DSML｜tool_calls>" not in prompt
    assert "<｜DSML｜invoke" not in prompt
    assert "<｜DSML｜parameter" not in prompt


def test_v41_renders_and_parses_spaced_dsml_roundtrip():
    messages = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "reasoning_content": "  reason  ",
            "content": "summary",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"query":"value","limit":2}',
                    },
                }
            ],
        },
    ]

    prompt = render_message(1, messages, thinking_mode="thinking")
    assert prompt == V41_TOOL_CALL_OUTPUT

    parsed = parse_message_from_completion_text(
        V41_TOOL_CALL_OUTPUT, thinking_mode="thinking"
    )
    assert parsed["role"] == "assistant"
    assert parsed["reasoning_content"] == "  reason  "
    assert parsed["content"] == "summary"
    assert parsed["tool_calls"][0]["function"]["name"] == "lookup"
    assert json.loads(parsed["tool_calls"][0]["function"]["arguments"]) == {
        "query": "value",
        "limit": 2,
    }


def test_v41_drop_thinking_without_tools():
    prompt = encode_messages(
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
            {"role": "user", "content": "q2"},
        ],
        thinking_mode="thinking",
        drop_thinking=True,
    )
    assert "<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>" in prompt
    assert "r1" not in prompt
    assert prompt.endswith("<｜User｜>q2<｜Assistant｜><think>")


def test_merge_tool_messages_creates_tool_result_blocks():
    merged = merge_tool_messages(
        [
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "tool_call_id": "a", "content": "r1"},
            {"role": "tool", "tool_call_id": "b", "content": "r2"},
        ]
    )
    assert len(merged) == 2
    assert merged[1]["role"] == "user"
    assert [b["type"] for b in merged[1]["content_blocks"]] == [
        "tool_result",
        "tool_result",
    ]


# ============================================================
# Tokenizer wrapper behavior
# ============================================================


def test_deepseek_v41_defaults_to_chat_mode():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
    )
    assert prompt == "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>"


@pytest.mark.parametrize("kwargs", [{"thinking": True}, {"enable_thinking": True}])
def test_deepseek_v41_thinking_defaults_to_high_effort(kwargs):
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        **kwargs,
    )
    # Default thinking-mode effort is "high" = 50, behind the system token.
    assert prompt == (
        "<｜begin▁of▁sentence｜>"
        f"{REASONING_EFFORT_TEMPLATE.format(budget=50)}"
        "<｜User｜>Hello<｜Assistant｜><think>"
    )


def test_deepseek_v41_none_reasoning_effort_disables_thinking():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="none",
    )
    assert prompt == "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>"


def test_deepseek_v41_integer_effort_not_dropped():
    """Regression test for the V4-mode bug: integer efforts were silently
    dropped by an isinstance(str) check and rendered nothing."""
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort=87,
    )
    assert "Reasoning Effort: 87 (range 1-100" in prompt
    assert prompt.startswith("<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 87 ")


def test_deepseek_v41_tools_attach_to_existing_system_message():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Weather?"},
    ]
    prompt = _tokenizer().apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
    )

    # One <｜System｜> token: system content and tools render together.
    assert prompt.count("<｜System｜>") == 1
    assert "<｜System｜>You are helpful.\n\n## Tools" in prompt
    assert "<｜DSML｜ calls>" in prompt
    assert prompt.endswith("<｜User｜>Weather?<｜Assistant｜></think>")
    # Caller's message list must not be mutated.
    assert "tools" not in messages[0]


def test_deepseek_v41_tools_without_system_message():
    tools = [
        {
            "type": "function",
            "function": {"name": "f", "parameters": {"type": "object"}},
        }
    ]
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tools=tools,
        tokenize=False,
    )
    assert prompt.startswith("<｜begin▁of▁sentence｜><｜System｜>\n\n## Tools")
    assert "<｜DSML｜ calls>" in prompt


@pytest.mark.parametrize(
    ("effort", "expected"),
    [
        (None, (None, False)),
        ("none", (None, True)),
        ("low", ("low", False)),
        ("high", ("high", False)),
        ("xhigh", ("xhigh", False)),
        ("max", ("max", False)),
        ("minimal", ("low", False)),
        ("medium", ("high", False)),
        ("unexpected", (None, False)),
        (42, (42, False)),
        (1, (1, False)),
        (100, (100, False)),
        (0, (None, False)),
        (101, (None, False)),
        (True, (None, False)),
        (0.5, (50, False)),
        ("42", (42, False)),
    ],
)
def test_deepseek_v41_wrapper_effort_resolution(effort, expected):
    assert _resolve_reasoning_effort(effort) == expected
