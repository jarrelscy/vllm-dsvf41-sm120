# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DeepSeek V4.1-specific parser engine semantics.

V4.1 output differs from V4 only in the DSML tag names: a ``calls`` wrapper
with leading-space tag names (``<｜DSML｜ calls>``, ``<｜DSML｜ invoke``,
``<｜DSML｜ parameter``).
"""

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import (
    collect_content,
    collect_function_name,
    collect_tool_arguments,
    simulate_reasoning_streaming,
    simulate_tool_streaming,
)
from vllm.parser.deepseek_v41 import (
    DSML_V41_INVOKE_NAME_END,
    DSML_V41_INVOKE_PREFIX,
    DSML_V41_THINK_END,
    DSML_V41_THINK_START,
    DSML_V41_TOOL_END,
    DSML_V41_TOOL_START,
    DeepSeekV41Parser,
    _dsml_v41_arg_converter,
    deepseek_v41_config,
)

_THINK_START_ID = 50
_THINK_END_ID = 51

_PARAM_OPEN = '｜DSML｜ parameter name="{name}" string="{is_str}">'
_PARAM_CLOSE = "</｜DSML｜ parameter>"


def _param(name: str, is_str: str, value: str) -> str:
    return f"<{_PARAM_OPEN.format(name=name, is_str=is_str)}{value}{_PARAM_CLOSE}"


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(
        {
            DSML_V41_THINK_START: _THINK_START_ID,
            DSML_V41_THINK_END: _THINK_END_ID,
        }
    )


class TestV41TagNames:
    def test_spaced_tag_names(self):
        assert DSML_V41_TOOL_START == "<｜DSML｜ calls>"
        assert DSML_V41_TOOL_END == "</｜DSML｜ calls>"
        assert DSML_V41_INVOKE_PREFIX == '<｜DSML｜ invoke name="'


class TestV41ArgConverter:
    def _raw(self, *params: tuple[str, str, str]) -> str:
        lines = [_param(n, s, v) for n, s, v in params]
        return "\n" + "\n".join(lines) + "\n"

    def test_string_param(self):
        raw = self._raw(("city", "true", "杭州"))
        result = json.loads(_dsml_v41_arg_converter(raw, partial=False))
        assert result == {"city": "杭州"}

    def test_mixed_types(self):
        raw = self._raw(
            ("query", "true", "value"),
            ("limit", "false", "2"),
            ("flag", "false", "true"),
        )
        result = json.loads(_dsml_v41_arg_converter(raw, partial=False))
        assert result == {"query": "value", "limit": 2, "flag": True}

    def test_partial_in_progress(self):
        raw = "\n" + _param("done", "true", "x") + '\n<｜DSML｜ parameter name="q" string="true">par'
        result = json.loads(_dsml_v41_arg_converter(raw, partial=True))
        assert result == {"done": "x", "q": "par"}

    def test_v4_unspaced_tags_not_matched(self):
        raw = '\n<｜DSML｜parameter name="city" string="true">NYC</｜DSML｜parameter>\n'
        result = json.loads(_dsml_v41_arg_converter(raw, partial=False))
        assert result == {}


class TestV41ThinkingModeConfig:
    def test_thinking_true_starts_in_reasoning(self):
        cfg = deepseek_v41_config(thinking=True)
        assert cfg.initial_state.name == "REASONING"

    def test_thinking_false_starts_in_content(self):
        cfg = deepseek_v41_config(thinking=False)
        assert cfg.initial_state.name == "CONTENT"

    def test_reasoning_effort_none_overrides_enable_thinking(self, mock_tokenizer):
        p = DeepSeekV41Parser(
            mock_tokenizer,
            chat_template_kwargs={
                "enable_thinking": True,
                "reasoning_effort": "none",
            },
        )
        assert p.parser_engine_config.initial_state.name == "CONTENT"

    def test_integer_reasoning_effort_keeps_thinking(self, mock_tokenizer):
        p = DeepSeekV41Parser(
            mock_tokenizer,
            chat_template_kwargs={"thinking": True, "reasoning_effort": 100},
        )
        assert p.parser_engine_config.initial_state.name == "REASONING"


class TestV41Reasoning:
    def test_thinking_reasoning_then_content(self, mock_tokenizer):
        parser = DeepSeekV41Parser(
            mock_tokenizer, chat_template_kwargs={"thinking": True}
        )
        chunks = ["Let me think.", "</think>", "The answer."]
        reasoning, content = simulate_reasoning_streaming(parser, chunks)
        assert "Let me think." in reasoning
        assert "The answer." in content


class TestV41ToolCalls:
    def test_non_streaming(self, mock_tokenizer, mock_request):
        parser = DeepSeekV41Parser(mock_tokenizer)
        text = (
            f"{DSML_V41_TOOL_START}\n"
            f"{DSML_V41_INVOKE_PREFIX}get_weather{DSML_V41_INVOKE_NAME_END}\n"
            f"{_param('location', 'true', 'NYC')}\n"
            f"{DSML_V41_TOOL_END}"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"location": "NYC"}

    def test_streaming(self, mock_tokenizer, mock_request):
        parser = DeepSeekV41Parser(mock_tokenizer)
        chunks = [
            DSML_V41_TOOL_START,
            f"\n{DSML_V41_INVOKE_PREFIX}get_weather{DSML_V41_INVOKE_NAME_END}\n"
            f"{_param('location', 'true', 'NYC')}\n",
            DSML_V41_TOOL_END,
            "Done.",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        assert collect_function_name(results) == "get_weather"
        args = json.loads(collect_tool_arguments(results))
        assert args == {"location": "NYC"}
        assert "Done." in collect_content(results)

    def test_reference_shaped_output(self, mock_tokenizer, mock_request):
        """Exactly the assistant completion shape the reference encoder
        renders (and its test_encoding.py round-trips)."""
        text = (
            "summary\n\n"
            "<｜DSML｜ calls>\n"
            '<｜DSML｜ invoke name="lookup">\n'
            '<｜DSML｜ parameter name="query" string="true">value'
            "</｜DSML｜ parameter>\n"
            '<｜DSML｜ parameter name="limit" string="false">2'
            "</｜DSML｜ parameter>\n"
            "</｜DSML｜ invoke>\n"
            "</｜DSML｜ calls>"
        )
        result = DeepSeekV41Parser(mock_tokenizer).extract_tool_calls(
            text, mock_request
        )
        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "lookup"
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "query": "value",
            "limit": 2,
        }
