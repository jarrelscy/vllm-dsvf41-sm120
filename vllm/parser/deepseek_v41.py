# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 parser: ``<think>``/``</think>``
reasoning plus DSML tool calls in a single state machine.

Identical to the DeepSeek V4 parser except for the DSML tag names, which in
V4.1 use a ``calls`` wrapper and leading-space tag names::

    <think>
    ...reasoning...
    </think>
    <｜DSML｜ calls>
    <｜DSML｜ invoke name="func_name">
    <｜DSML｜ parameter name="location" string="true">杭州</｜DSML｜ parameter>
    <｜DSML｜ parameter name="count" string="false">5</｜DSML｜ parameter>
    </｜DSML｜ invoke>
    </｜DSML｜ calls>
"""

from __future__ import annotations

import contextlib
import functools
import json
from typing import TYPE_CHECKING

import regex as re

from vllm.parser.deepseek_v4 import _unwrap_wrapper_args
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

_DSML = "｜DSML｜"

DSML_V41_THINK_START = "<think>"
DSML_V41_THINK_END = "</think>"
DSML_V41_TOOL_START = f"<{_DSML} calls>"
DSML_V41_TOOL_END = f"</{_DSML} calls>"
DSML_V41_INVOKE_PREFIX = f'<{_DSML} invoke name="'
DSML_V41_INVOKE_NAME_END = '">'
DSML_V41_INVOKE_END = f"</{_DSML} invoke>"
DSML_V41_PARAM_CLOSE = f"</{_DSML} parameter>"

_ESCAPED_DSML = re.escape(_DSML)
_V41_PARAM_RE = re.compile(
    rf'<{_ESCAPED_DSML} parameter\s+name="([^"]+)"\s+string="(true|false)">'
    rf"(.*?)</{_ESCAPED_DSML} parameter>",
    re.DOTALL,
)
_V41_PARTIAL_PARAM_RE = re.compile(
    rf'<{_ESCAPED_DSML} parameter\s+name="([^"]+)"\s+string="(true|false)">'
    rf"(.*)$",
    re.DOTALL,
)


def _dsml_v41_arg_converter(raw_args: str, partial: bool) -> str:
    params: dict[str, object] = {}

    last_end = 0
    for m in _V41_PARAM_RE.finditer(raw_args):
        name, is_str, value = m.group(1), m.group(2), m.group(3)
        if is_str == "true":
            params[name] = value
        else:
            try:
                params[name] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                params[name] = value
        last_end = m.end()

    if partial:
        pm = _V41_PARTIAL_PARAM_RE.search(raw_args, last_end)
        if pm:
            name, is_str, value = pm.group(1), pm.group(2), pm.group(3)
            if is_str == "true":
                params[name] = value
            else:
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    params[name] = json.loads(value)

    return json.dumps(params, ensure_ascii=False)


@functools.cache
def deepseek_v41_config(thinking: bool = False) -> ParserEngineConfig:
    return ParserEngineConfig(
        name="deepseek_v41",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        terminals={
            "THINK_START": DSML_V41_THINK_START,
            "THINK_END": DSML_V41_THINK_END,
            "TOOL_START": DSML_V41_TOOL_START,
            "TOOL_END": DSML_V41_TOOL_END,
            "INVOKE_PREFIX": DSML_V41_INVOKE_PREFIX,
            "INVOKE_NAME_END": DSML_V41_INVOKE_NAME_END,
            "INVOKE_END": DSML_V41_INVOKE_END,
            "PARAM_CLOSE": DSML_V41_PARAM_CLOSE,
        },
        token_id_terminals={
            "THINK_START": DSML_V41_THINK_START,
            "THINK_END": DSML_V41_THINK_END,
            "TOOL_START": DSML_V41_TOOL_START,
            "TOOL_END": DSML_V41_TOOL_END,
        },
        transitions={
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            # Absorb a bare </think> with no prior <think>
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            # Absorb a duplicate <think> while already reasoning
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            # Tool call beginning while still inside <think>
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
            (ParserState.TOOL_PREAMBLE, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_NAME, "INVOKE_NAME_END"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
            (ParserState.TOOL_ARGS, "INVOKE_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            # Parallel tool calls
            (ParserState.TOOL_BETWEEN, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
        },
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
        arg_converter=_dsml_v41_arg_converter,
        arg_structural_chars=frozenset(">"),
        strip_content_whitespace_with_tools=False,
        tool_args_json=False,
    )


class DeepSeekV41Parser(ParserEngine):
    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.pop("chat_template_kwargs", None) or {}
        thinking = (
            bool(chat_kwargs.get("thinking") or chat_kwargs.get("enable_thinking"))
            and chat_kwargs.get("reasoning_effort") != "none"
        )
        super().__init__(
            tokenizer,
            tools,
            parser_engine_config=deepseek_v41_config(thinking=thinking),
            **kwargs,
        )
        self._arg_converter = self._convert_args

    def _convert_args(self, raw_args: str, partial: bool) -> str:
        result = _dsml_v41_arg_converter(raw_args, partial)
        if not self._tools:
            return result
        func_name = next((s.name for s in self._tool_slots if s.args == raw_args), None)
        return _unwrap_wrapper_args(result, self._tools, func_name)
