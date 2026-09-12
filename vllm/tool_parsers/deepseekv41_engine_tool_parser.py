# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import DeepSeekV41ParserToolAdapter


class DeepSeekV41EngineToolParser(DeepSeekV41ParserToolAdapter):  # type: ignore[valid-type, misc]
    # No structural_tag_model: the xgrammar builtin "deepseek_v4" grammar uses
    # the V4 unspaced DSML tags, which would constrain V4.1 output to the
    # wrong dialect. Leave unset until an upstream "deepseek_v4_1" builtin
    # (spaced tags: "<｜DSML｜ calls>" etc.) exists.
    pass
