# Shipping the `deepseek_v41` tokenizer mode for DeepSeek-V4.1-Flash

## The bug

Serving DeepSeek-V4.1-Flash with vLLM's `--tokenizer-mode deepseek_v4` renders
prompts in the **V4 dialect**, which is not what the V4.1 checkpoint was
trained on:

1. **System messages lose their special token.** `{"role":"system"}` renders
   as bare text. V4.1 expects `<｜System｜>` to lead the conversation, and
   again before every mid-conversation system message.
2. **Reasoning effort is wrong in thinking mode.** V4.1 always expects a
   numeric effort prefix rendered behind the system token:

   ```
   <｜System｜>Reasoning Effort: {N} (range 1-100, the higher the value, the more thorough the reasoning)\n\n
   ```

   with tiers low/high/xhigh/max mapped to 25/50/75/100, default "high" (50),
   and integers 1-100 accepted. The V4 encoder instead emits *nothing* unless
   the effort is the string `'max'` — and then it emits a completely different
   V4 prose block. Integer efforts are silently dropped by an
   `isinstance(reasoning_effort, str)` check in
   `vllm/tokenizers/deepseek_v4.py`.
3. **Dialect details differ.** DSML tool tags are `<｜DSML｜ calls>` /
   `<｜DSML｜ invoke>` / `<｜DSML｜ parameter>` (leading-space names) rather
   than V4's `<｜DSML｜tool_calls>` / `<｜DSML｜invoke>` /
   `<｜DSML｜parameter>`; mid-conversation system messages count as
   user-position messages for the assistant generation header.

**Symptom:** a plain
`{"role":"system","content":"You are an AI assistant."}` produces garbage
output intermittently (~15% of requests in chat mode, ~80% with reasoning
enabled), because the model sees system text and effort prompts in a framing
it never saw in training.

**Ground truth:** the checkpoint ships its own reference encoder
(`encoding/encoding.py` with tests and five golden input/output pairs in
`encoding/tests/`). The fix below reproduces it byte-for-byte on all five
golden pairs.

## The fix (this branch)

One self-contained encoding module plus thin shims:

| File | Role |
|---|---|
| `vllm/tokenizers/deepseek_v41_encoding.py` | **The core.** Faithful transcription of the checkpoint's reference encoder. Pure stdlib, no vLLM imports — portable to any stack. |
| `vllm/tokenizers/deepseek_v41.py` | Thin tokenizer-mode wrapper (mirrors `deepseek_v4.py`): accepts int 1-100 and low/high/xhigh/max, `reasoning_effort="none"` → chat mode, default thinking effort "high" (50). |
| `vllm/parser/deepseek_v41.py` | Output parser for V4.1's spaced DSML tool tags (reasoning `<think>`/`</think>` handling unchanged from V4). |
| `vllm/reasoning/deepseek_v41_engine_reasoning_parser.py`, `vllm/tool_parsers/deepseekv41_engine_tool_parser.py` | Two-line adapter shims. |
| Registry edits | `vllm/tokenizers/registry.py`, `vllm/renderers/registry.py`, `vllm/config/model.py`, `vllm/parser/engine/registered_adapters.py`, `vllm/reasoning/__init__.py`, `vllm/tool_parsers/__init__.py`. |

Serve with:

```
--tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41
```

(`--tokenizer-mode auto` also resolves to `deepseek_v41` for the
`DeepseekV41ForCausalLM` architecture after this change.)

### Why new parsers too

Reasoning output (`<think>...</think>`) is identical to V4, so
`--reasoning-parser deepseek_v4` happens to work for text+reasoning. But V4.1
**tool calls** use the spaced DSML tags, which the V4 parser does not match,
and the V4 tool parser requests the xgrammar builtin `deepseek_v4` structural
tag, which would *constrain* generation to the wrong (unspaced) grammar under
strict tools. Use `deepseek_v41` for both; the V4.1 tool parser deliberately
sets no structural-tag model.

## Applying to vanilla vLLM

`deepseek_v41_encoding.py` and `deepseek_v41.py` drop into
`vllm/tokenizers/`; `deepseek_v41.py` (parser) into `vllm/parser/`. Then the
minimal registration diff:

```diff
--- a/vllm/tokenizers/registry.py
+++ b/vllm/tokenizers/registry.py
@@ _VLLM_TOKENIZERS = {
     "deepseek_v4": ("deepseek_v4", "DeepseekV4Tokenizer"),
+    "deepseek_v41": ("deepseek_v41", "DeepseekV41Tokenizer"),

--- a/vllm/renderers/registry.py
+++ b/vllm/renderers/registry.py
@@ _VLLM_RENDERERS = {
     "deepseek_v4": ("deepseek_v4", "DeepseekV4Renderer"),
+    # V4.1 reuses the V4 renderer; the dialect lives in the tokenizer wrapper.
+    "deepseek_v41": ("deepseek_v4", "DeepseekV4Renderer"),

--- a/vllm/config/model.py
+++ b/vllm/config/model.py
@@
-TokenizerMode = Literal["auto", "hf", "slow", "mistral", "deepseek_v32", "deepseek_v4"]
+TokenizerMode = Literal[
+    "auto", "hf", "slow", "mistral", "deepseek_v32", "deepseek_v4", "deepseek_v41"
+]
@@ (auto tokenizer-mode defaulting)
             elif arch == "DeepseekV4ForCausalLM":
                 self.tokenizer_mode = "deepseek_v4"
+            elif arch == "DeepseekV41ForCausalLM":
+                self.tokenizer_mode = "deepseek_v41"

--- a/vllm/parser/engine/registered_adapters.py
+++ b/vllm/parser/engine/registered_adapters.py
@@
 from vllm.parser.deepseek_v4 import DeepSeekV4Parser
+from vllm.parser.deepseek_v41 import DeepSeekV41Parser
@@
+(
+    DeepSeekV41ParserReasoningAdapter,
+    DeepSeekV41ParserToolAdapter,
+) = make_adapters(DeepSeekV41Parser)

--- a/vllm/reasoning/__init__.py
+++ b/vllm/reasoning/__init__.py
@@ _REASONING_PARSERS_TO_REGISTER = {
+    "deepseek_v41": (
+        "deepseek_v41_engine_reasoning_parser",
+        "DeepSeekV41ParserReasoningAdapter",
+    ),

--- a/vllm/tool_parsers/__init__.py
+++ b/vllm/tool_parsers/__init__.py
@@ _TOOL_PARSERS_TO_REGISTER = {
+    "deepseek_v41": (
+        "deepseekv41_engine_tool_parser",
+        "DeepSeekV41EngineToolParser",
+    ),
```

The two adapter shims are:

```python
# vllm/reasoning/deepseek_v41_engine_reasoning_parser.py
from vllm.parser.engine.registered_adapters import DeepSeekV41ParserReasoningAdapter
__all__ = ["DeepSeekV41ParserReasoningAdapter"]

# vllm/tool_parsers/deepseekv41_engine_tool_parser.py
from vllm.parser.engine.registered_adapters import DeepSeekV41ParserToolAdapter

class DeepSeekV41EngineToolParser(DeepSeekV41ParserToolAdapter):
    pass  # no structural_tag_model: the "deepseek_v4" builtin is the wrong grammar
```

On a vLLM build that predates the `deepseek_v4` tokenizer mode / parser
engine, you can still ship the core module unchanged: register
`DeepseekV41Tokenizer` through `TokenizerRegistry.register()` from a plugin
(or call `deepseek_v41_encoding.encode_messages()` from any custom
chat-template hook). The encoding module has no vLLM imports, so nothing else
is required.

## sglang

**Checked: recent sglang already renders V4.1 correctly — it does not have
this bug.** Its tree contains a faithful port of the reference encoder at
`sglang/srt/entrypoints/openai/encoding_dsv41.py` (same `<｜System｜>` token,
same numeric `Reasoning Effort: {N}` template, same low/high/xhigh/max =
25/50/75/100 mappings, spaced DSML tags), dispatched by
`sglang/srt/entrypoints/openai/chat_encoding.py` when
`model_type == "deepseek_v41"` (or arch contains `DeepseekV41`, or
`--tool-call-parser deepseekv41` is set). It serves with
`--tool-call-parser deepseekv41 --reasoning-parser deepseek-v41`.

If you run an older sglang without `encoding_dsv41.py`: check whether your
version routes DeepSeek-V4.1 through `encoding_dsv4.py` (V4 dialect — same
bug as vLLM's `deepseek_v4` mode) or through a generic HF chat template (the
V4.1 checkpoint ships **no** `chat_template` in `tokenizer_config.json`, so
that path is also broken). In either case, drop this repo's
`deepseek_v41_encoding.py` next to `encoding_dsv4.py` and dispatch to its
`encode_messages(messages, thinking_mode=..., reasoning_effort=...)` for
`model_type == "deepseek_v41"` — the module is dependency-free, and its
`parse_message_from_completion_text()` gives you output parsing if your
sglang lacks the `deepseekv41` parsers.

## Validation

- **Golden pairs:** all five checkpoint input/output pairs encode
  byte-for-byte (`tests/tokenizers_/test_deepseek_v41.py`, fixtures under
  `tests/tokenizers_/fixtures/deepseek_v41/`).
- **Probe conditions** (unit-tested, expected strings generated by the
  checkpoint's reference encoder):
  - system + chat: `<｜begin▁of▁sentence｜><｜System｜>You are an AI assistant.<｜User｜>Hello!<｜Assistant｜></think>`
  - system + thinking, effort=100: `<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 100 (range 1-100, the higher the value, the more thorough the reasoning)\n\nYou are an AI assistant.<｜User｜>Hello!<｜Assistant｜><think>`
  - no system + thinking, effort=100: `<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 100 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>Hello!<｜Assistant｜><think>`
- Ported reference tests: chat mode emits no effort/system token,
  mid-conversation system, numeric effort behind the system token, spaced
  DSML round-trip, drop-thinking multi-turn, tool-message merging.
- Parser tests: spaced-DSML tool-call parsing (streaming and non-streaming),
  V4 unspaced tags rejected, `reasoning_effort="none"` disables the
  reasoning initial state.
