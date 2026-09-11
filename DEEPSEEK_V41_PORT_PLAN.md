# DeepSeek-V4.1-Flash — vLLM SM120 port plan

Target: serve `deepseek-ai/DeepSeek-V4.1-Flash` on 4× RTX PRO 6000 (96 GB ea, SM120, no NVLink)
at **TP4, 1M context, native vision, MTP/DSpark spec-decode**, with the **engram memory tables
offloaded to host DRAM** (the only way it fits 384 GB).

Branch: `deepseek-v41-flash` off `dsv4pro0813` (fork `jarrelscy/vllm-glm52-sm120`, remote `glm52fork`).
Never push to `vllm-project/vllm`.

## Why engram-on-CPU is the fit lever
Full ckpt ~511 GB; official recipe wants 614 GB VRAM (GB200/H200). The `engram` is two
n-gram-indexed embedding tables (`engram_num_embeddings [384006168, 384016682]`, `engram_head_dim 256`)
≈ **196 GB at int8** — sparsely accessed by hash lookup. Move both tables to host DRAM and only the
**gathered rows** cross PCIe per token (per position: `engram_max_ngram_size-1` n-gram cols ×
`engram_n_heads 8` × `engram_head_dim 256`). GPU-resident weights then drop to ~315 GB ≈ 79 GB/GPU
at TP4; +KV(<1 GB/GPU @1M) +vision +graphs ≈ 86–88 GB/GPU. **Fits, ~8 GB headroom.**

Host-RAM budget is the tight real-world margin: 251 GB total, ~235 GB free with hybrid down;
196 GB engram leaves ~40 GB for the process + pinned staging + weight-load page cache. Swap is a dead
8 GB. Likely need non-pinned host tables + async row-gather (not full pinning) to stay safe.

## Reference implementation (ships in the HF repo — translate, don't reverse-engineer)
`…/DeepSeek-V4.1-Flash/snapshots/df42…/inference/`:
- `model.py` (62 KB) — authoritative arch (ModelArgs, CSA2 attn + 2-level indexer + candidate
  clustering, engram, MoE + hyper-connections, DSpark, MTP, top-level Transformer, self-test).
- `engram.py` — `NgramHashState` (hash-id computation; portable as-is). Big-table gather lives in model.py.
- `vision.py` — native ViT (32 L, 2D RoPE, bidirectional) + `Aligner` (downsample→MLP). Clean, self-contained.
- `kernel.py` (24 KB) — tilelang fp8/fp4 kernels.
- `convert.py` — HF → per-TP-rank sharded checkpoint; expert-dtype fp4.
- `generate.py`, `image_processor.py` — generation loop + vision preprocessing.
- `config.json` (flat, `inference/`) and the HF `config.json` (nested `text_config`/`vision_config`).

## Arch facts (from config.json)
dim 5120 · 40 layers · 64 Q heads / 1 KV head · head_dim 512 · qk_rope 64 · q_lora 1280 ·
o_lora 1024 · o_groups 8 · norm_eps 1e-20 · MoE 384 routed + 1 shared, top-6, `sqrtsoftplus`/`noaux_tc`,
route_scale 1.5, moe_inter 2304 · swiglu_limit 10 · rope_theta 10000, yarn factor 16 over 65536,
compress_rope_theta 160000 · window_size 128.
- CED: `kv_source_layers [2,8,14,20]` project the global KV consumed downstream.
- Indexer: `index_source_layers [2,8,14,20,24,28,32,36]`, index_n_heads 32, index_head_dim 128, index_topk 512.
- Candidate clustering: `candidate_source_layer 20`, candidate_topk_blocks 2048, candidate_block_size 8,
  hc_mult 4, hc_sinkhorn_iters 20, hc_eps 1e-6.
- `compress_ratios` (per layer, 0/1/2) select CSA2 Full/Reindex/Reuse.
- Engram at layers [1,14]; n_heads 8, head_dim 256, max_ngram 4, vocab 16M, compressed_vocab 99092.
- Spec: MTP `n_mtp_layers 3`; DSpark block 5, target_layers [37,38,39], markov_rank 256, 128 exp top-3.
- Vision: 32 L, dim 1024, 16 heads, inter 2816, patch 14, downsample 3, max 1024 img tokens, image_token_id 129264.
- Quant: FP8 attn/dense (ue8m0 block scales, weight_block_size [32,32]) + FP4 experts.

## Corrections after full model.py digest (supersede earlier assumptions)
- **compress_ratios do NOT select Full/Reindex/Reuse.** compress_ratio ∈{0,1,2} only sets KV pooling
  factor + rope theta (0=window-only+base rope; 2=pool 2→1+compress_rope 160000+YaRN; 1=1→1 project).
  The CSA2 mode is set by layer membership: **Full**=`kv_source_layers[2,8,14,20]` (own Compressor+index keys,
  publish shared compressed KV); **Reindex**=`index_source_layers\kv_source[24,28,32,36]` (own Indexer, re-score
  shared KV); **Reuse**=neither (read published `compress_kv`+`topk_idxs`). Sharing is by-reference through a
  single `SharedAttentionRuntime` scratch, ordered by layer execution — NOT a 0-20/20-40 encoder-decoder split.
- **Hyper-Connections are pervasive.** Residual stream = `hc_mult=4` parallel copies `[B,L,4,5120]`. Each sublayer:
  `hc_pre`(collapse 4→1 via pre_mix) → norm → attn/ffn → `hc_post`(expand+Sinkhorn-mix `comb[4,4]`). Coefficients
  computed by one sublayer feed the next (attn consumes previous ffn's pre_mix). `hc_split_sinkhorn` kernel,
  `hc_sinkhorn_iters=20`. deepseek_v4 has none of this → v41 DecoderLayer is new, not a v4 subclass.
- **Non-paged KV in reference.** State in per-layer buffers keyed by start_pos: `window_kv_cache[mb,128,512]`,
  `compress_kv_cache[mb,seq//ratio,512]`, indexer `k_cache[mb,seq//ratio,128]`, Compressor `kv_state`/`score_state`,
  `NgramHashState.cache`. Mapping these onto vLLM paged block-manager + cudagraph is the hardest port task.
- **o-proj is grouped block-diagonal LoRA** (o_groups=8): `wo_a` reshaped `[8,1024,4096]`, einsum per-group, then `wo_b`.
- **Engram tables are fp8 e4m3 + e8m0 scale** (~98 GB each, ~196 GB both). CPU-offload boundary = gathered
  `[24,256]` bf16 per token per engram layer (24 hash cols × head_dim 256). Gate: signed-sqrt→sigmoid dot of
  residual vs projected key, written into all 4 hc copies. Runs BEFORE the block at layers 1,14.
- **MoE gate**: sqrtsoftplus scores; **noaux_tc** selects top-6 via correction bias (`bias_vl` for image tokens)
  but weights come from UNBIASED scores; norm_topk_prob; route_scale 1.5. shared_expert on every token.
- **Spec-decode (DSpark/MTP) not production-wired even in reference** (`forward_spec` uncalled) → net-new plumbing.
- **lm_head kept fp32** (logits emerge fp32). embed sharded on vocab.

## Strategy: two phases (lossless bar forces reference-first)
- **Phase 1 — reference impl as oracle + fit proof.** Run the shipped `convert.py`(MP=4)+`generate.py` at TP4 with
  the engram `ParallelEngramEmbedding.weight` kept on host (CPU) + async row-gather. Proves 384 GB fit, engram-CPU
  thesis, vision, coherence; produces GOLDEN outputs to gate the port against. tilelang==0.1.8 env in a container.
  Converted TP4 ckpt needs ~510 GB more disk (have 2.6 TB). Gated on the HF download completing.
- **Phase 2 — vLLM production port** (below), each block diffed bit-exact vs Phase 1 (fp32 reference where feasible).
  This is the max-speed / paged-1M / spec-decode deliverable; large multi-week surface.

## v4 → v41 delta (reuse the existing `vllm/models/deepseek_v4/` package)
Existing v4 already provides: sparse-MLA + indexer (`attention.py`, `sparse_mla.py`,
`nvidia/flashinfer_sparse.py`), CSA compressor (`compressor.py`), MegaMoE fp4/fp8 (`nvidia/model.py`),
ue8m0 loading, MTP (`nvidia/mtp.py`), DSpark (`nvidia/dspark.py`), FP8+NVFP4 quant (`quant_config.py`).
**New for v41:**
1. **Engram memory layers** (host-offloaded) — biggest new piece. `NgramHashState` + big-table gather + project-into-residual.
2. **CED** global-KV projection from `kv_source_layers` (v4 shares differently — verify against digest).
3. **Candidate clustering** (sinkhorn) at `candidate_source_layer` feeding the indexer.
4. **o-projection LoRA** (`o_lora_rank`, `o_groups`) — new attention output structure.
5. **Native vision** (ViT+Aligner) replacing the v4 MoonViT graft.
6. Scoring `sqrtsoftplus`, `swiglu_limit`, norm_eps 1e-20.

## File plan (new package `vllm/models/deepseek_v41/`)
- `vllm/transformers_utils/configs/deepseek_v41.py` — `DeepseekV41Config` (promote `text_config` to top level, keep `.vision_config`). Register: `transformers_utils/config.py` + `_register_config_class`.
- `vllm/models/deepseek_v41/__init__.py` — platform-isolated re-export (mirror v4).
- `vllm/models/deepseek_v41/nvidia/model.py` — `DeepseekV41ForCausalLM` (extend v4 model; add engram, CED, o-lora).
- `vllm/models/deepseek_v41/engram.py` — `NgramHashState` (port) + `EngramMemory` host-offload module.
- `vllm/models/deepseek_v41/nvidia/vision.py` + `deepseek_v41_vision.py` — native ViT/Aligner + multimodal wrapper.
- `vllm/models/deepseek_v41/nvidia/{mtp,dspark}.py` — extend v4 spec-decode for v41 layer names.
- `vllm/models/deepseek_v41/quant_config.py` — subclass v4 FP8+FP4 config.
- Registry: `model_executor/models/registry.py` — `DeepseekV41ForCausalLM`, `DeepSeekV41MTP`, `DSparkDeepseekV41…`, vision cond-gen.

## Engram host-offload design
- Tables live in host DRAM (`torch.empty(num_embeddings, n_heads*head_dim, dtype=int8)` per engram layer),
  loaded from safetensors directly to CPU (skip the `to(device)` in `load_weights` for these params).
- Per forward: `NgramHashState` computes hash ids on GPU (cheap int ops) → copy the small id tensor to CPU →
  `index_select` rows from the host table → async H2D copy the gathered `[tokens, n_ngram_cols, n_heads, head_dim]`
  block on a side stream, overlap with attention/MoE of the surrounding layers → dequant + project + add to residual.
- Under cudagraph: the host gather + H2D is a host-sync — must run OUTSIDE the captured region (like the v4
  indexer python-metadata path). Gate engram in the eager/piecewise portion; do NOT put the CPU gather in FULL capture.
- Flag: `VLLM_DSV41_ENGRAM_DEVICE=cpu` (default cpu) / `cuda` to keep on GPU (needs the 614 GB config, won't fit here).

## Serving (new compose profile `dsv41flash`, folder `vllm-deepseek-v41flash-fp8fp4/`)
Template = the `dsv4flash-0731-tp4` block. Weights on `/data/huggingface` (cleared, 2.6 TB free).
Flags: `--tensor-parallel-size 4 --kv-cache-dtype fp8 --max-model-len 1048576 --max-num-seqs 4`
`--moe-backend flashinfer_b12x --linear-backend flashinfer_b12x --attention-backend FLASHINFER_MLA_SPARSE_SM120`
`--enable-chunked-prefill --enable-prefix-caching --gpu-memory-utilization 0.90`
`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` (DSpark variant later)
`VLLM_USE_V2_MODEL_RUNNER=1 VLLM_DSV41_ENGRAM_DEVICE=cpu`.

## Top risks
1. New-arch surface (CED + CSA2 modes + candidate clustering + engram + native ViT + o-lora) — largest lift.
2. Host RAM: 196 GB engram in 251 GB box, swap dead — may force non-pinned + slower gather, or block the fit.
3. Engram gather + spec-decode host-syncs vs cudagraph capture — expect the v4 "V2 runner only, gate out of FULL capture" cycle.

## POST-DELTA-MAP CORRECTIONS (2026-09-10, supersede the above where they conflict)
A structural v4→v41 diff (v4 vLLM package vs reference model.py/vision.py/engram.py) corrected
several plan assumptions:
- **v4 ALREADY has hyper-connections.** `DeepseekV4DecoderLayer` carries the `[tokens,hc_mult,dim]`
  residual stream + `mhc_pre/post/fused_post_pre/hc_head_fused` tilelang kernels (`post_mix`=post,
  `res_mix`=Sinkhorn `comb`). REUSE AS-IS. The earlier "v41 DecoderLayer is new, not a v4 subclass"
  is WRONG for the hc machinery.
- **Grouped o-LoRA is present-and-equivalent** in v4 (`wo_a` bmm ColumnParallel + `wo_b` RowParallel,
  `o_groups`/`o_lora_rank` match). REUSE AS-IS.
- **MoE gate core present** (`sqrtsoftplus` + `noaux_tc` `e_score_correction_bias` +
  `fused_topk_bias`, shared expert). Only delta = v41's `bias_vl` (image-token correction bias) —
  add ONLY when vision is wired.
- **MTP + DSpark present and close** (v4 `mtp.py`/`dspark.py` reuse the v4 DecoderLayer). Delta =
  v41 `DSparkConfidenceHead` is dropped in v4 (`confidence_head.*` skipped) — wire only if needed.
- **Candidate selection is block-amax two-level top-k, NOT Sinkhorn.** The only Sinkhorn in v41 is
  the hc mixing. Earlier "candidate clustering (sinkhorn)" is WRONG.
- **compress_ratios are incompatible.** v4 uses {1,4,128} with a learned APE + a `[4,128]` assert +
  indexer-only-when-ratio==4. v41 uses per-layer {0,1,2} (0=SWA-only, 1=plain bf16 projection no
  gate, 2=softmax pool-2 NO APE). Must add ratio-0/1 paths and drop the APE.

### Net-new subsystems (absent in v4) + build order
1. **Shared-KV runtime (CED membership)** — BIGGEST net-new. `kv_source_layers` publish `compress_kv`
   + `index_k`; `index_source_layers` publish `topk_idxs`; Reuse layers only read. In vLLM this must
   be a forward-context / registered-buffer object (NOT the reference Python `SharedAttentionRuntime`
   singleton) to survive cudagraph capture. Components 2/4/5 depend on it.
2. Compressor ratios {0,1,2} + drop APE.
3. Indexer decoupling + `owns_k` (derive keys only on kv_source layers, else read shared `index_k`);
   align index_n_heads 32/head_dim 128/topk 512 (already match).
4. Candidate blocks (`select_candidate_blocks` block-amax top-k at layer 20, publish
   `shared.candidates`, consumed by later indexers).
5. **Engram** — code ported to `deepseek_v41/engram.py`, hash + gate BIT-EXACT vs reference; wire at
   layers [1,14] before the block, on the `[tokens,hc,dim]` stream.
6. Native vision (ViT 2D-RoPE bidirectional + Aligner unfold/GELU MLP) + `merge_image_embeddings` +
   image_start/end/newline embeddings + `SupportsMultiModal` + image_mask → VL gate bias.

Reuse-as-is: hc (1), o-proj (6), gate core (7a), MTP/DSpark core (10), quant/rope/SWA infra,
weight-mapper/loader scaffolding.

## Status / next
- [x] Weights → `/data/huggingface`; reference impl read; TP4 golden captured (Phase 1 done).
- [x] `DeepseekV41Config` + registration — round-trip validated vs released config.json.
- [x] Engram module ported — NgramHashState + engram_gate BIT-EXACT vs reference.
- [x] Structural v4→v41 delta map (this section).
- [ ] Native vision port (self-contained, next validated increment).
- [ ] Shared-KV runtime (component 1) — critical path for CSA2.
- [ ] Assemble `nvidia/model.py` → `__init__.py` re-export + registry entries.
- [ ] Dockerfile + compose profile `dsv41flash`; TP4 1M serve gate vs golden.
