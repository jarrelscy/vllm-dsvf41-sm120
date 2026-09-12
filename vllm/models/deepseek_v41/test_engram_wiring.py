# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU validation of the DeepSeek-V4.1-Flash engram SERVING wiring.

Complements the existing bit-exactness harnesses (validate_engram.py for the
hash core, validate_engram_gate.py for the gate): this file validates the NEW
serving pieces added when the engram was wired into the vLLM port:

  1. EngramServingHasher (stateless per-token windows) vs the reference-shaped
     NgramHashState across one-shot prefill / chunked prefill / one-token decode
     / prefix-cache-hit windows (history tokens NOT in the forward batch) /
     spec-decode rollback (recompute after replacing rejected tokens) / image
     dead-token masking. All must be bit-exact.
  2. The full Engram module (gather -> fp8/e8m0 dequant -> wkv -> gate) vs a
     verbatim copy of the reference inference/model.py module on a synthetic
     table, bit-exact, including the serving [T, hc, dim] layout.
  3. The mmap-backed host table against the REAL checkpoint: shapes/dtypes and
     a row-gather dequant compared bit-exactly against bytes read straight from
     the safetensors shard through an independent code path.
  4. A loader dry-run: every ``.engram.`` checkpoint tensor name maps (through
     the hf_to_vllm mapper + the load_weights renames) onto exactly one module
     parameter with matching shape/dtype expectations, and the ram-mode row
     shard loader copies the right rows.

Run inside the serving image (host python lacks torch), no GPU needed:

  docker run --rm --entrypoint /opt/vllm/.venv/bin/python \
    -e HF_HUB_OFFLINE=1 -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH=/fork \
    -v /home/jarrelscy/dsv4pro/vllm:/fork:ro \
    -v /data/huggingface:/data/huggingface:ro \
    dsv4pro-sm120:latest /fork/vllm/models/deepseek_v41/test_engram_wiring.py
"""
import json
import struct
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

CKPT = "/data/huggingface/dsv41-serve"
TOK = "/data/huggingface/dsv41-flash-tp4"

failures: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  {name}: {'PASS' if ok else 'FAIL'} {detail}")
    if not ok:
        failures.append(name)


# ---------------------------------------------------------------------------
# distributed init (single rank, gloo) so the TP-aware modules construct
# ---------------------------------------------------------------------------
from vllm.distributed import (  # noqa: E402
    init_distributed_environment,
    initialize_model_parallel,
)

from vllm.config import (  # noqa: E402
    DeviceConfig,
    VllmConfig,
    set_current_vllm_config,
)

init_distributed_environment(
    world_size=1, rank=0,
    distributed_init_method="tcp://127.0.0.1:29765",
    local_rank=0, backend="gloo",
)
with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
    initialize_model_parallel(1, 1)

from vllm.models.deepseek_v41.engram import (  # noqa: E402
    Engram,
    EngramLayout,
    EngramServingHasher,
    NgramHashState,
    ParallelEngramEmbedding,
)
from vllm.transformers_utils.configs.deepseek_v41 import (  # noqa: E402
    DeepseekV41Config,
)

cfg = DeepseekV41Config(**json.load(open(f"{CKPT}/config.json")))
ra = cfg.reference_args
IMG_ID = ra["image_token_id"]

print("building layout + tokenizer (compressed token map)...")
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(TOK)
layout = EngramLayout.from_config(cfg)
N = layout.max_ngram_size

MBS, MSL = 1, 512
ref_state = NgramHashState(cfg, layout, tok, max_batch_size=MBS, max_seq_len=MSL)
hasher = EngramServingHasher(cfg, layout, tok, image_token_id=IMG_ID)


def windows_from_ids(ids_1d: torch.Tensor, positions: torch.Tensor,
                     full_ids: torch.Tensor) -> torch.Tensor:
    """Build [T, N] windows exactly as gpu_model_runner._prepare_inputs does:
    column 0 = the token, column s = full_ids[max(pos - s, 0)]."""
    cols = [ids_1d.to(torch.int32)]
    for s in range(1, N):
        prev = full_ids[(positions - s).clamp_min(0)]
        cols.append(prev.to(torch.int32))
    return torch.stack(cols, dim=-1)


# ---------------------------------------------------------------------------
# 1. hash-state equivalence
# ---------------------------------------------------------------------------
print("\n[1] serving hasher vs reference ring-cache state")
torch.manual_seed(0)
L = 96
full = torch.randint(0, len(tok), (MBS, L))

# reference: one-shot prefill from 0
ref_state.cache.zero_()
ref_all = ref_state.forward(full, 0, None)  # [1, L, 2, 24]

# (a) one-shot: serving windows over the whole sequence
pos = torch.arange(L)
win = windows_from_ids(full[0], pos, full[0])
srv_all, mask = hasher(win, pos)
check("one-shot prefill", torch.equal(srv_all, ref_all[0]))
check("text token_mask all-True", mask is not None and bool(mask.all()))

# (b) chunked prefill: chunk boundary at 41
srv_c2, _ = hasher(win[41:], pos[41:])
check("chunked prefill (2nd chunk)", torch.equal(srv_c2, ref_all[0, 41:]))

# (c) decode one token at a time (last 5 tokens)
ok = True
for t in range(L - 5, L):
    srv_t, _ = hasher(win[t : t + 1], pos[t : t + 1])
    ok &= torch.equal(srv_t[0], ref_all[0, t])
check("one-token decode", ok)

# (d) prefix-cache hit: only tokens [37:] are in the forward batch, but their
# windows carry predecessors 34..36 from the (cache-skipped) prompt history
srv_hit, _ = hasher(win[37:], pos[37:])
check("prefix-cache-hit windows", torch.equal(srv_hit, ref_all[0, 37:]))

# (e) spec-decode rollback: last 4 "draft" tokens rejected and replaced; the
# recomputed windows must equal a fresh one-shot over the corrected sequence
full2 = full.clone()
full2[0, -4:] = torch.randint(0, len(tok), (4,))
ref_state.cache.zero_()
ref_all2 = ref_state.forward(full2, 0, None)
win2 = windows_from_ids(full2[0], pos, full2[0])
srv_tail, _ = hasher(win2[-6:], pos[-6:])
check("spec rollback recompute", torch.equal(srv_tail, ref_all2[0, -6:]))

# (f) image span: positions 10..19 carry the image token id
full3 = full.clone()
full3[0, 10:20] = IMG_ID
tmask = full3 != IMG_ID
ref_state.cache.zero_()
ref_all3 = ref_state.forward(full3, 0, tmask)
win3 = windows_from_ids(full3[0], pos, full3[0])
srv3, mask3 = hasher(win3, pos)
check("image-span hashes", torch.equal(srv3, ref_all3[0]))
check("image-span token_mask", torch.equal(mask3, tmask[0]))

# ---------------------------------------------------------------------------
# 2. full Engram module vs verbatim reference (synthetic table, TP1)
# ---------------------------------------------------------------------------
print("\n[2] Engram module vs verbatim reference (synthetic table)")


class RefParallelEngramEmbedding(nn.Module):
    """Verbatim from inference/model.py (world_size=1, table on CPU)."""

    def __init__(self, num_embeddings, dim, block_size=32):
        super().__init__()
        self.vocab_start_idx, self.vocab_end_idx = 0, num_embeddings
        self.block_size = block_size
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, dim, dtype=torch.float8_e4m3fn),
            requires_grad=False)
        self.scale = nn.Parameter(
            torch.empty(num_embeddings, dim // block_size,
                        dtype=torch.float8_e8m0fnu), requires_grad=False)

    def forward(self, indices):
        compute_device = indices.device
        mask = (indices < self.vocab_start_idx) | (indices >= self.vocab_end_idx)
        local_indices = (indices - self.vocab_start_idx).masked_fill(mask, 0)
        gather_indices = local_indices.to(self.weight.device)
        values = F.embedding(gather_indices, self.weight)
        scales = F.embedding(gather_indices, self.scale)
        values = values.float().unflatten(-1, (-1, self.block_size)) \
            * scales.float().unsqueeze(-1)
        values = values.flatten(-2).to(torch.bfloat16).to(compute_device)
        return values.masked_fill(mask.unsqueeze(-1), 0)


class RefEngram(nn.Module):
    """Verbatim math from inference/model.py Engram (wkv = plain bf16 linear)."""

    def __init__(self, dim, hc_mult, eps, embed, wkv_weight):
        super().__init__()
        self.dim, self.hc_mult, self.eps, self.clamp_value = dim, hc_mult, eps, 1e-6
        self.embed = embed
        self.wkv_weight = wkv_weight
        self.q_weight = nn.Parameter(torch.ones(hc_mult, dim), requires_grad=False)
        self.k_weight = nn.Parameter(torch.ones(hc_mult, dim), requires_grad=False)

    def forward(self, x, hash_ids, token_mask=None):
        kv = F.linear(self.embed(hash_ids).flatten(-2), self.wkv_weight)
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()
        h, eps = x.float(), self.eps
        rstd = torch.rsqrt(h.square().mean(-1) + eps) \
            * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        gate = torch.sigmoid(
            torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


torch.manual_seed(1)
dim, hc, head_dim = ra["dim"], ra["hc_mult"], layout.head_dim
n_hash_cols = (N - 1) * layout.n_heads
SMALL_ROWS = 4096

# tiny config namespace so Engram builds a synthetic-size table
small_cfg = SimpleNamespace(reference_args={
    **ra, "engram_num_embeddings": (SMALL_ROWS, SMALL_ROWS)})
small_layout = EngramLayout(
    max_ngram_size=layout.max_ngram_size, layer_ids=layout.layer_ids,
    num_embeddings=(SMALL_ROWS, SMALL_ROWS), primes=layout.primes,
    n_heads=layout.n_heads, head_dim=layout.head_dim)

wkv_weight = (torch.randn(dim * (hc + 1), n_hash_cols * head_dim) / 64).to(
    torch.bfloat16)


class PlainWkv(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight)


port_eng = Engram(small_cfg, 1, small_layout, PlainWkv(wkv_weight),
                  table_device="cpu", table_source="ram")
w_bytes = torch.randint(0, 255, (SMALL_ROWS, head_dim), dtype=torch.uint8)
w_bytes[w_bytes == 0x7F] = 0  # avoid fp8 NaN encodings
w_bytes[w_bytes == 0xFF] = 0
s_bytes = torch.randint(118, 132, (SMALL_ROWS, head_dim // 32), dtype=torch.uint8)
port_eng.embed.weight.data.copy_(w_bytes.view(torch.float8_e4m3fn))
port_eng.embed.scale.data.copy_(s_bytes.view(torch.float8_e8m0fnu))

ref_embed = RefParallelEngramEmbedding(SMALL_ROWS, head_dim)
ref_embed.weight.data.copy_(w_bytes.view(torch.float8_e4m3fn))
ref_embed.scale.data.copy_(s_bytes.view(torch.float8_e8m0fnu))
ref_eng = RefEngram(dim, hc, ra["norm_eps"], ref_embed, wkv_weight)

T = 9
x = torch.randn(T, hc, dim, dtype=torch.bfloat16)
hash_ids = torch.randint(0, SMALL_ROWS, (T, n_hash_cols))
tmask = torch.ones(T, dtype=torch.bool)
tmask[3] = False
out_port = port_eng(x, hash_ids, tmask)
out_ref = ref_eng(x, hash_ids, tmask)
check("Engram forward [T,hc,dim] bit-exact", torch.equal(out_port, out_ref))
# reference batch layout [B, L, hc, dim] must give the same numbers
out_port_b = port_eng(x.unsqueeze(0), hash_ids.unsqueeze(0), tmask.unsqueeze(0))
check("Engram forward [B,L,hc,dim] == [T,hc,dim]",
      torch.equal(out_port_b[0], out_ref))

# ---------------------------------------------------------------------------
# 3. mmap host table on the REAL checkpoint
# ---------------------------------------------------------------------------
print("\n[3] mmap-backed host table vs raw checkpoint bytes (layer 1)")
emb = ParallelEngramEmbedding(
    layout.num_embeddings[0], head_dim, table_device="cpu",
    table_source="mmap", ckpt_dir=CKPT, ckpt_prefix="layers.1.engram.embed")
check("mmap weight shape/dtype",
      tuple(emb.weight.shape) == (layout.num_embeddings[0], head_dim)
      and emb.weight.dtype == torch.float8_e4m3fn)
check("mmap scale shape/dtype",
      tuple(emb.scale.shape) == (layout.num_embeddings[0], head_dim // 32)
      and emb.scale.dtype == torch.float8_e8m0fnu)

# independent read path: seek + read the same rows straight from the shard
index = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
shard = f"{CKPT}/{index['layers.1.engram.embed.weight']}"
with open(shard, "rb") as f:
    hlen = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(hlen))
    w_lo = 8 + hlen + hdr["layers.1.engram.embed.weight"]["data_offsets"][0]
    s_lo = 8 + hlen + hdr["layers.1.engram.embed.scale"]["data_offsets"][0]

    def read_row(base, row, width):
        f.seek(base + row * width)
        return torch.frombuffer(bytearray(f.read(width)), dtype=torch.uint8)

    torch.manual_seed(2)
    rows = torch.randint(0, layout.num_embeddings[0], (7,))
    ok_w = ok_deq = True
    for r in rows.tolist():
        w_raw = read_row(w_lo, r, head_dim)
        s_raw = read_row(s_lo, r, head_dim // 32)
        ok_w &= torch.equal(emb.weight[r].view(torch.uint8), w_raw)
        manual = (
            w_raw.view(torch.float8_e4m3fn).float().unflatten(-1, (-1, 32))
            * s_raw.view(torch.float8_e8m0fnu).float().unsqueeze(-1)
        ).flatten(-2).to(torch.bfloat16)
        got = emb(rows.new_tensor([[r]]))[0, 0]
        ok_deq &= torch.equal(got, manual)
check("mmap rows == raw shard bytes", ok_w)
check("mmap gather+dequant bit-exact", ok_deq)

# ---------------------------------------------------------------------------
# 4. loader dry-run: name mapping + row-shard loader
# ---------------------------------------------------------------------------
print("\n[4] loader dry-run")
ckpt_engram_names = sorted(k for k in index if ".engram." in k)


def map_name(name: str) -> str:
    """hf_to_vllm mapper (fp4 rules) + the load_weights engram renames."""
    if name.startswith("layers."):
        name = "model." + name
    if name.endswith(".scale") and ".experts." not in name:
        name = name[: -len(".scale")] + ".weight_scale_inv"
    if ".engram.embed." in name and name.endswith(".weight_scale_inv"):
        name = name[: -len(".weight_scale_inv")] + ".scale"
    return name


# expected module-side parameter names, per engram layer
expected = set()
for lid in layout.layer_ids:
    p = f"model.layers.{lid}.engram"
    expected |= {f"{p}.embed.weight", f"{p}.embed.scale",
                 f"{p}.wkv.weight", f"{p}.wkv.weight_scale_inv",
                 f"{p}.q_weight", f"{p}.k_weight"}
mapped = {map_name(n) for n in ckpt_engram_names}
check("every ckpt engram tensor maps to a param",
      mapped == expected, f"(mapped {len(mapped)}/{len(expected)})")
if mapped != expected:
    print("    missing:", sorted(expected - mapped))
    print("    unexpected:", sorted(mapped - expected))

# ram-mode row shard loader (ws=1: shard == full table)
emb_ram = ParallelEngramEmbedding(SMALL_ROWS, head_dim, table_device="cpu",
                                  table_source="ram")
full_w = torch.randint(0, 127, (SMALL_ROWS, head_dim), dtype=torch.uint8).view(
    torch.float8_e4m3fn)
emb_ram.weight.weight_loader(emb_ram.weight, full_w)
check("ram row-shard loader copies rows",
      torch.equal(emb_ram.weight.data.view(torch.uint8),
                  full_w.view(torch.uint8)))
bad = torch.zeros(SMALL_ROWS + 1, head_dim, dtype=torch.float8_e4m3fn)
try:
    emb_ram.weight.weight_loader(emb_ram.weight, bad)
    check("ram loader rejects row mismatch", False)
except ValueError:
    check("ram loader rejects row mismatch", True)

# mmap-mode loader validates and never copies
try:
    emb.weight.weight_loader(emb.weight, torch.empty(
        layout.num_embeddings[0], head_dim, dtype=torch.float8_e4m3fn,
        device="meta"))
    check("mmap loader validate-only accepts", True)
except Exception as e:  # noqa: BLE001
    check("mmap loader validate-only accepts", False, repr(e))
try:
    emb.weight.weight_loader(emb.weight, torch.empty(3, 3, device="meta"))
    check("mmap loader rejects bad shape", False)
except ValueError:
    check("mmap loader rejects bad shape", True)

# ---------------------------------------------------------------------------
# 5. V2-runner GPU window gather (gather_token_windows_gpu, on CPU tensors)
# ---------------------------------------------------------------------------
print("\n[5] V2 window gather: batch/committed split, drafts, stale history")
from vllm.models.deepseek_v41.engram import (  # noqa: E402
    gather_token_windows_gpu,
)

torch.manual_seed(3)
R, M = 8, 256
VOCAB = 1000
all_tok = torch.randint(0, VOCAB, (R, M), dtype=torch.int32)
# batch: req A = decode (1 token @ pos 100, state row 5),
#        req B = chunked prefill (7 tokens @ pos 40..46, state row 2),
#        req C = spec verify (1 anchor + 5 drafts @ pos 60..65, state row 0)
qsl = torch.tensor([0, 1, 8, 14], dtype=torch.int32)
idx_map = torch.tensor([5, 2, 0], dtype=torch.int32)
pos = torch.cat([
    torch.tensor([100]),
    torch.arange(40, 47),
    torch.arange(60, 66),
]).to(torch.int64)
num_tokens = 14
# input_ids: decode anchor + prefill chunk tokens agree with all_token_ids;
# req C's 5 DRAFT tokens (positions 61..65) exist ONLY in input_ids -- and we
# plant DIFFERENT (stale, "rejected") values at those all_token_ids slots to
# prove within-batch predecessors come from the batch, not history.
input_ids = torch.empty(num_tokens, dtype=torch.int32)
input_ids[0] = all_tok[5, 100]
input_ids[1:8] = all_tok[2, 40:47]
input_ids[8] = all_tok[0, 60]  # anchor (committed)
drafts = torch.randint(0, VOCAB, (5,), dtype=torch.int32)
input_ids[9:14] = drafts
all_tok[0, 61:66] = (drafts + 7) % VOCAB  # stale/rejected junk in history

windows = torch.zeros(32, N, dtype=torch.int32)
gather_token_windows_gpu(
    windows, input_ids, pos, qsl, idx_map, all_tok, num_tokens, 3)

# ground truth: per-request full sequences with drafts spliced in
seq_a = all_tok[5, :101]
seq_b = all_tok[2, :47]
seq_c = torch.cat([all_tok[0, :61], drafts])
expect = torch.stack([
    windows_from_ids(seq_a[100:101], torch.tensor([100]), seq_a)[0],
    *windows_from_ids(seq_b[40:47], torch.arange(40, 47), seq_b),
    *windows_from_ids(seq_c[60:66], torch.arange(60, 66), seq_c),
])
check("V2 gather windows (decode+chunk+verify)",
      torch.equal(windows[:num_tokens], expect))
# drafts' predecessors really came from the batch, not the stale history rows
check("V2 gather ignores stale rejected drafts",
      torch.equal(windows[10, 1], drafts[0])
      and torch.equal(windows[13, 3], drafts[1]))

print("\nRESULT:", "ALL PASS" if not failures else f"FAILURES: {failures}")
sys.exit(0 if not failures else 1)
