#!/usr/bin/env python3
"""Fold a MiniMax-H3 LoRA into the released BF16 DiT shards (SRC/transformer), in place: W += scale * B @ A.

h3 has no LoRA runtime; a folded checkpoint is then packed as usual (h3_pack_int8 + tools/align_int8_pack.py).
Only the byte ranges of the touched tensors are rewritten, so headers, order and alignment stay as they were.

Adapters come in two key spaces:
  native     the checkpoint's own names (blocks.N.attn.qkv_proj, mlp.fc1, adaln_proj.linear, final_layer...),
             e.g. larryvrh/MiniMax-H3-Turbo-Lora (<target>.lora_A.weight / .lora_B.weight, alpha = rank)
  diffusers  transformer_blocks.N.attn.to_q/to_k/to_v/to_out.0, ff.net.0.proj, ff.net.2, norm_out.linear,
             token_refiner.refiner_blocks.N... (<target>.lora_A[.default].weight, optional <target>.alpha),
             e.g. lightx2v/Minimax-h3-Turbo
Layout differences against the released shards (see diffusers' convert_minimax_h3_to_diffusers.py):
  - The shards store the fused QKV rows interleaved per head ([head: q k v]); the reference model reorders them at
    load time to [q_all; k_all; v_all], and adapters are trained on that layout. Every QKV delta is therefore
    interleaved back before folding (--qkv-layout interleaved skips this for an adapter already in shard layout).
  - diffusers' ff.net.0.proj is the shards' mlp.fc1 with its two halves swapped ([value; gate] vs [gate; value]).
A per-tensor probe checks the folded rows against W x + scale * B (A x) before any byte is written.

Rounding: a LoRA delta is ~0.01-0.05 % of W, mostly below half a BF16 ulp, and W is already on the BF16 grid, so
round-to-nearest gives W back for most weights: measured, only 10-55 % of the delta survives in blocks 0-25
(regression of the folded output on B(A x)). --rounding stochastic (the default) rounds W + delta up or down with
probability proportional to the distance, which keeps the delta in expectation (measured 1.00 +- 0.01, also after
INT8 packing); the added noise is 0.2-0.35x the INT8 rounding noise of the packed DiT and ~0.5x the BF16 rounding of
the adaLN table. It is seeded per tensor, so a fold is reproducible.
Folding is one-way: fold into a fresh copy of the released shards (subtracting the delta again leaves 0.03-0.2 %
of the values one ulp off, measured).

Usage: tools/fold_lora.py SRC/transformer LORA.safetensors [--scale 1.0] [--qkv-layout contiguous|interleaved]
       [--rounding stochastic|nearest] [--dry-run]   (--dry-run lists the mapping, writes nothing)
"""
import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np


def read_header(path):
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    meta = header.pop("__metadata__", None) or {}
    return header, 8 + length, meta


def load(path, spec, base):
    a, b = spec["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        raw = f.read(b - a)
    if spec["dtype"] == "BF16":
        return (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32).reshape(spec["shape"])
    if spec["dtype"] in ("F32", "F16"):
        return np.frombuffer(raw, "<f4" if spec["dtype"] == "F32" else "<f2").astype(np.float32).reshape(spec["shape"])
    sys.exit(f"unsupported dtype {spec['dtype']}")


def to_bf16(x, seed=None):
    """BF16 bytes of x: round to nearest even, or stochastically with a seeded generator."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    if seed is None:
        return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2").tobytes()
    noise = np.random.default_rng(seed).integers(0, 1 << 16, size=u.shape, dtype=np.uint32)
    return ((u + noise) >> 16).astype("<u2").tobytes()


DIFFUSERS = [  # (pattern, native replacement)
    (r"^transformer_blocks\.(\d+)\.", r"blocks.\1."),
    (r"^token_refiner\.refiner_blocks\.(\d+)\.", r"token_refiner.blocks.\1."),
    (r"^norm_out\.linear$", "final_layer.adaln_proj.linear"),
    (r"\.attn\.to_out\.0$", ".attn.out_proj"),
    (r"\.ff\.net\.0\.proj$", ".mlp.fc1"),
    (r"\.ff\.net\.2$", ".mlp.fc2"),
]


def adapter_modules(header, meta):
    """{module name: (A key, B key, alpha or None)} in the adapter's own key space."""
    mods = {}
    for key in header:
        m = re.match(r"^(.*)\.lora_A(?:\.default)?\.weight$", key)
        if not m:
            continue
        name = m.group(1)
        b = next((k for k in (f"{name}.lora_B.weight", f"{name}.lora_B.default.weight") if k in header), None)
        if b is None:
            sys.exit(f"{key}: no lora_B")
        mods[name] = (key, b, f"{name}.alpha" if f"{name}.alpha" in header else None)
    return mods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("lora", type=Path)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--qkv-layout", choices=("contiguous", "interleaved"), default="contiguous",
                    help="row layout of the adapter's QKV deltas (reference / diffusers models: contiguous)")
    ap.add_argument("--rounding", choices=("stochastic", "nearest"), default="stochastic")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = json.loads((args.checkpoint / "config.json").read_text())
    heads, hd = cfg["num_attention_heads"], cfg["attention_head_dim"]

    lhead, lbase, lmeta = read_header(args.lora)
    mods = adapter_modules(lhead, lmeta)
    diffusers = any(n.startswith("transformer_blocks.") for n in mods)
    meta_alpha = float(lmeta["alpha"]) if "alpha" in lmeta else None
    print(f"{args.lora.name}: {len(mods)} modules, {'diffusers' if diffusers else 'native'} keys, metadata {lmeta}")

    # target tensor -> list of (row slice in the target's adapter layout, A key, B key, alpha, fc1 swap)
    plan = {}
    for name, (a_key, b_key, alpha_key) in mods.items():
        rank = lhead[a_key]["shape"][0]
        if alpha_key:
            alpha = float(load(args.lora, lhead[alpha_key], lbase).reshape(-1)[0])
        else:
            alpha = meta_alpha if meta_alpha is not None else rank
        native, part = name, None
        if diffusers:
            m = re.match(r"^(.*)\.attn\.to_([qkv])$", name)
            if m:
                native, part = m.group(1) + ".attn.qkv_proj", "qkv".index(m.group(2))
            for pattern, repl in DIFFUSERS:
                native = re.sub(pattern, repl, native)
        plan.setdefault(native + ".weight", []).append(
            (part, a_key, b_key, alpha / rank, diffusers and native.endswith(".mlp.fc1")))

    shards = sorted(args.checkpoint.glob("*.safetensors"))
    where = {}
    for shard in shards:
        header, base, _ = read_header(shard)
        for key in header:
            if key in plan:
                where[key] = (shard, header[key], base)
    missing = sorted(set(plan) - set(where))
    if missing:
        sys.exit(f"{len(missing)} adapter targets not in the checkpoint, e.g. {missing[:5]}")
    print(f"{len(plan)} target tensors in {len({w[0] for w in where.values()})} shards; qkv layout {args.qkv_layout}")
    rng = np.random.default_rng(0)
    for n, key in enumerate(sorted(plan, key=lambda k: (str(where[k][0]), where[k][1]["data_offsets"][0]))):
        shard, spec, base = where[key]
        rows, cols = spec["shape"]
        qkv = key.endswith(".attn.qkv_proj.weight")
        delta = np.zeros((rows, cols), np.float32)
        for part, a_key, b_key, scale, swap in plan[key]:
            A = load(args.lora, lhead[a_key], lbase)
            B = load(args.lora, lhead[b_key], lbase)
            d = (B @ A) * (scale * args.scale)
            if swap:                                   # diffusers [value; gate] -> shards [gate; value]
                d = np.concatenate([d[rows // 2:], d[: rows // 2]])
            if part is not None:                       # one of to_q / to_k / to_v: its third of [q; k; v]
                delta[part * rows // 3:(part + 1) * rows // 3] += d
            else:
                delta += d
        if qkv and args.qkv_layout == "contiguous":     # [q_all; k_all; v_all] -> [head: q k v]
            delta = delta.reshape(3, heads, hd, cols).transpose(1, 0, 2, 3).reshape(rows, cols)
        if spec["dtype"] != "BF16" or delta.shape != (rows, cols):
            sys.exit(f"{key}: {spec['dtype']} {spec['shape']} vs delta {delta.shape}")
        if args.dry_run:
            print(f"  {key} {spec['shape']} <- {[p[1] for p in plan[key]]} scale {plan[key][0][3]:.3g}")
            continue
        W = load(shard, spec, base)
        folded = W + delta
        x = rng.standard_normal(cols).astype(np.float32)
        ref = W @ x + delta @ x
        rel = float(np.max(np.abs(folded @ x - ref)) / (np.max(np.abs(ref)) + 1e-9))
        if rel > 1e-4:
            sys.exit(f"{key}: probe failed (rel {rel:.2e})")
        a, b = spec["data_offsets"]
        data = to_bf16(folded, n if args.rounding == "stochastic" else None)
        if len(data) != b - a:
            sys.exit(f"{key}: byte count")
        with open(shard, "r+b") as f:
            f.seek(base + a)
            f.write(data)
        rms = float(np.sqrt(np.mean(delta * delta)) / (np.sqrt(np.mean(W * W)) + 1e-12))
        print(f"  [{n + 1}/{len(plan)}] {key}: |dW|/|W| {rms:.4f}", flush=True)
    print("dry run" if args.dry_run else f"folded ({args.rounding} rounding)")


if __name__ == "__main__":
    main()
