#!/usr/bin/env python3
"""Store the video VAE decoder block matrices in BF16 (2026-11-18).

With H3_VAE_BF16 (1 or 2) h3.c rounds decoder.transformer_blocks.*.{attn.to_qkv, attn.to_out, ff.w1, ff.w2}.weight
from F32 to BF16 at load (round to nearest even). Storing that rounding saves 4.8 of the 10.4 GB and gives the same
bytes on the GPU; every other tensor is copied unchanged. The F32 path (H3_VAE_BF16 unset) widens them again.

usage: vae_bf16.py SRC.safetensors DST.safetensors
"""
import json
import re
import struct
import sys

import numpy as np

BLOCK = re.compile(r"decoder\.transformer_blocks\.\d+\.(attn\.to_qkv|attn\.to_out|ff\.w1|ff\.w2)\.weight$")


def to_bf16(values):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")


def main(src, dst):
    with open(src, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    start = 8 + length
    data = np.memmap(src, dtype=np.uint8, mode="r")
    keys = [k for k in header if k != "__metadata__"]
    keys.sort(key=lambda k: header[k]["data_offsets"][0])
    out_header, offset = {}, 0
    if "__metadata__" in header:
        out_header["__metadata__"] = dict(header["__metadata__"], h3_vae_bf16_blocks="1")
    for key in keys:
        spec = dict(header[key])
        a, b = spec["data_offsets"]
        size = (b - a) // 2 if BLOCK.search(key) and spec["dtype"] == "F32" else b - a
        if BLOCK.search(key) and spec["dtype"] == "F32":
            spec["dtype"] = "BF16"
        spec["data_offsets"] = [offset, offset + size]
        out_header[key] = spec
        offset += size
    blob = json.dumps(out_header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(dst, "wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        for key in keys:
            a, b = header[key]["data_offsets"]
            raw = data[start + a:start + b]
            if out_header[key]["dtype"] == "BF16" and header[key]["dtype"] == "F32":
                raw = to_bf16(raw.view(np.float32)).view(np.uint8)
            out.write(raw.tobytes())


if __name__ == "__main__":
    main(*sys.argv[1:3])
