#!/usr/bin/env python3
"""Stamp the rotation of a Hadamard-rotated INT8 DiT pack into every weight file (h3_rot_check_pack).

h3_pack_int8 --rot writes the rotation in two places: the marker h3-int8-pack.json ("format": "h3-int8-pack-v2",
"rot": LIST) and the __metadata__ of every .safetensors file ("h3": "int8-pack-v2", "h3_rot": LIST); h3 refuses a
pack where they disagree. Packs written before that (2026-11-21, e.g. transformer-int8-rot256) carry the rotation
only in the marker and "h3": "int8-pack-v1" in their files. This tool rewrites just the __metadata__ object of each
file in place: the header keeps its length (the new text replaces the old one inside the space padding that
tools/align_int8_pack.py leaves), so no tensor byte and no data offset moves. Every tensor entry of the header is
compared before and after; a file that already carries the marker's identity is left alone.

Usage: tools/stamp_int8_pack_rot.py DIR [--backup-dir BACKUP]   (BACKUP: the original headers are saved there first)
"""

import json
import os
import struct
import sys

ORDER = ["qkv", "fc1", "out256", "out128", "fc2"]


def canonical(rot):
    names = rot.split(",")
    if not names or any(n not in ORDER for n in names) or len(set(names)) != len(names):
        sys.exit(f"invalid rotation list {rot!r}")
    return ",".join(n for n in ORDER if n in names)


def main():
    args = sys.argv[1:]
    backup = None
    if "--backup-dir" in args:
        i = args.index("--backup-dir")
        backup = args[i + 1]
        del args[i:i + 2]
    if len(args) != 1:
        sys.exit(__doc__)
    directory = args[0]
    with open(os.path.join(directory, "h3-int8-pack.json")) as f:
        marker = json.load(f)
    if marker.get("format") != "h3-int8-pack-v2" or not marker.get("rot"):
        sys.exit("the marker is not a rotated (h3-int8-pack-v2) pack; nothing to stamp")
    rot = canonical(marker["rot"])
    wanted = {"format": "pt", "h3": "int8-pack-v2", "h3_rot": rot}
    files = sorted(n for n in os.listdir(directory) if n.endswith(".safetensors") and not n.startswith("."))
    if not files:
        sys.exit("no safetensors files")
    if backup:
        os.makedirs(backup, exist_ok=True)
    changed = 0
    for name in files:
        path = os.path.join(directory, name)
        with open(path, "rb") as f:
            prefix = f.read(8)
            (length,) = struct.unpack("<Q", prefix)
            raw = f.read(length)
        header = json.loads(raw)
        meta = header.get("__metadata__") or {}
        if meta == wanted:
            continue
        if meta.get("h3") != "int8-pack-v1" or "h3_rot" in meta:
            sys.exit(f"{name}: unexpected metadata {meta}")
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        text = json.dumps({"__metadata__": wanted, **tensors}, separators=(",", ":"))
        new = text.encode()
        if len(new) > length:
            sys.exit(f"{name}: no room in the header padding ({len(new)} > {length})")
        new += b" " * (length - len(new))
        check = json.loads(new)
        if check.get("__metadata__") != wanted or {k: v for k, v in check.items() if k != "__metadata__"} != tensors:
            sys.exit(f"{name}: header round trip failed")
        if backup:
            with open(os.path.join(backup, name + ".header"), "wb") as f:
                f.write(prefix + raw)
        fd = os.open(path, os.O_WRONLY)
        try:
            if os.pwrite(fd, new, 8) != len(new):
                sys.exit(f"{name}: short write")
            os.fsync(fd)
        finally:
            os.close(fd)
        with open(path, "rb") as f:
            f.seek(8)
            if f.read(length) != new:
                sys.exit(f"{name}: verification read differs")
        changed += 1
    print(f"{directory}: {changed} of {len(files)} files stamped h3=int8-pack-v2 h3_rot={rot}")


if __name__ == "__main__":
    main()
