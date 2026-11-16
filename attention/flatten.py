# Flatten MLX steel/NAX Metal headers into one self-contained source (runtime newLibraryWithSource cannot resolve
# quoted includes). Usage: flatten.py ENTRY.metal OUT.metal ; quoted "mlx/..." includes resolve under MLX's include dir.
import re, sys, pathlib
INC = pathlib.Path.home() / "llm-serving/h3-image-mlx/.venv/lib/python3.15/site-packages/mlx/include"
seen, out = set(), []
def emit(path):
    path = path.resolve()
    if path in seen: return
    seen.add(path)
    for line in path.read_text().splitlines():
        m = re.match(r'\s*#\s*include\s+"([^"]+)"', line)
        if m:
            name = m.group(1)
            cand = [path.parent / name, INC / name]
            target = next((c for c in cand if c.exists()), None)
            if target is None: sys.exit(f"cannot resolve {name} from {path}")
            out.append(f"// ---- begin {name}")
            emit(target)
            out.append(f"// ---- end {name}")
        elif re.match(r'\s*#\s*pragma\s+once', line):
            continue
        else:
            out.append(line)
emit(pathlib.Path(sys.argv[1]))
pathlib.Path(sys.argv[2]).write_text("\n".join(out) + "\n")
print(f"{sys.argv[2]}: {len(out)} lines from {len(seen)} files")
