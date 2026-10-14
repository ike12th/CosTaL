"""Peak memory of each step, above the loaded input (user guide §10, "Memory").

Each (input, step) runs in a fresh interpreter: import costal, load the input from a cached .npy,
take the peak resident set size as the baseline, run the step, and report the new peak minus the
baseline. The inputs: Levine_32 (deduplicated, arcsinh, float32), Mosmann_rare (paper preset,
float32) and a dense random 100,000 × 2,000 float32 matrix. The kNN-graph steps include whatever
they import (igraph for `leiden="igraph"`).

Usage: python bench/memory.py
Output: bench/out/memory.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = HERE / "out" / "memory.json"
STEPS = {
    "knn": "cp.knn(x, 10)",
    "build_graph": "cp.build_graph(x, 10)",
    "find_communities(leiden='fast')": "cp.find_communities(x, leiden='fast', seed=0)",
    "find_communities(leiden='igraph')": "cp.find_communities(x, leiden='igraph', seed=0)",
    "find_communities_multiscale": "cp.find_communities_multiscale(x)",
}
CHILD = """
import resource, sys
import numpy as np
import costal as cp
x = np.load(sys.argv[1])
scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, KiB on Linux
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
{step}
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale - base)
"""


def inputs(cache: Path) -> dict[str, Path]:
    from datasets import load, preprocess
    from stage_a import load_levine

    paths = {}
    for name in ("Levine_32", "Mosmann_rare", "dense_100k_x_2000"):
        path = cache / f"memory_{name}.npy"
        if not path.exists():
            if name == "Levine_32":
                x = load_levine(None, dedup=True)[0]
            elif name == "Mosmann_rare":
                x = preprocess(load(name))[0]
            else:
                x = np.random.default_rng(0).random((100_000, 2_000), dtype=np.float32)
            np.save(path, np.ascontiguousarray(x, dtype=np.float32))
        paths[name] = path
    return paths


def main() -> None:
    from datasets import CACHE
    from stage_a import environment

    out = {"datasets": {}, "environment": environment()}
    for name, path in inputs(CACHE).items():
        mb = np.load(path, mmap_mode="r").nbytes / 2**20
        row = {"input_mb": mb}
        for step, code in STEPS.items():
            if name == "dense_100k_x_2000" and step == "find_communities_multiscale":
                continue
            res = subprocess.run([sys.executable, "-I", "-c", CHILD.format(step=code), str(path)],
                                 capture_output=True, text=True, check=True)
            row[step] = int(res.stdout.split()[-1]) / 2**20
            print(f"{name:<18} {step:<36} {row[step]:7.0f} MB above the input ({mb:.0f} MB)", flush=True)
        out["datasets"][name] = row
        OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
