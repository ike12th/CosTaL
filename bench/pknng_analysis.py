"""What does the original `p_knng_Linux` actually compute, and does it matter? (docs/04 §4)

Works on the `.clu` files produced by the `golden` workflow (`bench/golden_pknng.py`):

    gh run download <golden run id> -n golden-pknng -D /tmp/golden
    python bench/pknng_analysis.py /tmp/golden/golden

1. **Semantics.** Compares p_knng's reported similarity values with candidate definitions
   (true cosine; cosine after clipping negatives to 0; dropping negatives after normalization),
   and regresses the discrepancy on the per-feature products a_t·b_t to see whether p_knng
   ignores or reweights particular features.
2. **Recall.** Fraction of p_knng's neighbors that are in the exact cosine top-k (CosTaL+).
3. **Clustering impact.** Builds the graph exactly as the published Linux path does
   (p_knng's neighbors and values -> original `tani_from_cos` -> original `build_graph`),
   runs Leiden with several seeds next to the exact graph, and compares both with each other
   and with the gating labels.

Output: `bench/out/pknng_analysis.json`. Uses deduplicated Levine (docs/02 Addendum).
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stage_a import (  # noqa: E402
    SEEDS,
    label_scores,
    leiden,
    load_levine,
    original_clustering,
    ours_graph,
)

import CosTaL as cp  # noqa: E402

OUT = HERE / "out" / "pknng_analysis.json"


def parse_clu(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a .clu file; indices are 1-based (verified by the golden workflow)."""
    lines = [line.split() for line in path.read_text().splitlines()]
    meta = list(map(int, lines[0]))
    n, width = meta[0], meta[2] // meta[0]
    flat = [tok for row in lines[1:] for tok in row]
    cols = np.array(list(map(int, flat[::2])), dtype=np.int64).reshape(n, width) - 1
    vals = np.array(list(map(float, flat[1::2])), dtype=np.float64).reshape(n, width)
    return cols, vals


def semantics(x: np.ndarray, cols: np.ndarray, vals: np.ndarray) -> dict:
    n, k = cols.shape
    a, b = np.repeat(np.arange(n), k), cols.ravel()
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    pos = np.clip(x, 0, None)
    posn = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    xnp = np.clip(xn, 0, None)
    v = vals.ravel()
    candidates = {
        "cosine": np.einsum("ij,ij->i", xn[a], xn[b]),
        "clip_negatives_then_cosine": np.einsum("ij,ij->i", posn[a], posn[b]),
        "normalize_then_drop_negatives": np.einsum("ij,ij->i", xnp[a], xnp[b]),
    }
    max_abs = {name: float(np.max(np.abs(v - s))) for name, s in candidates.items()}
    prod = xn[a] * xn[b]
    diff = v - prod.sum(1)
    coef, *_ = np.linalg.lstsq(prod, diff, rcond=None)
    return {
        "max_abs_diff_vs_definition": max_abs,
        "abs_diff_vs_cosine_quantiles": dict(zip(
            ["50%", "90%", "99%", "99.9%", "max"],
            map(float, np.quantile(np.abs(diff), [0.5, 0.9, 0.99, 0.999, 1.0])),
        )),
        "fraction_above_cosine": float(np.mean(diff > 1e-6)),
        "fraction_below_cosine": float(np.mean(diff < -1e-6)),
        "per_feature_fit_max_abs_coef": float(np.max(np.abs(coef))),
        "per_feature_fit_residual_max": float(np.max(np.abs(diff - prod @ coef))),
    }


def impact(x: np.ndarray, labels: np.ndarray, cols: np.ndarray, vals: np.ndarray) -> dict:
    from sklearn.metrics import adjusted_rand_score

    clustering = original_clustering()
    n, k = cols.shape
    g, _, graph_exact, _ = ours_graph(x, k)
    exact = [set(r) for r in g.idx.tolist()]
    recall = np.array([len(set(c) & e) / k for c, e in zip(cols.tolist(), exact)])

    norms = np.linalg.norm(x, axis=1)
    rows = list(itertools.chain.from_iterable(itertools.repeat(i, k) for i in range(n)))
    columns = cols.ravel().tolist()
    weights = [clustering.tani_from_cos(norms[i], norms[j], v)
               for i, j, v in zip(rows, columns, vals.ravel())]
    graph_pknng = clustering.build_graph(rows, columns, weights, (n, n))

    m_p = [leiden(graph_pknng, s)[0] for s in SEEDS]
    m_e = [leiden(graph_exact, s)[0] for s in SEEDS]

    def mean_ari(pairs):
        return float(np.mean([adjusted_rand_score(p, q) for p, q in pairs]))

    def summary(ms):
        scores = [label_scores(labels, m) for m in ms]
        return {key: {"mean": float(np.mean([s[key] for s in scores])),
                      "sd": float(np.std([s[key] for s in scores]))}
                for key in ("ari", "nmi", "clusters")}

    return {
        "recall_at_k": {"mean": float(recall.mean()), "min": float(recall.min()),
                        "rows_with_any_difference": float((recall < 1).mean())},
        "edges": {"pknng": graph_pknng.ecount(), "exact": graph_exact.ecount()},
        "partition_ari_pknng_vs_exact_same_seed": mean_ari(zip(m_p, m_e)),
        "partition_ari_seed_to_seed": {"pknng": mean_ari(itertools.combinations(m_p, 2)),
                                       "exact": mean_ari(itertools.combinations(m_e, 2))},
        "labels": {"pknng": summary(m_p), "exact": summary(m_e)},
    }


def main(clu_dir: Path) -> None:
    result = {"seeds": list(SEEDS), "costal_plus": cp.__version__}
    for n in (20_000, 50_000):
        path = clu_dir / f"levine_n{n}_dedup.clu"
        if not path.exists():
            continue
        x, labels = load_levine(n, dedup=True)
        cols, vals = parse_clu(path)
        entry = {"semantics": semantics(x, cols, vals)}
        if n == 50_000:
            entry["impact"] = impact(x, labels, cols, vals)
        result[f"n{n}_dedup"] = entry
        print(json.dumps({f"n{n}_dedup": entry}, indent=2), flush=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
