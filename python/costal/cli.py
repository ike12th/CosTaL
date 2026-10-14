"""CosTaL command line: cluster a file without writing Python (docs/01 §3).

    costal cluster cells.fcs --platform mass -o labels.csv
    costal cluster counts.h5ad --platform scrnaseq --multiscale
    costal info

Inputs: CSV / TSV (cells × features, header row; non-numeric columns are dropped), FCS, ``.npy``,
and ``.h5ad`` (its ``.X``). FCS is read by ``costal.read_fcs``; pandas and anndata are installed with it.
The output has one row per input cell: its index and its cluster (``-1`` for a cell that scRNA-seq
preprocessing dropped).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__, core_version, default_matmul, hello, l2_bytes_per_thread, tile_budget_bytes


def _read(path: Path, columns: list[str] | None, exclude: list[str]):
    """The matrix (cells × features) and the feature names."""
    suffix = "".join(path.suffixes[-2:]).lower() if path.suffix.lower() == ".gz" else path.suffix.lower()
    if suffix == ".npy":
        x = np.load(path, allow_pickle=False)
        names = [str(i) for i in range(x.shape[1])]
    elif suffix == ".h5ad":
        try:
            import anndata
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise SystemExit("reading .h5ad needs anndata: pip install anndata") from exc
        ad = anndata.read_h5ad(path)
        x, names = ad.X, list(map(str, ad.var_names))
    elif suffix == ".fcs":
        from ._fcs import read_fcs

        _, df = read_fcs(path)
        x, names = df, list(df.columns)
    elif suffix in (".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz", ".txt.gz"):
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise SystemExit("reading text tables needs pandas: pip install pandas") from exc

        sep = "," if ".csv" in suffix else "\t"
        df = pd.read_csv(path, sep=sep)
        df = df.select_dtypes([np.number])
        x, names = df, list(df.columns)
    else:
        raise SystemExit(f"unsupported input {path.name}: use .csv, .tsv, .txt, .fcs, .npy or .h5ad")

    if columns or exclude:
        keep = [n for n in (columns or names) if n not in set(exclude)]
        missing = [n for n in keep if n not in names]
        if missing:
            raise SystemExit(f"columns not in the input: {missing}")
        idx = [names.index(n) for n in keep]
        x = x.iloc[:, idx] if hasattr(x, "iloc") else x[:, idx]
        names = keep
    if hasattr(x, "to_numpy"):
        x = x.to_numpy(dtype=np.float64)
    return x, names


def _cluster(args: argparse.Namespace) -> int:
    import costal as cp

    path = Path(args.input)
    t0 = time.perf_counter()
    x, names = _read(path, args.columns.split(",") if args.columns else None,
                     args.exclude.split(",") if args.exclude else [])
    n = x.shape[0]
    print(f"read {n} cells x {len(names)} features from {path.name} ({time.perf_counter() - t0:.1f} s)",
          file=sys.stderr)

    cells = np.arange(n)
    if args.platform != "none":
        x, cells = cp.preprocess(x, args.platform, preset=args.preset, return_cells=True, compat=args.compat)
    elif hasattr(x, "toarray"):
        x = x.toarray()

    t0 = time.perf_counter()
    common = dict(k=args.k, select=args.select, weight=args.weight, compat=args.compat, n_threads=args.threads)
    if args.multiscale:
        labels, info = cp.find_communities_multiscale(x, phi_max=args.phi_max, leiden=args.leiden,
                                                      seeds=args.seeds, return_info=True, **common)
        detail = f"multiscale, phi_max {args.phi_max}, {len(info['carved'])} clusters carved"
    else:
        labels = cp.find_communities(x, resolution=args.resolution, leiden=args.leiden, seed=args.seed, **common)
        detail = f"resolution {args.resolution}, leiden={args.leiden}"
    elapsed = time.perf_counter() - t0

    out = np.full(n, -1, dtype=np.int64)
    out[cells] = labels
    output = Path(args.output) if args.output else path.with_name(path.name.split(".")[0] + "_clusters.csv")
    if output.suffix.lower() == ".npy":
        np.save(output, out)
    else:
        with open(output, "w", encoding="ascii") as f:
            f.write("cell,cluster\n")
            f.writelines(f"{i},{c}\n" for i, c in enumerate(out))
    summary = {"cells": n, "clustered": int(len(cells)), "clusters": int(labels.max() + 1),
               "seconds": round(elapsed, 2), "output": str(output)}
    print(f"{summary['clusters']} clusters ({detail}) in {elapsed:.1f} s -> {output}", file=sys.stderr)
    if args.json:
        print(json.dumps(summary))
    return 0


def _info() -> int:
    print(hello())
    print(f"wheel version : {__version__}")
    print(f"core version  : {core_version()}")
    print(f"python        : {sys.version.split()[0]}")
    print(f"platform      : {platform.platform()} / {platform.machine()}")
    l2 = l2_bytes_per_thread()
    print(f"matmul        : {default_matmul()}")
    print(f"L2 per thread : {f'{l2 // 1024} KB' if l2 else 'unknown (using fallback)'}")
    print(f"tile budget   : {tile_budget_bytes() // 1024} KB")
    # Same seed, same partition holds only within one Leiden version: report them with results
    for name in ("leidenalg", "igraph"):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        print(f"{name:<14}: {version}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="costal", description="CosTaL command line")
    parser.add_argument("--version", action="version", version=f"costal (CosTaL) {__version__}")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("info", help="print build info to confirm the extension module is installed correctly")

    c = sub.add_parser("cluster", help="cluster the cells of a file and write their labels",
                       description="Cluster the cells of a file (rows) and write one label per cell. "
                                   "With no options this is the published CosTaL.")
    c.add_argument("input", help=".csv / .tsv / .txt, .fcs, .npy or .h5ad; cells are rows")
    c.add_argument("-o", "--output", help="labels file, .csv (default: <input>_clusters.csv) or .npy")
    c.add_argument("--platform", default="none", choices=["none", "mass", "flow", "spectral flow", "scrnaseq"],
                   help="the original pp(): arcsinh with cofactor 5 / 150 / 6000, or the scRNA-seq pipeline; "
                        "'none' uses the values as they are (default)")
    c.add_argument("--preset", default="paper", choices=["paper", "normalize_total"],
                   help="scRNA-seq preprocessing preset (docs/04 §6.7)")
    c.add_argument("--columns", help="comma-separated features to use (default: every numeric column)")
    c.add_argument("--exclude", help="comma-separated features to drop, e.g. Time,Event_length")
    c.add_argument("-k", type=int, default=10, help="neighbors per cell (default 10)")
    c.add_argument("--select", default="cosine", choices=["cosine", "tanimoto"], help="neighbor selection")
    c.add_argument("--weight", default="tanimoto", choices=["tanimoto", "cosine", "jaccard"], help="edge weights")
    c.add_argument("--resolution", type=float, default=0.8, help="Leiden resolution (default 0.8)")
    c.add_argument("--leiden", choices=["paper", "fast"],
                   help="'paper': leidenalg as published (default); 'fast': igraph's native Leiden "
                        "(the default with --multiscale)")
    c.add_argument("--seed", type=int, help="Leiden seed")
    c.add_argument("--multiscale", action="store_true",
                   help="multi-resolution clustering that keeps rare populations (docs/05 §2.3)")
    c.add_argument("--phi-max", type=float, default=0.04, help="with --multiscale: the leak threshold")
    c.add_argument("--seeds", type=int, default=4, help="with --multiscale: Leiden seeds per resolution")
    c.add_argument("--compat", choices=["paper"], help="refuse anything that departs from the published method")
    c.add_argument("--threads", type=int, help="threads for the kNN (default: all cores)")
    c.add_argument("--json", action="store_true", help="also print a JSON summary to stdout")

    args = parser.parse_args(argv)
    if args.cmd in (None, "info"):
        return _info()
    if args.cmd == "cluster":
        if args.leiden is None:
            args.leiden = "fast" if args.multiscale else "paper"
        try:
            return _cluster(args)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
