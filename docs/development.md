# Developing CosTaL

## Layout

```
crates/costal-core/   Rust: exact kNN kernel (knn.rs), matmul backends, cache detection,
                      edge weights (weight.rs), fuzzy-union symmetrization (graph.rs)
crates/costal-py/     PyO3 bindings → the costal._costal extension module
python/costal/        the `costal` package: _api.py (knn … find_communities), _preprocess.py, _fcs.py,
                      _multiscale.py, cli.py; 1.0's interface in _legacy.py, clustering.py, descriptions.py
python/CosTaL.py      `import CosTaL`, 1.0's import name: a one-file alias of `costal` (pyproject `include`)
tests/                pytest suite (also run against built wheels)
examples/             runnable examples, executed by the test suite
bench/                benchmarks and experiments behind the design documents (bench/original.py
                      loads the 0.0.2.4 source under its own name, next to 2.0)
docs/                 user guide, design documents 01–06, this file, release notes
```

## Build and test

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y   # Rust (stable; see rust-toolchain.toml)
python -m venv .venv && . .venv/bin/activate          # use a native interpreter (arm64 on Apple silicon)
pip install maturin pytest
pip install -e . --no-build-isolation        # also installs every runtime dependency
scripts/dev.sh                                         # fmt + clippy + cargo test + build + pytest
```

or step by step:

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
maturin develop --release
pytest -q
```

Wheels: `scripts/build-wheel.sh` (current platform) or `scripts/build-wheel.sh x86_64-unknown-linux-gnu --zig`.

## Invariants the tests guard

- **Exactness**: `knn` equals the brute force bit for bit on random, tie-heavy, signed and
  duplicate-row data, for every tile size, backend, layout and thread count (`tests/test_knn.py`,
  `crates/costal-core/src/knn.rs` tests).
- **Parity with the original**: edge weights and the fuzzy union match the original scipy code bit for
  bit; `find_communities` matches leidenalg on the same graph (`tests/test_knn.py`).
- **Defaults are the published method**: `compat="paper"` rejects every departure.

Never weaken these to make a change pass; the design documents explain each one (docs/04 §4,
docs/06 §1).

## Continuous integration (GitHub Actions)

| Workflow | When | What |
|---|---|---|
| `CI` | every push to main (not docs-only) | fmt, clippy, `cargo test`, pytest on Linux, Python 3.9 and 3.13 |
| `CI` with `all_os` | manual dispatch, `v*` tags | the same on macOS and Windows too |
| `wheels` | `v*` tags, manual dispatch | wheels for 8 platforms + sdist, each native wheel installed and tested; publishes to PyPI when a GitHub release is published |
| `bench` | manual | `bench/stage_a.py` on x86-64 Linux |
| `golden` | manual | 1.0's `p_knng_Linux` binary against 2.0 on Levine |

The repository's Actions minutes are limited (macOS counts 10x, Windows 2x): run the all-OS and
manual workflows when a change needs them.

## Benchmarks and experiments

Everything quoted in the design documents comes from a script in `bench/` writing JSON to
`bench/out/` (tracked). The datasets are public; `bench/datasets.py` lists their sources and reads
them from `$COSTAL_DATA` (each dataset in its own directory). Levine_32 lives in `bench/data/`
(`bench/fetch_reference.py` fetches it and the original CosTaL source).

| Script | Question |
|---|---|
| `datasets.py check` | does our preprocessing reproduce the paper's Table 2? |
| `stage_a.py` | kNN speed and exactness; parity with the original on Levine_32 |
| `small_datasets.py` | accuracy, parity, the select × weight 2×2 and the scRNA-seq presets |
| `timing_small.py` | end-to-end speed at identical results, all datasets |
| `magnitude_confirm.py` | does magnitude (Tanimoto) help? 10-seed paired check |
| `rare_populations.py`, `multiscale.py`, `two_level.py`, `local_scaling.py` | the resolution-limit work of docs/05 |
| `golden_pknng.py`, `pknng_analysis.py` | how the original `p_knng` binary deviates |

Performance numbers count only from a native interpreter on an otherwise idle machine.

## Releasing

See [release/checklist.md](release/checklist.md).
