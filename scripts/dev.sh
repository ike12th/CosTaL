#!/usr/bin/env bash
# Local dev loop: build + install into the current python environment + run the tests.
# Usage: scripts/dev.sh [python interpreter]  (defaults to python3 on PATH)
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"

PY="${1:-python3}"
echo "==> interpreter: $("$PY" -c 'import sys,platform;print(sys.executable, platform.machine(), sys.version.split()[0])')"

cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

"$PY" -m pip install -e . --no-build-isolation
"$PY" -m pytest -q
"$PY" -m CosTaL.cli info
