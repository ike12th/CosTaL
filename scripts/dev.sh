#!/usr/bin/env bash
# 本地开发循环：编译 + 装进当前 python 环境 + 跑测试。
# 用法：scripts/dev.sh [python 解释器]（默认用 PATH 上的 python3）
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"

PY="${1:-python3}"
echo "==> 解释器: $("$PY" -c 'import sys,platform;print(sys.executable, platform.machine(), sys.version.split()[0])')"

cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

"$PY" -m pip install -e . --no-build-isolation
"$PY" -m pytest -q
"$PY" -m costal_plus.cli info
