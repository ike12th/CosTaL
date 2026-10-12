#!/usr/bin/env bash
# Build a wheel locally. No argument = current platform; a target triple = cross-compile (needs zig).
#   scripts/build-wheel.sh
#   scripts/build-wheel.sh x86_64-unknown-linux-gnu --zig
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"

TARGET="${1:-}"
shift || true

if [ -n "$TARGET" ]; then
  rustup target add "$TARGET"
  maturin build --release --target "$TARGET" --out dist "$@"
else
  maturin build --release --out dist "$@"
fi
ls -la dist/
