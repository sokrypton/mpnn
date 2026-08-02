#!/bin/sh
set -e
cd "$(dirname "$0")"
rustup target add wasm32-unknown-unknown >/dev/null 2>&1 || true
RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown
for extra in "" "-ffast-math"; do
  out=$([ -z "$extra" ] && echo plain.wasm || echo plainfast.wasm)
  clang --target=wasm32 -O3 -msimd128 $extra -nostdlib -ffreestanding \
    -fvisibility=hidden -Wl,--no-entry -Wl,--export-dynamic -o "$out" plain.c
done
