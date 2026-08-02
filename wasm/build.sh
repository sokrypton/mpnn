#!/bin/sh
# Build the SIMD kernels. Requires clang with the wasm32 target and wasm-ld
# (both ship with a standard LLVM install; no Emscripten needed).
set -e
cd "$(dirname "$0")"
clang --target=wasm32 -O3 -msimd128 -nostdlib -ffreestanding \
  -fvisibility=hidden \
  -Wl,--no-entry -Wl,--export-dynamic -Wl,--export=__heap_base \
  -Wl,--initial-memory=16777216 -Wl,--max-memory=1073741824 \
  -Wl,--growable-table -Wl,--allow-undefined \
  -o kernels.wasm kernels.c
ls -l kernels.wasm
