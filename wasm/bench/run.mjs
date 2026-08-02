// Does the kernel's language matter? Build the same matmul five ways and time
// them. See wasm/bench/README.md for the answer.
//
//   ./wasm/bench/build.sh && node wasm/bench/run.mjs

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const HERE = fileURLToPath(new URL(".", import.meta.url));

async function load(path) {
  const buf = readFileSync(path);
  const { instance } = await WebAssembly.instantiate(
    buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), {},
  );
  return instance.exports;
}

const c = await load(new URL("../kernels.wasm", import.meta.url));
const rust = await load(`${HERE}target/wasm32-unknown-unknown/release/rustkern.wasm`);
const plain = await load(`${HERE}plain.wasm`);
const fast = await load(`${HERE}plainfast.wasm`);

/** Each module has its own memory; lay inputs out after a fixed base. */
function call(ex, fn, x, w, b, n, cin, cout) {
  const base = 1 << 16;
  const need = base + (n * cin + cout * cin + cout + n * cout) * 4 + 64;
  if (need > ex.memory.buffer.byteLength) {
    ex.memory.grow(Math.ceil((need - ex.memory.buffer.byteLength) / 65536));
  }
  let p = base;
  const xP = p; p += n * cin * 4;
  const wP = p; p += cout * cin * 4;
  const bP = p; p += cout * 4;
  const oP = p;
  const mem = new Float32Array(ex.memory.buffer);
  mem.set(x, xP >> 2);
  mem.set(w, wP >> 2);
  mem.set(b, bP >> 2);
  fn(xP, wP, bP, n, cin, cout, oP);
  return new Float32Array(ex.memory.buffer).slice(oP >> 2, (oP >> 2) + n * cout);
}

const IMPLS = [
  ["C  ptr+SIMD", c, c.linear_f32],
  ["Rust ptr+SIMD", rust, rust.linear_ptr],
  ["Rust safe", rust, rust.linear_safe],
  ["Rust safe blocked", rust, rust.linear_safe_blocked],
  ["C plain loop", plain, plain.linear_plain],
  ["C plain -ffast-math", fast, fast.linear_plain],
];

// The shapes the engine actually issues.
const SHAPES = [[3648, 384, 128], [3648, 128, 128], [3648, 512, 128],
  [3025, 148, 128], [608, 128, 512]];

const head = "shape".padEnd(21) + IMPLS.map(([n]) => n.padStart(12)).join("");
console.log(head);
console.log("-".repeat(head.length));

for (const [n, cin, cout] of SHAPES) {
  const x = new Float32Array(n * cin).map(() => Math.random() - 0.5);
  const w = new Float32Array(cout * cin).map(() => Math.random() - 0.5);
  const b = new Float32Array(cout).map(() => Math.random());
  const flops = 2 * n * cin * cout;
  let reference = null;
  const cells = [];
  for (const [, ex, fn] of IMPLS) {
    const got = call(ex, fn, x, w, b, n, cin, cout);
    reference ??= got;
    let maxAbs = 0;
    for (let i = 0; i < got.length; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(reference[i] - got[i]));
    }
    const R = 20;
    call(ex, fn, x, w, b, n, cin, cout);
    const t0 = performance.now();
    for (let q = 0; q < R; q++) call(ex, fn, x, w, b, n, cin, cout);
    const ms = (performance.now() - t0) / R;
    cells.push(`${(flops / ms / 1e6).toFixed(1)}${maxAbs > 1e-3 ? "!" : ""}`.padStart(12));
  }
  console.log(`[${n},${cin}]x[${cin},${cout}]`.padEnd(21) + cells.join(""));
}
console.log("\nGFLOP/s. For reference the blocked JS kernel in mpnn/ops.js does ~2.4.");
