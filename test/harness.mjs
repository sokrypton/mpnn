// Shared plumbing for the parity tests.
//
// Every one of them loads weights, boots the kernel, compares a float array
// against a golden one and prints PASS/FAIL. Those had drifted into three
// spellings with three different tolerances and four copies of the PRNG, which
// is exactly the kind of thing nobody notices because the shapes look alike.

import { readFileSync } from "node:fs";

import { enableAcceleration } from "../mpnn/accel.js";
import { Model } from "../mpnn/model.js";
import { Weights } from "../mpnn/weights.js";

/** Boot the WASM kernel unless MPNN_NO_SIMD is set, and say which ran. */
export async function startKernel() {
  const wasmPath = new URL("../wasm/kernels.wasm", import.meta.url);
  const simd = process.env.MPNN_NO_SIMD
    ? null
    : await enableAcceleration(readFileSync(wasmPath).buffer);
  console.log(`kernel: ${simd ? "wasm simd" : "javascript"}`);
  return Boolean(simd);
}

/**
 * Load a checkpoint, refusing float16.
 *
 * fp16's ~5e-4 relative error swamps every tolerance in these tests and reads
 * as a broken port; `weights/` is the fp16 build the page ships, so parity
 * needs `convert_weights.py --dtype float32` into a scratch directory.
 */
export function loadModel(weightsDir, name) {
  const buffer = readFileSync(`${weightsDir}/${name}.mpnn`);
  const weights = Weights.fromArrayBuffer(
    buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength),
  );
  if (weights.dtype !== "float32") {
    console.log(`\n${weightsDir} holds ${weights.dtype} weights. `
      + "Parity needs float32 -- rerun tools/convert_weights.py --dtype float32.");
    process.exit(2);
  }
  return new Model(weights);
}

/** Max absolute and relative (L2) deviation of `got` from `want`. */
export function stats(got, want) {
  let maxAbs = 0;
  let sumSq = 0;
  let refSq = 0;
  for (let i = 0; i < want.length; i++) {
    const d = Math.abs(got[i] - want[i]);
    if (d > maxAbs) maxAbs = d;
    sumSq += d * d;
    refSq += want[i] * want[i];
  }
  return { maxAbs, rel: Math.sqrt(sumSq / (refSq || 1)) };
}

/** Fraction of rows whose argmax over the first `n` columns agrees. */
export function argmaxAgreement(got, want, rows, cols, n = cols) {
  let same = 0;
  for (let i = 0; i < rows; i++) {
    let a = 0;
    let b = 0;
    for (let v = 1; v < n; v++) {
      if (got[i * cols + v] > got[i * cols + a]) a = v;
      if (want[i * cols + v] > want[i * cols + b]) b = v;
    }
    if (a === b) same++;
  }
  return same / rows;
}

/** Deterministic PRNG, so a run is reproducible across machines. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A PASS/FAIL tally shared by the parity tests, so "PASS" means one thing. */
export class Report {
  constructor(width = 28) {
    this.failures = 0;
    this.width = width;
  }

  /** Assert a boolean, with an optional detail column. */
  ok(label, passed, detail = "") {
    if (!passed) this.failures++;
    console.log(`    ${passed ? "PASS" : "FAIL"}  ${label.padEnd(this.width)} ${detail}`);
    return passed;
  }

  /** Assert two float arrays agree within `tol`. */
  close(label, got, want, tol = { maxAbs: 5e-4, rel: 5e-5 }) {
    const { maxAbs, rel } = stats(got, want);
    return this.ok(label, maxAbs <= tol.maxAbs && rel <= tol.rel,
      `maxAbs=${maxAbs.toExponential(2)} rel=${rel.toExponential(2)}`);
  }

  /** Print the footer and exit with the right status. */
  finish(passMessage, failMessage = "check(s) failed") {
    console.log(this.failures === 0
      ? `\n${passMessage}`
      : `\n${this.failures} ${failMessage}`);
    process.exit(this.failures === 0 ? 0 : 1);
  }
}
