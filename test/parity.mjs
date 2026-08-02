// Compare the JS engine against golden tensors from the reference PyTorch
// implementation (see tools/make_reference.py).
//
//   node test/parity.mjs <reference.json> <inputs.json> <weights dir>

import { readFileSync } from "node:fs";

import { AR, Model } from "../mpnn/model.js";
import { Weights } from "../mpnn/weights.js";
import { enableAcceleration } from "../mpnn/accel.js";

const [, , refPath, inputsPath, weightsDir] = process.argv;
const wasmPath = new URL("../wasm/kernels.wasm", import.meta.url);
const simd = process.env.MPNN_NO_SIMD
  ? null
  : await enableAcceleration(readFileSync(wasmPath).buffer);
console.log(`kernel: ${simd ? "wasm simd" : "javascript"}`);

const reference = JSON.parse(readFileSync(refPath, "utf8"));
const cases = JSON.parse(readFileSync(inputsPath, "utf8"));

function stats(a, b) {
  let maxAbs = 0;
  let sumSq = 0;
  let refSq = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > maxAbs) maxAbs = d;
    sumSq += d * d;
    refSq += b[i] * b[i];
  }
  return { maxAbs, rel: Math.sqrt(sumSq / Math.max(refSq, 1e-30)) };
}

/** Fraction of positions where the two logit rows pick the same argmax. */
function argmaxAgreement(a, b, rows, cols) {
  let same = 0;
  for (let i = 0; i < rows; i++) {
    let ai = 0;
    let bi = 0;
    for (let v = 1; v < cols; v++) {
      if (a[i * cols + v] > a[i * cols + ai]) ai = v;
      if (b[i * cols + v] > b[i * cols + bi]) bi = v;
    }
    if (ai === bi) same++;
  }
  return same / rows;
}

const TOL = { maxAbs: 2e-3, rel: 2e-4 };
let failures = 0;

function check(label, got, want, tol = TOL) {
  const { maxAbs, rel } = stats(got, want);
  const ok = maxAbs <= tol.maxAbs && rel <= tol.rel;
  if (!ok) failures++;
  console.log(
    `    ${ok ? "PASS" : "FAIL"}  ${label.padEnd(28)} maxAbs=${maxAbs.toExponential(2)} `
    + `rel=${rel.toExponential(2)}`,
  );
  return ok;
}

for (const ref of reference) {
  const testCase = cases.find((c) => c.name === ref.name);
  console.log(`\n${ref.name}  (${ref.modelType}, ${ref.checkpoint}, L=${ref.L}, K=${ref.K})`);

  const buffer = readFileSync(`${weightsDir}/${ref.checkpoint}.mpnn`);
  const weights = Weights.fromArrayBuffer(
    buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength),
  );
  const model = new Model(weights);

  const raw = testCase.inputs;
  const inputs = {
    X: Float32Array.from(raw.X),
    mask: Float32Array.from(raw.mask),
    residueIdx: Int32Array.from(raw.residueIdx),
    chainLabels: Int32Array.from(raw.chainLabels),
    ligandXyz: Float32Array.from(raw.ligandXyz),
    ligandType: Int32Array.from(raw.ligandType),
    ligandMask: Float32Array.from(raw.ligandMask),
    membraneLabels: Int32Array.from(raw.membraneLabels ?? new Array(raw.L).fill(0)),
  };

  const t0 = performance.now();
  const enc = model.encode(inputs);
  const encodeMs = performance.now() - t0;

  // Neighbour graph must match exactly -- everything downstream depends on it.
  const refEIdx = ref.EIdx.flat();
  let graphMismatch = 0;
  for (let i = 0; i < refEIdx.length; i++) if (enc.EIdx[i] !== refEIdx[i]) graphMismatch++;
  console.log(
    `    ${graphMismatch === 0 ? "PASS" : "FAIL"}  ${"neighbour graph".padEnd(28)} `
    + `${refEIdx.length - graphMismatch}/${refEIdx.length} edges identical`,
  );
  if (graphMismatch !== 0) failures++;

  check("encoder h_V", enc.hV, Float32Array.from(ref.hV.flat()));

  const S = Int32Array.from(raw.S);
  const order = Int32Array.from(testCase.order);
  const modes = [
    ["logits (backbone only)", { type: AR.NONE }, ref.logits.none],
    ["logits (autoregressive)", { type: AR.ORDER, order }, ref.logits.order],
    ["logits (all-but-self)", { type: AR.ALL_BUT_SELF }, ref.logits.all_but_self],
  ];
  for (const [label, ar, want] of modes) {
    const got = model.score(enc, S, ar);
    const flat = Float32Array.from(want.flat());
    check(label, got, flat);
    const agree = argmaxAgreement(got, flat, ref.L, 21);
    if (agree < 1) {
      console.log(`          argmax agreement ${(agree * 100).toFixed(1)}%`);
    }
  }

  // The autoregressive sampler and the teacher-forced scorer must agree: decode
  // a sequence, then score that same sequence under the same order.
  const rng = mulberry32(7);
  const drawn = model.sample(enc, { batch: 2, temperature: 0.3, rng });
  const rescored = model.score(enc, drawn.S[0], { type: AR.ORDER, order: drawn.order[0] });
  check("sampler == scorer", drawn.logits[0], rescored, { maxAbs: 5e-4, rel: 5e-5 });
  console.log(`    seq  ${drawn.seq[0].slice(0, 60)}`);
  console.log(
    `    encode ${encodeMs.toFixed(0)} ms   `
    + `sample(batch 2) ${drawn.score.map((s) => s.toFixed(3)).join(", ")} nll`,
  );
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

console.log(failures === 0 ? "\nall checks passed" : `\n${failures} check(s) failed`);
process.exit(failures === 0 ? 0 : 1);
