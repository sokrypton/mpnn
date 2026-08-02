// Sampling parity: the features the logit tests do not reach.
//
// KNOWN DIVERGENCE. A tied group containing both fixed and designed positions
// is the single place this engine deliberately does not match the reference.
// The reference reassigns its running `S_t` inside the per-group loop
// (model_utils.py, the symmetry branch of `sample`), so a fixed member
// overwrites it and every designed member after that one in the list inherits
// the *fixed member's input residue* instead of the sampled one. On ubiquitin
// with group [7, 27, 47, 60] and 27 fixed, it emits position 7 = E and position
// 47 = A, where A is position 27's native residue -- the group is not tied at
// all, and which member wins depends on list order. This engine samples one
// identity per group and applies it to the designed members, which is what the
// feature promises. Everything else matches exactly.
//
// Tied positions, per-amino-acid bias, fixed residues and the ligand-context
// switch all live in the sampler, which is stochastic. Both sides are made
// deterministic the same way -- an explicit decoding order and a temperature of
// 1e-6, which collapses the draw onto the argmax -- so the sequences have to
// match exactly, not merely resemble each other.
//
//   node test/sampling.mjs <golden.json> <cases.json> <weights dir>

import { readFileSync } from "node:fs";

import { enableAcceleration } from "../mpnn/accel.js";
import { Model, groupSteps, normaliseSymmetry, sequenceToString } from "../mpnn/model.js";
import { Weights } from "../mpnn/weights.js";

const [, , goldenPath, casesPath, weightsDir] = process.argv;
const wasmPath = new URL("../wasm/kernels.wasm", import.meta.url);
const simd = process.env.MPNN_NO_SIMD
  ? null
  : await enableAcceleration(readFileSync(wasmPath).buffer);
console.log(`kernel: ${simd ? "wasm simd" : "javascript"}`);

const golden = JSON.parse(readFileSync(goldenPath, "utf8"));
const cases = JSON.parse(readFileSync(casesPath, "utf8"));

const models = new Map();
function modelFor(name) {
  if (!models.has(name)) {
    const buf = readFileSync(`${weightsDir}/${name}.mpnn`);
    models.set(name, new Model(Weights.fromArrayBuffer(
      buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
    )));
  }
  return models.get(name);
}

/** Mirrors tools/sample_reference.py: |randn| encodes the requested ranks. */
function randnForOrder(order, L) {
  const randn = new Float64Array(L);
  order.forEach((pos, rank) => { randn[pos] = rank + 1; });
  return randn;
}

let failures = 0;
function check(ok, label, detail = "") {
  if (!ok) failures++;
  console.log(`    ${ok ? "PASS" : "FAIL"}  ${label.padEnd(34)}${detail}`);
}

for (const want of golden) {
  const [caseName, variantName] = want.name.split("/");
  const testCase = cases.find((c) => c.name === caseName);
  const variant = testCase.variants.find((v) => v.name === variantName);
  const raw = testCase.inputs;
  const L = raw.L;

  console.log(`\n${want.name}  (${want.modelType}, L=${L})`);
  const model = modelFor(want.checkpoint);
  const enc = model.encode({
    X: Float32Array.from(raw.X),
    mask: Float32Array.from(raw.mask),
    residueIdx: Int32Array.from(raw.residueIdx),
    chainLabels: Int32Array.from(raw.chainLabels),
    ligandXyz: Float32Array.from(raw.ligandXyz),
    ligandType: Int32Array.from(raw.ligandType),
    ligandMask: Float32Array.from(raw.ligandMask),
    membraneLabels: Int32Array.from(raw.membraneLabels ?? new Array(L).fill(0)),
    useAtomContext: variant.useAtomContext !== false,
  });

  const symmetry = (variant.symmetry ?? []).length && variant.symmetry[0].length
    ? variant.symmetry.map((group, gi) =>
      group.map((pos, pi) => ({ pos, weight: variant.symmetryWeights[gi][pi] })))
    : undefined;

  // The reference derives its own order from `(chain_mask + 1e-4) * |randn|`,
  // which deliberately pulls fixed residues to the front, then pulls tied
  // groups forward. Check that construction against the order it reports, then
  // hand that same order back so the sequence comparison isolates the decoder.
  const groups = normaliseSymmetry(symmetry, L);
  const keys = variant.chainMask.map(
    (cm, i) => (cm + 0.0001) * Math.abs(randnForOrder(variant.order, L)[i]),
  );
  const flat = Array.from({ length: L }, (_, i) => i).sort((a, b) => keys[a] - keys[b]);
  const builtOrder = groupSteps(flat, groups).flat();
  let buildDiff = 0;
  for (let i = 0; i < L; i++) if (builtOrder[i] !== want.decodingOrder[i]) buildDiff++;
  check(buildDiff === 0, "decoding order construction",
    buildDiff === 0 ? "" : `${buildDiff}/${L} differ`);

  const got = model.sample(enc, {
    batch: 1,
    temperature: 1e-6,
    order: Int32Array.from(want.decodingOrder),
    S: Int32Array.from(raw.S),
    chainMask: Float32Array.from(variant.chainMask),
    bias: Float32Array.from(variant.bias),
    symmetry,
  });

  const wantS = want.S;
  const gotS = Array.from(got.S[0]);

  // A tied group that mixes fixed and designed members is the one place the
  // two implementations are known to disagree, deliberately. See KNOWN
  // DIVERGENCE below.
  const mixedGroups = (variant.symmetry ?? []).filter(
    (g) => g.some((p) => variant.chainMask[p] === 0)
      && g.some((p) => variant.chainMask[p] > 0),
  );

  let diff = 0;
  for (let i = 0; i < L; i++) if (wantS[i] !== gotS[i]) diff++;
  if (mixedGroups.length === 0) {
    check(diff === 0, "sequence identical",
      diff === 0 ? sequenceToString(got.S[0]).slice(0, 46) : `${diff}/${L} positions differ`);
  } else {
    // Exact equality is the wrong thing to demand here, and the reference does
    // not always visibly break either -- when the residue it leaks happens to
    // equal the sampled one, the two agree by coincidence. So this is reported,
    // not asserted; "tied positions share identity" below is the real check.
    let refConsistent = true;
    for (const g of mixedGroups) {
      const designed = g.filter((p) => variant.chainMask[p] > 0);
      for (const p of designed) if (wantS[p] !== wantS[designed[0]]) refConsistent = false;
    }
    console.log(
      `    NOTE  ${"mixed fixed/tied group".padEnd(34)}`
      + `${diff}/${L} differ from the reference; its own tie `
      + `${refConsistent ? "held by coincidence" : "came apart"} (see KNOWN DIVERGENCE)`,
    );
  }

  if (symmetry) {
    // Every designed member of every tied group must agree, including in the
    // mixed groups where the reference does not. Fixed members keep their input
    // residue, which is the point of fixing them.
    let tiedOk = true;
    for (const group of variant.symmetry) {
      const designed = group.filter((pos) => variant.chainMask[pos] > 0);
      for (const pos of designed) if (gotS[pos] !== gotS[designed[0]]) tiedOk = false;
    }
    check(tiedOk, "tied positions share identity");
  }

  const omitted = [];
  for (let v = 0; v < 20; v++) if (variant.bias[v] < -1e8) omitted.push(v);
  if (omitted.length) {
    let leaked = 0;
    for (let i = 0; i < L; i++) {
      if (variant.chainMask[i] > 0 && omitted.includes(gotS[i])) leaked++;
    }
    check(leaked === 0, "omitted amino acids absent",
      leaked === 0 ? `${omitted.length} omitted` : `${leaked} leaked`);
  }

  let fixedBroken = 0;
  for (let i = 0; i < L; i++) {
    if (variant.chainMask[i] === 0 && gotS[i] !== raw.S[i]) fixedBroken++;
  }
  check(fixedBroken === 0, "fixed residues untouched",
    fixedBroken === 0 ? "" : `${fixedBroken} changed`);
}

console.log(failures === 0 ? "\nsampling parity holds" : `\n${failures} check(s) failed`);
process.exit(failures === 0 ? 0 : 1);
