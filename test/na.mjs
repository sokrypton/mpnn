// NA-MPNN parity against the real PyTorch model.
//
//   node test/na.mjs <na_golden.json> <weights dir> [pdb dir]
//
// Two halves, checked separately, because NA-MPNN parses structures with ProDy
// and its own polymer-type rules:
//
//   * model maths -- feed the tensors `tools/na_reference.py` dumped straight
//     into the engine and compare the neighbour graph, encoder and logits;
//   * the parser -- read the same PDB with `structureFromText` and check it
//     reproduces those tensors.
//
// Keeping them apart means a parser disagreement cannot masquerade as a model
// bug, which is the failure mode that wasted the most time on LigandMPNN.

import { readFileSync } from "node:fs";

import { enableAcceleration } from "../mpnn/accel.js";
import { AR, Model } from "../mpnn/model.js";
import { NA_ALPHABET } from "../mpnn/na.js";
import { structureFromText } from "../mpnn/pdb.js";
import { Weights } from "../mpnn/weights.js";

const [, , goldenPath, weightsDir, pdbDir] = process.argv;
const wasmPath = new URL("../wasm/kernels.wasm", import.meta.url);
const simd = process.env.MPNN_NO_SIMD
  ? null
  : await enableAcceleration(readFileSync(wasmPath).buffer);
console.log(`kernel: ${simd ? "wasm simd" : "javascript"}`);

const golden = JSON.parse(readFileSync(goldenPath, "utf8"));
const buffer = readFileSync(`${weightsDir}/na_mpnn_design.mpnn`);
const weights = Weights.fromArrayBuffer(
  buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength),
);
if (weights.dtype !== "float32") {
  console.log(`${weightsDir} holds ${weights.dtype} weights; parity needs float32.`);
  process.exit(2);
}
const model = new Model(weights);

let failures = 0;
function check(label, got, want, tol = { maxAbs: 5e-4, rel: 5e-5 }) {
  let maxAbs = 0;
  let sumSq = 0;
  let refSq = 0;
  for (let i = 0; i < want.length; i++) {
    const d = Math.abs(got[i] - want[i]);
    if (d > maxAbs) maxAbs = d;
    sumSq += d * d;
    refSq += want[i] * want[i];
  }
  const rel = Math.sqrt(sumSq / (refSq || 1));
  const ok = maxAbs <= tol.maxAbs && rel <= tol.rel;
  if (!ok) failures++;
  console.log(`    ${ok ? "PASS" : "FAIL"}  ${label.padEnd(28)} `
    + `maxAbs=${maxAbs.toExponential(2)} rel=${rel.toExponential(2)}`);
}

function checkEqual(label, ok, detail = "") {
  if (!ok) failures++;
  console.log(`    ${ok ? "PASS" : "FAIL"}  ${label.padEnd(28)} ${detail}`);
}

for (const ref of golden) {
  const raw = ref.inputs;
  const L = ref.L;
  console.log(`\n${ref.name}  (na_mpnn, L=${L}, K=${ref.K})`);

  const inputs = {
    mask: Float32Array.from(raw.mask),
    residueIdx: Int32Array.from(raw.residueIdx),
    chainLabels: Int32Array.from(raw.chainLabels),
    X16: Float32Array.from(raw.X16),
    X16Mask: Float32Array.from(raw.X16Mask),
    polytype: Int32Array.from(raw.polytype),
  };

  const t0 = performance.now();
  const enc = model.encode(inputs);
  const encodeMs = performance.now() - t0;

  // A fully masked residue's row of `D_adjust` is all zeros, so the reference's
  // topk breaks an L-way tie however it likes and there is no answer to match.
  // Those rows are zeroed by mask_attend before they reach anything, so the
  // assertion covers unmasked rows and the rest is reported. 1am9 has four
  // UNK residues that exercise this; every other case has none.
  let graphMismatch = 0;
  let maskedMismatch = 0;
  let live = 0;
  for (let i = 0; i < L; i++) {
    for (let k = 0; k < ref.K; k++) {
      const differs = enc.EIdx[i * ref.K + k] !== ref.EIdx[i * ref.K + k];
      if (raw.mask[i] === 0) {
        if (differs) maskedMismatch++;
      } else {
        live++;
        if (differs) graphMismatch++;
      }
    }
  }
  checkEqual("neighbour graph", graphMismatch === 0,
    `${live - graphMismatch}/${live} unmasked edges identical`
    + (maskedMismatch ? `; ${maskedMismatch} masked-row edges tie differently` : ""));

  check("encoder h_V", enc.hV, Float32Array.from(ref.hV));

  const S = Int32Array.from(raw.S);
  const order = Int32Array.from(ref.order);
  for (const [label, ar, want] of [
    ["logits (backbone only)", { type: AR.NONE }, ref.logits.none],
    ["logits (autoregressive)", { type: AR.ORDER, order }, ref.logits.order],
    ["logits (all-but-self)", { type: AR.ALL_BUT_SELF }, ref.logits.all_but_self],
  ]) {
    check(label, model.score(enc, S, ar), Float32Array.from(want));
  }

  // Sampling must reproduce teacher-forced scoring under the same order.
  const rng = mulberry32(7);
  const drawn = model.sample(enc, { batch: 2, temperature: 0.3, rng });
  const rescored = model.score(enc, drawn.S[0], { type: AR.ORDER, order: drawn.order[0] });
  check("sampler == scorer", drawn.logits[0], rescored, { maxAbs: 5e-4, rel: 5e-5 });

  const seq = [...drawn.S[0]].map((v) => NA_ALPHABET[v]).join("");
  console.log(`    seq  ${seq.slice(0, 60)}`);
  console.log(`    encode ${encodeMs.toFixed(0)} ms`);

  // --- the parser, against the same structure -------------------------------
  if (!pdbDir) continue;
  const text = readFileSync(`${pdbDir}/${ref.name}.pdb`, "utf8");
  const s = structureFromText(text, { nucleicAsResidues: true });
  if (s.L !== L) {
    checkEqual("parser: residue count", false, `${s.L} parsed, reference has ${L}`);
    continue;
  }
  checkEqual("parser: residue count", true, `${L} residues`);

  let polyDiff = 0;
  let seqDiff = 0;
  let idxDiff = 0;
  let chainDiff = 0;
  for (let i = 0; i < L; i++) {
    if (s.polytype[i] !== raw.polytype[i]) polyDiff++;
    if (s.S[i] !== raw.S[i]) seqDiff++;
    if (s.residueIdx[i] !== raw.residueIdx[i]) idxDiff++;
    if (s.chainLabels[i] !== raw.chainLabels[i]) chainDiff++;
  }
  checkEqual("parser: polymer types", polyDiff === 0, polyDiff ? `${polyDiff} differ` : "");
  checkEqual("parser: sequence", seqDiff === 0, seqDiff ? `${seqDiff} differ` : "");
  checkEqual("parser: residue index", idxDiff === 0, idxDiff ? `${idxDiff} differ` : "");
  checkEqual("parser: chain labels", chainDiff === 0, chainDiff ? `${chainDiff} differ` : "");

  let maskDiff = 0;
  let coordMax = 0;
  for (let i = 0; i < L * 16; i++) {
    if (s.X16Mask[i] !== raw.X16Mask[i]) maskDiff++;
    for (let d = 0; d < 3; d++) {
      // Only compare coordinates that both sides consider present.
      if (raw.X16Mask[i]) {
        coordMax = Math.max(coordMax, Math.abs(s.X16[i * 3 + d] - raw.X16[i * 3 + d]));
      }
    }
  }
  checkEqual("parser: atom mask", maskDiff === 0, maskDiff ? `${maskDiff}/${L * 16} differ` : "");
  checkEqual("parser: coordinates", coordMax < 1e-3, `max ${coordMax.toExponential(2)} A`);

  // The end-to-end statement: parsing here and encoding here reproduces the
  // reference logits, with no tensors handed over.
  const encFromParse = model.encode({
    mask: s.mask,
    residueIdx: s.residueIdx,
    chainLabels: s.chainLabels,
    X16: s.X16,
    X16Mask: s.X16Mask,
    polytype: s.polytype,
  });
  check("end to end logits", model.score(encFromParse, s.S, { type: AR.NONE }),
    Float32Array.from(ref.logits.none));
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

console.log(failures === 0 ? "\nNA-MPNN parity holds" : `\n${failures} check(s) failed`);
process.exit(failures === 0 ? 0 : 1);
