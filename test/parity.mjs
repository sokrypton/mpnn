// Compare the JS engine against golden tensors from the reference PyTorch
// implementation (see tools/make_reference.py).
//
//   node test/parity.mjs <reference.json> <inputs.json> <weights dir>

import { readFileSync } from "node:fs";

import { AR } from "../mpnn/model.js";
import { argmaxAgreement, loadModel, mulberry32, Report, startKernel } from "./harness.mjs";

const [, , refPath, inputsPath, weightsDir] = process.argv;
await startKernel();

const reference = JSON.parse(readFileSync(refPath, "utf8"));
const cases = JSON.parse(readFileSync(inputsPath, "utf8"));
const report = new Report();
const check = (label, got, want, tol) => report.close(label, got, want, tol);

for (const ref of reference) {
  const testCase = cases.find((c) => c.name === ref.name);
  console.log(`\n${ref.name}  (${ref.modelType}, ${ref.checkpoint}, L=${ref.L}, K=${ref.K})`);

  const model = loadModel(weightsDir, ref.checkpoint);

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
    useSideChains: Boolean(testCase.useSideChains),
    xyz37: raw.xyz37 ? Float32Array.from(raw.xyz37) : undefined,
    xyz37Mask: raw.xyz37Mask ? Float32Array.from(raw.xyz37Mask) : undefined,
    chainMask: raw.chainMask ? Float32Array.from(raw.chainMask) : undefined,
  };

  const t0 = performance.now();
  const enc = model.encode(inputs);
  const encodeMs = performance.now() - t0;

  // Neighbour graph must match exactly -- everything downstream depends on it.
  const refEIdx = ref.EIdx.flat();
  let graphMismatch = 0;
  for (let i = 0; i < refEIdx.length; i++) if (enc.EIdx[i] !== refEIdx[i]) graphMismatch++;
  report.ok("neighbour graph", graphMismatch === 0,
    `${refEIdx.length - graphMismatch}/${refEIdx.length} edges identical`);

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

report.finish("all checks passed");
