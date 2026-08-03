// Is one pass with `ar_mask = 1 - I` good enough to score a sequence?
//
//   node test/pseudolikelihood.mjs <weights dir> [reps]
//
// The obvious objection to the single-pass pseudo-likelihood is that the mask
// is cyclic: i tells j its identity at layer 0, and j tells it back to i at
// layer 1, so across three decoder layers a residue partly sees itself. The
// L-pass alternative -- decode position t last, once per t -- has no such leak.
//
// But it is not a clean reference point either, because "t last" does not
// pin down the order of the other L-1, and the decoder is not invariant to it:
// t reads its neighbours' layer-1 states, which depend on *their* masks. So the
// L-pass profile is a family of answers, not one answer.
//
// This measures both spreads on the same structure: how far two L-pass profiles
// sit from each other when only the others' order changes, and how far the
// single pass sits from them. If the second is the same order of magnitude as
// the first, the leak costs about as much as the arbitrary choice the L-pass
// version has to make anyway -- at 1/L of the cost.

import { readFileSync } from "node:fs";

import { AR, perPositionNLL } from "../mpnn/model.js";
import { structureFromText } from "../mpnn/pdb.js";
import { loadModel, mulberry32, startKernel } from "./harness.mjs";

const [, , weightsDir, repsArg] = process.argv;
const REPS = parseInt(repsArg, 10) || 3;
await startKernel();

const CASES = [
  ["assets/1ubq.pdb", "proteinmpnn_v_48_020"],
  ["assets/1stp.pdb", "ligandmpnn_v_32_010_25"],
];

function shuffled(L, rng) {
  const out = Int32Array.from({ length: L }, (_, i) => i);
  for (let i = L - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const t = out[i];
    out[i] = out[j];
    out[j] = t;
  }
  return out;
}

function argmax20(logits, i, V) {
  let best = 0;
  for (let v = 1; v < 20; v++) if (logits[i * V + v] > logits[i * V + best]) best = v;
  return best;
}

function compare(a, b, L, V, S) {
  const na = perPositionNLL(a, S, L, V);
  const nb = perPositionNLL(b, S, L, V);
  let sum = 0;
  let max = 0;
  let agree = 0;
  for (let i = 0; i < L; i++) {
    const d = Math.abs(na[i] - nb[i]);
    sum += d;
    if (d > max) max = d;
    if (argmax20(a, i, V) === argmax20(b, i, V)) agree++;
  }
  return { mae: sum / L, max, agree: agree / L };
}

let failures = 0;
for (const [path, modelName] of CASES) {
  const s = structureFromText(readFileSync(path, "utf8"));
  const model = loadModel(weightsDir, modelName);
  const enc = model.encode(s);
  const { L } = enc;
  const V = model.numLetters;
  const S = s.S;

  console.log(`\n${path}  ${modelName}  L=${L}`);

  const t0 = performance.now();
  const onePass = model.score(enc, S, { type: AR.ALL_BUT_SELF });
  const onePassMs = performance.now() - t0;
  const backbone = model.score(enc, S, { type: AR.NONE });

  const rng = mulberry32(1);
  const reps = [];
  let lPassMs = 0;
  for (let r = 0; r < REPS; r++) {
    const others = r === 0 ? null : shuffled(L, rng);
    const t = performance.now();
    reps.push(model.profile(enc, { S, exact: true, order: others }).logits);
    lPassMs += performance.now() - t;
  }

  const mean = (logits) => {
    const n = perPositionNLL(logits, S, L, V);
    let total = 0;
    for (let i = 0; i < L; i++) total += n[i];
    return total / L;
  };

  // Spread among the L-pass profiles: same quantity, different arbitrary order.
  let repMae = 0;
  let repMax = 0;
  let repAgree = 0;
  let pairs = 0;
  for (let i = 0; i < REPS; i++) {
    for (let j = i + 1; j < REPS; j++) {
      const c = compare(reps[i], reps[j], L, V, S);
      repMae += c.mae;
      repMax = Math.max(repMax, c.max);
      repAgree += c.agree;
      pairs++;
    }
  }
  repMae /= pairs;
  repAgree /= pairs;

  let oneMae = 0;
  let oneAgree = 0;
  for (const rep of reps) {
    const c = compare(onePass, rep, L, V, S);
    oneMae += c.mae;
    oneAgree += c.agree;
  }
  oneMae /= REPS;
  oneAgree /= REPS;

  console.log(`    mean nll     one pass ${mean(onePass).toFixed(4)}  `
    + `L passes ${reps.map((r) => mean(r).toFixed(4)).join(" ")}  `
    + `backbone ${mean(backbone).toFixed(4)}`);
  console.log(`    L vs L       mae ${repMae.toFixed(4)}  max ${repMax.toFixed(3)}  `
    + `argmax ${(repAgree * 100).toFixed(1)}%`);
  console.log(`    1 vs L       mae ${oneMae.toFixed(4)}  `
    + `argmax ${(oneAgree * 100).toFixed(1)}%  `
    + `ratio ${(oneMae / repMae).toFixed(1)}x the L-vs-L spread`);
  console.log(`    cost         ${onePassMs.toFixed(0)} ms vs `
    + `${(lPassMs / REPS).toFixed(0)} ms (${(lPassMs / REPS / onePassMs).toFixed(0)}x)`);

  // The claims the docs make, as assertions. Generous bounds -- this is
  // guarding against a regression that breaks the mask, not pinning digits.
  const checks = [
    [repMae > 0, "the L-pass profile really is order-dependent"],
    [oneMae < 8 * repMae, "one pass stays within an order of magnitude of that spread"],
    [oneAgree > 0.85, "one pass agrees with the L-pass argmax"],
    [mean(onePass) > mean(reps[0]) - 0.01, "one pass does not flatter the sequence"],
    [mean(reps[0]) < mean(backbone), "conditioning on the rest beats the backbone alone"],
  ];
  for (const [ok, label] of checks) {
    if (!ok) failures++;
    console.log(`    ${ok ? "PASS" : "FAIL"}  ${label}`);
  }
}

console.log(failures === 0
  ? "\npseudo-likelihood behaves as documented"
  : `\n${failures} check(s) failed`);
process.exit(failures === 0 ? 0 : 1);
