// The neighbour graph indexes every downstream feature, so the grid-accelerated
// search has to agree with the exact sweep edge for edge -- not approximately.
//
//   node test/neighbors.mjs [structure.pdb ...]

import { readFileSync } from "node:fs";

import { computeBackbone, neighborGraph } from "../mpnn/features.js";
import { argTopKSmallest } from "../mpnn/ops.js";
import { structureFromText } from "../mpnn/pdb.js";

/** The original O(L²·K) sweep, kept here purely as the oracle. */
function exactGraph(CA, mask, L, topK) {
  const K = Math.min(topK, L);
  const EIdx = new Int32Array(L * K);
  const DNeighbors = new Float32Array(L * K);
  const row = new Float32Array(L);
  const d = (a, b) => Math.sqrt(
    (CA[a * 3] - CA[b * 3]) ** 2
    + (CA[a * 3 + 1] - CA[b * 3 + 1]) ** 2
    + (CA[a * 3 + 2] - CA[b * 3 + 2]) ** 2 + 1e-6,
  );
  for (let i = 0; i < L; i++) {
    let rowMax = 0;
    for (let j = 0; j < L; j++) {
      const v = mask[i] * mask[j] === 0 ? 0 : d(i, j);
      row[j] = v;
      if (v > rowMax) rowMax = v;
    }
    for (let j = 0; j < L; j++) if (mask[i] * mask[j] === 0) row[j] += rowMax;
    argTopKSmallest(row, L, K, EIdx.subarray(i * K, (i + 1) * K));
    for (let k = 0; k < K; k++) DNeighbors[i * K + k] = row[EIdx[i * K + k]];
  }
  return { EIdx, DNeighbors, K };
}

function compare(label, CA, mask, L, K) {
  let t0 = performance.now();
  const want = exactGraph(CA, mask, L, K);
  const exactMs = performance.now() - t0;
  t0 = performance.now();
  const got = neighborGraph(CA, mask, L, K);
  const gridMs = performance.now() - t0;

  let badIdx = 0;
  let maxAbs = 0;
  for (let i = 0; i < L * want.K; i++) {
    if (want.EIdx[i] !== got.EIdx[i]) badIdx++;
    maxAbs = Math.max(maxAbs, Math.abs(want.DNeighbors[i] - got.DNeighbors[i]));
  }
  const ok = badIdx === 0 && maxAbs === 0;
  console.log(
    `  ${ok ? "PASS" : "FAIL"}  ${label.padEnd(30)} L=${String(L).padStart(5)} `
    + `edges=${L * want.K} mismatched=${badIdx} maxAbsD=${maxAbs}  `
    + `exact ${exactMs.toFixed(0)}ms  grid ${gridMs.toFixed(0)}ms  `
    + `${(exactMs / Math.max(gridMs, 0.001)).toFixed(1)}x`,
  );
  return ok;
}

const paths = process.argv.slice(2);
let failures = 0;

// Synthetic cases first: they cover the shapes real files rarely hit.
const rng = (() => {
  let a = 12345;
  return () => {
    a = (a * 1664525 + 1013904223) >>> 0;
    return a / 4294967296;
  };
})();

for (const [label, L, K, maskFrac] of [
  ["random cloud", 300, 48, 0],
  ["random cloud, 10% masked", 300, 48, 0.1],
  ["K larger than chain", 20, 48, 0],
  ["more masked than K", 60, 48, 0.9],
  ["all masked", 80, 48, 1],
  ["duplicate coordinates", 200, 48, 0],
  ["single point repeated", 60, 32, 0],
]) {
  const CA = new Float32Array(L * 3);
  for (let i = 0; i < L * 3; i++) {
    CA[i] = label === "single point repeated" ? 1
      : label === "duplicate coordinates" ? Math.floor(rng() * 5) * 3
        : (rng() - 0.5) * 60;
  }
  const mask = new Float32Array(L).fill(1);
  for (let i = 0; i < L; i++) if (rng() < maskFrac) mask[i] = 0;
  if (!compare(label, CA, mask, L, K)) failures++;
}

for (const path of paths) {
  const s = structureFromText(readFileSync(path, "utf8"));
  const bb = computeBackbone(s.X, s.L);
  for (const K of [32, 48]) {
    if (!compare(`${path.split("/").pop()} K=${K}`, bb.CA, s.mask, s.L, K)) failures++;
  }
}

console.log(failures === 0 ? "\nneighbour graphs identical" : `\n${failures} mismatch(es)`);
process.exit(failures === 0 ? 0 : 1);
