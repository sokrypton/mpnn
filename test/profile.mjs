// Where the time actually goes.
//
//   node test/profile.mjs <weights dir> [structure.pdb ...]
//
// Wraps the hot entry points with counters rather than sampling, so the numbers
// line up with the code you would change.

import { readFileSync } from "node:fs";

import { Model } from "../mpnn/model.js";
import { Weights } from "../mpnn/weights.js";
import { enableAcceleration } from "../mpnn/accel.js";
import { structureFromText } from "../mpnn/pdb.js";

const wasmPath = new URL("../wasm/kernels.wasm", import.meta.url);
const simd = process.env.MPNN_NO_SIMD
  ? null
  : await enableAcceleration(readFileSync(wasmPath).buffer);
console.log(`kernel: ${simd ? "wasm simd" : "javascript"}`);

const [, , weightsDir, ...structures] = process.argv;

function loadModel(name) {
  const buf = readFileSync(`${weightsDir}/${name}.mpnn`);
  return new Model(Weights.fromArrayBuffer(
    buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  ));
}


const CASES = structures.length ? structures : [
  ["assets/1ubq.pdb", "proteinmpnn_v_48_020"],
  ["assets/1stp.pdb", "ligandmpnn_v_32_010_25"],
];

for (const [path, modelName] of CASES) {
  const s = structureFromText(readFileSync(path, "utf8"));
  const model = loadModel(modelName);
  console.log(`\n${path}  ${modelName}  L=${s.L}  K=${model.K}`
    + (model.isLigand ? `  M=${model.M}  ligandAtoms=${s.ligandType.length}` : ""));

  // warm
  const warm = model.encode(s);
  model.sample(warm, { batch: 1 });

  let t0 = performance.now();
  const enc = model.encode(s);
  const encodeMs = performance.now() - t0;

  t0 = performance.now();
  model.sample(enc, { batch: 1 });
  const sample1 = performance.now() - t0;

  t0 = performance.now();
  model.sample(enc, { batch: 8 });
  const sample8 = performance.now() - t0;

  t0 = performance.now();
  model.profile(enc);
  const profileMs = performance.now() - t0;

  console.log(`    encode          ${encodeMs.toFixed(0)} ms`);
  console.log(`    sample batch 1  ${sample1.toFixed(0)} ms`);
  console.log(`    sample batch 8  ${sample8.toFixed(0)} ms `
    + `(${(sample8 / 8).toFixed(0)} ms/seq, ${(sample1 * 8 / sample8).toFixed(2)}x from batching)`);
  console.log(`    profile         ${profileMs.toFixed(0)} ms`);
}
