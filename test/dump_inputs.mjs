// Parse the test structures with the JS parser and write the exact tensors the
// PyTorch reference should be fed, so tools/make_reference.py compares model
// maths rather than two independent PDB readers.

import { readFileSync, writeFileSync } from "node:fs";

import { structureFromText } from "../mpnn/pdb.js";

const [, , outPath, ...structures] = process.argv;
if (!outPath) {
  console.error("usage: node test/dump_inputs.mjs <out.json> <name=path=checkpoint> ...");
  process.exit(1);
}

// A fixed decoding order per case, so "order" conditioning is reproducible.
function fixedOrder(L, seed) {
  const order = Array.from({ length: L }, (_, i) => i);
  let state = seed >>> 0;
  for (let i = L - 1; i > 0; i--) {
    state = (state * 1664525 + 1013904223) >>> 0;
    const j = state % (i + 1);
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}

const cases = [];
for (const spec of structures) {
  const [name, path, checkpoint, modelType] = spec.split("=");
  const text = readFileSync(path, "utf8");
  const s = structureFromText(text);
  cases.push({
    name,
    checkpoint,
    modelType,
    order: fixedOrder(s.L, 12345),
    inputs: {
      L: s.L,
      X: [...s.X],
      mask: [...s.mask],
      S: [...s.S],
      residueIdx: [...s.residueIdx],
      chainLabels: [...s.chainLabels],
      ligandXyz: [...s.ligandXyz],
      ligandType: [...s.ligandType],
      ligandMask: [...s.ligandMask],
      // 0 soluble, 1 interface, 2 buried -- a deterministic mix, so the
      // membrane models are exercised with something other than all zeros.
      membraneLabels: Array.from({ length: s.L }, (_, i) => (i * 7) % 11 < 3 ? 2 : (i % 5 === 0 ? 1 : 0)),
    },
  });
  console.log(
    `${name}: L=${s.L} chains=${s.chainList.join(",")} ligandAtoms=${s.ligandType.length} `
    + `seq=${s.sequence.slice(0, 40)}${s.sequence.length > 40 ? "..." : ""}`,
  );
}

writeFileSync(outPath, JSON.stringify(cases));
console.log(`wrote ${outPath}`);
