// Structures smaller than K.
//
//   node test/small.mjs [weights dir]
//
// The neighbour graph clamps to `min(K, L)`, matching the reference's
// `topk(..., min(top_k, L))`, but every layer used to take the checkpoint's K
// from its closure and index `hE` with it. That read past the end of a
// too-narrow edge tensor, so a 24-base-pair duplex or any short peptide died
// with "offset is out of bounds" -- in every model family, not just NA-MPNN.
//
// Nothing else in the suite goes below L = 76, which is how it survived.
//
// This test is numeric only in the loose sense: it checks the encode runs, that
// K comes out as `min(K, L)`, and that sampling produces letters in range. That
// needs no golden tensors, so unlike the parity tests it runs against the fp16
// weights the page ships.

import { readFileSync } from "node:fs";

import { Model } from "../mpnn/model.js";
import { NA_ALPHABET } from "../mpnn/na.js";
import { structureFromText } from "../mpnn/pdb.js";
import { Weights } from "../mpnn/weights.js";
import { startKernel } from "./harness.mjs";

const weightsDir = process.argv[2] ?? new URL("../weights", import.meta.url).pathname;
await startKernel();

function load(name) {
  const buf = readFileSync(`${weightsDir}/${name}.mpnn`);
  return new Model(Weights.fromArrayBuffer(
    buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  ));
}

/** 1UBQ truncated to its first `n` residues. */
const ubq = readFileSync(new URL("../assets/1ubq.pdb", import.meta.url), "utf8");
function truncated(n) {
  return ubq.split("\n")
    .filter((line) => !line.startsWith("ATOM") || Number(line.slice(22, 26)) <= n)
    .join("\n");
}

let failures = 0;
function check(label, ok, detail) {
  console.log(`${ok ? "  ok  " : "  FAIL"} ${label}${detail ? `  ${detail}` : ""}`);
  if (!ok) failures++;
}

// Either side of both checkpoint K values, plus the degenerate small end.
for (const name of ["proteinmpnn_v_48_020", "ligandmpnn_v_32_020_25", "solublempnn_v_48_020"]) {
  const model = load(name);
  console.log(`\n${name} (K=${model.K})`);
  for (const n of [2, 4, 10, 20, 31, 32, 33, 47, 48, 49]) {
    const s = structureFromText(truncated(n));
    try {
      const enc = model.encode(s);
      const want = Math.min(model.K, s.L);
      const out = model.sample(enc, { batch: 2, temperature: 0.1, rng: () => 0.5 });
      const inRange = out.S.every((seq) => seq.every((v) => v >= 0 && v < model.numLetters));
      check(`L=${s.L}`, enc.K === want && enc.L === s.L && inRange,
        `K=${enc.K} (want ${want}) nll=${out.score[0].toFixed(3)}`);
    } catch (error) {
      check(`L=${s.L}`, false, String(error.message));
    }
  }
}

// NA-MPNN reaches the same code by a different featuriser, and a short duplex
// is the case that turned this up: 1BNA is 24 bases against its K of 32.
{
  const model = load("na_mpnn_design");
  console.log(`\nna_mpnn_design (K=${model.K})`);
  for (const n of [12, 24, 40]) {
    const s = structureFromText(truncated(n), { nucleicAsResidues: true });
    try {
      const enc = model.encode({
        X: s.X, mask: s.mask, residueIdx: s.residueIdx, chainLabels: s.chainLabels,
        ligandXyz: s.ligandXyz, ligandType: s.ligandType, ligandMask: s.ligandMask,
        X16: s.X16, X16Mask: s.X16Mask, polytype: s.polytype,
      });
      const out = model.sample(enc, { batch: 2, temperature: 0.1, rng: () => 0.5 });
      const inRange = out.S.every((seq) => seq.every((v) => v >= 0 && v < NA_ALPHABET.length));
      check(`L=${s.L}`, enc.K === Math.min(model.K, s.L) && inRange,
        `K=${enc.K} (want ${Math.min(model.K, s.L)})`);
    } catch (error) {
      check(`L=${s.L}`, false, String(error.message));
    }
  }
}

console.log(failures === 0
  ? "\nsmall structures encode and sample"
  : `\n${failures} failure(s)`);
process.exit(failures === 0 ? 0 : 1);
