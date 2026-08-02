// Build the sampling-parity cases: bias, omissions, fixed residues, tied
// positions and the ligand-context switch, across two model families.
//
//   node test/make_sampling_cases.mjs <out.json>
//
// Feeds tools/sample_reference.py and test/sampling.mjs.
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { structureFromText } from "../mpnn/pdb.js";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const OUT = process.argv[2];
const rng = (() => { let a = 987654321; return () => { a = (a*1664525+1013904223)>>>0; return a/4294967296; }; })();

function shuffled(L) {
  const o = Array.from({length:L},(_,i)=>i);
  for (let i=L-1;i>0;i--){const j=Math.floor(rng()*(i+1));[o[i],o[j]]=[o[j],o[i]];}
  return o;
}
const zeroBias = (L) => Array.from({length:L*21},()=>0);

function biasWith(L, spec) {
  const b = zeroBias(L);
  for (let i=0;i<L;i++) for (const [aa,v] of Object.entries(spec)) b[i*21+Number(aa)] = v;
  return b;
}

const cases = [];
for (const [name, file, ckpt, type] of [
  ["ubq", "1ubq.pdb", "proteinmpnn_v_48_020", "protein_mpnn"],
  ["stp", "1stp.pdb", "ligandmpnn_v_32_010_25", "ligand_mpnn"],
]) {
  const s = structureFromText(readFileSync(`${ROOT}assets/${file}`, "utf8"));
  const L = s.L;
  const all1 = Array.from({length:L},()=>1);

  // half the chain fixed
  const halfFixed = Array.from({length:L},(_,i)=> i%3===0 ? 0 : 1);

  // three tied groups with non-uniform weights
  const sym = [[2, 20, 40], [5, 25], [7, 27, 47, 60]].filter(g=>g.every(p=>p<L));
  const symW = sym.map(g=>g.map((_,i)=> i===0 ? 1.0 : (i===1 ? 2.0 : 0.5)));

  // omit C and M, favour W
  const omitBias = biasWith(L, {1: -1e9, 10: -1e9, 18: 2.0});

  const variants = [
    { name:"plain", order: shuffled(L), chainMask: all1, bias: zeroBias(L) },
    { name:"bias-omit", order: shuffled(L), chainMask: all1, bias: omitBias },
    { name:"fixed", order: shuffled(L), chainMask: halfFixed, bias: zeroBias(L) },
    { name:"symmetry", order: shuffled(L), chainMask: all1, bias: zeroBias(L),
      symmetry: sym, symmetryWeights: symW },
    { name:"symmetry+bias+fixed", order: shuffled(L), chainMask: halfFixed, bias: omitBias,
      symmetry: sym, symmetryWeights: symW },
  ];
  if (type === "ligand_mpnn") {
    variants.push({ name:"no-atom-context", order: shuffled(L), chainMask: all1,
      bias: zeroBias(L), useAtomContext: false });
  }

  cases.push({ name, checkpoint: ckpt, modelType: type, variants,
    inputs: { L, X:[...s.X], mask:[...s.mask], S:[...s.S], residueIdx:[...s.residueIdx],
      chainLabels:[...s.chainLabels], ligandXyz:[...s.ligandXyz],
      ligandType:[...s.ligandType], ligandMask:[...s.ligandMask] } });
  console.log(`${name}: L=${L}, ${variants.length} variants`);
}
writeFileSync(OUT, JSON.stringify(cases));
console.log(`wrote ${OUT}`);
