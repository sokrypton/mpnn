// One-hot times a matrix, or a gather? ColabDesign does the former; this engine
// does the latter. Run with: node wasm/bench/onehot.mjs
import { readFileSync } from "node:fs";
import { enableAcceleration } from "../../mpnn/accel.js";
import { linear } from "../../mpnn/ops.js";
const buf = readFileSync(new URL("../kernels.wasm", import.meta.url));
await enableAcceleration(buf.buffer.slice(buf.byteOffset, buf.byteOffset+buf.byteLength));

// LigandMPNN's atom-type encoding: one-hot over [type 120 | group 19 | period 8]
// = 147 wide with exactly THREE non-zeros, projected to 128.
const N = 3025;           // L*M for streptavidin
const IN = 147, OUT = 128;
const W = new Float32Array(OUT*IN).map(()=>Math.random()-0.5);
const bias = new Float32Array(OUT).map(()=>Math.random());
const types = Int32Array.from({length:N}, ()=>1+Math.floor(Math.random()*30));
const GROUP = Int32Array.from({length:120},(_, i)=>i%18+1);
const PERIOD = Int32Array.from({length:120},(_, i)=>i%7+1);

// (a) ColabDesign's form: materialise the one-hot, then one dense matmul.
const onehot = new Float32Array(N*IN);
function buildOneHot(){
  onehot.fill(0);
  for (let i=0;i<N;i++){
    const t=types[i];
    onehot[i*IN+t]=1; onehot[i*IN+120+GROUP[t]]=1; onehot[i*IN+139+PERIOD[t]]=1;
  }
}
const outA = new Float32Array(N*OUT);
function matmulWay(){ buildOneHot(); linear(onehot, W, bias, N, IN, OUT, outA); }

// (b) three column reads plus the bias -- what mpnn/features.js does.
const outB = new Float32Array(N*OUT);
function gatherWay(){
  for (let i=0;i<N;i++){
    const t=types[i], g=120+GROUP[t], p=139+PERIOD[t], off=i*OUT;
    for (let o=0;o<OUT;o++){ const r=o*IN; outB[off+o]=W[r+t]+W[r+g]+W[r+p]+bias[o]; }
  }
}
// (c) dedupe by type first (what the engine actually does), then gather rows.
const outC = new Float32Array(N*OUT);
function dedupWay(){
  const table = new Float32Array(120*OUT);
  for (let t=0;t<120;t++){ const g=120+GROUP[t], p=139+PERIOD[t];
    for (let o=0;o<OUT;o++){ const r=o*IN; table[t*OUT+o]=W[r+t]+W[r+g]+W[r+p]+bias[o]; } }
  for (let i=0;i<N;i++) outC.set(table.subarray(types[i]*OUT,(types[i]+1)*OUT), i*OUT);
}

for (const [name,f,out] of [["one-hot @ matrix (ColabDesign form)",matmulWay,outA],
                            ["3 column reads",gatherWay,outB],
                            ["dedupe by type, then gather",dedupWay,outC]]) {
  f();
  let maxAbs=0; for(let i=0;i<outA.length;i++) maxAbs=Math.max(maxAbs,Math.abs(outA[i]-out[i]));
  const R=200; const t=performance.now(); for(let r=0;r<R;r++) f();
  const ms=(performance.now()-t)/R;
  console.log(`${name.padEnd(38)} ${ms.toFixed(3)} ms   ${(N*IN*OUT/ms/1e6).toFixed(1)} equiv-GMAC/s   maxAbs=${maxAbs.toExponential(1)}`);
}
