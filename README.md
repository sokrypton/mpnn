# mpnn.web

ProteinMPNN, SolubleMPNN, LigandMPNN and MembraneMPNN running entirely in a
browser tab. No server, no upload, no build step — open `index.html` and the
weights come down once, then every forward pass happens on the visitor's
machine.

**[Live page](https://sokrypton.github.io/mpnn/)** · works offline after the
first load.

---

## What it does

- **Load a structure** by PDB ID, UniProt accession (AlphaFold DB), file drop,
  or drag-and-drop. PDB and mmCIF both parse.
- **Pick any of 13 checkpoints** — the four ProteinMPNN and four SolubleMPNN
  noise levels, four LigandMPNN variants, and the two membrane models.
- **Choose what to design** by clicking residues in the 3D view or the sequence
  track, shift-dragging a box, toggling whole chains, or selecting everything
  within 6 Å of a ligand.
- **Design** with a temperature, a batch size, per-amino-acid bias and
  omissions, tied/symmetric positions, and a seed that makes a run reproducible.
- **Read a position profile** as a sequence logo, in three flavours (see
  *Conditioning* below), and colour the structure by the model's confidence.

Everything is plain ES modules. There is no bundler, no framework, and no
runtime dependency — the only external resource the page ever fetches is a
structure file, and only when you ask it to.

## Layout

```
index.html          the page
app/                UI: main.js, worker.js, viewer.js, style.css
mpnn/               the engine, usable on its own from Node or a browser
  constants.js        generated tables (see tools/gen_constants.py)
  ops.js              dense kernels: linear, GELU, LayerNorm, gathers
  arena.js            named-slot scratch allocator
  features.js         backbone frames, kNN graph, edge/ligand features
  layers.js           encoder and decoder message-passing blocks
  model.js            encode / score / profile / sample
  pdb.js              PDB + mmCIF parsing
  weights.js          .mpnn file reader
weights/            13 converted checkpoints (~55 MB total, fp16)
tools/              checkpoint conversion, constant generation, golden tensors
test/               parity against PyTorch, plus a browser smoke test
```

## Using the engine directly

```js
import { Model } from "./mpnn/model.js";
import { Weights } from "./mpnn/weights.js";
import { structureFromText } from "./mpnn/pdb.js";

const model = new Model(await Weights.fetch("weights/proteinmpnn_v_48_020.mpnn"));
const s = structureFromText(await (await fetch("1ubq.pdb")).text());

// The encoder does not depend on the sequence, so it runs once.
const enc = model.encode(s);

const out = model.sample(enc, { batch: 8, temperature: 0.1 });
console.log(out.seq, out.score);          // sequences and mean negative log-likelihood

const { probs } = model.profile(enc);      // [L, 21] backbone-only distribution
```

## Conditioning

Following ColabDesign, conditioning is expressed as an **autoregressive mask**
rather than a decoding order: `arMask[i][j] = 1` means position `i` may see
position `j`'s amino acid. That collapses three separate code paths into one and
makes the cheap cases obvious.

| Mode | Cost | Exact? |
| --- | --- | --- |
| `AR.NONE` — nobody sees any sequence | 1 decoder pass | yes |
| `AR.ORDER` — the usual triangular mask from a decoding order | 1 pass | yes |
| `AR.ALL_BUT_SELF` — everyone sees everyone else | 1 pass | **no** |
| `profile({exact: true})` — each position decoded last in its own pass | L passes | yes |

The caveat on `ALL_BUT_SELF` is real and the page says so in the UI: the decoder
is three layers deep, so with a cyclic mask a residue's own identity reaches it
again through two-hop paths. It is a good fast approximation of the conditional
profile, not the thing itself. The reference implementation's `single_aa_score`
computes the exact version, and so does `profile({exact: true})` — at L times
the cost.

## Correctness

`test/parity.mjs` checks the port against golden tensors produced by the real
PyTorch model on the same inputs (`tools/make_reference.py`). Current state, on
ubiquitin and on streptavidin + biotin:

```
1ubq  (protein_mpnn, L=76, K=48)
    PASS  neighbour graph              3648/3648 edges identical
    PASS  encoder h_V                  maxAbs=4.11e-6 rel=2.60e-6
    PASS  logits (backbone only)       maxAbs=2.98e-5 rel=4.77e-6
    PASS  logits (autoregressive)      maxAbs=2.69e-5 rel=4.41e-6
    PASS  logits (all-but-self)        maxAbs=3.86e-5 rel=4.41e-6
    PASS  sampler == scorer            maxAbs=0.00e+0 rel=0.00e+0

1stp  (ligand_mpnn, L=121, K=32)
    PASS  neighbour graph              3872/3872 edges identical
    PASS  encoder h_V                  maxAbs=3.34e-6 rel=1.22e-6
    PASS  logits (backbone only)       maxAbs=1.19e-5 rel=1.31e-6
    ...
```

All four model families pass. The neighbour graph has to match *exactly* —
every downstream feature is indexed by it — and it does. The last check is an
internal invariant rather than a comparison: autoregressive sampling and
teacher-forced scoring of the resulting sequence under the same order must give
identical logits, which catches decoder bugs the golden tensors would not.

Two smaller guards worth mentioning, because both are places a port silently
drifts:

- GELU uses the **erf** form, matching `torch.nn.GELU()`. The tanh
  approximation is off by ~1e-3, which is visible in the logits.
- The element and periodic-table tables are **generated** from the reference
  source by `tools/gen_constants.py`, not transcribed. CI diffs the generated
  file against the committed one.

Run it yourself:

```bash
pip install ligandmpnn torch numpy
REF=$(python -c 'import ligandmpnn,pathlib;print(pathlib.Path(ligandmpnn.__file__).parent)')

python tools/convert_weights.py --src "$REF/data/model_params" --out /tmp/w32 --dtype float32
node test/dump_inputs.mjs /tmp/inputs.json \
  "1ubq=assets/1ubq.pdb=proteinmpnn_v_48_020=protein_mpnn" \
  "1stp=assets/1stp.pdb=ligandmpnn_v_32_010_25=ligand_mpnn"
python tools/make_reference.py --ref "$REF" --checkpoints "$REF/data/model_params" \
  --inputs /tmp/inputs.json --out /tmp/reference.json
node test/parity.mjs /tmp/reference.json /tmp/inputs.json /tmp/w32
```

And the page itself, end to end in real Chromium:

```bash
npm install --no-save playwright && npx playwright install chromium
node test/browser.mjs --shot page.png
```

## Speed

Measured in Node 22 on one core of the development container, so treat these as
a pessimistic floor rather than a benchmark.

| | L = 76 (ProteinMPNN) | L = 121 (LigandMPNN) |
| --- | --- | --- |
| encode | 1.5 s | 14 s |
| sample | ~1.1 s / sequence | ~1.2 s / sequence |
| profile (backbone only) | 1.1 s | — |

Three things are already done, in descending order of how much they mattered:

1. **The encoder runs once per structure.** It does not depend on the sequence,
   so `encode()` returns a handle that every sample, score and profile reuses.
   This is the ColabDesign split and it is worth more than any kernel work.

2. **A Linear over a concatenation is split into per-block projections.** Every
   message MLP starts with `W1 · [h_V_i ‖ h_E_ij ‖ h_V_j]`, and a Linear
   distributes over a concatenation:

   ```
   W1 · [a ‖ b ‖ c]  =  W1a · a  +  W1b · b  +  W1c · c
   ```

   Two of those three blocks are per-*residue*, not per-*edge*. Projecting them
   once per residue and gathering onto edges turns an `[L·K, 384] × [384, 128]`
   product into an `[L·K, 128] × [128, 128]` one plus two tiny `[L, 128]` ones —
   about 1.6× fewer multiply-adds through the encoder.

3. **The dense kernel blocks 8 rows against each pass over the weights.** The
   naive triple loop streams the whole weight matrix from L2 once per row and is
   bandwidth bound; blocking roughly doubles throughput (1.15 → 2.43 GFLOP/s on
   a representative `[3648, 384] × [384, 128]`).

### What is next, and why

These are ordered by expected payoff. Nothing here is started; the notes are
here so the reasoning does not have to be rediscovered.

- **A WASM SIMD kernel.** The obvious next step and the largest remaining lever.
  The shape to copy is CIRPIN-web's: one coarse entry point, all weights and
  intermediates living in a single pre-allocated `WebAssembly.Memory` arena so
  nothing is copied per call, and the JS staying as the reference implementation
  and the fallback when instantiation fails. `mpnn/arena.js` already centralises
  every scratch buffer, which is the part that would otherwise need unpicking.
  Expect 2-4× on top of the blocked JS kernel.

- **Extend the block split into the decoder.** The decoder's W1 takes
  `[h_V_i ‖ h_E ‖ h_S_j ‖ h_V_j]` — *three* of four blocks are per-node, and the
  `h_E` block never changes at all, so `W1e · h_E` can be computed once per
  structure and shared across every layer, every decode step and the whole
  batch. During sampling `h_S` and the layer stacks change one row per step, so
  their projections can be maintained incrementally, which removes the W1
  matmul from the inner loop almost entirely. Roughly 2× on scoring and ~3× on
  sampling. `makeDecoderLayer` already exposes `.blocks` and `.applyPre()` for
  exactly this; only `model.js` needs to change.

- **Cache the ligand atom-pair messages.** This is why LigandMPNN's encode is
  ~10× ProteinMPNN's. The two `y_context_encoder` layers message-pass among the
  M = 25 nearest ligand atoms *per residue*, which is `L · M · M` = 75 625 rows
  for streptavidin. But biotin only has 16 atoms, so there are at most 256
  distinct atom pairs — and in the first of the two layers the entire message is
  a pure function of the atom pair, not of the residue. Deduplicating would cut
  that layer by two orders of magnitude on small ligands.

- **Stream results.** `sample()` currently returns the whole batch at once; the
  page could show sequences as they finish.

## Weights

`weights/` holds all 13 checkpoints converted to a flat binary format, fp16, 3.3
to 5.3 MB each and about 55 MB in total. Only one is downloaded at a time, and
the browser caches it. The format is a magic number, a JSON header describing
every tensor, and the tensors concatenated — see `tools/convert_weights.py`,
which regenerates them from the upstream `.pt` files. CI checks the committed
files still match.

`ligandmpnn_sc_v_32_002_16` (the sidechain packer) is deliberately not
converted; nothing here runs it.

## Credit

Model architecture and weights are from
[dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN) (MIT) — Dauparas
et al., *Atomic context-conditioned protein sequence design using LigandMPNN*,
and Dauparas et al., *Robust deep learning-based protein sequence design using
ProteinMPNN*, Science 2022.

The autoregressive-mask API, the encode-once/decode-many split, and Gumbel-max
sampling are lifted from
[ColabDesign](https://github.com/sokrypton/ColabDesign). The overall shape of
the page — no build step, JS as the reference implementation, a WASM kernel as
an optional accelerator — follows
[CIRPIN-web](https://github.com/sokrypton/CIRPIN-web).

`assets/1ubq.pdb` and `assets/1stp.pdb` are from the RCSB PDB and are used as
test fixtures.
