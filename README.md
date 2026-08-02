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
- **Read a position profile** as a sequence logo -- bits on the y axis, glyphs
  stretched to their share of the column, the native residue underneath -- in
  three flavours (see *Conditioning* below). Hovering a column highlights that
  residue in 3D and vice versa; clicking one toggles whether it is designed.
  The structure can also be coloured by the model's confidence.

Everything is plain ES modules. There is no bundler, no framework, and no
runtime dependency — the only external resource the page ever fetches is a
structure file, and only when you ask it to.

## The renderer is vendored, not reimplemented

`app/trace3d.js` and `app/sec.js` are copied **verbatim** from
[CIRPIN-web](https://github.com/sokrypton/CIRPIN-web). Please do not rewrite
them.

There was a hand-written cartoon renderer here first and it was worse in ways
that are not obvious until you look at a render: no background halo, so crossing
elements blended into each other instead of occluding; whole elements sharing
one depth, so a strand passing through a helix drew entirely in front of it;
ribbon widths in screen units rather than angstroms, so a small domain looked
chunky and a large one spindly; and a secondary-structure heuristic invented on
the spot instead of TM-align's `make_sec`, which CIRPIN's parity suite already
checks against the C++ to 5e-11. Every one of those is a comment in the vendored
file explaining what the fix was for.

The only modification is `setPaper()` at the bottom of `trace3d.js`, plus
`PAPER_CSS` becoming `let` so it can follow. `shade()` expresses depth by
blending toward the page background and returning an *opaque* colour, which only
works if it knows what that background is; CIRPIN-web is a light page with a
fixed paper colour and this one has a dark mode. `diff` against upstream shows
exactly that one line changed.

Everything else lives in `app/viewer.js`, which is an adapter: it decides
colours, draws the ligand, and hit-tests residues (the renderer has no picking,
and the design UI needs it). Its projection is copied line for line from
`drawTraces` -- it has to be, or clicks land somewhere other than where the
ribbon was drawn.

## Layout

```
index.html          the page
app/                UI
  main.js             page controller: selection, worker, colour modes
  worker.js           runs the engine off the main thread
  trace3d.js          cartoon renderer -- VENDORED from CIRPIN-web
  sec.js              C-alpha secondary structure -- VENDORED from CIRPIN-web
  viewer.js           adapter: colours, ligand discs, residue picking
  logo.js             sequence logo on a canvas
  style.css
mpnn/               the engine, usable on its own from Node or a browser
  constants.js        generated tables (see tools/gen_constants.py)
  ops.js              dense kernels: linear, GELU, LayerNorm, gathers
  arena.js            named-slot scratch allocator
  features.js         backbone frames, kNN graph, edge/ligand features
  layers.js           encoder and decoder message-passing blocks
  model.js            encode / score / profile / sample
  pdb.js              PDB + mmCIF parsing
  weights.js          .mpnn file reader
  accel.js            optional WebAssembly SIMD accelerator
wasm/kernels.c      the SIMD kernels; build.sh rebuilds kernels.wasm
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

Measured on one core of the development container with Node 22, so treat these
as a pessimistic floor rather than a benchmark.

| | ProteinMPNN, L = 76 | LigandMPNN, L = 121 |
| --- | --- | --- |
| encode | **0.52 s** (was 1.4) | **1.04 s** (was 13.3) |
| sample | **76 ms/seq** (was 1014) | **91 ms/seq** (was 1113) |
| profile | **0.13 s** (was 1.04) | **0.09 s** (was 1.12) |

Between 2.7x and 13x depending on what you ask for. Four things got it there,
and the order matters: every algorithmic change came before any kernel work,
because a faster kernel only makes the remaining multiply-adds cheaper.

**1. The encoder runs once per structure.** It does not depend on the sequence,
so `encode()` returns a handle that every sample, score and profile reuses.
This is the ColabDesign split and it is worth more than any kernel.

**2. A Linear over a concatenation splits into per-block projections.** Every
message MLP begins with `W1 · [a ‖ b ‖ …]`, and a Linear distributes over a
concatenation: `W1 · [a ‖ b ‖ c] = W1a·a + W1b·b + W1c·c`.

In the *encoder* two of the three blocks are per-residue rather than per-edge.
Projecting those once and gathering them onto edges turns an
`[L·K, 384] × [384, 128]` product into an `[L·K, 128] × [128, 128]` one — about
1.6x fewer multiply-adds.

In the *decoder* it is better than that. W1 there takes
`[h_V_i ‖ h_E ‖ h_S_j ‖ h_V_j]`: three of the four blocks are per-node, and the
`h_E` block never changes — not between layers, not between decode steps, not
across the batch. So `W1e · h_E` is computed once per structure and cached on
the encoding. During sampling, `h_S` and the layer stacks change exactly one row
per step, so their projections are maintained incrementally instead of being
recomputed. The decoder's inner loop ends up with **no per-edge matmul at all**,
just gathers of four precomputed tables. 2.5x on sampling, before any SIMD.

**3. Ligand atom pairs are deduplicated.** This is why LigandMPNN used to cost
ten times ProteinMPNN. Its two atom-context layers message-pass among the M = 25
nearest ligand atoms *of every residue* — 75,625 rows for streptavidin. But an
atom pair's contribution depends only on which two ligand atoms it is, never on
the residue looking at them, and biotin has 16 atoms: **289 distinct pairs, not
75,625**. The pair edge embedding is pair-pure, and so is the entire message of
the first atom layer, because that layer's node input is the atom-type embedding
— also a function of the atom alone. The second layer's node input has absorbed
a per-residue sum and is no longer pair-pure, but its edge half still is, and
its rows are restricted to pairs whose mask is 1 (59% of biotin's slots are zero
padding). 3.1x on LigandMPNN's encode.

**4. A WebAssembly SIMD kernel.** After all of that, a CPU profile put 90% of
the remaining time in a single function. `wasm/kernels.c` is ~200 lines of
`-msimd128` C, built with plain clang and wasm-ld — no Emscripten — and reaches
**~22 GFLOP/s against the JS kernel's ~2.4**, roughly 9x on the shapes that
matter.

Two things about how it is integrated, both taken from CIRPIN-web:

- **JS stays the reference.** `mpnn/ops.js` defines what the model computes; the
  kernel has to agree with it, and the parity suite runs against both
  (`MPNN_NO_SIMD=1` selects the JS path). There is no feature probe — a runtime
  without SIMD fails to validate a module containing `v128`, so instantiation
  *is* the probe, and any failure falls back silently.
- **The entry points are coarse.** A bare `gelu` export would not pay: GELU is
  memory bound, so staging its array in and back out costs roughly what the
  arithmetic does. Instead `tail2_f32` runs a whole `gelu → W2 → gelu → W3`
  chain inside wasm memory, and `ff_f32` a whole feed-forward. That turned GELU
  from 29% of runtime into part of the matmul.

`exp` in the kernel is range reduction plus a degree-6 series, good to about
1e-7 relative — float32 epsilon. Agreement with PyTorch is unchanged by any of
this: still ~1e-5 on the logits, on both kernels.

Rebuild with `./wasm/build.sh` (needs clang with the wasm32 target). The
committed `kernels.wasm` is 10 KB, so most people never will.

### What is left

- **Batching helps less than it looks.** Eight sequences cost about five times
  one, not one: the encoder is shared, but decode work is genuinely per-sample.
  There is no more amortisation to find there — only cheaper steps.
- **Threads.** Nothing here uses more than one core. A `SharedArrayBuffer`
  worker pool would scale nearly linearly over a batch, since samples are
  independent, but it needs COOP/COEP headers that GitHub Pages does not send.
- **The rest of the encoder** is three `[L·K, 128] × [128, 128]` products per
  layer, which no rearrangement removes.

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
