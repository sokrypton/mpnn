# mpnn.web

ProteinMPNN, SolubleMPNN, LigandMPNN, MembraneMPNN and NA-MPNN running entirely
in a browser tab. No server, no upload, no build step — open `index.html` and the
weights come down once, then every forward pass happens on the visitor's
machine.

**[Live page](https://sokrypton.github.io/mpnn/)** · works offline after the
first load.

---

## What it does

- **Load a structure** by PDB ID, UniProt accession (AlphaFold DB), file drop,
  or drag-and-drop. PDB and mmCIF both parse, including modified residues that
  the file declares via `MODRES` or `_chem_comp` — without which a
  pseudouridine drops out of an RNA chain and the graph bridges the hole.
- **Pick any of 15 checkpoints** — the four ProteinMPNN and four SolubleMPNN
  noise levels, four LigandMPNN variants, the two membrane models, and NA-MPNN
  for RNA and protein–DNA.
- **Choose what to design** by clicking residues in the 3D view or the sequence
  track, shift-dragging a box, toggling whole chains, or selecting everything
  within 6 Å of a ligand.
- **Design** with a temperature, a batch size, per-amino-acid bias and
  omissions, tied/symmetric positions, and a seed that makes a run reproducible.
  Bias can be global or scoped to the residues you have selected.
- **Score a sequence** you paste in, per position and averaged, by
  pseudo-likelihood or true autoregressive likelihood (see *Conditioning*).
- **Feed LigandMPNN the side chains** of the residues you are not designing,
  as extra atom context. Heteroatoms are drawn only for LigandMPNN, because it
  is the only family whose encoder reads them.
- **Tie a homo-oligomer** with one checkbox, or tie arbitrary positions by hand.
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

There are two modifications. `setPaper()` at the bottom of `trace3d.js`, plus
`PAPER_CSS` becoming `let` so it can follow. `shade()` expresses depth by
blending toward the page background and returning an *opaque* colour, which only
works if it knows what that background is; CIRPIN-web is a light page with a
fixed paper colour and this one has a dark mode. `diff` against upstream shows
exactly that one line changed.

And a nucleic-acid block, for NA-MPNN. A nucleotide trace steps along C1',
which sits 5.5-6.5 Å from its neighbour where a C-alpha sits 3.8 Å — past the
5 Å chain-break threshold, so a nucleic chain was split into one run per
residue and drew *nothing*. Measured on 4oqu, 94 of 96 steps. A layer may now
carry a `nucleic` flag array, which widens the break allowance to 8 Å for any
step touching a nucleotide and thickens the tube from 0.27 Å — right for a
protein loop, invisible beside a duplex — to 1.0 Å. `viewer.js` also forces
those positions to coil, because `make_sec` reads C1' spacing as helix. A layer
that sets no `nucleic` renders exactly as before: verified byte-identical
screenshots for ubiquitin and for streptavidin + biotin.

Everything else lives in `app/viewer.js`, which is an adapter: it decides
colours, draws the ligand as ball-and-stick (bonds inferred at py2Dmol's 2 Å
heavy-atom cutoff, since CONECT records are usually absent for the ligands that
matter) and hit-tests residues (the renderer has no picking,
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
  na.js               NA-MPNN: 33 letters, 18 atom slots, all-pairs edges
  pdb.js              PDB + mmCIF parsing
  weights.js          .mpnn file reader
  accel.js            optional WebAssembly SIMD accelerator
wasm/kernels.c      the SIMD kernels; build.sh rebuilds kernels.wasm
weights/            15 converted checkpoints (~60 MB total, fp16)
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

| Mode | Cost |
| --- | --- |
| `AR.NONE` — nobody sees any sequence | 1 decoder pass |
| `AR.ORDER` — the usual triangular mask from a decoding order | 1 pass |
| `AR.ALL_BUT_SELF` — pseudo-likelihood, `ar_mask = 1 - I` | 1 pass |
| `profile({exact: true})` — each position decoded last in its own pass | L passes |

### Scoring a sequence

The default is `AR.ALL_BUT_SELF`: every position scored as if it were decoded
last, all in one pass. This is what the [ProteinMPNN-in-JAX
notebook](https://github.com/sokrypton/ColabDesign/blob/main/mpnn/examples/proteinmpnn_in_jax.ipynb)
calls `conditional`, and `score()` here takes the same `ar_mask` argument
ColabDesign's does.

The usual objection is that the mask is cyclic — `i` tells `j` its identity at
layer 0 and `j` hands it back at layer 1, so across three decoder layers a
residue partly sees itself. That is true. But the L-pass alternative is not a
clean reference point either: putting position `t` last leaves the order of the
other L−1 free, and the decoder is not invariant to it, because `t` reads its
neighbours' layer-1 states and those depend on *their* masks. So the "exact"
profile is a family of answers.

`test/pseudolikelihood.mjs` measures both spreads on the same structure:

```
assets/1ubq.pdb  proteinmpnn_v_48_020  L=76
    mean nll     one pass 1.3280  L passes 1.2973 1.3053 1.3047  backbone 1.3846
    L vs L       mae 0.0449  max 0.329  argmax 93.0%
    1 vs L       mae 0.1015  argmax 91.7%  ratio 2.3x the L-vs-L spread
    cost         218 ms vs 6377 ms (29x)

assets/1stp.pdb  ligandmpnn_v_32_010_25  L=121
    mean nll     one pass 1.3075  L passes 1.2957 1.2847 1.2895  backbone 1.3723
    L vs L       mae 0.0448  max 0.568  argmax 97.2%
    1 vs L       mae 0.1146  argmax 96.4%  ratio 2.6x the L-vs-L spread
    cost         175 ms vs 11486 ms (66x)
```

Two L-pass profiles that differ only in the arbitrary part disagree by 0.045
nats per position. The single pass sits 0.10–0.11 nats from either — the same
kind of number, about 2.5×, for 1/L of the work. It errs *high* (1.328 against
1.297), so it does not flatter a sequence. Those figures were cross-checked
against PyTorch and agree to four decimals.

`AR.ORDER` averaged over random orders is still offered, because it is a
genuinely different quantity: the true autoregressive likelihood, and the one
the sampler reports for its own designs.

> The reference implementation has its `single_aa_score` flags backwards.
> `--use_sequence 1`, the default and documented as
> `p(AA_i | backbone, AA_{all except i})`, builds `order_mask` with a zero at
> the target, which sorts the target to the *front* of the decoding order — so
> it returns the backbone-only logits, bit for bit (verified: max difference
> 0.0 against an `ar_mask = 0` pass). `--use_sequence 0`, documented as
> "backbone info only", is the one that conditions on the rest. This port does
> not reproduce that.

## Side-chain context

LigandMPNN's `--ligand_mpnn_use_side_chain_context`, on the *Model* panel. Each
residue takes the 32 side-chain slots of its 16 nearest backbone neighbours,
drops any belonging to a residue being designed -- that side chain is about to
change -- appends the ligand atoms it already selected, and keeps the M nearest
to C-beta. It is worth what it costs on a binding site: redesigning the 14
residues around biotin in 1STP, mean NLL over those positions goes 1.185 to
1.064, and in the page's own smoke test 0.504 to 0.370.

Two things follow from the design that are easy to trip over.

**The encoder now depends on the selection.** Everywhere else `chainMask` is a
sampling-time argument and one encoding serves every run; here it is an encoder
input, so clicking a residue re-encodes. The page does that automatically and
says so in the hint -- on a trailing 350 ms debounce, so refining a selection
posts one encode rather than one per click, and Design flushes the debounce
before it runs. With *nothing* fixed the result is bit-identical to leaving the
flag off, which is the natural check and the one the browser test makes.

**Side chains crowd out the ligand.** The context is capped at M atoms, and
side-chain atoms are much closer to C-beta than the ligand is. On 1STP with
everything fixed, the average residue keeps 1.2 of biotin's 16 atoms in context,
against 16 with the flag off -- and the score gets *worse*, 1.31 to 1.84. So
this is a tool for redesigning a pocket inside a fixed scaffold, not something
to leave on.

## Homo-oligomer tying

One checkbox on the *Design* panel, LigandMPNN's `--homo_oligomer`. Tied
positions are decoded in the same step and their logits averaged (weight
1/chains), so every copy comes out with the same sequence.

Like the reference, chains are matched by residue *number* rather than by
position, so a complex whose chains share a numbering ties correctly even when
one of them has a gap. Unlike the reference, chains with no numbers in common
but equal length fall back to tying end to end rather than raising a KeyError,
and the panel says which of the two ran — they disagree precisely when it
matters.

## NA-MPNN

[NA-MPNN](https://github.com/baker-laboratory/NA-MPNN) (MIT,
[preprint](https://www.biorxiv.org/content/10.1101/2025.10.03.679414v2)) designs
RNA and protein–DNA complexes. Selecting it re-parses the structure, because
nucleic acids become model positions rather than ligand context — which changes
the length, the selection and the alphabet.

It is the same network: 3 encoder and 3 decoder layers at hidden 128, the same
message scale, the same autoregressive decoder. `layers.js`, the whole decoding
half of `model.js` and every WASM kernel are reused untouched. Three things
differ, all in `mpnn/na.js`.

**Eighteen atom slots, all pairs.** N/CA/C/O for protein and
`OP1 OP2 P O5' C5' C4' O4' C3' O3' C2' O2' C1'` for a nucleotide, plus the
familiar virtual C-beta and a nucleic-acid pseudo-N placed from O4'/C1'/C2' by
the same construction with its own coefficients. Edge features are all 324
ordered pairs, so the edge embedding takes 5200 inputs against 416. Each block
is masked by both endpoints, though, so a protein–protein edge fills 25 of the
324 and a nucleotide–nucleotide edge 169; only the live blocks are written.
Neighbours are found on `CA + C1'`, which works because the two are disjoint.

Which blocks are live depends only on the two endpoints' atom masks, and a real
structure has a handful of distinct masks, so edges are bucketed by that pair
and each bucket multiplies a **column-compacted copy of the weight** — 416
columns for a protein–protein edge, 2720 for the RNA–RNA worst case, instead of
5200 for everything. On 6VXX (L = 2916, all protein) that is 124.2 GFLOP and
1941 MB of staged scratch down to 9.9 GFLOP and 155 MB, and the featuriser from
5.13 s to 1.40 s. It is bit-identical, not merely close: every dropped run is a
whole 16-column RBF block, so each surviving column keeps its index mod 4 and
therefore its SIMD lane, and all a lane loses is `acc + 0*w` steps.

**Polymer-type nodes**, a 6-class one-hot — the same shape as the membrane
models' per-residue label.

**Thirty-three letters.** Note the amino acids are in three-letter alphabetical
order (`ARNDCQEGHILKMFPSTWYV`), *not* MPNN's `ACDEFGHIKLMNPQRSTVWY`, so `W_s`
and `W_out` rows do not line up with the other models'. Lower case is
nucleotide. By default `--na_shared_tokens` stores an RNA base as the
corresponding DNA token and omits the legacy RNA letters, so a uracil is held
as DT and converted back for display using the presence of an O2'.

`test/na.mjs` checks it against the real PyTorch model in two halves — the model
maths from dumped tensors, and separately that the JS parser reproduces those
tensors from the same PDB — on a pure-RNA structure (4oqu, 97 nt) and a
protein–DNA complex (1am9, 313 aa + 72 nt + 4 unknown). Both match; the parser
agrees with ProDy on residue count, polymer type, sequence, numbering and
coordinates exactly.

Two notes. Encoding costs about 2x ProteinMPNN at the same length (2.5 s at
L = 389) and scales worse: at L = 1000 the edge-embedding matmul alone is 3.1 s,
running at 13.8 GFLOP/s of which only 8% is useful work — the other 92% is
multiplying the structural zeros left by the masked-out blocks. Bucketing edges
by atom-mask pattern and compacting the weight matrix per bucket would fix it
and is bit-identical (every dropped run is a multiple of four columns, so each
SIMD lane's addend sequence is unchanged). And 1am9 has four residues with no complete backbone
of any kind, whose `D_adjust` row is all zeros — the reference's `topk` breaks
that tie arbitrarily, so the neighbour-graph assertion covers unmasked rows and
reports the rest.

## Correctness

`test/parity.mjs` checks the port against golden tensors produced by the real
PyTorch model on the same inputs (`tools/make_reference.py`). Point it at a
**float32** weights directory — `weights/` here is the float16 build the page
downloads, and fp16's ~5e-4 relative error swamps every tolerance below and
reads as a broken port. Rerun `tools/convert_weights.py --dtype float32` into a
scratch directory for testing; the test refuses to run on float16 rather than
failing mysteriously. Current state, on ubiquitin and on streptavidin + biotin:

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

`test/pseudolikelihood.mjs` backs the numbers in *Scoring a sequence* above,
and asserts the properties they imply rather than the digits themselves.

`test/sampling.mjs` covers what logits cannot reach — per-amino-acid bias,
omissions, fixed residues, tied positions and the ligand-context switch all live
in the sampler, which is stochastic. Both sides are pinned the same way: an
explicit decoding order, and a temperature of 1e-6 that collapses the draw onto
the argmax. The sequences then have to match *exactly*.

### One deliberate divergence

A tied group containing both fixed and designed positions is the single place
this engine does not match the reference, on purpose.

The reference reassigns its running `S_t` inside the per-group loop, so a fixed
member overwrites it and every designed member after it in the list inherits the
*fixed member's input residue* rather than the sampled one. On ubiquitin with
group `[7, 27, 47, 60]` and 27 fixed, it emits position 7 = E and position 47 =
A, where A is position 27's native residue. The group is not tied at all, and
which member wins depends on the order they were listed in.

This engine samples one identity per group and applies it to the designed
members, leaving fixed ones alone. The test asserts that invariant, and reports
the reference's drift rather than reproducing it. Everything else matches
exactly, including the decoding-order construction — which does reproduce the
reference's ordering rules, fixed residues pulled to the front and tied groups
pulled forward to their first member.

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
| encode | **0.22 s** (was 1.4) | **0.60 s** (was 13.3) |
| sample | **59 ms/seq** (was 1014) | **60 ms/seq** (was 1113) |
| profile | **0.13 s** (was 1.04) | **0.09 s** (was 1.12) |

Across sizes, ProteinMPNN, single thread, sampling as ms per sequence at batch 8:

| structure | L | encode | sample | profile | peak RSS |
| --- | --- | --- | --- | --- | --- |
| 1UBQ ubiquitin | 76 | 0.22 s | 60 ms | 0.12 s | 118 MB |
| 1STP + biotin (LigandMPNN) | 121 | 0.60 s | 60 ms | 0.08 s | 191 MB |
| 1BL8 K⁺ channel, 4 chains | 388 | 0.93 s | 0.29 s | 0.34 s | 193 MB |
| 4HHB haemoglobin, 4 chains | 574 | 1.25 s | 0.41 s | 0.43 s | 266 MB |
| 6VXX spike trimer | 2916 | 6.54 s | 2.09 s | 2.04 s | 760 MB |
| 1AON GroEL/GroES, 21 chains | 8015 | 17.4 s | — | — | ~2 GB |

Scaling is linear in L. Past ~3000 residues memory bites before arithmetic
does: the encoder's edge tensor is L·K·128 floats and the cached decoder
projection three times that, so a browser tab around 8000 residues is close to
the practical limit.

### Compared with the reference on CPU

Same structures, same inputs, PyTorch 2.13 on the same machine
(`tools/bench_reference.py`). Both sides single threaded — this engine has no
choice, and PyTorch is pinned with `torch.set_num_threads(1)` plus
`OMP_NUM_THREADS=1`. Seconds; sampling is per sequence at batch 8. "Score" is
one teacher-forced decoder pass over every position, which is the same work as
this engine's `profile`.

| | | this engine | PyTorch | |
| --- | --- | --- | --- | --- |
| L = 76 | encode | 0.22 | **0.09** | 2.4x slower |
| | sample | **0.060** | 0.08 | 1.3x faster |
| | score | 0.12 | 0.12 | equal |
| L = 121, ligand | encode | **0.60** | 0.72 | 1.2x faster |
| | sample | **0.060** | 0.16 | 2.7x faster |
| | score | **0.08** | 0.78 | 9.8x faster |
| L = 388 | encode | 0.93 | **0.53** | 1.8x slower |
| | sample | **0.29** | 0.41 | 1.4x faster |
| | score | **0.34** | 0.77 | 2.3x faster |
| L = 574 | encode | 1.25 | **0.89** | 1.4x slower |
| | sample | **0.41** | 0.59 | 1.4x faster |
| | score | **0.43** | 1.38 | 3.2x faster |
| L = 2916 | encode | **6.54** | 10.98 | 1.7x faster |
| | sample | **2.09** | 6.67 | 3.2x faster |
| | score | **2.04** | 15.20 | 7.5x faster |
| L = 8015 | encode | **17.4** | 74.8 | 4.3x faster |

On one thread this engine is **faster than the reference at everything except
encoding small proteins**. Encoding is 2.4x behind at 76 residues, crosses over
somewhere around a thousand, and is 4.3x ahead by eight thousand. Sampling and
scoring are ahead at every size measured.

Three separate things produce that shape.

**Encoding at small L** is the one real deficit, and it is the kernel: a dense
sweep where AVX2 gives oneDNN 8 lanes with a fused multiply-add against wasm's 4
lanes with neither. The matmul here already runs at ~90% of what that
instruction set allows, so the remaining ~1.5x is hardware.

**Encoding at large L** inverts because the reference's featuriser builds full
`[L, L]` distance matrices for each of 25 atom pairs, which is O(L²), while this
one touches only the K neighbours it kept and finds them with a grid, which is
O(L·K). PyTorch's encode grows 8.3x from L = 2916 to L = 8015; this one grows
2.7x.

**Sampling and scoring** are ahead everywhere because they are not throughput
problems. The reference walks L autoregressive steps in Python and pays
interpreter and dispatch overhead on each; the decoder here has no per-edge
matmul left in that loop. The 9.8x on LigandMPNN's score is the atom-pair
deduplication on top of that.

For the record, PyTorch on four threads is 3-4x faster than PyTorch on one, so
it stays ahead on encode until about L = 8000. Threads here would be worth the
same factor and are the obvious next step.

### Why encoding is still slower at small and medium L

Two reasons, in proportion. Measured at L = 574: 1.15 s inside the kernel and
0.57 s outside it.

**The kernel gap is hardware, and it is nearly exhausted.** In situ the matmul
runs at 18-19 GFLOP/s, about 85% of what wasm SIMD can reach — 128-bit vectors,
4 lanes, and *no fused multiply-add*, which caps a 3 GHz core near 24. AVX2
gives oneDNN 8 lanes with FMA, so its ceiling is 2-4x higher, and it achieves
~23 GFLOP/s on the same work. Roughly a 1.5x gap that no amount of tuning
closes; relaxed-SIMD FMA, the one instruction that would help, measured 5-8%
because the loop is load-bound rather than FLOP-bound.

**The rest is JavaScript, and it is shrinking.** 28% of the encode at L = 2916,
down from 37%. Two of the three items named here have been dealt with:

- the neighbour search was O(L²·K); it is now a uniform bucket grid, **22x
  faster** at L = 2916 and gone from the profile's top ten. It has its own test
  (`test/neighbors.mjs`) asserting the graph is *edge-for-edge identical* to the
  exact sweep, because everything downstream is indexed by it. That test caught
  a real bug: the exact sweep buffers its row in a `Float32Array`, so ties are
  decided at float32 precision, and comparing doubles reordered a couple of
  edges in every few thousand
- the masked neighbour sum, both residual LayerNorms, the feed-forward and the
  output mask now ride inside `message_block_f32` and `edge_block_f32`, one
  accelerator call each for a whole layer half, so the intermediates never
  cross the boundary

- the radial basis evaluation followed, into `edge_features_f32`. It was the
  largest remaining JavaScript item at every size — 400 `exp` calls per edge, 56
  million on a spike trimer — and it is fused with the 416-wide projection and
  LayerNorm that consume it, so the encoder's widest intermediate never crosses
  the boundary. ~12% off encode

Encode is now ~75% inside the kernel, and what remains there is at the hardware
ceiling. **Threads are the only multiplicative lever left**, and they are worth
most exactly where this is weakest: a worker pool would take L = 76 encode from
0.22 s to roughly 0.06 s, past single-threaded PyTorch and level with four
threads. Sampling parallelises without any shared memory at all, since the
samples are independent and each worker needs only a copy of the encoding.

Between 5x and 17x depending on what you ask for. Five things got it there, and
the order matters: every algorithmic change came before any kernel work, because
a faster kernel only makes the remaining multiply-adds cheaper.

**0. Denormals.** Worth stating first because it was invisible and cost more
than anything else at scale. The radial basis features are `exp(-z²)`, which
underflows past float32's smallest normal for atom pairs far from a basis
centre: about 4% of a real edge-feature block lands in the denormal range.
Denormal arithmetic traps to microcode on x86, and wasm — unlike every native
BLAS, which quietly sets flush-to-zero — is required to honour it. That single
detail ran the edge-embedding matmul at **0.9 GFLOP/s instead of 20.6**, and it
is most of why the reference looked so far ahead at encoding. Flushing values
below 1e-30 in `rbfInto` costs nothing measurable in accuracy and took the
2916-residue encode from 20.7 s to 8.9 s.

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

The distinct-pair count is a property of the structure, not of the model, and
side-chain context makes it large — every fixed residue's atoms join the pool,
730,805 pairs on 6VXX with all 2916 residues fixed. So both atom layers run
their message MLP over a fixed 8192-row window (`PAIR_CHUNK`) rather than over
one row per pair; without that, staging the first layer's pre-activation asked
for 1.1 GB and the encode died on `WebAssembly.Memory.grow`.

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

**Gathers, not one-hot products.** ColabDesign turns every embedding lookup
into a dense `one_hot @ W`, primarily so a *soft* sequence can flow through it
under a gradient, and secondarily because that suits a TPU. Neither holds here —
inference only, one wasm thread — so this engine goes the other way. LigandMPNN's
atom-type one-hot is 147 wide with three non-zeros, and reading three columns
beats multiplying by it 5x; deduplicating by element first makes it 26x. Details
and the reverse case in [`wasm/bench/`](wasm/bench/).

**Why C and not Rust?** Only that clang and wasm-ld ship with LLVM and were
already there, so the build is one command with no package manager and no extra
target to install. It is not a performance argument:
[`wasm/bench/`](wasm/bench/) builds the same matmul five ways and Rust with
`core::arch::wasm32` intrinsics measures **identical** to the C — same LLVM,
same emitted code. What actually matters is writing the intrinsics at all.
Idiomatic safe Rust manages 3-4 GFLOP/s against the hand-written 20+, and so
does the equivalent plain C loop, because LLVM will not reassociate a float
reduction without being asked. Switching languages is free; switching away from
explicit SIMD costs 5-6x.

### What is left

**Would porting the whole engine to wasm help?** Not much, and it was measured
rather than assumed. After the above, a profile splits as **71% wasm, 25% JS,
4% runtime**, so moving *all* remaining JS into wasm caps at **1.34x**, and a
realistic 3x on that glue is **1.20x**. The usual argument for going all-in —
staging copies at the JS/wasm boundary — does not apply here: they measure
**0.6%**, because the arrays that cross are small next to the arithmetic done
on them. Against that, the JS would stop being the reference implementation
that makes the parity story checkable, and changing a line of model code would
start requiring a C toolchain. The coarse-entry-point design gets most of the
benefit for none of that.

Still open, roughly in order of payoff:

- **Threads.** Nothing here uses more than one core, and samples are completely
  independent. A worker pool would scale nearly linearly over a batch — a much
  bigger win than the 1.2x a full wasm port would buy. It does *not* need
  `SharedArrayBuffer` or COOP/COEP: each worker can be handed a structured
  clone of the encoding and asked for its share of the sequences.
- **Batching still helps less than it looks.** Eight sequences cost about 2.7x
  one, not 1x: the encoder is shared, but decode work is genuinely per-sample.
- **The rest of the encoder** is three `[L·K, 128] × [128, 128]` products per
  layer, which no rearrangement removes — only more cores or a better kernel.

## Weights

`weights/` holds all 13 checkpoints converted to a flat binary format, fp16, 3.3
to 5.3 MB each and about 55 MB in total. Only one is downloaded at a time, and
the browser caches it. The format is a magic number, a JSON header describing
every tensor, and the tensors concatenated — see `tools/convert_weights.py`,
which regenerates them from the upstream `.pt` files. CI checks the committed
files still match.

`ligandmpnn_sc_v_32_002_16` (the sidechain packer) is deliberately not
converted; nothing here runs it.

## What is not done

[`FUTURE.md`](FUTURE.md) lists everything known-open with the evidence for each
— a crash on any structure with fewer residues than K neighbours, what memory
side-chain context still holds proportional to the structure, side-chain
packing and NA-MPNN's specificity mode, and a pass of UI cleanup.

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
