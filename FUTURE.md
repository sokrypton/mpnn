# Future directions

Everything known-open, with the evidence for each. Ordered roughly by how much
it matters. Numbers here were measured on this machine unless marked otherwise;
where a claim is *not* verified it says so, because an earlier version of this
codebase carried a performance claim nobody had measured and it was wrong by an
order of magnitude.

---

## 1. Defects

### 1.0 Any structure with fewer residues than K neighbours crashes — *fixed*

**Was:** 1BNA, a 24-base-pair DNA duplex — L = 24 against NA-MPNN's K = 32 —
or 1L2Y, the 20-residue Trp-cage, under any model:

```
RangeError: offset is out of bounds
    at Accelerator.edgeBlock (mpnn/accel.js:239)
    at mpnn/layers.js:160  (encoder edge update)
```

`neighborGraph` already clamps to `min(K, L)` and returns what it used,
matching the reference's `topk(..., min(top_k, L))`. But `makeEncoderLayer`
closed over the *checkpoint's* K and indexed `hE` with it, so every layer read
past the end of a graph narrower than the model nominally asks for. K is a
property of the graph, not of the layer, so it is now an argument to the call
like L. The decoder already took it from the encoding and was never wrong.

Long-standing, not a regression, and it survived because nothing in the suite
went below L = 76 — `test/small.mjs` now covers L = 2 to 49 across four
checkpoints on both kernels, and fails 23 ways against the old engine. Verified
in the page too: 1L2Y designs, scores and profiles.

Structures at L ≥ K are untouched, by sha256 over `hV`, `hE` and `mask` on
1STP (half- and all-fixed side chains) and 3HDD on the JS kernel.

### 1.1 Side-chain context runs out of WASM memory on a large complex — *fixed*

**Was:** encoding 6VXX (L = 2916) with LigandMPNN, `useSideChains: true` and
`chainMask` all zeros — every residue fixed, so every side chain contributes —
died after 4.7 s with

```
RangeError: WebAssembly.Memory.grow(): Maximum memory size exceeded
```

It needed the worst case; half the residues fixed encoded fine either way.
The overflowing call was the accelerator's `tail2` on `pairH1`: 730805 ordered
pairs at that selection, so 374 MB of pre-activation and three staged copies of
it, 1.12 GB, on top of the 383 MB already resident.

`pairH1`'s rows are independent, so the fix is a fixed 8192-row window over
them (`PAIR_CHUNK`), each batch written straight into its slice of the layer-0
message. Same arithmetic in the same order on the same rows: encoder output is
**bit-identical**, checked by sha256 over `hV`, `hE` and `mask` before and
after on 1STP all-fixed (WASM and JS kernels), 1UBQ all-fixed, and 6VXX
half-fixed. The worst case now encodes in 14.2 s in node and 86.4 s in the
page.

**Still scaled by the structure**, and the reason this is not the clean
"bounded by the chunk size" the section originally claimed — arena at 6VXX
all-fixed, 978 MB total:

| slot | size | keyed on |
| --- | --- | --- |
| `lig.pairMsg0` | 374 MB | ordered pairs (730805) |
| `lig.edgePart` | 189 MB | unordered pairs (373k) |
| `lig.pairEdge1` | 189 MB | unordered pairs |

`pairMsg0` is the one worth taking next. It is consumed exactly once, by the
layer-0 gather loop, so it does not have to exist globally: dedup pairs within
a block of residues instead of across the whole structure, and hold only that
block's messages. Blocking preserves the per-row summation order, so it stays
bit-identical, and the extra tail work is small — measured on 6VXX all-fixed,
a 256-residue block needs 734463 tail rows against the global 730805, **1.01×**
(64 residues is 1.19×, and 1024 is 0.86× because unused pairs never get
computed at all). That caps the block at 256 × 25² = 160000 pairs, 82 MB.
It does not touch the other two, which are keyed on the unordered pair and
would need the same treatment separately.

### 1.2 The selection re-encode is not debounced — *fixed*

Was: with side-chain context on, `refreshSelection()` posted a full encode on
every residue click. Measured in the page on 1STP with
`ligandmpnn_v_32_020_25`, side chains on: six clicks 30 ms apart posted six
encodes, and the last one landed 14.91 s after it was asked for. Now one encode,
against the final selection, and Design flushes the pending debounce rather than
racing it.

The fix is a trailing 350 ms debounce plus a promise chain that serialises what
does get posted, both on the page side — not the worker-side generation counter
this section originally proposed. For the worker to drop a superseded encode it
would have to see the newer generation while blocked inside `Model.encode`,
which means shared memory, and `SharedArrayBuffer` needs the COOP/COEP headers
GitHub Pages does not send (§2.2). Coalescing before the post costs nothing and
needs no headers.

---

## 2. Performance

### 2.1 NA-MPNN's edge matmul is 92% structural zeros — *fixed*

`naEdgeFeatures` built a 5200-wide input and multiplied densely, though for a
protein–protein edge only `16 + 5·5·16 = 416` of those columns are ever
written. Edges are now bucketed by the pair of 18-bit atom-mask patterns at
their endpoints and each bucket multiplies a column-compacted copy of
`edge_embedding.weight`, cached against the weight array.

Measured on `naEdgeFeatures` alone, K = 32:

| | before | after |
| --- | --- | --- |
| 6VXX, L = 2916, all protein | 5.13 s, 124.2 GFLOP, 1941 MB staged | **1.40 s**, 9.9 GFLOP, 155 MB |
| 3HDD, L = 153, protein + DNA | 0.27 s, 6.5 GFLOP, 102 MB | **0.13 s**, 1.0 GFLOP, 16 MB |
| 4OQU, L = 97, RNA | 0.23 s, 4.1 GFLOP, 65 MB | 0.23 s, 2.2 GFLOP, 34 MB |

The FLOP counts fall by 12.5× on protein and the staging with them. Wall clock
does not follow all the way: throughput drops from ~24 GFLOP/s to ~7, because
what is left is a smaller matmul against the same scalar RBF fill, and on
RNA — 13×13 live blocks, 2720 of 5200 columns — the fill now dominates
entirely and the time does not move at all. Whole-encode: 3HDD 0.42 → 0.20 s
on the WASM kernel and 2.56 → 1.23 s on the JS one.

**Bit-identical**, and checked rather than only argued: sha256 over `hV`, `hE`
and `mask` matches before and after on 3HDD, 4OQU, 1UBQ and 1STP, on both
kernels. The argument is in `compactedEdgeWeight` — every dropped run is a
whole 16-column RBF block, so each surviving column keeps its index mod 4 and
so its SIMD lane, and what a lane loses is `acc + 0*w` steps.

The next thing in this path is the RBF fill, not the matmul.

### 2.2 Ideas not yet investigated

- **There is no one owner for "derived once from the weights".** `naEdgeFeatures`
  caches its column-compacted copies of `edge_embedding.weight` in a module-level
  `WeakMap` keyed on the weight array. That works, but `Arena` is already the one
  owner of per-call scratch and `Weights` owns the tensors, so this is a third
  convention. §2.1 says the same compaction trick wants applying elsewhere; the
  second user is the point at which a keyed memo slot on `Weights` or `Model`
  earns itself, and inventing it for one user would not.
- **`redraw()` runs on every raw pointer move**, with no `requestAnimationFrame`
  gate, so a rotate drag redraws the structure, the logo and the sequence track
  once per event rather than once per frame. Pre-existing. The obvious cheap
  fixes went in with the cleanup — the selection model and the ligand geometry
  are both cached now rather than rebuilt per event — but the gate itself is
  still missing and nobody has measured what it costs.


- The decoder's `DEFAULT_PREP_BUDGET` fallback is 2.5× slower and triggers
  silently past its ceiling. It logs `prepSkippedBytes` but nothing surfaces it.
- Multi-threading via `SharedArrayBuffer` + several workers. Needs COOP/COEP
  headers, which GitHub Pages does not send — likely a non-starter as hosted.

**Not worth doing** (measured, negative or negligible):
`naEdgeFeatures`'s `scratch.fill(0)` was 27 ms of 3541 (0.8%) — moot now, the
compaction in §2.1 leaves nothing to zero;
`naBackbone`'s six `pick()` closures are 1–4 ms of a 5.4 s encode; hoisting the
`liveJ` recompute and precomputing the position table were inside run-to-run
variance. Making the *weights* fp16 in RAM saves ~5 MB against ~305 MB of
activations at L = 2916 — the wrong lever, and it would cost accuracy
everywhere.

---

## 3. Features

### 3.1 Side-chain packing (LigandMPNN)

The largest gap. `--pack_side_chains` uses a separate checkpoint,
`ligandmpnn_sc_v_32_002_16.pt`, which `tools/convert_weights.py` still skips by
default. It is a denoising network that emits full-atom coordinates, bringing
`--number_of_packs_per_design`, `--sc_num_denoising_steps`,
`--pack_with_ligand_context` and `--repack_everything`, plus a way to render and
download packed structures. Comparable in size to the NA-MPNN work.

### 3.2 NA-MPNN specificity mode

The second checkpoint (`s_70114.pt`) and `--mode specificity`, which outputs a
PPM rather than a sequence, plus `--design_na_only`. Two-thirds of the machinery
already exists — same architecture, different head usage — so this is mostly a
new output path and a heatmap for reading off predicted DNA base preferences.

### 3.3 NA-MPNN pair bias

`--pair_bias_AA`: a 33×33 bias applied between *neighbouring* positions, e.g. to
discourage KK/KE/EK adjacencies. Small — `make_pair_bias` plus one term in the
sampler.

### 3.4 Smaller reference flags with no equivalent here

- `--parse_these_chains_only`: the engine supports it
  (`structureFromText({chains})`); nothing in the UI exposes it. Note this is
  *not* the same as the chain toggles, which fix a chain but still feed it to
  the graph.
- `--ligand_mpnn_cutoff_for_score`: the reference also reports a score
  restricted to residues within 8 Å of the ligand.
- `--k_neighbors`, `--na_shared_tokens 0`, `--load_residues_with_missing_atoms`:
  non-default knobs. We implement the defaults.

---

## 4. UI

The page shows every affordance at once. `4aa6158` gated the ones that read
absent data; what remains:

1. **The progress bar is in the wrong panel.** `#progress` lives inside the
   Design panel, but Score and Profile both drive it, so clicking Score animates
   a bar somewhere unrelated. One global strip under the header — the worker is
   serial, so there is only ever one job.
2. **Four status lines** (`load`, `model`, `design`, `score`) are permanently on
   screen and usually three are stale. Only one operation can run; they could
   collapse into one area.
3. **The kernel line is permanent but only interesting when it is bad.**
   "Running on the WebAssembly SIMD kernel" never changes. Show it only on the
   JS fallback, which is the ~5×-slower case worth knowing about.
4. **The seed field is inert while "random" is ticked**, which is the default.
   Disable it, or fold the seed control into the advanced `<details>`.
5. **The numbered headings imply a wizard.** "1 · Structure … 4 · Design", then
   Results, Score and Profile unnumbered. Number all or none.
6. **Redundant text.** Every colour option repeats "colour:" and every profile
   view repeats "view:" — the second was added knowingly, to match the first
   rather than leave one bare dropdown next to one labelled one, so fix them
   together. `#score-hint` reprints a full mode description plus paste
   instructions on every change.
7. **Stale examples.** 1UBQ, 1STP, 4KT0, 1BL8 — none nucleic, though 4OQU is
   already in `assets/`. 4KT0 and 1BL8 only make sense with particular models.

Biggest win for least risk: 1–3 together, since they are one idea (a single
status/progress area instead of five scattered ones).

The sequence track and the profile display were reworked after py2Dmol's; the
README has what changed and why. Two things left in that area:

- **The track is upstream's now**, so its gaps are upstream's: see
  `app/viewer-seq.js` and the README. The one thing lost in the swap is this
  page's lower-case convention for nucleotides — `naDisplaySequence` writes
  `acgu`, and `viewer-seq.js` upper-cases them from the residue name, so
  the track and the FASTA now disagree in case. The letters are right and the
  chains are separated and coloured, so nothing is ambiguous; it is a
  consistency wart, and fixing it is a small change in `viewer-seq.js` -- which
  is ours to change.
- **Nothing checks the adapter.** `app/seqview.js` is the only new logic and it
  has no test. The bug it already had — taking the position type from
  `polytype`, which calls a 5'-terminal nucleotide UNK, so the first base of
  each DNA strand typed as protein and the viewer drew a polymer-change spacer
  where the numbering is contiguous — was found by looking at 3HDD, not by
  anything automatic. A test over the frame it builds (types, names, numbering,
  ligand grouping) for 1STP/3HDD/4KT0 would be cheap and needs no weights.
- **The heatmap has no scale.** Cell alpha is `sqrt(p)` and nothing says so, so
  it reads as ordering rather than magnitude. A short legend, or numbers in the
  tooltip, would fix it — the tooltip already reports the top few letters with
  percentages, so the information is one step away.

### 4.1 Rendering

- **Large structures sit off-centre and clip.** `DEFAULT_ZOOM = 1.5` was raised
  so small proteins did not look lost; it over-crops long complexes. Pre-existing
  and not nucleic-specific.
- **No base-pair ladder.** Nucleic acids draw as a backbone tube — fine for
  picking positions, not a duplex representation.

---

## 5. Testing gaps

- **mmCIF + nucleic acids is untested.** The code path is shared with PDB and
  should work, but nothing exercises it.
- **The `_chem_comp` branch of the modified-residue work is unverified.** The
  PDB `MODRES` path was checked against a real case (`U/PSU/OMC/G`: 2/4
  positions before, 4/4 after, and a free PSU 500 Å away correctly *not*
  absorbed). The mmCIF branch was only reasoned through.
- **No test covers the memory ceiling in §1.1.** It was fixed against an
  ad-hoc script, not a checked-in case: 6VXX is 2.2 MB and the encode is 14 s
  in node, so neither the structure nor the runtime fits the existing suite as
  written. A cheaper regression would be a direct assertion that no arena slot
  or staged buffer exceeds a fixed multiple of `PAIR_CHUNK`, which needs no
  large structure at all.
- NA-MPNN scoring is not in the browser test, though the display↔parse
  round-trip was verified exact over all 97 residues of 4OQU.
- `test/small.mjs` covers L = 2 to 49 (§1.0) but only checks that the encode
  runs, that K clamps, and that the letters are in range. Whether a short
  structure agrees *numerically* with the reference is unverified — it needs a
  golden case from `tools/make_reference.py` at L < K.

---

## 6. Notes for whoever picks this up

- **`weights/` is the fp16 build the page ships.** The parity tests refuse to
  run on it and say so. Build float32 into a scratch directory first:
  `python tools/convert_weights.py --src <ckpts> --out /tmp/w32 --dtype float32`.
  fp16's ~5e-4 relative error swamps every tolerance in the suite and reads as a
  broken port.
- **`app/trace3d.js` and `app/sec.js` came from CIRPIN-web, `app/viewer-seq.js`
  and `app/ligandgroups.js` from py2Dmol.** They are this repo's now — refactor
  or rewrite them freely. What is worth preserving is the knowledge in them, not
  their byte order: `trace3d.js` in particular encodes a long list of rendering
  fixes that a from-scratch version reproduced as bugs once already, and each
  one is a comment saying what it was for. Keeping additions opt-in per layer is
  still worth doing where it is cheap, because it lets a change be verified by
  screenshot hash against the untouched path.
- **One deliberate divergence from the reference** remains: a tied group mixing
  fixed and designed positions. The reference reassigns `S_t` inside its group
  loop so a fixed member leaks its native residue into the others; this engine
  samples one identity per group. Documented in `test/sampling.mjs`.
- All of §1 and §2.1 are done. What is left of §1.1 is memory *headroom*, not
  a defect — nothing fails, but three buffers still grow with the pair count.
- **Every "bit-identical" claim in here was checked, not just argued.** The
  method: sha256 the encoder's `hV`, `hE` and `mask` on a spread of structures,
  on both kernels, with the change stashed and unstashed. It costs a few
  minutes and it is the only thing that would have caught a lane-assignment
  mistake in §2.1.
