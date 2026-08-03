# Future directions

Everything known-open, with the evidence for each. Ordered roughly by how much
it matters. Numbers here were measured on this machine unless marked otherwise;
where a claim is *not* verified it says so, because an earlier version of this
codebase carried a performance claim nobody had measured and it was wrong by an
order of magnitude.

---

## 1. Defects

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

### 2.1 NA-MPNN's edge matmul is 92% structural zeros

`naEdgeFeatures` builds a 5200-wide input and multiplies densely, but for a
protein–protein edge only `16 + 5·5·16 = 416` of those columns are ever
written. Measured at L = 1000, K = 32, all protein: the `linear` call issues
42.6 GFLOP in 3.08 s — 13.8 GFLOP/s, of which **8% is useful**, i.e. 1.1 useful
GFLOP/s. It also stages ~668 MB of mostly-zero scratch into WASM memory per
encode.

**Fix:** the live column set is a pure function of the two endpoints' 18-bit
atom masks, and a real structure has only a handful of distinct masks
(protein-complete, DNA, RNA, plus a few with a missing atom). Bucket each
chunk's edges by `(maskPattern(i), maskPattern(j))`, build one column-compacted
copy of `edge_embedding.weight` per bucket (cache it on the model — it is
weight-derived, not input-derived), and call the existing `linear` at the
compacted width.

This is **bit-identical**, not merely within tolerance: every RBF block is 16
columns and the position prefix is 16, so every dropped run has length ≡ 0
(mod 4), and `linear_f32` assigns lane `l` the terms with `k ≡ l (mod 4)` —
compaction leaves each lane's addend sequence unchanged except for removing
`acc + 0.0f*w`, an exact no-op for finite `w`.

Expected: protein–protein edges back to 416 columns (today's ProteinMPNN cost),
worst case RNA–RNA 2720, and staging down to ~53 MB.

### 2.2 Ideas not yet investigated

- The decoder's `DEFAULT_PREP_BUDGET` fallback is 2.5× slower and triggers
  silently past its ceiling. It logs `prepSkippedBytes` but nothing surfaces it.
- Multi-threading via `SharedArrayBuffer` + several workers. Needs COOP/COEP
  headers, which GitHub Pages does not send — likely a non-starter as hosted.

**Not worth doing** (measured, negative or negligible):
`scratch.fill(0)` in `naEdgeFeatures` is 27 ms of 3541 (0.8%);
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
6. **Redundant text.** Every colour option repeats "colour:"; `#score-hint`
   reprints a full mode description plus paste instructions on every change.
7. **Stale examples.** 1UBQ, 1STP, 4KT0, 1BL8 — none nucleic, though 4OQU is
   already in `assets/`. 4KT0 and 1BL8 only make sense with particular models.

Biggest win for least risk: 1–3 together, since they are one idea (a single
status/progress area instead of five scattered ones).

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

---

## 6. Notes for whoever picks this up

- **`weights/` is the fp16 build the page ships.** The parity tests refuse to
  run on it and say so. Build float32 into a scratch directory first:
  `python tools/convert_weights.py --src <ckpts> --out /tmp/w32 --dtype float32`.
  fp16's ~5e-4 relative error swamps every tolerance in the suite and reads as a
  broken port.
- **`app/trace3d.js` and `app/sec.js` are vendored from CIRPIN-web.** Two
  deliberate modifications, both documented in the README: `setPaper()`, and the
  nucleic-acid block. Keep additions opt-in per layer so a layer that sets no
  flags renders byte-identically — that property is what made the nucleic change
  safe to verify by screenshot hash.
- **One deliberate divergence from the reference** remains: a tied group mixing
  fixed and designed positions. The reference reassigns `S_t` inside its group
  loop so a fixed member leaks its native residue into the others; this engine
  samples one identity per group. Documented in `test/sampling.mjs`.
- §1 is done. What is left of it is the memory *headroom* noted under §1.1, not
  a defect: nothing crashes, but three buffers still grow with the pair count.
