// Adapter between this page and the py2Dmol sequence viewer in viewer-seq.js.
//
// `viewer-seq.js` talks to a "renderer": an object with
// a handful of properties and methods, and a frame made of plain parallel
// arrays. None of that is py2Dmol-specific once it is written down, so this
// file supplies it from `state` -- the same relationship `viewer.js` has with
// trace3d.js.
//
// Two things need translating rather than passing through.
//
// **Selection means something different.** py2Dmol's `positions` set is "the
// residues you have selected"; here the equivalent is "the residues you are
// designing", which is also an encoder input under side-chain context. So
// `getSelection` reads the design mask and `setSelection` writes it and then
// asks the page to refresh -- which is where the encode debounce lives.
//
// **Heteroatoms are not model positions here.** LigandMPNN reads them as atom
// context, not as things to design, so they are not in `designMask` and there
// is no letter for them. The viewer wants ligands *as positions* with type 'L'
// so it can group them into tokens, so this appends them after the L real
// positions and drops anything past L on the way back. That is what makes the
// upstream ligand grouping and its BTN/CLA/SF4 tokens work unchanged.

import { groupLigandAtoms } from "./ligandgroups.js";

const OBJECT = "structure";

export class SequenceView {
  /**
   * @param {object} deps everything the adapter needs from the page
   * @param {() => object} deps.getState
   * @param {(i: number) => {rgb: number[], dim: number}} deps.colourFor
   * @param {(i: number) => string} deps.nameAt
   * @param {(i: number) => string} deps.typeAt "P", "D" or "R"
   * @param {() => boolean} deps.showLigands whether heteroatoms are in scope
   * @param {(chain: string) => number[]} deps.chainColour
   * @param {() => void} deps.onSelectionChange called after the mask is written
   * @param {() => void} deps.onHoverChange
   */
  constructor(deps) {
    this.deps = deps;
    this.frame = null;
    /** Total display positions: the L residues, then the heteroatoms. */
    this.positions = 0;
    this.ligandGroups = new Map();
    /** py2Dmol reads this off a `<select>`; one structure, so one option. */
    this.objectSelect = { value: OBJECT };

    if (typeof window !== "undefined" && window.SEQ) {
      window.SEQ.setCallbacks({
        getRenderer: () => this,
        getObjectSelect: () => this.objectSelect,
        highlightAtom: (i) => this._hover(i),
        highlightAtoms: (list) => this._hover(list?.length ? list[0] : -1),
        clearHighlight: () => this._hover(-1),
        toggleChainResidues: (chain) => this._toggleChain(chain),
        setChainResiduesSelected: (chain, on) => this._setChain(chain, on),
        getPreviewSelectionSet: () => this._preview,
        setPreviewSelectionSet: (set) => { this._preview = set; },
      });
    }
    this._preview = null;
  }

  // --- what the viewer calls the renderer -----------------------------------

  get currentObjectName() { return OBJECT; }
  get currentFrame() { return 0; }
  get objectsData() {
    return { [OBJECT]: { frames: this.frame ? [this.frame] : [], ligandGroups: this.ligandGroups } };
  }
  /** null means "everything is visible", which is always true here. */
  get visibilityMask() { return null; }
  get coords() { return this.frame?.coords ?? []; }
  get highlightedAtom() { return this.deps.getState().hover; }
  get highlightedAtoms() { return []; }
  get isDragging() { return false; }
  get positionScreenPositions() { return null; }
  /** The 3D highlight is this page's own job, so the overlay draws nothing. */
  get canvas() { return null; }
  get displayWidth() { return 0; }
  get displayHeight() { return 0; }
  getHighlightCoordinates() { return null; }

  /**
   * The selection, in the shape the viewer expects.
   *
   * `chains` and `selectionMode` are what the chain badges in the track are
   * drawn from, and they follow upstream's own rule in `applySelection`: a
   * chain is listed if anything in it is selected, and when *everything* is
   * selected the mode goes to "default" with an empty set. Returning an empty
   * set unconditionally, which is what this did first, left every chain badge
   * looking unselected no matter what.
   */
  /**
   * The selection, in the shape the viewer expects.
   *
   * `chains` and `selectionMode` are what the chain badges in the track are
   * drawn from, and they follow upstream's own rule in `applySelection`: a
   * chain is listed if anything in it is selected, and when *everything* is
   * selected the mode goes to "default" with an empty set.
   *
   * Cached, because the viewer reads `selectionModel` several times per render
   * -- its own comment there says it does so to avoid "the expensive
   * getSelection() copy" -- and the page redraws on every pointer move while
   * rotating, when the selection has not changed at all. Rebuilding two sets
   * over L + heteroatoms each time is exactly the copy that comment is trying
   * to dodge. `invalidate()` is the one way it goes stale.
   */
  getSelection() {
    if (this._selection) return this._selection;
    const state = this.deps.getState();
    const s = state.structure;
    const positions = new Set();
    const chains = new Set();
    if (!s) return { positions, chains, selectionMode: "explicit", paeBoxes: [] };

    for (let i = 0; i < s.L; i++) {
      if (state.designMask[i] > 0) {
        positions.add(i);
        chains.add(s.chainIds[i]);
      }
    }
    // The appended heteroatoms count as selected, always. They are not
    // designable so the mask says nothing about them, but the viewer dims
    // anything outside this set, and a permanently greyed-out ligand token
    // would read as "excluded" rather than "not a position you can design".
    for (let i = s.L; i < this.positions; i++) {
      positions.add(i);
      chains.add(this.frame.chains[i]);
    }

    const partial = positions.size > 0 && positions.size < this.positions;
    const allChains = new Set(this.frame ? this.frame.chains : s.chainIds);
    const whole = chains.size === allChains.size && !partial && positions.size > 0;
    this._selection = {
      positions,
      chains: whole ? new Set() : chains,
      selectionMode: whole ? "default" : "explicit",
      paeBoxes: [],
    };
    return this._selection;
  }

  /** The cached selection is stale; the mask changed. */
  invalidate() {
    this._selection = null;
  }

  setSelection({ positions }) {
    const state = this.deps.getState();
    const s = state.structure;
    if (!s) return;
    for (let i = 0; i < s.L; i++) state.designMask[i] = positions.has(i) ? 1 : 0;
    this.invalidate();
    this.deps.onSelectionChange();
  }

  get selectionModel() { return this.getSelection(); }

  /** True where the position is an appended heteroatom, not a residue. */
  _isLigand(i) {
    return i >= this.deps.getState().structure.L;
  }

  /**
   * The colour of a position, in whatever mode the 3D view is using.
   *
   * The viewer dims unselected positions itself, so this hands back the
   * saturated colour and lets it do that -- otherwise the two dimmings
   * compound and a fixed residue goes to mud.
   */
  getAtomColor(i) {
    const s = this.deps.getState().structure;
    if (s && this._isLigand(i)) {
      const rgb = this.deps.ligandColour(i - s.L);
      return { r: rgb[0], g: rgb[1], b: rgb[2] };
    }
    const { rgb } = this.deps.colourFor(i);
    return { r: Math.round(rgb[0]), g: Math.round(rgb[1]), b: Math.round(rgb[2]) };
  }

  getChainColorForChainId(chain) {
    const rgb = this.deps.chainColour(chain);
    return { r: rgb[0], g: rgb[1], b: rgb[2] };
  }

  // --- what the page calls ---------------------------------------------------

  /** Rebuild from the current structure. */
  build() {
    const state = this.deps.getState();
    const s = state.structure;
    const host = document.getElementById("sequenceView");
    if (!host) return;
    if (!s) {
      host.innerHTML = "";
      this.frame = null;
      this.positions = 0;
      this.invalidate();
      return;
    }

    const het = this.deps.showLigands() ? s.ligandType.length : 0;
    const n = s.L + het;
    this.positions = n;
    this.invalidate();
    const coords = new Float32Array(n * 3);
    const chains = new Array(n);
    const names = new Array(n);
    const numbers = new Array(n);
    const types = new Array(n);

    for (let i = 0; i < s.L; i++) {
      coords[i * 3] = s.X[i * 12 + 3];
      coords[i * 3 + 1] = s.X[i * 12 + 4];
      coords[i * 3 + 2] = s.X[i * 12 + 5];
      chains[i] = s.chainIds[i];
      // Names, not letters: the viewer runs its own three-to-one table and
      // sniffs protein/DNA/RNA from these to pick which one. Passing the
      // *displayed* name is also what makes a painted design show up here.
      names[i] = this.deps.nameAt(i);
      numbers[i] = s.resSeq[i];
      types[i] = this.deps.typeAt(i);
    }
    for (let a = 0; a < het; a++) {
      const i = s.L + a;
      coords[i * 3] = s.ligandXyz[a * 3];
      coords[i * 3 + 1] = s.ligandXyz[a * 3 + 1];
      coords[i * 3 + 2] = s.ligandXyz[a * 3 + 2];
      chains[i] = s.ligandChains[a];
      names[i] = s.ligandResNames[a];
      numbers[i] = s.ligandResSeq[a];
      types[i] = "L";
    }

    this.frame = {
      coords, chains, position_names: names, residue_numbers: numbers,
      position_types: types,
    };
    // Upstream computes this on its renderer and the sequence viewer reads it
    // off `currentObject.ligandGroups`; same function, same arguments.
    this.ligandGroups = groupLigandAtoms(chains, types, numbers, names);

    window.SEQ?.clear();
    window.SEQ?.buildView();
  }

  /** Selection or colours changed, but the structure did not. */
  refresh() {
    window.SEQ?.updateColors();
    window.SEQ?.updateSelection();
  }

  _hover(i) {
    const state = this.deps.getState();
    const s = state.structure;
    // A ligand token hover reports one of the appended positions; there is no
    // residue to highlight in 3D for it.
    const at = s && i >= 0 && !this._isLigand(i) ? i : -1;
    if (at === state.hover) return;
    state.hover = at;
    this.deps.onHoverChange();
  }

  _toggleChain(chain) {
    const state = this.deps.getState();
    const s = state.structure;
    if (!s) return;
    let all = true;
    for (let i = 0; i < s.L; i++) {
      if (s.chainIds[i] === chain && state.designMask[i] === 0) { all = false; break; }
    }
    this._setChain(chain, !all);
  }

  _setChain(chain, on) {
    const state = this.deps.getState();
    const s = state.structure;
    if (!s) return;
    for (let i = 0; i < s.L; i++) {
      if (s.chainIds[i] === chain) state.designMask[i] = on ? 1 : 0;
    }
    this.invalidate();
    this.deps.onSelectionChange();
  }
}
