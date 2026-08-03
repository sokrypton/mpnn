// The sequence, as a wrapped grid of residues you can select by dragging.
//
// This replaces a single non-wrapping line of DOM spans. Two reasons.
//
// It has to wrap. A 2916-residue complex on one line is 30 screens of sideways
// scrolling, and the whole point of the track is to see the selection at once.
//
// And it has to be a canvas. The DOM version rebuilt every span on every
// selection change -- `innerHTML = ""` then L nodes -- which measured about a
// second per click at L = 121, before any model work. Here a redraw is one
// `fillRect` plus one `fillText` per cell and the geometry is exact, which is
// also what makes hit-testing a drag straightforward.
//
// The colouring follows the 3D view: the same `colourAt(i)` the renderer uses,
// so the track reads as a legend for the structure rather than a second scheme
// to learn. Selection rides on that colour's alpha -- a designed residue is
// saturated, a fixed one washed out -- which is the mechanism py2Dmol's
// sequence viewer uses and it survives every colour mode without needing a
// reserved hue.
//
// The layout is a list of display *items* rather than a run of residues,
// again following py2Dmol: a chain is not always a contiguous stretch of
// model positions, and pretending it is hides things that matter for design.

const CELL_W = 11;
const CELL_H = 17;
const ROW_GAP = 6;
const GUTTER = 52;
const PAD_R = 8;
/** Cells a ligand token spans. Enough for a three-letter PDB chemical id. */
const LIG_CELLS = 4;

export class SequenceTrack {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.data = null;
    this.hover = -1;
    this.hoverLigand = -1;
    this.rows = [];
    this.perRow = 1;
    this.theme = {
      ink: "#e6edf7", dim: "#93a4c0", line: "#24324f", accent: "#38bdf8",
    };
  }

  /**
   * @param {{L: number, chainIds: string[], resSeq: Int32Array,
   *          letterAt: (i: number) => string, isDesigned: (i: number) => boolean,
   *          isChanged: (i: number) => boolean,
   *          colourAt: (i: number) => {rgb: number[], dim: number},
   *          polytypeAt?: (i: number) => number,
   *          ligands?: {label: string, name: string, count: number,
   *                     chain: string, rgb: number[]}[]}} data
   */
  setData(data) {
    this.data = data;
    this.rows = [];
  }

  readTheme(element) {
    const style = getComputedStyle(element);
    const read = (name, fallback) => style.getPropertyValue(name).trim() || fallback;
    this.theme = {
      ink: read("--ink", "#e6edf7"),
      dim: read("--ink-dim", "#93a4c0"),
      line: read("--line", "#24324f"),
      accent: read("--accent-2", "#38bdf8"),
    };
  }

  /**
   * Turn the structure into display items.
   *
   * Three kinds beyond the residue itself, all taken from py2Dmol's viewer:
   *
   * `gap` -- a residue the file numbers but does not contain. A disordered
   * loop leaves a hole, and the model does not see a hole: the k-nearest
   * graph simply joins the two sides, so positions that look adjacent in the
   * track may be 30 Å apart. 6VXX is full of these. Drawing one dash per
   * missing residue puts the track on *residue numbering* rather than on model
   * index, which is the numbering people actually talk in.
   *
   * `ligand` -- one token per heteroatom residue rather than one cell per
   * atom, keyed on `resName chain+resSeq` exactly as py2Dmol keys its ligand
   * groups. Biotin is one BTN, not sixteen carbons.
   *
   * `spacer` -- a blank cell where the polymer type changes, so a protein run
   * and a DNA run inside one chain do not read as one word.
   */
  _items() {
    const d = this.data;
    const items = [];
    for (let i = 0; i < d.L; i++) {
      if (i > 0 && d.chainIds[i] === d.chainIds[i - 1]) {
        const missing = d.resSeq[i] - d.resSeq[i - 1] - 1;
        for (let g = 0; g < missing; g++) {
          items.push({ kind: "gap", chain: d.chainIds[i], resSeq: d.resSeq[i - 1] + 1 + g });
        }
        if (missing <= 0 && d.polytypeAt && d.polytypeAt(i) !== d.polytypeAt(i - 1)) {
          items.push({ kind: "spacer", chain: d.chainIds[i] });
        }
      }
      items.push({ kind: "res", chain: d.chainIds[i], i });
    }
    (d.ligands ?? []).forEach((lig, g) => {
      // Its own row, labelled `lig <chain>` -- py2Dmol gives a ligand its own
      // chain row too, and a token inline with the sequence would imply it is
      // a position in the chain, which it is not.
      items.push({ kind: "ligand", chain: `lig ${lig.chain}`, g });
    });
    return items;
  }

  /** Cells an item occupies. */
  static _cells(item) {
    return item.kind === "ligand" ? LIG_CELLS : 1;
  }

  /**
   * Wrap into rows of whatever fits, breaking whenever the chain changes.
   *
   * A row is one chain, so the gutter can name it without ambiguity.
   */
  layout(width) {
    const d = this.data;
    if (!d) return 0;
    this.perRow = Math.max(LIG_CELLS, Math.floor((width - GUTTER - PAD_R) / CELL_W));
    this.rows = [];
    let row = null;
    for (const item of this._items()) {
      const cells = SequenceTrack._cells(item);
      if (row === null || row.chain !== item.chain || row.cells + cells > this.perRow) {
        row = { chain: item.chain, items: [], cells: 0 };
        this.rows.push(row);
      }
      item.cell = row.cells;
      row.items.push(item);
      row.cells += cells;
    }
    return this.rows.length * (CELL_H + ROW_GAP);
  }

  draw(width) {
    const d = this.data;
    if (!d) return;
    const height = this.layout(width);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.textBaseline = "middle";

    this.rows.forEach((row, r) => {
      const y = r * (CELL_H + ROW_GAP);
      this._drawGutter(ctx, row, y);
      ctx.textAlign = "center";
      for (const item of row.items) {
        const x = GUTTER + item.cell * CELL_W;
        if (item.kind === "res") this._drawResidue(ctx, item.i, x, y);
        else if (item.kind === "gap") this._drawGap(ctx, x, y);
        else if (item.kind === "ligand") this._drawLigand(ctx, item.g, x, y);
      }
    });
    ctx.globalAlpha = 1;
  }

  /** The row's chain and the residue number it starts at. */
  _drawGutter(ctx, row, y) {
    const d = this.data;
    const first = row.items.find((it) => it.kind === "res" || it.kind === "gap");
    const at = first === undefined ? ""
      : first.kind === "res" ? d.resSeq[first.i] : first.resSeq;
    ctx.textAlign = "right";
    ctx.fillStyle = this.theme.dim;
    ctx.font = "10px ui-monospace, Menlo, Consolas, monospace";
    ctx.fillText(`${row.chain} ${at}`.trim(), GUTTER - 8, y + CELL_H / 2);
  }

  _drawResidue(ctx, i, x, y) {
    const d = this.data;
    const { rgb, dim } = d.colourAt(i);
    const designed = d.isDesigned(i);

    ctx.globalAlpha = designed ? 1 : 0.3 * dim;
    ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
    ctx.fillRect(x, y, CELL_W, CELL_H);
    ctx.globalAlpha = 1;

    if (i === this.hover) {
      ctx.strokeStyle = this.theme.ink;
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, y + 0.5, CELL_W - 1, CELL_H - 1);
    }

    // Dark ink on a saturated cell, dim ink on a washed-out one: the letter
    // has to stay legible in both states and against every colour mode.
    ctx.font = "12px ui-monospace, Menlo, Consolas, monospace";
    ctx.fillStyle = designed ? "#04121e" : this.theme.dim;
    ctx.fillText(d.letterAt(i), x + CELL_W / 2, y + CELL_H / 2 + 0.5);

    // Where a painted design differs from the input structure.
    if (d.isChanged(i)) {
      ctx.fillStyle = this.theme.accent;
      ctx.fillRect(x, y + CELL_H - 2, CELL_W, 2);
    }
  }

  /** A residue the numbering claims and the file does not contain. */
  _drawGap(ctx, x, y) {
    ctx.strokeStyle = this.theme.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x + 1.5, y + CELL_H / 2 + 0.5);
    ctx.lineTo(x + CELL_W - 1.5, y + CELL_H / 2 + 0.5);
    ctx.stroke();
  }

  _drawLigand(ctx, g, x, y) {
    const lig = this.data.ligands[g];
    const w = LIG_CELLS * CELL_W;
    ctx.fillStyle = `rgb(${lig.rgb[0]},${lig.rgb[1]},${lig.rgb[2]})`;
    ctx.fillRect(x, y, w - 2, CELL_H);
    if (g === this.hoverLigand) {
      ctx.strokeStyle = this.theme.ink;
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, y + 0.5, w - 3, CELL_H - 1);
    }
    ctx.fillStyle = "#04121e";
    ctx.font = "10px ui-monospace, Menlo, Consolas, monospace";
    ctx.fillText(lig.name, x + (w - 2) / 2, y + CELL_H / 2 + 0.5);
  }

  /** The item under a canvas-space point, or null. */
  pick(x, y) {
    if (!this.data) return null;
    const r = Math.floor(y / (CELL_H + ROW_GAP));
    if (r < 0 || r >= this.rows.length) return null;
    if (y - r * (CELL_H + ROW_GAP) > CELL_H) return null;
    const cell = Math.floor((x - GUTTER) / CELL_W);
    if (cell < 0) return null;
    for (const item of this.rows[r].items) {
      if (cell >= item.cell && cell < item.cell + SequenceTrack._cells(item)) return item;
    }
    return null;
  }

  /** Residue index under a point, or -1. */
  pickResidue(x, y) {
    const item = this.pick(x, y);
    return item !== null && item.kind === "res" ? item.i : -1;
  }

  /**
   * The nearest residue to a point, for dragging.
   *
   * `pick` returns nothing in the gaps between rows, over a gap dash and past
   * the end of a short row, which would make a drag stutter every time the
   * pointer crossed one. Here the point is clamped into the grid and snapped
   * to the closest residue item instead.
   */
  nearest(x, y) {
    if (!this.data || this.rows.length === 0) return -1;
    const r = Math.min(this.rows.length - 1,
      Math.max(0, Math.floor(y / (CELL_H + ROW_GAP))));
    const cell = Math.floor((x - GUTTER) / CELL_W);
    let best = -1;
    let bestDist = Infinity;
    // Search outwards from the pointer's row so a drag over a run of gap dashes
    // or a ligand token keeps extending rather than freezing.
    for (let step = 0; step < this.rows.length; step++) {
      for (const dir of step === 0 ? [0] : [-step, step]) {
        const row = this.rows[r + dir];
        if (row === undefined) continue;
        for (const item of row.items) {
          if (item.kind !== "res") continue;
          const d = dir !== 0 ? Math.abs(dir) * this.perRow
            : Math.abs(item.cell - cell);
          if (d < bestDist) { bestDist = d; best = item.i; }
        }
      }
      if (best >= 0) return best;
    }
    return best;
  }
}
