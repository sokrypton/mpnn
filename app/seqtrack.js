// The sequence, as a wrapped grid of residues you can select by dragging.
//
// This replaces a single non-wrapping line of DOM spans. Two reasons.
//
// It has to wrap. A 2916-residue complex on one line is 30 screens of sideways
// scrolling, and the whole point of the track is to see the selection at once.
//
// And it has to be a canvas. The DOM version rebuilt every span on every
// selection change -- `innerHTML = ""` then L nodes -- which measured about a
// second per click at L = 121, before any model work. Here a redraw is L
// `fillRect` plus L `fillText` and the geometry is exact, which is also what
// makes hit-testing a drag straightforward.
//
// The colouring follows the 3D view: the same `colourAt(i)` the renderer uses,
// so the track reads as a legend for the structure rather than a second scheme
// to learn. Selection rides on that colour's alpha -- a designed residue is
// saturated, a fixed one washed out -- which is the mechanism py2Dmol's
// sequence viewer uses and it survives every colour mode without needing a
// reserved hue.

const CELL_W = 11;
const CELL_H = 17;
const ROW_GAP = 6;
const GUTTER = 52;
const PAD_R = 8;

export class SequenceTrack {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.data = null;
    this.hover = -1;
    this.rows = [];
    this.perRow = 1;
    this.theme = { ink: "#e6edf7", dim: "#93a4c0", line: "#24324f", accent: "#38bdf8" };
  }

  /**
   * @param {{L: number, chainIds: string[], resSeq: Int32Array,
   *          letterAt: (i: number) => string, isDesigned: (i: number) => boolean,
   *          isChanged: (i: number) => boolean,
   *          colourAt: (i: number) => {rgb: number[], dim: number}}} data
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
   * Wrap into rows of whatever fits, breaking at chain boundaries.
   *
   * A chain always starts a new row, so a row is one chain and the gutter can
   * name it without ambiguity.
   */
  layout(width) {
    const d = this.data;
    if (!d) return 0;
    this.perRow = Math.max(1, Math.floor((width - GUTTER - PAD_R) / CELL_W));
    this.rows = [];
    let start = 0;
    while (start < d.L) {
      let end = Math.min(d.L, start + this.perRow);
      for (let i = start + 1; i < end; i++) {
        if (d.chainIds[i] !== d.chainIds[start]) { end = i; break; }
      }
      this.rows.push({ start, end });
      start = end;
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
    ctx.font = "12px ui-monospace, Menlo, Consolas, monospace";
    ctx.textBaseline = "middle";

    this.rows.forEach((row, r) => {
      const y = r * (CELL_H + ROW_GAP);

      ctx.textAlign = "right";
      ctx.fillStyle = this.theme.dim;
      ctx.font = "10px ui-monospace, Menlo, Consolas, monospace";
      ctx.fillText(`${d.chainIds[row.start]} ${d.resSeq[row.start]}`,
        GUTTER - 8, y + CELL_H / 2);

      ctx.textAlign = "center";
      ctx.font = "12px ui-monospace, Menlo, Consolas, monospace";
      for (let i = row.start; i < row.end; i++) {
        const x = GUTTER + (i - row.start) * CELL_W;
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
        ctx.fillStyle = designed ? "#04121e" : this.theme.dim;
        ctx.fillText(d.letterAt(i), x + CELL_W / 2, y + CELL_H / 2 + 0.5);

        // Where a painted design differs from the input structure.
        if (d.isChanged(i)) {
          ctx.fillStyle = this.theme.accent;
          ctx.fillRect(x, y + CELL_H - 2, CELL_W, 2);
        }
      }
    });
    ctx.globalAlpha = 1;
  }

  /** Residue index under a canvas-space point, or -1. */
  pick(x, y) {
    if (!this.data) return -1;
    const r = Math.floor(y / (CELL_H + ROW_GAP));
    if (r < 0 || r >= this.rows.length) return -1;
    if (y - r * (CELL_H + ROW_GAP) > CELL_H) return -1;
    const row = this.rows[r];
    const c = Math.floor((x - GUTTER) / CELL_W);
    if (c < 0) return -1;
    const i = row.start + c;
    return i < row.end ? i : -1;
  }

  /**
   * The nearest residue to a point, for dragging.
   *
   * `pick` returns -1 in the gaps between rows and past the end of a short row,
   * which would make a drag stutter every time the pointer crossed one. Here
   * the point is clamped into the grid instead.
   */
  nearest(x, y) {
    if (!this.data || this.rows.length === 0) return -1;
    const r = Math.min(this.rows.length - 1,
      Math.max(0, Math.floor(y / (CELL_H + ROW_GAP))));
    const row = this.rows[r];
    const c = Math.floor((x - GUTTER) / CELL_W);
    return Math.min(row.end - 1, Math.max(row.start, row.start + c));
  }
}
