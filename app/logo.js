// A sequence logo for the per-position distribution.
//
// Drawn on a canvas rather than assembled from DOM nodes, because a logo needs
// glyphs stretched to an exact height and CSS can only clip them. Column height
// is the information content in bits and each letter's share of that column is
// its probability, which is what makes a logo readable at a glance: a tall
// column is a position the model is sure about.
//
// The native residue is shown underneath, so "what is there" and "what the
// model wants" can be compared without counting along.

import { ALPHABET } from "../mpnn/constants.js";

export const AA_COLORS = {
  A: "#8ecae6", V: "#8ecae6", L: "#8ecae6", I: "#8ecae6", M: "#8ecae6",
  C: "#ffd166",
  F: "#a78bfa", W: "#a78bfa", Y: "#a78bfa",
  S: "#4ade80", T: "#4ade80", N: "#4ade80", Q: "#4ade80",
  D: "#f87171", E: "#f87171",
  K: "#60a5fa", R: "#60a5fa", H: "#60a5fa",
  G: "#d4d4d8", P: "#fb923c", X: "#64748b",
};

const MAX_BITS = Math.log2(20);
const GUTTER = 38;      // y-axis
const LOGO_H = 118;
const NATIVE_H = 18;
const RULER_H = 16;
const TOTAL_H = LOGO_H + NATIVE_H + RULER_H;
const BASE_FONT = 100;  // glyphs are measured at this size then scaled

export class Logo {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.data = null;
    this.columnWidth = 18;
    this.hover = -1;
    /** (i) => boolean */
    this.isDesigned = () => true;
    this.theme = { ink: "#e6edf7", dim: "#93a4c0", line: "#24324f", bg: "#111a2e" };
    this._ascent = new Map();
  }

  /**
   * @param {{probs: Float32Array, entropy: Float32Array, L: number,
   *          native: Int32Array, resSeq: Int32Array, chainIds: string[]}} data
   */
  setData(data) {
    this.data = data;
  }

  get width() {
    return GUTTER + (this.data ? this.data.L * this.columnWidth : 0) + 8;
  }

  /** Cached ascent of a glyph at BASE_FONT, used to scale it to a target height. */
  _ascentOf(letter) {
    if (this._ascent.has(letter)) return this._ascent.get(letter);
    const ctx = this.ctx;
    ctx.font = `700 ${BASE_FONT}px ui-monospace, Menlo, Consolas, monospace`;
    const m = ctx.measureText(letter);
    const a = m.actualBoundingBoxAscent || BASE_FONT * 0.72;
    this._ascent.set(letter, a);
    return a;
  }

  readTheme(element) {
    const style = getComputedStyle(element);
    this.theme = {
      ink: style.getPropertyValue("--ink").trim() || "#e6edf7",
      dim: style.getPropertyValue("--ink-dim").trim() || "#93a4c0",
      line: style.getPropertyValue("--line").trim() || "#24324f",
      bg: style.getPropertyValue("--panel").trim() || "#111a2e",
    };
  }

  draw() {
    const d = this.data;
    if (!d) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = this.width;
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${TOTAL_H}px`;
    this.canvas.width = w * dpr;
    this.canvas.height = TOTAL_H * dpr;
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, TOTAL_H);

    this._drawAxis(ctx, w);

    for (let i = 0; i < d.L; i++) {
      const x = GUTTER + i * this.columnWidth;
      const designed = this.isDesigned(i);

      if (i === this.hover) {
        ctx.fillStyle = this.theme.line;
        ctx.fillRect(x, 0, this.columnWidth, TOTAL_H);
      }

      // Information content: the column's total height.
      const bits = Math.max(0, MAX_BITS - d.entropy[i] / Math.LN2);
      const height = (bits / MAX_BITS) * LOGO_H;

      // Least probable at the top, so the dominant letter sits on the baseline.
      const order = [];
      let z = 0;
      for (let v = 0; v < 20; v++) z += d.probs[i * 21 + v];
      for (let v = 0; v < 20; v++) order.push([v, d.probs[i * 21 + v] / (z || 1)]);
      order.sort((a, b) => a[1] - b[1]);

      let y = LOGO_H;
      for (const [v, p] of order) {
        const h = p * height;
        if (h < 0.7) continue;
        const letter = ALPHABET[v];
        ctx.save();
        ctx.globalAlpha = designed ? 1 : 0.4;
        ctx.fillStyle = AA_COLORS[letter] ?? "#64748b";
        ctx.translate(x + this.columnWidth / 2, y);
        ctx.scale(this.columnWidth / (BASE_FONT * 0.62), h / this._ascentOf(letter));
        ctx.font = `700 ${BASE_FONT}px ui-monospace, Menlo, Consolas, monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "alphabetic";
        ctx.fillText(letter, 0, 0);
        ctx.restore();
        y -= h;
      }

      // Native residue.
      ctx.font = "11px ui-monospace, Menlo, Consolas, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = designed ? this.theme.ink : this.theme.dim;
      ctx.fillText(ALPHABET[d.native[i]] ?? "X", x + this.columnWidth / 2, LOGO_H + NATIVE_H / 2);

      // A tick under every position the design is allowed to change.
      if (designed) {
        ctx.fillStyle = "#38bdf8";
        ctx.fillRect(x + 1, LOGO_H + NATIVE_H - 2, this.columnWidth - 2, 2);
      }

      if (i % 10 === 0 || i === d.L - 1) {
        ctx.fillStyle = this.theme.dim;
        ctx.font = "9px ui-monospace, Menlo, Consolas, monospace";
        ctx.fillText(
          String(d.resSeq[i]), x + this.columnWidth / 2, LOGO_H + NATIVE_H + RULER_H / 2,
        );
      }
    }
  }

  _drawAxis(ctx, w) {
    ctx.strokeStyle = this.theme.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(GUTTER - 0.5, 0);
    ctx.lineTo(GUTTER - 0.5, LOGO_H);
    ctx.moveTo(GUTTER - 0.5, LOGO_H + 0.5);
    ctx.lineTo(w, LOGO_H + 0.5);
    ctx.stroke();

    ctx.fillStyle = this.theme.dim;
    ctx.font = "9px ui-monospace, Menlo, Consolas, monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let b = 0; b <= 4; b++) {
      const y = LOGO_H - (b / MAX_BITS) * LOGO_H;
      ctx.fillText(String(b), GUTTER - 6, y);
      ctx.strokeStyle = this.theme.line;
      ctx.beginPath();
      ctx.moveTo(GUTTER - 4, y + 0.5);
      ctx.lineTo(GUTTER, y + 0.5);
      ctx.stroke();
    }
    ctx.save();
    ctx.translate(9, LOGO_H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("bits", 0, 0);
    ctx.restore();
  }

  /** Position under a canvas-space x, or -1. */
  pick(x) {
    if (!this.data) return -1;
    const i = Math.floor((x - GUTTER) / this.columnWidth);
    return i >= 0 && i < this.data.L ? i : -1;
  }

  /** Text summary of a position, for a tooltip. */
  describe(i, n = 4) {
    const d = this.data;
    const order = [];
    for (let v = 0; v < 20; v++) order.push([ALPHABET[v], d.probs[i * 21 + v]]);
    order.sort((a, b) => b[1] - a[1]);
    const bits = Math.max(0, MAX_BITS - d.entropy[i] / Math.LN2);
    return `${d.chainIds[i]}${d.resSeq[i]} ${ALPHABET[d.native[i]]} — ${bits.toFixed(2)} bits\n`
      + order.slice(0, n).map(([aa, p]) => `${aa} ${(p * 100).toFixed(0)}%`).join("  ");
  }
}
