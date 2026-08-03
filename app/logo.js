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
  // NA-MPNN's nucleotides. Lower case is DNA, and under shared tokens the RNA
  // bases render as b/d/h/u/y. Purine/pyrimidine pairs share a hue.
  a: "#22d3ee", b: "#22d3ee",
  t: "#f472b6", u: "#f472b6",
  g: "#facc15", h: "#facc15",
  c: "#34d399", d: "#34d399",
  x: "#64748b", y: "#64748b",
};

const MAX_BITS = Math.log2(20);
const GUTTER = 38;      // y-axis
const LOGO_H = 118;
const NATIVE_H = 18;
const RULER_H = 16;
const BASE_FONT = 100;  // glyphs are measured at this size then scaled
const PSSM_ROW_H = 13;

/**
 * Row order for the heatmap, grouped by chemistry.
 *
 * Alphabetical order would scatter the answer: the interesting thing about a
 * position is usually "the model wants something hydrophobic here", and that
 * reads off a grouped axis as one bright band. The groups are the ones
 * `AA_COLORS` already shares a hue within, so the letters down the side band
 * together as well.
 */
const PSSM_GROUPS = [
  ["A", "V", "L", "I", "M"],
  ["F", "W", "Y"],
  ["S", "T", "N", "Q"],
  ["D", "E"],
  ["K", "R", "H"],
  ["C", "G", "P"],
];

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
    this._ink = new Map();
    /** The model's alphabet, its width, and which letters get a glyph. */
    this.alphabet = ALPHABET;
    this.V = ALPHABET.length;
    this.letters = [...Array(20).keys()];
    /** "logo" or "pssm". */
    this.mode = "logo";
  }

  /**
   * The rows of the heatmap: letter indices, plus where to rule a line.
   *
   * Falls back to whatever `letters` holds when the alphabet is not the usual
   * 20 -- NA-MPNN's has nucleotides in it and no chemistry grouping to apply.
   */
  _pssmRows() {
    if (this.alphabet !== ALPHABET) {
      return { rows: this.letters, breaks: new Set() };
    }
    const rows = [];
    const breaks = new Set();
    for (const group of PSSM_GROUPS) {
      for (const letter of group) rows.push(this.alphabet.indexOf(letter));
      breaks.add(rows.length);
    }
    breaks.delete(rows.length);
    return { rows, breaks };
  }

  get totalHeight() {
    if (this.mode !== "pssm") return LOGO_H + NATIVE_H + RULER_H;
    return this._pssmRows().rows.length * PSSM_ROW_H + NATIVE_H + RULER_H;
  }

  /**
   * Point the logo at a different alphabet -- NA-MPNN's 33 letters rather than
   * the usual 21. `letters` are the indices worth drawing: the placeholders
   * (UNK, MAS, PAD) carry no probability mass worth a column.
   */
  setAlphabet(alphabet, letters) {
    this.alphabet = alphabet;
    this.V = alphabet.length;
    this.letters = letters;
    this._ink.clear();
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

  /**
   * The glyph's *ink* box at BASE_FONT: how far it actually extends from the
   * draw point, on all four sides.
   *
   * Not the advance width and not the font's ascent. A logo stretches every
   * letter to exactly fill its column, so what has to be measured is the mark
   * on the page. Bold `W` and `Q` ink wider than a monospace advance, which
   * spilled them into the neighbouring columns; `Q` and `J` ink below the
   * baseline, which spilled them over the letter underneath. Scaling by the
   * advance and positioning by the ascent got both wrong.
   */
  _inkOf(letter) {
    if (this._ink.has(letter)) return this._ink.get(letter);
    const ctx = this.ctx;
    ctx.font = `700 ${BASE_FONT}px ui-monospace, Menlo, Consolas, monospace`;
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    const m = ctx.measureText(letter);
    // The fallbacks are for engines without the actualBoundingBox* metrics;
    // they put the glyph roughly right rather than not at all.
    const ink = {
      left: m.actualBoundingBoxLeft ?? 0,
      right: m.actualBoundingBoxRight ?? m.width,
      ascent: m.actualBoundingBoxAscent || BASE_FONT * 0.72,
      descent: m.actualBoundingBoxDescent || 0,
    };
    ink.width = ink.right + ink.left || m.width || BASE_FONT * 0.6;
    ink.height = ink.ascent + ink.descent;
    this._ink.set(letter, ink);
    return ink;
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
    const total = this.totalHeight;
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${total}px`;
    this.canvas.width = w * dpr;
    this.canvas.height = total * dpr;
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, total);

    const bodyH = total - NATIVE_H - RULER_H;
    if (this.mode === "pssm") this._drawPssm(ctx, w, bodyH);
    else this._drawLogo(ctx, w);
    this._drawFooter(ctx, bodyH);
  }

  _drawLogo(ctx, w) {
    const d = this.data;
    this._drawAxis(ctx, w);

    for (let i = 0; i < d.L; i++) {
      const x = GUTTER + i * this.columnWidth;
      const designed = this.isDesigned(i);

      if (i === this.hover) {
        ctx.fillStyle = this.theme.line;
        ctx.fillRect(x, 0, this.columnWidth, LOGO_H);
      }

      // Information content: the column's total height.
      const bits = Math.max(0, MAX_BITS - d.entropy[i] / Math.LN2);
      const height = (bits / MAX_BITS) * LOGO_H;

      // Least probable at the top, so the dominant letter sits on the baseline.
      const order = [];
      let z = 0;
      for (const v of this.letters) z += d.probs[i * this.V + v];
      for (const v of this.letters) order.push([v, d.probs[i * this.V + v] / (z || 1)]);
      order.sort((a, b) => a[1] - b[1]);

      let y = LOGO_H;
      for (const [v, p] of order) {
        const h = p * height;
        if (h < 0.7) continue;
        const letter = this.alphabet[v];
        const ink = this._inkOf(letter);
        // Map the ink box onto the cell exactly: [x, x+columnWidth] across and
        // [y-h, y] down. Anything less and neighbouring letters collide.
        ctx.save();
        ctx.globalAlpha = designed ? 1 : 0.4;
        ctx.fillStyle = AA_COLORS[letter] ?? "#64748b";
        ctx.translate(x, y);
        ctx.scale(this.columnWidth / ink.width, h / ink.height);
        ctx.font = `700 ${BASE_FONT}px ui-monospace, Menlo, Consolas, monospace`;
        ctx.textAlign = "left";
        ctx.textBaseline = "alphabetic";
        ctx.fillText(letter, ink.left, -ink.descent);
        ctx.restore();
        y -= h;
      }
    }
  }

  /**
   * The same distribution as a heatmap: one row per letter, one column per
   * position, cell brightness the probability.
   *
   * The logo answers "how sure is the model here" at a glance and buries
   * everything below a few percent -- a glyph under about a pixel is not drawn
   * at all. The heatmap answers the other question: which letters are in play
   * across a whole run of positions, including the quiet ones. Same numbers,
   * and the mode switch is the cheapest way to have both.
   */
  _drawPssm(ctx, w, bodyH) {
    const d = this.data;
    const { rows, breaks } = this._pssmRows();
    const h = PSSM_ROW_H;

    ctx.fillStyle = this.theme.bg;
    ctx.fillRect(GUTTER, 0, w - GUTTER, bodyH);

    ctx.font = "10px ui-monospace, Menlo, Consolas, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    rows.forEach((v, r) => {
      const letter = this.alphabet[v];
      ctx.fillStyle = AA_COLORS[letter] ?? "#64748b";
      ctx.fillRect(GUTTER - 14, r * h, 12, h - 1);
      ctx.fillStyle = "#04121e";
      ctx.fillText(letter, GUTTER - 8, r * h + h / 2);
    });

    for (let i = 0; i < d.L; i++) {
      const x = GUTTER + i * this.columnWidth;
      const designed = this.isDesigned(i);
      rows.forEach((v, r) => {
        const p = d.probs[i * this.V + v];
        if (p > 0.004) {
          // sqrt, because the interesting range is the bottom of it: a 4%
          // alternative is worth seeing and would be invisible on a linear ramp.
          ctx.globalAlpha = (designed ? 1 : 0.45) * Math.min(1, Math.sqrt(p));
          ctx.fillStyle = AA_COLORS[this.alphabet[v]] ?? "#64748b";
          ctx.fillRect(x, r * h, this.columnWidth, h - 1);
          ctx.globalAlpha = 1;
        }
      });
      if (i === this.hover) {
        ctx.strokeStyle = this.theme.ink;
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, 0.5, this.columnWidth - 1, bodyH - 1);
      }
    }

    ctx.strokeStyle = this.theme.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const r of breaks) {
      ctx.moveTo(GUTTER - 14, r * h - 0.5);
      ctx.lineTo(w, r * h - 0.5);
    }
    ctx.stroke();
  }

  /** The native sequence and the residue-number ruler, under either view. */
  _drawFooter(ctx, bodyH) {
    const d = this.data;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let i = 0; i < d.L; i++) {
      const x = GUTTER + i * this.columnWidth;
      const designed = this.isDesigned(i);
      const letter = this.alphabet[d.native[i]] ?? "X";

      // Coloured like the logo's glyphs, so the eye can match a column to the
      // residue that is actually there without reading the letter.
      ctx.globalAlpha = designed ? 1 : 0.35;
      ctx.fillStyle = AA_COLORS[letter] ?? "#64748b";
      ctx.fillRect(x, bodyH + 1, Math.max(1, this.columnWidth - 1), NATIVE_H - 4);
      ctx.globalAlpha = 1;

      if (this.columnWidth >= 8) {
        ctx.font = "11px ui-monospace, Menlo, Consolas, monospace";
        ctx.fillStyle = designed ? "#04121e" : this.theme.dim;
        ctx.fillText(letter, x + this.columnWidth / 2, bodyH + NATIVE_H / 2);
      }

      // A tick under every position the design is allowed to change.
      if (designed) {
        ctx.fillStyle = "#38bdf8";
        ctx.fillRect(x + 1, bodyH + NATIVE_H - 2, this.columnWidth - 2, 2);
      }

      if (i % 10 === 0 || i === d.L - 1) {
        ctx.fillStyle = this.theme.dim;
        ctx.font = "9px ui-monospace, Menlo, Consolas, monospace";
        ctx.fillText(String(d.resSeq[i]),
          x + this.columnWidth / 2, bodyH + NATIVE_H + RULER_H / 2);
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

  /**
   * The whole matrix as CSV: one row per position, one column per letter.
   *
   * The heatmap is for reading; this is for everything else people do with a
   * profile -- a logo of their own, a threshold, a comparison against another
   * run. Writing the probabilities out means nobody has to reverse them out of
   * pixel values.
   */
  toCsv() {
    const d = this.data;
    if (!d) return "";
    const letters = this.mode === "pssm" ? this._pssmRows().rows : this.letters;
    const lines = [
      ["chain", "resSeq", "native", "designed", "bits",
        ...letters.map((v) => this.alphabet[v])].join(","),
    ];
    for (let i = 0; i < d.L; i++) {
      const bits = Math.max(0, MAX_BITS - d.entropy[i] / Math.LN2);
      lines.push([
        d.chainIds[i], d.resSeq[i], this.alphabet[d.native[i]] ?? "X",
        this.isDesigned(i) ? 1 : 0, bits.toFixed(4),
        ...letters.map((v) => d.probs[i * this.V + v].toFixed(6)),
      ].join(","));
    }
    return `${lines.join("\n")}\n`;
  }

  /** Text summary of a position, for a tooltip. */
  describe(i, n = 4) {
    const d = this.data;
    const order = [];
    for (const v of this.letters) order.push([this.alphabet[v], d.probs[i * this.V + v]]);
    order.sort((a, b) => b[1] - a[1]);
    const bits = Math.max(0, MAX_BITS - d.entropy[i] / Math.LN2);
    return `${d.chainIds[i]}${d.resSeq[i]} ${this.alphabet[d.native[i]]} — ${bits.toFixed(2)} bits\n`
      + order.slice(0, n).map(([aa, p]) => `${aa} ${(p * 100).toFixed(0)}%`).join("  ");
  }
}
