// Adapter between the page's state and the vendored CIRPIN renderer.
//
// All the drawing lives in trace3d.js and all the secondary-structure
// assignment in sec.js, both copied verbatim. This file only does the things
// those two do not: hold the current structure, decide each residue's colour,
// draw the ligand, and hit-test residues so the design UI can be clicked.
//
// The projection here mirrors `drawTraces` exactly -- same normalisation onto
// the unit sphere, same perspective factor, same box. It has to, or picking
// lands somewhere other than where the ribbon was drawn.

import {
  PAPER,
  PE_MAX,
  drawTraces,
  fitOf,
  hexToRgb,
  makeCamera,
  orbit,
  prep,
  setPaper,
  shade,
  spectrumRgb,
} from "./trace3d.js";
import { makeSec, smoothSec } from "./sec.js";

export { hexToRgb, spectrumRgb, makeCamera, orbit };

const INSET = 10;

// The renderer normalises onto the unit sphere and divides by the perspective
// factor's maximum, which is the value at the nearest pole -- a point no real
// atom occupies while also being at full radius. The result is correct but
// conservative: at zoom 1 a structure sits inside a wide margin. This is the
// caller's business, not the renderer's, so it is fixed here.
const DEFAULT_ZOOM = 1.5;

/** Ligand atoms are drawn as discs, coloured by element. */
const ELEMENT_RGB = {
  C: [148, 163, 184], N: [96, 165, 250], O: [248, 113, 113], S: [251, 191, 36],
  P: [251, 146, 60], F: [74, 222, 128], CL: [74, 222, 128], BR: [249, 115, 22],
  I: [167, 139, 250], FE: [249, 115, 22], ZN: [163, 163, 163], MG: [134, 239, 172],
  CA: [212, 212, 216], MN: [192, 132, 252], NA: [147, 197, 253], K: [196, 181, 253],
};
const ELEMENT_FALLBACK = [192, 132, 252];

export class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.camera = makeCamera();
    this.structure = null;
    this.ca = null;
    this.sec = "";
    this.fit = null;
    /** (residueIndex) => {rgb: [r, g, b], dim: number} | null */
    this.colourAt = () => ({ rgb: [125, 211, 252], dim: 1 });
    this.highlight = -1;
    this.showLigand = true;
    this.box = null;
    this._projected = null;
    this._geom = null;
  }

  setStructure(structure) {
    this.structure = structure;
    const L = structure.L;
    // trace3d wants Float64 xyz triples.
    const ca = new Float64Array(L * 3);
    for (let i = 0; i < L; i++) {
      ca[i * 3] = structure.X[i * 12 + 3];
      ca[i * 3 + 1] = structure.X[i * 12 + 4];
      ca[i * 3 + 2] = structure.X[i * 12 + 5];
    }
    this.ca = ca;
    this.sec = smoothSec(makeSec(ca, L));

    // Fit over the ligand too, so a cofactor sticking out of the pocket is not
    // cropped. pct < 1 keeps a single stray residue from shrinking everything.
    const clouds = [ca];
    if (structure.ligandXyz.length) clouds.push(Float64Array.from(structure.ligandXyz));
    this.fit = fitOf(clouds, 0.985);

    this.resetCamera();
  }

  resetCamera() {
    this.camera.rot = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    this.camera.zoom = DEFAULT_ZOOM;
    this.camera.tx = 0;
    this.camera.ty = 0;
  }

  /** Tell the renderer what colour the page actually is behind the canvas. */
  syncBackground() {
    const css = getComputedStyle(this.canvas.parentElement).backgroundColor;
    const m = css.match(/\d+/g);
    if (m && m.length >= 3) setPaper([+m[0], +m[1], +m[2]]);
  }

  /** Fraction of secondary structure assigned, for the status line. */
  secondaryStructureFractions() {
    let h = 0;
    let e = 0;
    for (const c of this.sec) {
      if (c === "H") h++;
      else if (c === "E") e++;
    }
    const n = this.sec.length || 1;
    return { helix: h / n, strand: e / n, loop: 1 - (h + e) / n };
  }

  draw() {
    if (!this.structure) return;
    this.syncBackground();

    const side = Math.min(this.canvas.clientWidth, this.canvas.clientHeight);
    const p = prep(this.canvas, side);
    if (!p) return;
    const box = {
      x: (p.w - side) / 2 + this.camera.tx,
      y: this.camera.ty,
      size: side,
    };
    this._geom = { box, p };

    drawTraces(
      this.canvas,
      [{ coords: this.ca, sec: this.sec, colourAt: (i) => this.colourAt(i) }],
      { into: p, box, fit: this.fit, rot: this.camera.rot, zoom: this.camera.zoom, inset: INSET },
    );

    this._projectAll();
    if (this.showLigand) this._drawLigand(p.ctx);
    if (this.highlight >= 0) this._drawHighlight(p.ctx);
    if (this.box) this._drawSelectionBox(p.ctx);
  }

  /** Same maths as `drawTraces`'s inner `project`, for picking and ligands. */
  _project(x, y, z) {
    const { cx, cy, cz, r } = this.fit;
    const M = this.camera.rot;
    const { box } = this._geom;
    const R = ((box.size / 2 - INSET) / PE_MAX) * this.camera.zoom;
    const nx = (x - cx) / r;
    const ny = (y - cy) / r;
    const nz = (z - cz) / r;
    const vx = M[0] * nx + M[1] * ny + M[2] * nz;
    const vy = M[3] * nx + M[4] * ny + M[5] * nz;
    const vz = M[6] * nx + M[7] * ny + M[8] * nz;
    const pe = 1 / (1.9 - vz * 0.55);
    return [
      box.x + box.size / 2 + vx * R * pe,
      box.y + box.size / 2 - vy * R * pe,
      vz,
      pe,
      R,
    ];
  }

  _projectAll() {
    const L = this.structure.L;
    const out = new Float64Array(L * 3);
    for (let i = 0; i < L; i++) {
      const q = this._project(this.ca[i * 3], this.ca[i * 3 + 1], this.ca[i * 3 + 2]);
      out[i * 3] = q[0];
      out[i * 3 + 1] = q[1];
      out[i * 3 + 2] = q[2];
    }
    this._projected = out;
  }

  _drawLigand(ctx) {
    const s = this.structure;
    const n = s.ligandType.length;
    if (!n) return;
    // Painted after the cartoon rather than interleaved with it. The ligand is
    // the thing you are looking at with LigandMPNN, and depth-sorting it into
    // the ribbon would mean reaching inside the vendored renderer.
    const discs = [];
    for (let i = 0; i < n; i++) {
      const q = this._project(s.ligandXyz[i * 3], s.ligandXyz[i * 3 + 1], s.ligandXyz[i * 3 + 2]);
      discs.push({ q, rgb: ELEMENT_RGB[s.ligandElements[i]] ?? ELEMENT_FALLBACK });
    }
    discs.sort((a, b) => a.q[2] - b.q[2]);

    for (const { q, rgb } of discs) {
      const near = Math.min(Math.max((q[2] + 1) / 2, 0), 1);
      // 1.55 Å is a reasonable heavy-atom radius on a cartoon's scale.
      const radius = Math.max(2, (1.55 / this.fit.r) * q[4] * q[3]);
      ctx.beginPath();
      ctx.arc(q[0], q[1], radius + 1.6, 0, Math.PI * 2);
      ctx.fillStyle = `rgb(${PAPER[0]},${PAPER[1]},${PAPER[2]})`;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(q[0], q[1], radius, 0, Math.PI * 2);
      ctx.fillStyle = shade(rgb, near, 1);
      ctx.fill();
      ctx.strokeStyle = shade(rgb, near, 1, 0.62);
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  _drawHighlight(ctx) {
    const i = this.highlight;
    if (!this._projected || i < 0 || i >= this.structure.L) return;
    ctx.beginPath();
    ctx.arc(this._projected[i * 3], this._projected[i * 3 + 1], 7, 0, Math.PI * 2);
    ctx.strokeStyle = "#fbbf24";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  _drawSelectionBox(ctx) {
    const { from, to } = this.box;
    ctx.save();
    ctx.strokeStyle = "#fbbf24";
    ctx.setLineDash([4, 3]);
    ctx.lineWidth = 1;
    ctx.strokeRect(
      Math.min(from[0], to[0]), Math.min(from[1], to[1]),
      Math.abs(to[0] - from[0]), Math.abs(to[1] - from[1]),
    );
    ctx.restore();
  }

  /** Nearest residue to a canvas point, or -1. */
  pick(point, tolerance = 12) {
    if (!this._projected) return -1;
    let best = -1;
    let bestScore = Infinity;
    for (let i = 0; i < this.structure.L; i++) {
      const dx = this._projected[i * 3] - point[0];
      const dy = this._projected[i * 3 + 1] - point[1];
      const d2 = dx * dx + dy * dy;
      if (d2 > tolerance * tolerance) continue;
      // Break ties toward the camera so a click lands on the near strand.
      const score = d2 - this._projected[i * 3 + 2] * 40;
      if (score < bestScore) {
        bestScore = score;
        best = i;
      }
    }
    return best;
  }

  /** Residues whose C-alpha projects inside a screen rectangle. */
  pickBox(from, to) {
    const hits = new Set();
    if (!this._projected) return hits;
    const x0 = Math.min(from[0], to[0]);
    const x1 = Math.max(from[0], to[0]);
    const y0 = Math.min(from[1], to[1]);
    const y1 = Math.max(from[1], to[1]);
    for (let i = 0; i < this.structure.L; i++) {
      const x = this._projected[i * 3];
      const y = this._projected[i * 3 + 1];
      if (x >= x0 && x <= x1 && y >= y0 && y <= y1) hits.add(i);
    }
    return hits;
  }
}
