// Adapter between the page's state and the CIRPIN cartoon renderer.
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
  PE_MAX,
  drawTraces,
  fitOf,
  hexToRgb,
  makeCamera,
  orbit,
  prep,
  setPaper,
  spectrumRgb,
} from "./trace3d.js";
import { POLYTYPE } from "../mpnn/na.js";

/** Heavy atoms this close are bonded, as py2Dmol's `ligand_bond` cutoff has it. */
const BOND_A2 = 2.0 * 2.0;
/** Bond stick radius, angstroms. */
const BOND_TUBE_A = 0.34;
/** Lone-atom disc radius. Bigger than a bond, so an ion reads as an atom. */
const ATOM_DISC_A = 0.75;
/**
 * Elements drawn as a disc and never bonded to anything.
 *
 * Coordination is not a covalent bond, and a zinc joined by sticks to the four
 * cysteines around it would read as a molecule that is not there. Anything else
 * that ends up with no bond within the cutoff gets a disc too -- a chloride, a
 * lone water -- since a bare stub is not a useful drawing of an atom.
 */
const METALS = new Set([
  "LI", "NA", "K", "RB", "CS", "MG", "CA", "SR", "BA",
  "MN", "FE", "CO", "NI", "CU", "ZN", "CD", "HG", "PT", "AU", "AG", "PB", "MO", "W",
]);
import { makeSec, smoothSec } from "./sec.js";

export { hexToRgb, spectrumRgb, makeCamera, orbit };

const INSET = 10;

// The renderer normalises onto the unit sphere and divides by the perspective
// factor's maximum, which is the value at the nearest pole -- a point no real
// atom occupies while also being at full radius. The result is correct but
// conservative: at zoom 1 a structure sits inside a wide margin. This is the
// caller's business, not the renderer's, so it is fixed here.
const DEFAULT_ZOOM = 1.5;

/** Bond colours, by the element at that end. */
const ELEMENT_RGB = {
  C: [148, 163, 184], N: [96, 165, 250], O: [248, 113, 113], S: [251, 191, 36],
  P: [251, 146, 60], F: [74, 222, 128], CL: [74, 222, 128], BR: [249, 115, 22],
  I: [167, 139, 250], FE: [249, 115, 22], ZN: [163, 163, 163], MG: [134, 239, 172],
  CA: [212, 212, 216], MN: [192, 132, 252], NA: [147, 197, 253], K: [196, 181, 253],
};
const ELEMENT_FALLBACK = [192, 132, 252];

/** The colour this element is drawn in, for anything that has to agree. */
export function elementRgb(element) {
  return ELEMENT_RGB[element] ?? ELEMENT_FALLBACK;
}

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
    /**
     * Only LigandMPNN reads heteroatoms -- `Model.encode` gates the whole
     * atom-context path on its model type. Drawing a cofactor the model is
     * blind to invites you to design a pocket around something it cannot see,
     * so the page turns this off for every other family.
     */
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
    // `makeSec` classifies helix/strand from C-alpha spacing, which means
    // nothing on a C1' trace -- left to itself it invents helices in a duplex.
    // Nucleotides are coil, and the renderer draws them as a fatter tube.
    this.nucleic = structure.nucleicAsResidues
      ? Uint8Array.from(structure.polytype, (p) => (p === POLYTYPE.PP ? 0 : 1))
      : null;
    const sec = smoothSec(makeSec(ca, L));
    this.sec = this.nucleic
      ? [...sec].map((c, i) => (this.nucleic[i] ? "C" : c)).join("")
      : sec;

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
      [
        {
          coords: this.ca,
          sec: this.sec,
          nucleic: this.nucleic,
          colourAt: (i) => this.colourAt(i),
        },
        ...this._ligandLayers(),
      ],
      { into: p, box, fit: this.fit, rot: this.camera.rot, zoom: this.camera.zoom, inset: INSET },
    );

    this._projectAll();
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

  /**
   * Ligand bonds, as layers for the renderer rather than a pass painted after
   * it.
   *
   * They used to be drawn on top of the finished cartoon, which meant a ligand
   * behind the fold still drew in front of it -- always, from every angle. The
   * comment excusing that said depth-sorting into the ribbon would mean
   * reaching inside the renderer. It does not: `drawTraces` already
   * sorts every segment of every layer it is given, so a bond expressed as a
   * two-point layer sorts against the ribbon for free, and picks up the same
   * halo and depth shading as everything else.
   *
   * Bonds only, no atom discs -- py2Dmol draws ligands as a bond skeleton and
   * it reads better: discs at this scale hide the pocket the ligand sits in.
   * Each bond is split at its midpoint into two half-bonds coloured by the
   * element at their own end, which is the usual convention and keeps the
   * element information the discs used to carry.
   */
  _ligandLayers() {
    const s = this.structure;
    const n = s.ligandType.length;
    if (!this.showLigand || !n) return [];
    if (this._bondCache?.n !== n || this._bondCache?.xyz !== s.ligandXyz) {
      const bonds = [];
      const bonded = new Uint8Array(n);
      for (let i = 0; i < n; i++) {
        if (METALS.has(s.ligandElements[i])) continue;
        for (let j = i + 1; j < n; j++) {
          if (METALS.has(s.ligandElements[j])) continue;
          const dx = s.ligandXyz[i * 3] - s.ligandXyz[j * 3];
          const dy = s.ligandXyz[i * 3 + 1] - s.ligandXyz[j * 3 + 1];
          const dz = s.ligandXyz[i * 3 + 2] - s.ligandXyz[j * 3 + 2];
          if (dx * dx + dy * dy + dz * dz <= BOND_A2) {
            bonds.push(i, j);
            bonded[i] = 1;
            bonded[j] = 1;
          }
        }
      }
      this._bondCache = { n, xyz: s.ligandXyz, bonds: Int32Array.from(bonds), bonded };
    }

    const { bonds, bonded } = this._bondCache;
    const layers = [];
    const rgbOf = (a) => ELEMENT_RGB[s.ligandElements[a]] ?? ELEMENT_FALLBACK;

    for (let b = 0; b < bonds.length; b += 2) {
      const i = bonds[b];
      const j = bonds[b + 1];
      const mx = (s.ligandXyz[i * 3] + s.ligandXyz[j * 3]) / 2;
      const my = (s.ligandXyz[i * 3 + 1] + s.ligandXyz[j * 3 + 1]) / 2;
      const mz = (s.ligandXyz[i * 3 + 2] + s.ligandXyz[j * 3 + 2]) / 2;
      for (const end of [i, j]) {
        const rgb = rgbOf(end);
        layers.push({
          coords: Float64Array.of(
            s.ligandXyz[end * 3], s.ligandXyz[end * 3 + 1], s.ligandXyz[end * 3 + 2],
            mx, my, mz,
          ),
          // Coil, so it comes out as a round tube -- a bond stick.
          sec: "CC",
          tubeA: BOND_TUBE_A,
          colourAt: () => ({ rgb, dim: 1 }),
        });
      }
    }

    // Whatever has no stick to sit on: metals, and any atom the cutoff left
    // unbonded. One-point layers, which the renderer draws as discs.
    for (let a = 0; a < n; a++) {
      if (bonded[a]) continue;
      const rgb = rgbOf(a);
      layers.push({
        coords: Float64Array.of(
          s.ligandXyz[a * 3], s.ligandXyz[a * 3 + 1], s.ligandXyz[a * 3 + 2],
        ),
        sec: "C",
        tubeA: ATOM_DISC_A,
        colourAt: () => ({ rgb, dim: 1 }),
      });
    }
    return layers;
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
