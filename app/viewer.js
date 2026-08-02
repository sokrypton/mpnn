// A dependency-free cartoon renderer on a 2D canvas.
//
// Everything is software projected and painter-sorted. That sounds slow, but a
// protein is a few thousand quads and Canvas2D handles it at 60 fps while
// costing nothing in bundle size and working identically everywhere.
//
// Secondary structure drives the geometry: flat ribbons for helices and
// strands, round tubes for loops. Residue picking is done against the projected
// spline, so clicking anywhere on the ribbon selects the residue under it.

const SUBDIV = 6;          // spline samples per residue
const HELIX_HALF = 1.5;    // ribbon half-width, angstroms
const STRAND_HALF = 1.3;
const LOOP_RADIUS = 0.35;

export const SS = { LOOP: 0, HELIX: 1, STRAND: 2 };

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function norm(a) {
  const n = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
}
function dist(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

/** Virtual CA torsion, in degrees. */
function torsion(p0, p1, p2, p3) {
  const b0 = sub(p1, p0);
  const b1 = sub(p2, p1);
  const b2 = sub(p3, p2);
  const n1 = cross(b0, b1);
  const n2 = cross(b1, b2);
  const m = cross(n1, norm(b1));
  const x = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];
  const y = m[0] * n2[0] + m[1] * n2[1] + m[2] * n2[2];
  return (Math.atan2(y, x) * 180) / Math.PI;
}

/**
 * Assign secondary structure from C-alpha geometry alone (a P-SEA style rule).
 *
 * We never have sidechains or hydrogens here, and DSSP would need both, so this
 * trades a little accuracy for not needing them. Short runs are dissolved so the
 * ribbon does not flicker between states.
 *
 * @param {Float32Array} CA [L, 3]
 * @returns {Uint8Array} [L] one of SS.*
 */
export function assignSecondaryStructure(CA, L, chainLabels) {
  const ss = new Uint8Array(L);
  const at = (i) => [CA[i * 3], CA[i * 3 + 1], CA[i * 3 + 2]];
  const sameChain = (i, j) => !chainLabels || chainLabels[i] === chainLabels[j];

  for (let i = 0; i < L; i++) {
    if (i < 1 || i + 3 >= L || !sameChain(i - 1, i + 3)) continue;
    const d13 = dist(at(i), at(i + 3));
    const d14 = i + 4 < L && sameChain(i, i + 4) ? dist(at(i), at(i + 4)) : 99;
    const tau = torsion(at(i - 1), at(i), at(i + 1), at(i + 2));

    if (d13 > 4.5 && d13 < 6.6 && d14 > 5.0 && d14 < 7.6 && tau > 30 && tau < 100) {
      ss[i] = SS.HELIX;
    } else if (d13 > 8.5 && (tau < -140 || tau > 140)) {
      ss[i] = SS.STRAND;
    }
  }

  // Dissolve runs shorter than three residues.
  for (let i = 0; i < L;) {
    let j = i;
    while (j < L && ss[j] === ss[i]) j++;
    if (ss[i] !== SS.LOOP && j - i < 3) ss.fill(SS.LOOP, i, j);
    i = j;
  }
  return ss;
}

/** Centre and radius of the coordinate cloud. */
export function boundingSphere(CA, L) {
  const c = [0, 0, 0];
  for (let i = 0; i < L; i++) {
    c[0] += CA[i * 3];
    c[1] += CA[i * 3 + 1];
    c[2] += CA[i * 3 + 2];
  }
  c[0] /= L || 1;
  c[1] /= L || 1;
  c[2] /= L || 1;
  // 97th percentile rather than the maximum, so one flapping terminus does not
  // shrink the whole structure.
  const radii = new Float64Array(L);
  for (let i = 0; i < L; i++) {
    radii[i] = dist([CA[i * 3], CA[i * 3 + 1], CA[i * 3 + 2]], c);
  }
  radii.sort();
  return { center: c, radius: Math.max(radii[Math.floor(0.97 * (L - 1))] || 1, 1) };
}

/**
 * Catmull-Rom through the C-alpha trace, with a carried reference frame so the
 * ribbon does not twist between consecutive residues.
 *
 * @returns {{pts: Float32Array, side: Float32Array, residue: Int32Array, n: number}}
 */
function buildSpline(CA, L, chainLabels) {
  const segments = [];
  let start = 0;
  for (let i = 1; i <= L; i++) {
    if (i === L || (chainLabels && chainLabels[i] !== chainLabels[i - 1])) {
      if (i - start >= 2) segments.push([start, i]);
      start = i;
    }
  }

  const total = segments.reduce((acc, [a, b]) => acc + (b - a - 1) * SUBDIV + 1, 0);
  const pts = new Float32Array(total * 3);
  const side = new Float32Array(total * 3);
  const residue = new Int32Array(total);
  let w = 0;

  for (const [a, b] of segments) {
    const get = (i) => {
      const c = Math.min(Math.max(i, a), b - 1);
      return [CA[c * 3], CA[c * 3 + 1], CA[c * 3 + 2]];
    };
    let prevSide = null;
    for (let i = a; i < b - 1; i++) {
      for (let s = 0; s < SUBDIV; s++) {
        const t = s / SUBDIV;
        const p0 = get(i - 1), p1 = get(i), p2 = get(i + 1), p3 = get(i + 2);
        const t2 = t * t;
        const t3 = t2 * t;
        const p = [0, 0, 0];
        const tangent = [0, 0, 0];
        for (let d = 0; d < 3; d++) {
          p[d] = 0.5 * ((2 * p1[d])
            + (-p0[d] + p2[d]) * t
            + (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * t2
            + (-p0[d] + 3 * p1[d] - 3 * p2[d] + p3[d]) * t3);
          tangent[d] = 0.5 * ((-p0[d] + p2[d])
            + 2 * (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * t
            + 3 * (-p0[d] + 3 * p1[d] - 3 * p2[d] + p3[d]) * t2);
        }
        // Ribbon normal: perpendicular to the local curve plane, flipped to
        // stay on the same side as the previous sample.
        let n = norm(cross(tangent, sub(p2, p0)));
        if (!Number.isFinite(n[0])) n = [0, 0, 1];
        if (prevSide && (n[0] * prevSide[0] + n[1] * prevSide[1] + n[2] * prevSide[2]) < 0) {
          n = [-n[0], -n[1], -n[2]];
        }
        prevSide = n;
        pts.set(p, w * 3);
        side.set(n, w * 3);
        residue[w] = t < 0.5 ? i : i + 1;
        w++;
      }
    }
    const last = get(b - 1);
    pts.set(last, w * 3);
    side.set(prevSide ?? [0, 0, 1], w * 3);
    residue[w] = b - 1;
    w++;
  }
  return { pts, side, residue, n: w };
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

export function identity() {
  return [1, 0, 0, 0, 1, 0, 0, 0, 1];
}

function matMul(a, b) {
  const out = new Array(9).fill(0);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < 3; k++) out[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
    }
  }
  return out;
}

function rotX(t) {
  const c = Math.cos(t), s = Math.sin(t);
  return [1, 0, 0, 0, c, -s, 0, s, c];
}

function rotY(t) {
  const c = Math.cos(t), s = Math.sin(t);
  return [c, 0, s, 0, 1, 0, -s, 0, c];
}

export class Camera {
  constructor() {
    this.rot = identity();
    this.zoom = 1;
    this.pan = [0, 0];
  }

  reset() {
    this.rot = identity();
    this.zoom = 1;
    this.pan = [0, 0];
  }

  /** Accumulate into the matrix rather than tracking Euler angles. */
  drag(dx, dy) {
    this.rot = matMul(matMul(rotY(dx * 0.01), rotX(dy * 0.01)), this.rot);
  }
}

/** Attach pointer handlers. Returns a detach function. */
export function attachControls(canvas, camera, redraw, { onHover, onPick, onBoxSelect } = {}) {
  let dragging = null;
  let moved = 0;

  const localPoint = (event) => {
    const rect = canvas.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  };

  const onDown = (event) => {
    canvas.setPointerCapture(event.pointerId);
    const p = localPoint(event);
    dragging = { x: event.clientX, y: event.clientY, shift: event.shiftKey, from: p, to: p };
    moved = 0;
  };

  const onMove = (event) => {
    const p = localPoint(event);
    if (!dragging) {
      if (onHover) onHover(p, event);
      return;
    }
    const dx = event.clientX - dragging.x;
    const dy = event.clientY - dragging.y;
    moved += Math.abs(dx) + Math.abs(dy);
    dragging.x = event.clientX;
    dragging.y = event.clientY;
    dragging.to = p;
    if (dragging.shift) {
      canvas.dispatchEvent(new CustomEvent("boxupdate", { detail: dragging }));
    } else {
      camera.drag(dx, dy);
    }
    redraw();
  };

  const onUp = (event) => {
    if (dragging && dragging.shift && moved > 4 && onBoxSelect) {
      onBoxSelect(dragging.from, dragging.to, event);
    } else if (dragging && moved < 4 && onPick) {
      onPick(localPoint(event), event);
    }
    dragging = null;
    redraw();
  };

  const onWheel = (event) => {
    event.preventDefault();
    // Trackpads deliver many small deltas; scroll wheels a few large ones.
    const step = event.deltaMode === 0 ? event.deltaY * 0.002 : event.deltaY * 0.05;
    camera.zoom = Math.min(Math.max(camera.zoom * Math.exp(-step), 0.2), 12);
    redraw();
  };

  const onDouble = () => {
    camera.reset();
    redraw();
  };

  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerup", onUp);
  canvas.addEventListener("pointercancel", onUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });
  canvas.addEventListener("dblclick", onDouble);

  return () => {
    canvas.removeEventListener("pointerdown", onDown);
    canvas.removeEventListener("pointermove", onMove);
    canvas.removeEventListener("pointerup", onUp);
    canvas.removeEventListener("pointercancel", onUp);
    canvas.removeEventListener("wheel", onWheel);
    canvas.removeEventListener("dblclick", onDouble);
  };
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

export class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.camera = new Camera();
    this.structure = null;
    this.spline = null;
    this.ss = null;
    /** (residueIndex) => "#rrggbb" */
    this.colorAt = () => "#7dd3fc";
    this.highlight = new Set();
    this.dim = new Set();
    this.background = "#0b1220";
    this.showLigand = true;
    this.projected = null;
    this.box = null;
  }

  setStructure(structure) {
    this.structure = structure;
    const CA = new Float32Array(structure.L * 3);
    for (let i = 0; i < structure.L; i++) {
      CA[i * 3] = structure.X[i * 12 + 3];
      CA[i * 3 + 1] = structure.X[i * 12 + 4];
      CA[i * 3 + 2] = structure.X[i * 12 + 5];
    }
    this.CA = CA;
    this.ss = assignSecondaryStructure(CA, structure.L, structure.chainLabels);
    this.spline = buildSpline(CA, structure.L, structure.chainLabels);
    this.sphere = boundingSphere(CA, structure.L);
    this.camera.reset();
  }

  /** Screen-space transform for the current canvas size and camera. */
  _transform() {
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (this.canvas.width !== w * dpr || this.canvas.height !== h * dpr) {
      this.canvas.width = w * dpr;
      this.canvas.height = h * dpr;
    }
    const scale = (Math.min(w, h) / (2.05 * this.sphere.radius)) * this.camera.zoom;
    return { dpr, w, h, scale, cx: w / 2 + this.camera.pan[0], cy: h / 2 + this.camera.pan[1] };
  }

  _project(p, t) {
    const r = this.camera.rot;
    const x = p[0] - this.sphere.center[0];
    const y = p[1] - this.sphere.center[1];
    const z = p[2] - this.sphere.center[2];
    const rx = r[0] * x + r[1] * y + r[2] * z;
    const ry = r[3] * x + r[4] * y + r[5] * z;
    const rz = r[6] * x + r[7] * y + r[8] * z;
    return [t.cx + rx * t.scale, t.cy - ry * t.scale, rz];
  }

  /** Nearest residue to a canvas point, or -1. */
  pick(point, tolerance = 14) {
    if (!this.projected) return -1;
    let best = -1;
    let bestScore = Infinity;
    for (let i = 0; i < this.projected.length; i += 4) {
      const dx = this.projected[i] - point[0];
      const dy = this.projected[i + 1] - point[1];
      const d2 = dx * dx + dy * dy;
      if (d2 > tolerance * tolerance) continue;
      // Prefer nearer-to-camera hits when several overlap.
      const score = d2 - this.projected[i + 2] * 2;
      if (score < bestScore) {
        bestScore = score;
        best = this.projected[i + 3];
      }
    }
    return best;
  }

  /** All residues whose projected point falls inside a screen rectangle. */
  pickBox(from, to) {
    const hits = new Set();
    if (!this.projected) return hits;
    const x0 = Math.min(from[0], to[0]);
    const x1 = Math.max(from[0], to[0]);
    const y0 = Math.min(from[1], to[1]);
    const y1 = Math.max(from[1], to[1]);
    for (let i = 0; i < this.projected.length; i += 4) {
      const x = this.projected[i];
      const y = this.projected[i + 1];
      if (x >= x0 && x <= x1 && y >= y0 && y <= y1) hits.add(this.projected[i + 3]);
    }
    return hits;
  }

  draw() {
    const ctx = this.ctx;
    const t = this._transform();
    ctx.setTransform(t.dpr, 0, 0, t.dpr, 0, 0);
    ctx.clearRect(0, 0, t.w, t.h);
    if (!this.structure) return;

    const { pts, side, residue, n } = this.spline;
    const projected = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      const p = this._project([pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]], t);
      projected[i * 4] = p[0];
      projected[i * 4 + 1] = p[1];
      projected[i * 4 + 2] = p[2];
      projected[i * 4 + 3] = residue[i];
    }
    this.projected = projected;

    // Painter's algorithm over ribbon quads.
    const quads = [];
    for (let i = 0; i + 1 < n; i++) {
      const r = residue[i];
      const kind = this.ss[r];
      const half = kind === SS.HELIX ? HELIX_HALF : kind === SS.STRAND ? STRAND_HALF : LOOP_RADIUS;
      const a = i * 3;
      const b = (i + 1) * 3;
      const corners = [
        [pts[a] + side[a] * half, pts[a + 1] + side[a + 1] * half, pts[a + 2] + side[a + 2] * half],
        [pts[a] - side[a] * half, pts[a + 1] - side[a + 1] * half, pts[a + 2] - side[a + 2] * half],
        [pts[b] - side[b] * half, pts[b + 1] - side[b + 1] * half, pts[b + 2] - side[b + 2] * half],
        [pts[b] + side[b] * half, pts[b + 1] + side[b + 1] * half, pts[b + 2] + side[b + 2] * half],
      ].map((c) => this._project(c, t));
      const depth = (corners[0][2] + corners[2][2]) / 2;
      quads.push({ corners, depth, residue: r, kind });
    }

    if (this.showLigand && this.structure.ligandType.length) {
      const lig = this.structure.ligandXyz;
      for (let i = 0; i < this.structure.ligandType.length; i++) {
        const p = this._project([lig[i * 3], lig[i * 3 + 1], lig[i * 3 + 2]], t);
        quads.push({ ligand: true, at: p, depth: p[2], element: this.structure.ligandElements[i] });
      }
    }

    quads.sort((x, y) => x.depth - y.depth);

    const near = -this.sphere.radius;
    const far = this.sphere.radius;
    for (const q of quads) {
      if (q.ligand) {
        const shade = depthMix(ELEMENT_COLORS[q.element] ?? "#c084fc", q.depth, near, far,
          this.background);
        ctx.fillStyle = shade;
        ctx.beginPath();
        ctx.arc(q.at[0], q.at[1], 3.2 * t.scale * 0.55 + 1.5, 0, Math.PI * 2);
        ctx.fill();
        continue;
      }
      const dimmed = this.dim.has(q.residue);
      let color = this.colorAt(q.residue);
      if (dimmed) color = mix(color, this.background, 0.62);
      ctx.fillStyle = depthMix(color, q.depth, near, far, this.background);
      ctx.beginPath();
      ctx.moveTo(q.corners[0][0], q.corners[0][1]);
      for (let c = 1; c < 4; c++) ctx.lineTo(q.corners[c][0], q.corners[c][1]);
      ctx.closePath();
      ctx.fill();
      if (this.highlight.has(q.residue)) {
        ctx.strokeStyle = "#fbbf24";
        ctx.lineWidth = 1.6;
        ctx.stroke();
      }
    }

    if (this.box) {
      ctx.strokeStyle = "#fbbf24";
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1;
      ctx.strokeRect(
        Math.min(this.box.from[0], this.box.to[0]),
        Math.min(this.box.from[1], this.box.to[1]),
        Math.abs(this.box.to[0] - this.box.from[0]),
        Math.abs(this.box.to[1] - this.box.from[1]),
      );
      ctx.setLineDash([]);
    }
  }
}

const ELEMENT_COLORS = {
  C: "#94a3b8", N: "#60a5fa", O: "#f87171", S: "#fbbf24", P: "#fb923c",
  F: "#4ade80", CL: "#4ade80", BR: "#f97316", I: "#a78bfa",
  FE: "#f97316", ZN: "#a3a3a3", MG: "#86efac", CA: "#d4d4d8", MN: "#c084fc",
};

export function hexToRgb(hex) {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

export function rgbToHex([r, g, b]) {
  return `#${((1 << 24) | (Math.round(r) << 16) | (Math.round(g) << 8) | Math.round(b))
    .toString(16).slice(1)}`;
}

export function mix(a, b, amount) {
  const ca = hexToRgb(a);
  const cb = hexToRgb(b);
  return rgbToHex(ca.map((v, i) => v + (cb[i] - v) * amount));
}

/** Blend toward the background with depth so far geometry recedes. */
function depthMix(color, depth, near, far, background) {
  const t = Math.min(Math.max((depth - near) / (far - near), 0), 1);
  return mix(color, background, 0.55 * (1 - t));
}
