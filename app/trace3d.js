// VENDORED VERBATIM from sokrypton/CIRPIN-web, src/trace3d.js.
//
// Do not rewrite this file. It encodes a long list of specific fixes -- the
// background halo that makes crossing elements read as occluding rather than
// blending, per-quad depth sorting, ribbon widths in angstroms divided by the
// fit radius, the 0.45 depth floor, butt-capped loop halos -- and every one of
// them was arrived at by looking at renders that were wrong. A from-scratch
// version reproduces the bugs, which is exactly what happened here before this
// file was vendored.
//
// The ONLY modification is at the bottom: `setPaper()`, plus `PAPER_CSS` being
// `let` instead of `const` so it can follow it. CIRPIN-web is a light page with
// a fixed paper colour; this one has a dark mode, and `shade()` blends toward
// the page background by design, so the background has to be settable.
//
// Upstream: https://github.com/sokrypton/CIRPIN-web/blob/main/src/trace3d.js

// The Cα cartoon renderer, and the gesture that turns it.
//
// This was inside app.js, where it was reachable only by the search page. The atlas needs the same
// two things — a structure drawn the way the app draws it, and a view that rotates the way the
// app's does — and a second implementation of either would drift: the atlas had its own polyline
// trace and its own yaw/pitch camera, and the camera in particular was worse in a way that showed.
// So the renderer moved here unchanged and both pages import it.
//
// One substantive change in the move: drawTraces took the rotation and zoom from module-level
// globals in app.js. They are now opts, because two pages cannot share one global camera and the
// caller is the only one who knows which view is being drawn.

/** Size a canvas for the device pixel ratio and clear it. */
export function prep(canvas, cssH) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth;
  if (!w) return null;
  canvas.style.height = `${cssH}px`;
  canvas.width = w * dpr;
  canvas.height = cssH * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, cssH);
  return { ctx, w, h: cssH };
}



export function mul3(a, b) {
  const o = new Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      o[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
    }
  }
  return o;
}
export const rotY = (t) => [Math.cos(t), 0, Math.sin(t), 0, 1, 0, -Math.sin(t), 0, Math.cos(t)];
export const rotX = (t) => [1, 0, 0, 0, Math.cos(t), -Math.sin(t), 0, Math.sin(t), Math.cos(t)];

/**
 * Centre and radius for a set of coordinate arrays.
 *
 * pct sets which distance from the centre becomes the radius: 1 is the furthest
 * atom, which guarantees nothing is cropped but lets one outlying residue shrink
 * everything else. Below 1 it takes that quantile instead, so the bulk fills the
 * frame and the few stragglers run past the edge — the right trade when the view
 * is about the part in the middle.
 */
export function fitOf(arrays, pct = 1) {
  let cx = 0; let cy = 0; let cz = 0; let m = 0;
  for (const arr of arrays) {
    for (let i = 0; i < arr.length; i += 3) { cx += arr[i]; cy += arr[i + 1]; cz += arr[i + 2]; m++; }
  }
  cx /= m; cy /= m; cz /= m;
  const d = new Float64Array(m);
  let k = 0;
  for (const arr of arrays) {
    for (let i = 0; i < arr.length; i += 3) {
      const dx = arr[i] - cx; const dy = arr[i + 1] - cy; const dz = arr[i + 2] - cz;
      d[k++] = Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
  }
  d.sort();
  const r = Math.max(1, d[Math.min(m - 1, Math.floor((m - 1) * pct))] || 1);
  return { cx, cy, cz, r };
}

/**
 * How far a chain reaches from someone else's centre.
 *
 * fitOf finds a centre and a radius together; this answers the other question — given a centre
 * already chosen, what radius covers this chain? Needed because both side-by-side panels share the
 * query's centre, so the hit's extent has to be measured about that point rather than its own.
 * pct trims outliers the same way fitOf does.
 */
export function radiusAbout(arr, c, pct = 1) {
  const n = arr.length / 3;
  const d = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    d[i] = Math.hypot(arr[i * 3] - c.cx, arr[i * 3 + 1] - c.cy, arr[i * 3 + 2] - c.cz);
  }
  d.sort();
  return Math.max(1, d[Math.min(n - 1, Math.floor((n - 1) * pct))] || 1);
}

export function hexToRgb(hex) {
  return [
    parseInt(hex.slice(1, 3), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(5, 7), 16),
  ];
}

// Rendering technique borrowed from solab's contact viewer
// (solab/assets/js/contact.js:390-575), minus its cartoon ribbons:
//
//   shade()   depth is expressed by blending toward the BACKGROUND and returning
//             an opaque colour, not by lowering alpha. This is the one that
//             matters. Translucent strokes blend where the chain crosses itself
//             and the result reads as mush; opaque ones occlude, so the nearer
//             pass genuinely covers the farther one.
//   pe        a perspective factor 1/(1.9 - z*0.55) applied to both position and
//             stroke width, so near parts of the chain are drawn thicker.
//   width     a fraction of the projection radius rather than a pixel count, so
//             the trace keeps its weight at any canvas size.
//
// Catmull-Rom smoothing is also from there: sampling between Cα positions turns
// an angular polyline into something that reads as a tube.

export const PAPER = [237, 240, 244];

// Panel margin in css px. Was 2, which is visually none — a structure that reached the edge
// looked cropped even when it was not.
export const SIDE_INSET = 8;

// Perspective factor, written for coordinates on the unit sphere. Its maximum
// (nearest face, z = 1) is used to scale the projection so the structure fills
// the frame instead of sitting inside a margin the factor creates.
export const PE_MAX = 1 / (1.9 - 0.55);

// Secondary structure drives the GEOMETRY in every colour mode — flat ribbons
// for helix and strand, round tubes for loops — because that is what makes a
// fold readable: a protein is the packing of its secondary structure, not a
// wandering line. Assignment is Cα-only, from makeSec (a port of TM-align's
// make_sec, which classifies H/E/T/C from the distance pattern over five
// consecutive Cα). It needs no side chains and no model, and it is already
// exercised by the TM-align parity suite, whose SS-seeded stage matches the C++
// exactly — so the assignment here is known good rather than hoped good.
/**
 * Cartoon dimensions in ANGSTROMS, not in units of the structure's own size.
 *
 * These used to be fractions of the bounding sphere, which meant a helix drawn on
 * a 60-residue domain and one drawn on a 400-residue chain were the same width on
 * screen and therefore very different widths in the protein — the ribbon looked
 * chunky on something small and spindly on something large, and two structures
 * could not be compared side by side. Divided by the fit radius at use, they are
 * the same physical width everywhere, which is what a cartoon means.
 *
 * 1.3 Å half-width for a helix and 1.1 for a strand match what the old fractions
 * happened to give on a typical 200-residue domain, so nothing shifts for the
 * common case; it is the extremes that stop drifting.
 */
const SS_HALF_A = { H: 1.3, E: 1.1 };     // ribbon half-width, angstroms
const SS_TUBE_A = 0.27;                   // loop tube radius, angstroms

// --- nucleic acids -------------------------------------------------------
// Added for NA-MPNN; everything above is CIRPIN-web verbatim and a layer that
// does not set `nucleic` renders exactly as it did before.
//
// A nucleotide trace is stepped along C1', which sits 5.5-6.5 A from its
// neighbour where a C-alpha sits 3.8 A. That is past BREAK_A2 below, so every
// nucleotide became its own run and a nucleic chain drew nothing at all. And a
// 0.27 A tube is right for a protein loop but invisible next to a duplex.
const BREAK_A2 = 25;                      // protein: split past 5 A
const NA_BREAK_A2 = 64;                   // nucleic: split past 8 A
const NA_TUBE_A = 1.0;                    // fatter backbone, angstroms
// 0.85, not 0.70. Loops should recede, but most of a trace IS loop, so this multiplies almost
// everything on screen — and stacked on the depth shading it was the largest part of why the render
// looked washed out. Enough to let a helix or a strand come forward, not enough to bleach the rest.
const LOOP_DIM = 0.85;
// A helix turns every ~3.6 residues, so one quad per residue facets it visibly.
// Subdividing the centre line and the ribbon normal together smooths it.
const HELIX_SUB = 5;

// Strands: flat plates, or pleated as the Cα actually zigzag.
//
// The pleat is real geometry, so drawing it is not wrong — but it competes with
// the packing, which is what a fold is read by, and every other cartoon
// convention flattens it for that reason. Flip this to false to get the zigzag
// back; it is the only thing that has to change.
//
// Flattening needs both halves of the pleat removed. Smoothing only the centre
// line still leaves the ribbon twisting, because the side vector comes from local
// curvature and curvature is what alternates along a pleat. So the side vectors
// are averaged along the strand as well.
const FLAT_SHEETS = true;
const SHEET_SUB = 4;          // samples per residue when flattening
const SHEET_SMOOTH = 3;       // half-window for averaging strand side vectors

/**
 * Blend toward the page background; opaque, so overlaps occlude.
 *
 * The range is wider than the viewer this is adapted from, because that one
 * colours each secondary-structure type differently and gets its separation from
 * hue. Here every element in a domain shares one colour, so depth has to do the
 * work that hue was doing.
 */
export function shade(rgb, near, dim, extra = 1) {
  // The depth floor is 0.45, not 0.26. Three factors multiply here — depth, domain dim and the loop
  // dim — so a floor that looks reasonable alone put the back of an unparsed chain at a fifth of its
  // colour. 0.45 still separates front from back; it just does not bleach the far side.
  const f = (0.45 + 0.55 * near) * dim * extra;
  return `rgb(${Math.round(rgb[0] * f + PAPER[0] * (1 - f))},`
    + `${Math.round(rgb[1] * f + PAPER[1] * (1 - f))},`
    + `${Math.round(rgb[2] * f + PAPER[2] * (1 - f))})`;
}

export let PAPER_CSS = `rgb(${PAPER[0]},${PAPER[1]},${PAPER[2]})`;
const HALO = 3.2;   // css px of background drawn around each element

/** Catmull-Rom point between p1 and p2 (p0, p3 are the neighbours). */
function catmull(p0, p1, p2, p3, t, o) {
  const t2 = t * t;
  const t3 = t2 * t;
  for (let k = 0; k < 3; k++) {
    o[k] = 0.5 * ((2 * p1[k])
      + (-p0[k] + p2[k]) * t
      + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2
      + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3);
  }
}

const SUB = 4;          // samples per residue interval

/** No rotation, for a caller that has not chosen a view yet. */
export const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

/**
 * Draw one or more Cα traces into a canvas.
 *
 * `opts.rot` and `opts.zoom` are the camera. They used to be module globals shared by the search
 * page's three views, which is why they read like an afterthought here — but a global camera cannot
 * be shared by two pages, and a renderer that reaches outside itself for the view it is drawing
 * cannot be reused. Passed in, `orbit` below owns them.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {Array<{coords: Float64Array, sec?: string, colourAt: function}>} layers
 * @param {{rot?: number[], zoom?: number, maxH?: number, fit?: object, into?: object,
 *          box?: object, inset?: number}} opts
 */
export function drawTraces(canvas, layers, opts = {}) {
  const wCss = canvas.clientWidth;
  if (!wCss) return;
  const side = Math.min(wCss, opts.maxH ?? canvas.clientHeight ?? wCss);
  // opts.into lets a caller that has already sized and cleared the canvas draw
  // into part of it, which is how side-by-side gets two panels without two
  // canvases: one clear, two calls, one shared rotation and scale.
  const p = opts.into || prep(canvas, side);
  if (!p || !layers.length) return;
  const { ctx } = p;
  // centred, because the canvas stays full width while the square that holds the
  // structure may be narrower
  const box = opts.box || { x: (p.w - side) / 2, y: 0, size: side };

  // opts.fit shares one centre and radius across panels, so a small domain does
  // not get blown up to match a large one and the comparison stays honest.
  const { cx, cy, cz, r } = opts.fit || fitOf(layers.map((l) => l.coords));
  // Coordinates are normalised onto the unit sphere before projecting. The
  // perspective factor is written for that range — fed Ångströms it goes negative
  // past z of about 3.5 and turns the structure inside out.
  const R = ((box.size / 2 - (opts.inset ?? 10)) / PE_MAX) * (opts.zoom ?? 1);
  const M = opts.rot || IDENTITY;
  const segs = [];
  const q0 = [0, 0, 0]; const q1 = [0, 0, 0];

  for (const layer of layers) {
    const arr = layer.coords;
    const n = arr.length / 3;
    // Optional [n] flags marking nucleotides; absent means all protein.
    const nucleic = layer.nucleic || null;

    // rotate into view space once
    const P = new Float64Array(n * 3);
    for (let i = 0; i < n; i++) {
      const x = (arr[i * 3] - cx) / r;
      const y = (arr[i * 3 + 1] - cy) / r;
      const z = (arr[i * 3 + 2] - cz) / r;
      P[i * 3] = M[0] * x + M[1] * y + M[2] * z;
      P[i * 3 + 1] = M[3] * x + M[4] * y + M[5] * z;
      P[i * 3 + 2] = M[6] * x + M[7] * y + M[8] * z;
    }

    // contiguous runs, split where the chain genuinely breaks — a domain
    // boundary, a permutation join, or a gap in the model
    const runs = [];
    let start = 0;
    for (let i = 0; i + 1 < n; i++) {
      const dx = arr[(i + 1) * 3] - arr[i * 3];
      const dy = arr[(i + 1) * 3 + 1] - arr[i * 3 + 1];
      const dz = arr[(i + 1) * 3 + 2] - arr[i * 3 + 2];
      // A step touching a nucleotide gets the wider allowance.
      const limit = nucleic && (nucleic[i] || nucleic[i + 1]) ? NA_BREAK_A2 : BREAK_A2;
      if (dx * dx + dy * dy + dz * dz > limit) { runs.push([start, i]); start = i + 1; }
    }
    runs.push([start, n - 1]);

    const sec = layer.sec || '';
    const at = (i) => [P[i * 3], P[i * 3 + 1], P[i * 3 + 2]];
    const project = (v) => {
      const pe = 1 / (1.9 - v[2] * 0.55);
      return [box.x + box.size / 2 + v[0] * R * pe,
        box.y + box.size / 2 - v[1] * R * pe, v[2], pe];
    };

    // Side vector for the ribbon face. The curvature direction — where the two
    // neighbours sit relative to this residue — is what orients a real cartoon:
    // for a helix it points at the axis, for a strand it lies in the pleat. Cross
    // it with the tangent and the ribbon twists the way the backbone does.
    const sideOf = (i, lo, hi) => {
      const a = at(Math.max(lo, i - 1));
      const b = at(i);
      const c = at(Math.min(hi, i + 1));
      const t = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
      const k = [a[0] + c[0] - 2 * b[0], a[1] + c[1] - 2 * b[1], a[2] + c[2] - 2 * b[2]];
      let sx = t[1] * k[2] - t[2] * k[1];
      let sy = t[2] * k[0] - t[0] * k[2];
      let sz = t[0] * k[1] - t[1] * k[0];
      const m = Math.hypot(sx, sy, sz);
      if (m < 1e-9) return null;
      sx /= m; sy /= m; sz /= m;
      return [sx, sy, sz];
    };

    for (const [lo, hi] of runs) {
      if (hi <= lo) continue;
      // keep the ribbon face continuous: flip a side vector that reversed
      let prevSide = null;
      const sides = [];
      for (let i = lo; i <= hi; i++) {
        let sv = sideOf(i, lo, hi) || prevSide || [0, 0, 1];
        if (prevSide && (sv[0] * prevSide[0] + sv[1] * prevSide[1] + sv[2] * prevSide[2]) < 0) {
          sv = [-sv[0], -sv[1], -sv[2]];
        }
        sides.push(sv);
        prevSide = sv;
      }

      // Average side vectors within each strand so the plate does not twist with
      // the pleat. Done after the flip pass above, or averaging would cancel
      // vectors that merely point opposite ways.
      if (FLAT_SHEETS) {
        const src = sides.slice();
        for (let i = lo; i <= hi; i++) {
          if ((sec[i] || 'C') !== 'E') continue;
          let ax = 0; let ay = 0; let az = 0;
          for (let j = i - SHEET_SMOOTH; j <= i + SHEET_SMOOTH; j++) {
            if (j < lo || j > hi || (sec[j] || 'C') !== 'E') continue;
            const v = src[j - lo];
            ax += v[0]; ay += v[1]; az += v[2];
          }
          const m = Math.hypot(ax, ay, az);
          if (m > 1e-9) sides[i - lo] = [ax / m, ay / m, az / m];
        }
      }

      // Loops are emitted per segment, like the ribbon quads, so each sorts on
      // its own depth. Grouping a whole loop under one mean depth made a loop
      // that passes both in front of and behind a sheet draw entirely in front
      // of it. The beading that grouping was meant to avoid is handled instead
      // by butt-capping the halo, so a segment's halo widens sideways but never
      // reaches back along the chain over its neighbour's fill.
      let prevTube = null;
      for (let i = lo; i < hi; i++) {
        const c = layer.colourAt(i);
        if (!c) { prevTube = null; continue; }
        const t0 = sec[i] || 'C';
        const t1 = sec[i + 1] || 'C';
        const ribbon = (t0 === 'H' || t0 === 'E') && t0 === t1;

        if (ribbon) {
          prevTube = null;
          // Quads accumulate into ONE primitive per element. Drawing them as
          // separate primitives means the next quad's halo eats the previous
          // quad's fill, so a helix comes out looking like a row of beads — the
          // same failure the loops had. Grouped, every halo in the element is
          // stroked before any of its fills, which gives an outline around the
          // element's exterior and none along its internal joints. It also stops
          // another primitive sorting between two quads of the same ribbon and
          // slicing it open.
          const hw = SS_HALF_A[t0] / r;
          const s1 = sides[i - lo]; const s2 = sides[i + 1 - lo];
          const p0 = at(Math.max(lo, i - 1));
          const pa = at(i);
          const pb = at(i + 1);
          const p3 = at(Math.min(hi, i + 2));
          // Subdivide helices always, strands only when flattening them — the
          // pleated version wants exactly one quad per residue, since that is
          // where the real zigzag lives.
          const nsub = t0 === 'H' ? HELIX_SUB : (FLAT_SHEETS ? SHEET_SUB : 1);
          const edge = (u) => {
            catmull(p0, pa, pb, p3, u, q0);
            // the ribbon normal is carried along with the centre line, or the
            // extra quads would all lie in the plane of the first one
            let nx = s1[0] + (s2[0] - s1[0]) * u;
            let ny = s1[1] + (s2[1] - s1[1]) * u;
            let nz = s1[2] + (s2[2] - s1[2]) * u;
            const m = Math.hypot(nx, ny, nz) || 1;
            nx /= m; ny /= m; nz /= m;
            return [
              project([q0[0] + nx * hw, q0[1] + ny * hw, q0[2] + nz * hw]),
              project([q0[0] - nx * hw, q0[1] - ny * hw, q0[2] - nz * hw]),
            ];
          };
          let prevEdge = edge(0);
          for (let k = 1; k <= nsub; k++) {
            const nextEdge = edge(k / nsub);
            // Each quad is its own primitive, so depth sorting stays per quad and
            // a strand passing through a helix interleaves correctly. Grouping a
            // whole element under one mean depth was what made the sorting look
            // wrong: a helix running front to back got a single z, so anything
            // crossing it was drawn entirely in front of or behind all of it.
            segs.push({
              quad: [prevEdge[0], prevEdge[1], nextEdge[1], nextEdge[0]],
              z: (prevEdge[0][2] + nextEdge[0][2]) / 2,
              pe: (prevEdge[0][3] + nextEdge[0][3]) / 2,
              c,
              ss: t0,
            });
            prevEdge = nextEdge;
          }
        } else {
          // loop, or the junction between two different elements: round tube,
          // smoothed so it does not read as a chain of straight sticks
          const p0 = at(Math.max(lo, i - 1));
          const p1 = at(i);
          const p2 = at(i + 1);
          const p3 = at(Math.min(hi, i + 2));
          for (let k = 0; k <= SUB; k++) {
            catmull(p0, p1, p2, p3, k / SUB, q0);
            const A = project(q0);
            if (prevTube) {
              segs.push({
                x1: prevTube[0], y1: prevTube[1], x2: A[0], y2: A[1],
                z: (prevTube[2] + A[2]) / 2, pe: (prevTube[3] + A[3]) / 2, c, ss: 'C',
                na: nucleic ? Boolean(nucleic[i]) : false,
              });
            }
            prevTube = A;
          }
        }
      }
    }
  }

  segs.sort((a2, b2) => a2.z - b2.z);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (const g of segs) {
    let near = (g.z + 1) / 2;
    near = near < 0 ? 0 : near > 1 ? 1 : near;
    // Loops recede: the fold is carried by how its helices and sheets pack, and
    // a loop at full weight competes with the elements it merely connects.
    const weight = (g.c.dim ?? 1) * (g.quad ? 1 : LOOP_DIM);
    const col = shade(g.c.rgb, near, weight);
    // Halo first: a band of page background around this element. Because
    // primitives are painted back to front, it carves the nearer element out of
    // whatever lies behind, which is what separates a helix from the strand it
    // crosses when both are the same colour. Then the element itself, edged in a
    // darker tint so a ribbon reads as a surface rather than a flat patch.
    const dark = shade(g.c.rgb, near, weight, 0.62);
    if (g.quad) {
      const q = g.quad;
      // The halo is a copy of the quad widened PERPENDICULAR to the ribbon axis,
      // filled rather than stroked.
      //
      // Stroking the long edges instead puts half the halo's width inward, and
      // where the ribbon bends that inward half lands on the neighbouring quad's
      // fill — a background-coloured sliver at every corner, which is what the
      // white speckles were. Widening only across the ribbon keeps the halo
      // within this quad's own span along the chain, so it can never reach a
      // neighbour of the same element, while still separating this element from
      // anything else drawn behind it.
      const push = (pa, pb) => {
        const dx = pa[0] - pb[0];
        const dy = pa[1] - pb[1];
        const m = Math.hypot(dx, dy) || 1;
        return [pa[0] + (dx / m) * (HALO / 2), pa[1] + (dy / m) * (HALO / 2)];
      };
      const e0 = push(q[0], q[1]);
      const e1 = push(q[1], q[0]);
      const e2 = push(q[2], q[3]);
      const e3 = push(q[3], q[2]);
      ctx.fillStyle = PAPER_CSS;
      ctx.beginPath();
      ctx.moveTo(e0[0], e0[1]);
      ctx.lineTo(e1[0], e1[1]);
      ctx.lineTo(e2[0], e2[1]);
      ctx.lineTo(e3[0], e3[1]);
      ctx.closePath();
      ctx.fill();
      // convex, so it always fills; one polygon down an element's whole length
      // self-intersects on a pleated strand and leaves holes
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.moveTo(q[0][0], q[0][1]);
      for (let j = 1; j < 4; j++) ctx.lineTo(q[j][0], q[j][1]);
      ctx.closePath();
      ctx.fill();
      // long edges only: the short ones would draw a rung at every residue
      ctx.strokeStyle = dark;
      ctx.lineWidth = 1.1;
      ctx.lineCap = 'butt';
      ctx.beginPath();
      ctx.moveTo(q[0][0], q[0][1]); ctx.lineTo(q[3][0], q[3][1]);
      ctx.moveTo(q[1][0], q[1][1]); ctx.lineTo(q[2][0], q[2][1]);
      ctx.stroke();
      ctx.lineCap = 'round';
    } else {
      const tube = g.na ? NA_TUBE_A : SS_TUBE_A;
      const lw = Math.max(1.5, (tube / r) * 2 * R * g.pe);
      ctx.beginPath();
      ctx.moveTo(g.x1, g.y1);
      ctx.lineTo(g.x2, g.y2);
      // butt so the halo cannot extend past the segment's ends onto the
      // neighbouring segment's fill; round on the fill so the joints close up
      ctx.lineCap = 'butt';
      ctx.strokeStyle = PAPER_CSS;
      ctx.lineWidth = lw + HALO;
      ctx.stroke();
      ctx.lineCap = 'round';
      ctx.strokeStyle = col;
      ctx.lineWidth = lw;
      ctx.stroke();
    }
  }
}

// --- colour -----------------------------------------------------------------

/**
 * Position along the chain as colour: blue at the N terminus, red at the C.
 *
 * The default for a single-domain structure, where a by-domain colouring says
 * nothing — one colour for everything. It gives eight or so distinguishable
 * steps, enough to say "the green strand packs against the orange one" across a
 * 300-residue chain, and the legend carries the direction so the hues do not have
 * to be learned.
 */
export function hslRgb(h, sat, lit) {
  const c = (1 - Math.abs(2 * lit - 1)) * sat;
  const hp = ((h % 360) + 360) % 360 / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  const m = lit - c / 2;
  const t = [[c, x, 0], [x, c, 0], [0, c, x], [0, x, c], [x, 0, c], [c, 0, x]][Math.floor(hp) % 6];
  return [Math.round((t[0] + m) * 255), Math.round((t[1] + m) * 255), Math.round((t[2] + m) * 255)];
}

// 245 down to 0: blue, cyan, green, yellow, red. Lightness dips in the middle
// because pure yellow and cyan read far lighter than blue at a fixed value, and
// a ribbon that goes pale mid-chain looks like it is behind something.
export const spectrumRgb = (t) => {
  const u = Math.min(1, Math.max(0, t));
  return hslRgb(245 - 245 * u, 0.72, 0.52 - 0.10 * Math.sin(Math.PI * u));
};

// --- the gesture ------------------------------------------------------------

/**
 * A camera: an orientation, a zoom, and optionally a pan.
 *
 * Held as an object rather than as loose variables so several views can share one — turning the
 * superposition and then switching to side-by-side keeps the orientation you just found, which is
 * the whole reason it is shared.
 */
export function makeCamera(rot = IDENTITY, zoom = 1) {
  return { rot: rot.slice(), zoom, tx: 0, ty: 0 };
}

/**
 * Drag to rotate, wheel or pinch to zoom, double-click to reset.
 *
 * Rotation ACCUMULATES INTO A MATRIX rather than tracking a yaw and a pitch, and that is the
 * substance of this function rather than an implementation detail. Euler angles have to be clamped
 * near the poles or the view flips, and a clamp is felt as the drag going dead — push down past
 * vertical and nothing happens. Multiplying an incremental rotation onto the matrix has no poles:
 * every drag is a rotation about a screen axis, from wherever the view already is, so the gesture
 * never runs out of travel and never gains an unwanted roll.
 *
 * Pointer events with capture, not mouse events. A drag that leaves the element keeps arriving,
 * so releasing the button outside the canvas — which happens constantly on a small panel — ends
 * the drag properly instead of leaving the view believing a button is still down.
 *
 * @param {Element} el
 * @param {object} camera from makeCamera, mutated in place
 * @param {function} paint called after every change
 * @param {{zoomMin?: number, zoomMax?: number, gain?: number, wheelGain?: number, pan?: boolean,
 *          panOnly?: boolean, clickSlop?: number, onClick?: function, onReset?: function,
 *          onFirstDrag?: function}} opts
 */
export function orbit(el, camera, paint, opts = {}) {
  const zoomMin = opts.zoomMin ?? 0.5;
  const zoomMax = opts.zoomMax ?? 8;
  const gain = opts.gain ?? 0.01;
  const pts = new Map();
  let pinch0 = 0;
  let zoom0 = 1;
  let last = null;
  let downAt = null;
  let moved = false;

  const setZoom = (z) => { camera.zoom = Math.max(zoomMin, Math.min(zoomMax, z)); };
  const spread = () => {
    const [a, b] = [...pts.values()];
    return Math.hypot(a[0] - b[0], a[1] - b[1]);
  };

  el.style.touchAction = 'none';
  if (!el.style.cursor) el.style.cursor = 'grab';

  el.addEventListener('pointerdown', (e) => {
    pts.set(e.pointerId, [e.clientX, e.clientY]);
    if (pts.size === 2) { pinch0 = spread(); zoom0 = camera.zoom; }
    last = [e.clientX, e.clientY, e.shiftKey];
    downAt = [e.clientX, e.clientY];
    moved = false;
    try { el.setPointerCapture(e.pointerId); } catch { /* not capturable */ }
  });

  el.addEventListener('pointermove', (e) => {
    if (!last) return;
    if (pts.has(e.pointerId)) pts.set(e.pointerId, [e.clientX, e.clientY]);
    // A pinch must not also rotate, or the structure spins while being scaled.
    if (pts.size >= 2) {
      if (pinch0 > 4) setZoom(zoom0 * (spread() / pinch0));
      paint();
      return;
    }
    const dx = e.clientX - last[0];
    const dy = e.clientY - last[1];
    if (!moved && Math.hypot(dx, dy) > 0) {
      moved = true;
      if (opts.onFirstDrag) opts.onFirstDrag();
    }
    last = [e.clientX, e.clientY, last[2]];
    // panOnly is for a view with nothing to rotate — a 2D layout, where a drag can only mean move.
    if (opts.pan && (last[2] || opts.panOnly)) {
      camera.tx += dx;
      camera.ty += dy;
    } else {
      camera.rot = mul3(mul3(rotX(dy * gain), rotY(dx * gain)), camera.rot);
    }
    paint();
  });

  const stop = (e) => {
    // A press that did not travel is a click, not a rotation. The threshold is generous because the
    // same gesture does both and a shaky click still has to select.
    if (opts.onClick && downAt && pts.size <= 1
        && Math.abs(e.clientX - downAt[0]) < (opts.clickSlop ?? 4)
        && Math.abs(e.clientY - downAt[1]) < (opts.clickSlop ?? 4)) {
      opts.onClick(e);
    }
    downAt = null;
    pts.delete(e.pointerId);
    if (pts.size < 2) pinch0 = 0;
    // the remaining finger becomes the rotation anchor rather than jumping
    if (pts.size === 1) { const [p] = [...pts.values()]; last = [p[0], p[1], false]; }
    if (!pts.size) last = null;
  };
  el.addEventListener('pointerup', stop);
  el.addEventListener('pointercancel', stop);

  el.addEventListener('wheel', (e) => {
    e.preventDefault();
    if (opts.onFirstDrag) opts.onFirstDrag();
    // A trackpad pinch arrives as a wheel event with ctrlKey set, and wants a larger gain than a
    // scroll wheel does.
    setZoom(camera.zoom * Math.exp(-e.deltaY * (e.ctrlKey ? 0.012 : (opts.wheelGain ?? 0.0022))));
    paint();
  }, { passive: false });

  el.addEventListener('dblclick', (e) => {
    e.preventDefault();
    if (opts.onReset) opts.onReset();
    paint();
  });

  return { camera };
}


// --- local addition ---------------------------------------------------------

/**
 * Point the depth blend and the halo at a different page background.
 *
 * Everything above blends toward PAPER to express depth and to carve elements
 * out of what lies behind them. That only works if PAPER is actually the colour
 * behind the canvas, so a page with a dark mode has to say so.
 */
export function setPaper(rgb) {
  PAPER[0] = rgb[0];
  PAPER[1] = rgb[1];
  PAPER[2] = rgb[2];
  PAPER_CSS = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
}
