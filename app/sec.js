// VENDORED VERBATIM from sokrypton/CIRPIN-web, src/tmalign.js.
//
// `makeSec` is that repository's port of TM-align's `make_sec` (TMalign.cpp:2466):
// secondary structure from C-alpha geometry alone, classified from the distance
// pattern over five consecutive C-alpha. No side chains, no hydrogens, no model
// -- exactly the situation here, since MPNN is handed a backbone.
//
// It is also already checked: CIRPIN-web's TM-align parity suite exercises the
// SS-seeded alignment stage against the C++ to 5e-11, so this assignment is
// known good rather than hoped good. The hand-rolled heuristic that used to
// live in app/viewer.js was neither.
//
// Upstream: https://github.com/sokrypton/CIRPIN-web/blob/main/src/tmalign.js

function dist2(x, xi, y, yi) {
  const d1 = x[xi] - y[yi];
  const d2 = x[xi + 1] - y[yi + 1];
  const d3 = x[xi + 2] - y[yi + 2];
  return d1 * d1 + d2 * d2 + d3 * d3;
}

function secStr(dis13, dis14, dis15, dis24, dis25, dis35) {
  let delta = 2.1;
  if (Math.abs(dis15 - 6.37) < delta && Math.abs(dis14 - 5.18) < delta
    && Math.abs(dis25 - 5.18) < delta && Math.abs(dis13 - 5.45) < delta
    && Math.abs(dis24 - 5.45) < delta && Math.abs(dis35 - 5.45) < delta) return 'H';
  delta = 1.42;
  if (Math.abs(dis15 - 13) < delta && Math.abs(dis14 - 10.4) < delta
    && Math.abs(dis25 - 10.4) < delta && Math.abs(dis13 - 6.1) < delta
    && Math.abs(dis24 - 6.1) < delta && Math.abs(dis35 - 6.1) < delta) return 'E';
  if (dis15 < 8) return 'T';
  return 'C';
}

/** make_sec (TMalign.cpp:2466). Returns a string of length len. */
export function makeSec(x, len) {
  const sec = new Array(len);
  for (let i = 0; i < len; i++) {
    sec[i] = 'C';
    const j1 = i - 2; const j2 = i - 1; const j3 = i; const j4 = i + 1; const j5 = i + 2;
    if (j1 >= 0 && j5 < len) {
      sec[i] = secStr(
        Math.sqrt(dist2(x, j1 * 3, x, j3 * 3)),
        Math.sqrt(dist2(x, j1 * 3, x, j4 * 3)),
        Math.sqrt(dist2(x, j1 * 3, x, j5 * 3)),
        Math.sqrt(dist2(x, j2 * 3, x, j4 * 3)),
        Math.sqrt(dist2(x, j2 * 3, x, j5 * 3)),
        Math.sqrt(dist2(x, j3 * 3, x, j5 * 3)),
      );
    }
  }
  return sec.join('');
}

/**
 * Tidy a secondary-structure string for DISPLAY.
 *
 * make_sec is deliberately conservative — it needs five consecutive Cα matching
 * an idealised geometry, so it marks element cores and leaves their ends coil.
 * Good for seeding an alignment, poor for drawing: on a PDZ domain only about a
 * quarter of residues land in a run long enough to be a ribbon, so a mostly-β
 * fold renders as mostly loop.
 *
 * TM-align's own smooth() (TMalign.cpp:2392) is NOT the fix, though it looks
 * like it. It is a pruning pass built to clean up alignment seeds: it deletes
 * isolated singles AND isolated pairs, and bridging single gaps does not recover
 * what that costs. Measured, it moves the ribbon fraction the wrong way — 30% to
 * 23% on a PDZ domain, 59% to 58% on Ras.
 *
 * So this does the two defensible things only: bridge a one-residue gap inside an
 * element, since a run interrupted by one marginal residue is one element; and
 * drop a lone residue with no element neighbours, since that is speckle. It does
 * not grow element ends. Inflating them would draw structure that was not
 * assigned, which is worse than a fold that looks loopy.
 *
 * NOT used by the aligner. tmalignMain must keep the raw assignment or the
 * SS-seeded stage stops matching the C++, which is checked to 5e-11.
 */
export function smoothSec(sec) {
  const a = [...sec];
  // bridge a single-residue gap: x-x => xxx
  for (let i = 0; i + 2 < a.length; i++) {
    for (const j of ['H', 'E']) {
      if (a[i] === j && a[i + 1] !== j && a[i + 2] === j) a[i + 1] = j;
    }
  }
  // drop a lone element residue with no element neighbour on either side
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== 'H' && a[i] !== 'E') continue;
    const before = i > 0 && a[i - 1] === a[i];
    const after = i + 1 < a.length && a[i + 1] === a[i];
    if (!before && !after) a[i] = 'C';
  }
  return a.join('');
}
