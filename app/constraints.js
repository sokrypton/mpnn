// Per-position amino-acid bias and omissions.
//
// No DOM and no page state, so it can be exercised from node.
//
// Two things here are load-bearing and were both bugs before.
//
// **Letters are keyed canonically.** The alphabets are not merely different
// lengths, they are in different *orders*: `ACDEFGHIKLMNPQRSTVWYX` for the
// protein models against `ARNDCQEGHILKMFPSTWYVXacgtx...` for NA-MPNN. Index 1
// is C in one and R in the other. Storing a model-relative index meant that
// omitting cysteine and then switching model family silently omitted arginine
// instead -- the structure is re-parsed on that switch, which cleared the
// per-position overrides but not the row they were derived from. Everything
// stored here is indexed into `CANON` and translated at the boundary.
//
// **Overrides are sparse and per letter, not row snapshots.** The old model
// copied the whole default row the first time a position was touched, which
// froze that position against every later edit of the defaults -- surprising,
// undocumented, and impossible to undo one letter at a time. Here a position
// holds only the letters it actually overrides, so editing a default still
// shows through everywhere it was not specifically contradicted.

import { NA_ALPHABET } from "../mpnn/na.js";

/**
 * The widest alphabet, used as the canonical index space.
 *
 * NA-MPNN's 33 letters are a superset of the protein models' 21 in content, if
 * not in order, so every letter either family can name has a place here.
 */
export const CANON = NA_ALPHABET;
const CANON_INDEX = new Map([...CANON].map((letter, i) => [letter, i]));

/** The canonical index of a letter in some model's alphabet. */
export function canonOf(alphabet, v) {
  const ci = CANON_INDEX.get(alphabet[v]);
  if (ci === undefined) throw new Error(`letter ${alphabet[v]} is not in CANON`);
  return ci;
}

export function createConstraints() {
  return {
    /** The default row: what applies to a position that says nothing else. */
    bias: new Float32Array(CANON.length),
    omit: new Uint8Array(CANON.length),
    /** @type {Map<number, Map<number, number>>} position -> canonical index -> bias */
    posBias: new Map(),
    /**
     * Position -> canonical index -> 1 omit, 0 allow.
     *
     * Tri-state on purpose: absent means "inherit the default", so a position
     * can *un*-omit a letter the defaults omit. "Omit cysteine everywhere
     * except the catalytic one" needs that and could not be said before.
     *
     * @type {Map<number, Map<number, 0|1>>}
     */
    posOmit: new Map(),
  };
}

/** The bias in force at a position, for one canonical letter. */
export function effBias(c, pos, ci) {
  return c.posBias.get(pos)?.get(ci) ?? c.bias[ci];
}

/** Whether a letter is omitted at a position. */
export function effOmit(c, pos, ci) {
  return (c.posOmit.get(pos)?.get(ci) ?? c.omit[ci]) === 1;
}

/** Positions carrying any override at all. */
export function overriddenPositions(c) {
  return new Set([...c.posBias.keys(), ...c.posOmit.keys()]);
}

function setDelta(map, pos, ci, value) {
  let row = map.get(pos);
  if (row === undefined) {
    row = new Map();
    map.set(pos, row);
  }
  row.set(ci, value);
}

function dropDelta(map, pos, ci) {
  const row = map.get(pos);
  if (row === undefined) return;
  row.delete(ci);
  if (row.size === 0) map.delete(pos);
}

/**
 * Set a bias. `positions` null means the default row.
 *
 * Writing the default's own value at a position deletes the override instead of
 * storing a redundant one, so typing a value back is the same gesture as
 * clearing it and the "N positions carry an override" count stays honest.
 */
export function setBias(c, positions, ci, value) {
  if (positions === null) {
    c.bias[ci] = value;
    return;
  }
  for (const pos of positions) {
    if (value === c.bias[ci]) dropDelta(c.posBias, pos, ci);
    else setDelta(c.posBias, pos, ci, value);
  }
}

/** Set an omission. `positions` null means the default row. */
export function setOmit(c, positions, ci, on) {
  if (positions === null) {
    c.omit[ci] = on ? 1 : 0;
    return;
  }
  for (const pos of positions) {
    if ((on ? 1 : 0) === c.omit[ci]) dropDelta(c.posOmit, pos, ci);
    else setDelta(c.posOmit, pos, ci, on ? 1 : 0);
  }
}

/** Forget one letter's overrides at a position, so the default shows through. */
export function clearLetterAt(c, pos, ci) {
  dropDelta(c.posBias, pos, ci);
  dropDelta(c.posOmit, pos, ci);
}

/** Forget every override at a position. */
export function clearPosition(c, pos) {
  c.posBias.delete(pos);
  c.posOmit.delete(pos);
}

/** Forget everything, defaults included. */
export function clearAll(c) {
  c.bias.fill(0);
  c.omit.fill(0);
  c.posBias.clear();
  c.posOmit.clear();
}

/** Forget the per-position overrides, keeping the defaults. */
export function clearOverrides(c) {
  c.posBias.clear();
  c.posOmit.clear();
}

/**
 * What to show for one letter across a set of positions.
 *
 * `positions` null means the default row. `mixed` is the honest answer when
 * the selection disagrees -- the old code claimed to report this and in fact
 * read whichever position happened to be first, so a heterogeneous selection
 * displayed one member's value and editing silently flattened the rest.
 *
 * @returns {{value: number, mixed: boolean, omit: boolean|"mixed",
 *            overridden: "none"|"some"|"all"}}
 */
export function resolveOver(c, positions, ci) {
  if (positions === null || positions.length === 0) {
    return {
      value: c.bias[ci],
      mixed: false,
      omit: c.omit[ci] === 1,
      overridden: "none",
    };
  }
  let value = effBias(c, positions[0], ci);
  let omit = effOmit(c, positions[0], ci);
  let mixed = false;
  let omitMixed = false;
  let overrides = 0;
  for (const pos of positions) {
    const b = effBias(c, pos, ci);
    const o = effOmit(c, pos, ci);
    if (b !== value) mixed = true;
    if (o !== omit) omitMixed = true;
    if (c.posBias.get(pos)?.has(ci) || c.posOmit.get(pos)?.has(ci)) overrides++;
  }
  return {
    value,
    mixed,
    omit: omitMixed ? "mixed" : omit,
    overridden: overrides === 0 ? "none"
      : overrides === positions.length ? "all" : "some",
  };
}

/**
 * What one position overrides, for a table row.
 *
 * @param {string} alphabet the current model's alphabet
 * @param {number[]} letters model indices the sampler may draw (`biasLetters`)
 */
export function rowSummary(c, pos, alphabet, letters) {
  const biasEntries = [];
  const omitLetters = [];
  for (const v of letters) {
    const ci = canonOf(alphabet, v);
    const b = effBias(c, pos, ci);
    if (b !== 0) biasEntries.push({ letter: alphabet[v], value: b });
    if (effOmit(c, pos, ci)) omitLetters.push(alphabet[v]);
  }
  return {
    biasEntries,
    omitLetters,
    hasOverride: c.posBias.has(pos) || c.posOmit.has(pos),
  };
}

/**
 * Expand into the [L, V] array the sampler wants.
 *
 * The structural omissions have to be written here rather than left to the
 * model: `Model.sample` replaces its own `defaultOmit()` with zeros the moment
 * any bias is supplied, so whatever this returns is the *whole* truth about
 * what may be drawn. Anything outside `letters` is omitted outright -- "X" for
 * the protein models, and for NA-MPNN also the legacy RNA tokens and the
 * MAS/PAD placeholders that never name a real residue.
 *
 * @param {{L: number, alphabet: string, letters: number[]}} shape
 */
export function buildBias(c, { L, alphabet, letters }) {
  const V = alphabet.length;
  const allowed = new Set(letters);
  // One lookup per letter rather than per (position, letter).
  const canon = new Int32Array(V);
  for (let v = 0; v < V; v++) canon[v] = allowed.has(v) ? canonOf(alphabet, v) : -1;

  const bias = new Float32Array(L * V);
  for (let i = 0; i < L; i++) {
    for (let v = 0; v < V; v++) {
      const ci = canon[v];
      bias[i * V + v] = ci < 0 || effOmit(c, i, ci) ? -1e9 : effBias(c, i, ci);
    }
  }
  return bias;
}
