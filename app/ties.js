// Tied positions: groups that must share one identity.
//
// A group is an object you can see, name, edit and delete, rather than a string
// in a text box or a checkbox that disables the text box.
//
// The important change is that **the maths is now stated**. `normaliseSymmetry`
// in the engine accepts a per-position weight and defaults it to 1, so a group
// of three built from bare indices *sums* three logits, while the homo-oligomer
// path emitted `1/nChains` and *averaged* them. Both were reachable from the
// same panel, they disagree exactly when the group is bigger than two, and
// nothing said which you were getting. Here every group carries an explicit
// mode and defaults to `average`.
//
// No DOM, so this is exercisable from node.

/** Weight per member for each mode. */
const WEIGHT = {
  average: (n) => 1 / n,
  sum: () => 1,
};

export function createTies() {
  return { groups: [], nextId: 1 };
}

/**
 * Add a group. Positions are de-duplicated and sorted; fewer than two is not a
 * group and is rejected rather than silently dropped.
 *
 * @returns {{ok: true, group: object} | {ok: false, reason: string}}
 */
export function addGroup(ties, positions, { mode = "average", source = "manual", label } = {}) {
  const unique = [...new Set(positions)].sort((a, b) => a - b);
  if (unique.length < 2) {
    return { ok: false, reason: "A tie needs at least two positions." };
  }
  const group = {
    id: ties.nextId++,
    label: label ?? `g${ties.groups.length + 1}`,
    positions: unique,
    mode,
    source,
  };
  ties.groups.push(group);
  return { ok: true, group };
}

export function removeGroup(ties, id) {
  ties.groups = ties.groups.filter((g) => g.id !== id);
}

export function clearTies(ties) {
  ties.groups = [];
  ties.nextId = 1;
}

/** Position -> the group holding it, for the table's tie column. */
export function groupByPosition(ties) {
  const out = new Map();
  for (const group of ties.groups) {
    for (const pos of group.positions) {
      if (!out.has(pos)) out.set(pos, []);
      out.get(pos).push(group);
    }
  }
  return out;
}

/**
 * What the engine will quietly ignore, so the UI can say it instead.
 *
 * `normaliseSymmetry` drops out-of-range positions and any position already
 * claimed by an earlier group, first group wins, with no report. Keeping that
 * rule is right -- something has to win -- but it should be visible.
 *
 * @returns {{conflicts: {pos: number, keeps: object, loses: object[]}[],
 *            outOfRange: {group: object, positions: number[]}[]}}
 */
export function validateTies(ties, L) {
  const seen = new Map();
  const conflicts = [];
  const outOfRange = [];
  for (const group of ties.groups) {
    const bad = group.positions.filter((p) => !Number.isInteger(p) || p < 0 || p >= L);
    if (bad.length) outOfRange.push({ group, positions: bad });
    for (const pos of group.positions) {
      if (pos < 0 || pos >= L) continue;
      if (seen.has(pos)) {
        let entry = conflicts.find((c) => c.pos === pos);
        if (!entry) {
          entry = { pos, keeps: seen.get(pos), loses: [] };
          conflicts.push(entry);
        }
        entry.loses.push(group);
      } else {
        seen.set(pos, group);
      }
    }
  }
  return { conflicts, outOfRange };
}

/** The `symmetry` argument `Model.sample` wants, or null when there is none. */
export function toEngineSymmetry(ties, L) {
  const out = [];
  for (const group of ties.groups) {
    const positions = group.positions.filter((p) => p >= 0 && p < L);
    if (positions.length < 2) continue;
    const weight = (WEIGHT[group.mode] ?? WEIGHT.average)(positions.length);
    out.push(positions.map((pos) => ({ pos, weight })));
  }
  return out.length ? out : null;
}

/**
 * Tie every chain to every other, LigandMPNN's `--homo_oligomer`.
 *
 * A generator now, not a mode: it returns groups you can then edit or delete,
 * and it composes with hand-made ones. The matching logic is unchanged and
 * deliberately so.
 *
 * The reference matches residues by *number*, not by position in the chain, so
 * a complex whose chains share a numbering ties correctly even when one of them
 * has a gap. When the numbering does not line up at all -- chain B continuing
 * where A left off, say -- that finds nothing, so equal-length chains fall back
 * to tying by position. Which one ran is reported, because the two disagree
 * exactly when it matters.
 *
 * @returns {{groups: number[][] | null, note: string}}
 */
export function chainGroups(s) {
  const chains = s?.chainList ?? [];
  if (chains.length < 2) return { groups: null, note: "Needs at least two chains." };

  // By residue number: chain -> "resSeq+iCode" -> position.
  const byChain = new Map(chains.map((c) => [c, new Map()]));
  for (let i = 0; i < s.L; i++) {
    byChain.get(s.chainIds[i]).set(`${s.resSeq[i]}${s.iCodes[i]}`, i);
  }
  const reference = byChain.get(chains[0]);
  const byNumber = [];
  let unmatched = 0;
  for (const [key, pos] of reference) {
    const group = [pos];
    for (let c = 1; c < chains.length; c++) {
      const other = byChain.get(chains[c]).get(key);
      if (other === undefined) break;
      group.push(other);
    }
    if (group.length === chains.length) byNumber.push(group);
    else unmatched++;
  }

  const lengths = chains.map((c) => byChain.get(c).size);
  const equalLength = lengths.every((n) => n === lengths[0]);

  if (byNumber.length) {
    return {
      groups: byNumber,
      note: `${byNumber.length} group(s) of ${chains.length}, matched by residue number`
        + (unmatched
          ? `; ${unmatched} residue(s) of chain ${chains[0]} have no counterpart`
          : ""),
    };
  }
  if (!equalLength) {
    return {
      groups: null,
      note: `Chains have different lengths (${lengths.join(", ")}) and no residue numbers `
        + "in common, so there is nothing to tie.",
    };
  }
  // Positional fallback: the i-th residue of every chain.
  const perChain = chains.map(() => []);
  for (let i = 0; i < s.L; i++) perChain[s.chainLabels[i]].push(i);
  const byPosition = perChain[0].map((_, i) => perChain.map((c) => c[i]));
  return {
    groups: byPosition,
    note: `${byPosition.length} group(s) of ${chains.length}, matched by position — the chains `
      + "share no residue numbers, so this assumes they are aligned end to end.",
  };
}

/**
 * Pair each selected position with its residue-number counterpart in every
 * other chain.
 *
 * "Pick four residues in chain A and tie the tetramer" is the case people
 * actually want; a flat group of the four selected residues in one chain is the
 * wrong answer to it.
 *
 * @returns {{groups: number[][], unmatched: number[]}}
 */
export function acrossChains(s, positions) {
  const chains = s.chainList;
  const byChain = new Map(chains.map((c) => [c, new Map()]));
  for (let i = 0; i < s.L; i++) {
    byChain.get(s.chainIds[i]).set(`${s.resSeq[i]}${s.iCodes[i]}`, i);
  }
  const groups = [];
  const unmatched = [];
  for (const pos of positions) {
    const key = `${s.resSeq[pos]}${s.iCodes[pos]}`;
    const group = [];
    for (const chain of chains) {
      const other = byChain.get(chain).get(key);
      if (other !== undefined) group.push(other);
    }
    if (group.length > 1) groups.push(group);
    else unmatched.push(pos);
  }
  return { groups, unmatched };
}

/**
 * Resolve chain-qualified position text: `A12`, `A12-A20`, `A12,B12`.
 *
 * The old box took flat 1-based indices across the whole structure, which at
 * L = 2916 meant counting, and disagreed with every label the rest of the page
 * shows. Unresolved tokens are returned rather than dropped.
 *
 * @returns {{positions: number[], unresolved: string[]}}
 */
export function parsePositions(s, text) {
  const index = new Map();
  for (let i = 0; i < s.L; i++) {
    index.set(`${s.chainIds[i]}${s.resSeq[i]}${s.iCodes[i]}`.trim(), i);
  }
  const positions = [];
  const unresolved = [];
  for (const raw of text.split(/[,\s]+/)) {
    const token = raw.trim();
    if (!token) continue;
    const range = token.split("-");
    if (range.length === 2) {
      const from = index.get(range[0]);
      const to = index.get(range[1]);
      if (from === undefined || to === undefined) unresolved.push(token);
      else for (let i = Math.min(from, to); i <= Math.max(from, to); i++) positions.push(i);
      continue;
    }
    const at = index.get(token);
    if (at === undefined) unresolved.push(token);
    else positions.push(at);
  }
  return { positions: [...new Set(positions)].sort((a, b) => a - b), unresolved };
}
