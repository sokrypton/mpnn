// NA-MPNN: protein, DNA and RNA in one graph.
//
// https://github.com/baker-laboratory/NA-MPNN (MIT), preprint
// https://www.biorxiv.org/content/10.1101/2025.10.03.679414v2
//
// The network is the ordinary MPNN: 3 encoder and 3 decoder layers at hidden
// 128, the same message scale, the same autoregressive decoder. Only the
// featuriser and the alphabet change, so everything in layers.js and the whole
// decoding half of model.js is reused untouched.
//
// What is different:
//
//   * 16 backbone atoms rather than 4 -- N/CA/C/O for protein, and
//     OP1/OP2/P/O5'/C5'/C4'/O4'/C3'/O3'/C2'/O2'/C1' for a nucleotide -- plus
//     two virtual atoms, the familiar C-beta and a nucleic-acid pseudo-N built
//     from O4'/C1'/C2' with its own coefficients. Eighteen slots in all.
//
//   * Edge features are *all 324 ordered pairs* of those slots, not the 25
//     hand-listed ones, so the edge embedding takes 16 + 16·18·18 = 5200
//     inputs against 416. That sounds ruinous and is not, because each block is
//     masked by both endpoints' atom masks: a protein residue has no C1', a
//     nucleotide has no CA. A protein-protein edge therefore has 5x5 = 25 live
//     blocks -- exactly today's cost -- and a nucleotide-nucleotide edge 13x13.
//     `naEdgeFeatures` walks only the live blocks, so the fat embedding costs
//     what the geometry actually contains.
//
//   * Nodes carry a polymer-type one-hot instead of nothing, which is the same
//     shape as the membrane models' per-residue label.
//
//   * 33 letters: 20 amino acids, UNK, the four DNA bases plus DX, the four RNA
//     bases plus RX, and MAS/PAD which never appear in a real structure.

import { CB_COEFF, MAX_RELATIVE_FEATURE, RBF } from "./constants.js";
import { dist, rbfInto } from "./features.js";
import { layerNorm, linear } from "./ops.js";

/**
 * One letter per residue type, in the checkpoint's own order.
 *
 * Note this is *not* MPNN's `ACDEFGHIKLMNPQRSTVWYX`: NA-MPNN indexes amino
 * acids in three-letter alphabetical order (ALA ARG ASN ASP CYS GLN ...), so
 * W_s and W_out rows do not line up with the other models'. Lower case is DNA,
 * and the second run of lower case is RNA.
 */
export const NA_ALPHABET = "ARNDCQEGHILKMFPSTWYVXacgtxbdhuy-+";

/** Three-letter residue name -> index, matching `restype_to_int`. */
export const NA_RESTYPES = [
  "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
  "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK",
  "DA", "DC", "DG", "DT", "DX",
  "A", "C", "G", "U", "RX",
  "MAS", "PAD",
];
export const NA_RESTYPE_TO_INT = Object.fromEntries(
  NA_RESTYPES.map((name, i) => [name, i]),
);
export const NA_UNKNOWN = { PP: NA_RESTYPE_TO_INT.UNK, DNA: NA_RESTYPE_TO_INT.DX, RNA: NA_RESTYPE_TO_INT.RX };

/**
 * What `--omit_AA` excludes by default.
 *
 * `X` (UNK) is the CLI's own default, and `--na_shared_tokens` -- which is
 * *on* by default -- aliases the RNA tokens onto the DNA ones and omits the
 * legacy RNA letters `bdhuy`. So a designed RNA base comes out as a DNA token
 * and is converted back for display; see `naDisplaySequence`.
 */
export const NA_OMIT_UNK = NA_RESTYPE_TO_INT.UNK;
export const NA_OMIT_LEGACY_RNA = ["A", "C", "G", "U", "RX"].map((n) => NA_RESTYPE_TO_INT[n]);

/** DNA token -> the RNA token it stands in for under shared tokens. */
export const NA_DNA_TO_RNA = new Map([
  [NA_RESTYPE_TO_INT.DA, NA_RESTYPE_TO_INT.A],
  [NA_RESTYPE_TO_INT.DC, NA_RESTYPE_TO_INT.C],
  [NA_RESTYPE_TO_INT.DG, NA_RESTYPE_TO_INT.G],
  [NA_RESTYPE_TO_INT.DT, NA_RESTYPE_TO_INT.U],
  [NA_RESTYPE_TO_INT.DX, NA_RESTYPE_TO_INT.RX],
]);

/**
 * Render a sequence, converting DNA tokens back to RNA letters wherever the
 * residue is actually RNA -- which the reference decides by the presence of an
 * O2', not by the token.
 *
 * @param {Int32Array} S
 * @param {Uint8Array|Float32Array} isRNA [L] 1 where the residue has an O2'
 */
export function naDisplaySequence(S, isRNA) {
  let out = "";
  for (let i = 0; i < S.length; i++) {
    const v = isRNA && isRNA[i] ? (NA_DNA_TO_RNA.get(S[i]) ?? S[i]) : S[i];
    out += NA_ALPHABET[v] ?? "?";
  }
  return out;
}

/** Polymer type of each residue, the 6 classes the node embedding takes. */
export const POLYTYPE = { PP: 0, DNA: 1, RNA: 2, UNK: 3, MAS: 4, PAD: 5 };
export const N_POLYTYPES = 6;

/** The 16 parsed backbone atoms, in the order the featuriser indexes them. */
export const NA_ATOMS = [
  "N", "CA", "C", "O",
  "OP1", "OP2", "P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
];
const AT = Object.fromEntries(NA_ATOMS.map((n, i) => [n, i]));

/** Atoms whose presence defines each polymer type, per `parse_PDB`. */
export const PROTEIN_BACKBONE = ["N", "CA", "C", "O"];
export const DNA_BACKBONE = ["OP1", "OP2", "P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"];
export const RNA_BACKBONE = [...DNA_BACKBONE.slice(0, 10), "O2'", "C1'"];

/** Parsed atoms plus the two virtual ones. */
export const NA_SLOTS = NA_ATOMS.length + 2;
const CB_SLOT = NA_ATOMS.length;
const NNA_SLOT = NA_ATOMS.length + 1;

/**
 * The nucleic-acid pseudo-N, placed from O4'/C1'/C2' exactly as C-beta is
 * placed from N/CA/C -- same construction, different coefficients.
 */
const NNA_COEFF = { a: -0.56967352, b: 0.51055973, c: -0.53122153 };

/** `w_a·(b×c) + w_b·b + w_c·c + origin`, the reference's `get_Cb`. */
function virtualAtom(out, o, p1, p2, p3, i, coeff) {
  const s = i * 3;
  const bx = p2[s] - p1[s];
  const by = p2[s + 1] - p1[s + 1];
  const bz = p2[s + 2] - p1[s + 2];
  const cx = p3[s] - p2[s];
  const cy = p3[s + 1] - p2[s + 1];
  const cz = p3[s + 2] - p2[s + 2];
  out[o] = coeff.a * (by * cz - bz * cy) + coeff.b * bx + coeff.c * cx + p2[s];
  out[o + 1] = coeff.a * (bz * cx - bx * cz) + coeff.b * by + coeff.c * cy + p2[s + 1];
  out[o + 2] = coeff.a * (bx * cy - by * cx) + coeff.b * bz + coeff.c * cz + p2[s + 2];
}

/**
 * Expand the 16 parsed atoms into the 18 slots the edge featuriser uses, and
 * derive the k-nearest-neighbour centre.
 *
 * That centre is `CA + C1'`, which the reference explains as "these vectors are
 * disjoint": a protein residue's C1' slot is zero and a nucleotide's CA slot is
 * zero, so the sum is whichever of the two exists.
 *
 * @param {Float32Array} X16     [L, 16, 3]
 * @param {Float32Array} X16Mask [L, 16]
 * @param {Int32Array}   polytype [L] see POLYTYPE
 * @returns {{xyz: Float32Array, xyzMask: Float32Array, centre: Float32Array}}
 *          xyz is [L, 18, 3], xyzMask [L, 18], centre [L, 3]
 */
export function naBackbone(X16, X16Mask, polytype, L) {
  const A = NA_ATOMS.length;
  const xyz = new Float32Array(L * NA_SLOTS * 3);
  const xyzMask = new Float32Array(L * NA_SLOTS);
  const centre = new Float32Array(L * 3);

  // Scratch views so virtualAtom can address one atom across all residues.
  const pick = (slot) => {
    const out = new Float32Array(L * 3);
    for (let i = 0; i < L; i++) {
      for (let d = 0; d < 3; d++) out[i * 3 + d] = X16[(i * A + slot) * 3 + d];
    }
    return out;
  };
  const N = pick(AT.N);
  const CA = pick(AT.CA);
  const C = pick(AT.C);
  const O4 = pick(AT["O4'"]);
  const C1 = pick(AT["C1'"]);
  const C2 = pick(AT["C2'"]);

  for (let i = 0; i < L; i++) {
    for (let a = 0; a < A; a++) {
      const src = (i * A + a) * 3;
      const dst = (i * NA_SLOTS + a) * 3;
      xyz[dst] = X16[src];
      xyz[dst + 1] = X16[src + 1];
      xyz[dst + 2] = X16[src + 2];
      xyzMask[i * NA_SLOTS + a] = X16Mask[i * A + a];
    }
    virtualAtom(xyz, (i * NA_SLOTS + CB_SLOT) * 3, N, CA, C, i, CB_COEFF);
    virtualAtom(xyz, (i * NA_SLOTS + NNA_SLOT) * 3, O4, C1, C2, i, NNA_COEFF);
    // C-beta exists for protein residues, the pseudo-N for nucleotides.
    const t = polytype[i];
    xyzMask[i * NA_SLOTS + CB_SLOT] = t === POLYTYPE.PP ? 1 : 0;
    xyzMask[i * NA_SLOTS + NNA_SLOT] = (t === POLYTYPE.DNA || t === POLYTYPE.RNA) ? 1 : 0;
    for (let d = 0; d < 3; d++) centre[i * 3 + d] = CA[i * 3 + d] + C1[i * 3 + d];
  }
  return { xyz, xyzMask, centre };
}

/** Polymer-type one-hot pushed through node_embedding and norm_nodes. */
export function naNodeFeatures(w, polytype, L, hidden) {
  const embed = w.linear(`${w.featurePrefix}.node_embedding`);
  const norm = w.norm(`${w.featurePrefix}.norm_nodes`);
  const oneHot = new Float32Array(L * N_POLYTYPES);
  for (let i = 0; i < L; i++) oneHot[i * N_POLYTYPES + polytype[i]] = 1;
  const V = linear(oneHot, embed.weight, embed.bias, L, N_POLYTYPES, hidden);
  layerNorm(V, norm.gamma, norm.beta, L, hidden, V);
  return V;
}

/**
 * Edge features for NA-MPNN: relative position, then every live atom-pair RBF.
 *
 * Two things make the 5200-wide input affordable.
 *
 * It is built a chunk of residues at a time, never all at once: the full
 * tensor would be 666 MB at L = 1000, while a 16-residue chunk is 10 MB and
 * stays in cache.
 *
 * And only the live (a, b) blocks are written. Each block is masked by both
 * endpoints' atom masks, so a protein-protein edge fills 5x5 of the 324 and a
 * nucleotide-nucleotide edge 13x13; the rest of the row is left at zero from
 * the clear. The matmul that follows is dense over all 5200 columns, which
 * sounds wasteful and is not -- it is the SIMD kernel at ~22 GFLOP/s against
 * roughly 0.9 for a scalar walk over just the live blocks, and multiplying by
 * a zero that is already in a register costs nothing extra.
 *
 * @returns {Float32Array} [L, K, hidden]
 */
export function naEdgeFeatures(w, bb, EIdx, residueIdx, chainLabels, L, K, chunk = 16) {
  const edgeEmbed = w.linear(`${w.featurePrefix}.edge_embedding`);
  const normEdges = w.norm(`${w.featurePrefix}.norm_edges`);
  const posLinear = w.linear(`${w.featurePrefix}.embeddings.linear`);
  const posIn = 2 * MAX_RELATIVE_FEATURE + 2; // 66
  const hidden = edgeEmbed.shape[0];
  const edgeIn = edgeEmbed.shape[1]; // 16 + 16·18·18 = 5200
  const { xyz, xyzMask } = bb;

  const E = new Float32Array(L * K * hidden);
  const scratch = new Float32Array(chunk * K * edgeIn);
  const projected = new Float32Array(chunk * K * hidden);
  const live = new Int32Array(NA_SLOTS);
  const liveJ = new Int32Array(NA_SLOTS);

  for (let start = 0; start < L; start += chunk) {
    const end = Math.min(L, start + chunk);
    const rows = end - start;
    scratch.fill(0, 0, rows * K * edgeIn);

    for (let i = start; i < end; i++) {
      let nLive = 0;
      for (let a = 0; a < NA_SLOTS; a++) if (xyzMask[i * NA_SLOTS + a]) live[nLive++] = a;

      for (let k = 0; k < K; k++) {
        const j = EIdx[i * K + k];
        const base = ((i - start) * K + k) * edgeIn;

        // Relative position: a one-hot into `embeddings.linear`, a column read.
        const sameChain = chainLabels[i] === chainLabels[j] ? 1 : 0;
        const offset = residueIdx[i] - residueIdx[j];
        const clipped = Math.min(
          Math.max(offset + MAX_RELATIVE_FEATURE, 0),
          2 * MAX_RELATIVE_FEATURE,
        );
        const d = sameChain ? clipped : 2 * MAX_RELATIVE_FEATURE + 1;
        for (let o = 0; o < 16; o++) {
          scratch[base + o] = posLinear.weight[o * posIn + d] + posLinear.bias[o];
        }

        let nLiveJ = 0;
        for (let b = 0; b < NA_SLOTS; b++) if (xyzMask[j * NA_SLOTS + b]) liveJ[nLiveJ++] = b;

        for (let ai = 0; ai < nLive; ai++) {
          const a = live[ai];
          const ao = (i * NA_SLOTS + a) * 3;
          for (let bi = 0; bi < nLiveJ; bi++) {
            const b = liveJ[bi];
            rbfInto(scratch, base + 16 + (a * NA_SLOTS + b) * RBF.count,
              dist(xyz, ao, xyz, (j * NA_SLOTS + b) * 3));
          }
        }
      }
    }

    linear(scratch, edgeEmbed.weight, edgeEmbed.bias, rows * K, edgeIn, hidden, projected);
    layerNorm(projected, normEdges.gamma, normEdges.beta, rows * K, hidden,
      E.subarray(start * K * hidden, end * K * hidden));
  }
  return E;
}
