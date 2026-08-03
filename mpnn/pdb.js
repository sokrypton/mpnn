// PDB / mmCIF parsing, producing exactly the tensors `Model.encode` expects.
//
// The residue selection follows LigandMPNN's `parse_PDB`: one entry per C-alpha,
// zero-occupancy atoms dropped, and everything that is neither protein nor water
// (ligands, cofactors, ions, nucleic acids) collected as ligand context.

import { ATOM37, ELEMENT_TO_INT, THREE_TO_ONE } from "./constants.js";
import {
  DNA_BACKBONE, naDisplaySequence, NA_ATOMS, NA_RESTYPE_TO_INT, NA_RNA_TO_DNA,
  NA_UNKNOWN, POLYTYPE, PROTEIN_BACKBONE, RNA_BACKBONE,
} from "./na.js";

/** Residues treated as protein even though they carry a non-standard name. */
const EXTRA_PROTEIN = new Set([
  "MSE", "SEC", "PYL", "ASX", "GLX", "UNK",
  "HSD", "HSE", "HSP", "HID", "HIE", "HIP", "CYX", "CSO", "PTR", "SEP", "TPO",
]);
const WATER = new Set(["HOH", "DOD", "WAT", "H2O", "TIP", "TIP3", "SOL"]);
const PROTEIN = new Set([...Object.keys(THREE_TO_ONE), ...EXTRA_PROTEIN]);

const BACKBONE = ["N", "CA", "C", "O"];

/** What a nucleotide contributes to `X`, purely so the renderer can trace it. */
const NUCLEIC_TRACE = ["P", "C1'", "C3'", "O4'"];

/**
 * Copy a residue's named atoms into a strided [L, names.length, 3] array.
 *
 * @param {Float32Array} [mask] set to 1 per present atom, when supplied
 * @returns {number} 1 if every name was present
 */
function fillSlots(res, names, out, mask, i) {
  let complete = 1;
  names.forEach((name, slot) => {
    const atom = res.atoms.get(name);
    if (atom === undefined) {
      complete = 0;
      return;
    }
    const off = (i * names.length + slot) * 3;
    out[off] = atom.x;
    out[off + 1] = atom.y;
    out[off + 2] = atom.z;
    if (mask) mask[i * names.length + slot] = 1;
  });
  return complete;
}

/**
 * Nucleic-acid residue names, as ProDy's `nucleic` selection understands them.
 *
 * These only matter under `nucleicAsResidues`; every other model still sees
 * them as ligand context, which is what LigandMPNN expects.
 */
const NUCLEIC = new Set([
  "DA", "DC", "DG", "DT", "DI", "DU",
  "A", "C", "G", "U", "I", "T",
  "ADE", "CYT", "GUA", "THY", "URA",
  "RA", "RC", "RG", "RU",
  "DA3", "DA5", "DC3", "DC5", "DG3", "DG5", "DT3", "DT5",
  "A3", "A5", "C3", "C5", "G3", "G5", "U3", "U5",
]);

/** Guess an element symbol from a PDB atom name when the column is blank. */
function elementFromAtomName(name) {
  const trimmed = name.trim();
  if (/^\d/.test(trimmed)) return trimmed.slice(1, 2).toUpperCase();
  const two = trimmed.slice(0, 2).toUpperCase();
  if (ELEMENT_TO_INT[two] !== undefined && trimmed.length > 1 && /[A-Z]/i.test(trimmed[1])) {
    return two;
  }
  return trimmed.slice(0, 1).toUpperCase();
}

/**
 * @typedef {object} RawAtom
 * @property {boolean} hetero
 * @property {string} name
 * @property {string} altLoc
 * @property {string} resName
 * @property {string} chain
 * @property {number} resSeq
 * @property {string} iCode
 * @property {number} x
 * @property {number} y
 * @property {number} z
 * @property {number} occupancy
 * @property {string} element
 */

/** @returns {RawAtom[]} */
export function parsePDBAtoms(text) {
  const atoms = [];
  for (const line of text.split("\n")) {
    const isAtom = line.startsWith("ATOM  ");
    const isHet = line.startsWith("HETATM");
    if (!isAtom && !isHet) {
      // Only the first model of a multi-model file is used.
      if (line.startsWith("ENDMDL")) break;
      continue;
    }
    const occupancy = parseFloat(line.slice(54, 60)) || 0;
    const element = (line.slice(76, 78).trim() || elementFromAtomName(line.slice(12, 16)))
      .toUpperCase();
    atoms.push({
      hetero: isHet,
      name: line.slice(12, 16).trim(),
      altLoc: line.slice(16, 17).trim(),
      resName: line.slice(17, 20).trim().toUpperCase(),
      chain: line.slice(21, 22).trim() || "A",
      resSeq: parseInt(line.slice(22, 26), 10),
      iCode: line.slice(26, 27).trim(),
      x: parseFloat(line.slice(30, 38)),
      y: parseFloat(line.slice(38, 46)),
      z: parseFloat(line.slice(46, 54)),
      occupancy,
      element,
    });
  }
  return atoms;
}

/** @returns {RawAtom[]} */
export function parseCIFAtoms(text) {
  const lines = text.split("\n");
  const atoms = [];
  let i = 0;
  while (i < lines.length) {
    if (lines[i].trim() !== "loop_") {
      i++;
      continue;
    }
    // Collect the column names of this loop.
    let j = i + 1;
    const columns = [];
    while (j < lines.length && lines[j].trim().startsWith("_")) {
      columns.push(lines[j].trim().split(/\s+/)[0]);
      j++;
    }
    if (!columns[0]?.startsWith("_atom_site.")) {
      i = j;
      continue;
    }
    const col = Object.fromEntries(columns.map((c, k) => [c.replace("_atom_site.", ""), k]));
    const pick = (row, key, fallback = "") => (col[key] === undefined ? fallback : row[col[key]]);

    let firstModel = null;
    for (; j < lines.length; j++) {
      const line = lines[j].trim();
      if (line === "" || line.startsWith("#") || line.startsWith("loop_")) break;
      const row = line.match(/'[^']*'|"[^"]*"|\S+/g)?.map((s) => s.replace(/^['"]|['"]$/g, ""));
      if (!row || row.length < columns.length) continue;

      const model = pick(row, "pdbx_PDB_model_num", "1");
      if (firstModel === null) firstModel = model;
      if (model !== firstModel) break;

      const iCode = pick(row, "pdbx_PDB_ins_code", "");
      const altLoc = pick(row, "label_alt_id", "");
      atoms.push({
        hetero: pick(row, "group_PDB", "ATOM") === "HETATM",
        name: pick(row, "auth_atom_id", pick(row, "label_atom_id", "")),
        altLoc: altLoc === "." || altLoc === "?" ? "" : altLoc,
        resName: pick(row, "auth_comp_id", pick(row, "label_comp_id", "")).toUpperCase(),
        chain: pick(row, "auth_asym_id", pick(row, "label_asym_id", "A")),
        resSeq: parseInt(pick(row, "auth_seq_id", pick(row, "label_seq_id", "0")), 10),
        iCode: iCode === "." || iCode === "?" ? "" : iCode,
        x: parseFloat(pick(row, "Cartn_x", "0")),
        y: parseFloat(pick(row, "Cartn_y", "0")),
        z: parseFloat(pick(row, "Cartn_z", "0")),
        occupancy: parseFloat(pick(row, "occupancy", "1")) || 0,
        element: pick(row, "type_symbol", "").toUpperCase(),
      });
    }
    i = j;
  }
  return atoms;
}

export function parseAtoms(text) {
  const looksCIF = /^\s*(data_|#)/.test(text) || text.includes("_atom_site.");
  return looksCIF ? parseCIFAtoms(text) : parsePDBAtoms(text);
}

/**
 * Turn raw atoms into model inputs.
 *
 * @param {string} text  PDB or mmCIF
 * @param {object} [opts]
 * @param {string[]} [opts.chains]  restrict to these chain ids
 * @param {boolean}  [opts.ligands=true] collect heteroatoms as ligand context
 * @returns {{
 *   X: Float32Array, mask: Float32Array, S: Int32Array,
 *   xyz37: Float32Array, xyz37Mask: Float32Array,
 *   residueIdx: Int32Array, chainLabels: Int32Array, L: number,
 *   chainIds: string[], resSeq: Int32Array, iCodes: string[], resNames: string[],
 *   ligandXyz: Float32Array, ligandType: Int32Array, ligandMask: Float32Array,
 *   ligandNames: string[], ligandElements: string[], ligandResidues: string[],
 *   chainList: string[], sequence: string,
 * }}
 */
export function structureFromText(text, opts = {}) {
  const wantLigands = opts.ligands !== false;
  // NA-MPNN only: nucleic acids become model positions instead of ligand atoms.
  const nucleicAsResidues = opts.nucleicAsResidues === true;
  const atoms = parseAtoms(text).filter((a) => a.occupancy > 0 && Number.isFinite(a.x));
  const chainFilter = opts.chains ? new Set(opts.chains) : null;

  /** @type {Map<string, {atoms: Map<string, RawAtom>, meta: RawAtom}>} */
  const residues = new Map();
  const residueOrder = [];
  const ligandAtoms = [];

  for (const atom of atoms) {
    if (chainFilter && !chainFilter.has(atom.chain)) continue;
    if (WATER.has(atom.resName)) continue;

    const isNucleic = nucleicAsResidues && NUCLEIC.has(atom.resName);
    if ((PROTEIN.has(atom.resName) || isNucleic) && !atom.name.startsWith("H")) {
      const key = `${atom.chain}|${atom.resSeq}|${atom.iCode}`;
      let res = residues.get(key);
      if (res === undefined) {
        res = { atoms: new Map(), meta: atom, nucleic: isNucleic };
        residues.set(key, res);
        residueOrder.push(key);
      }
      // First altloc wins, matching the reference's occupancy-filtered selection.
      if (!res.atoms.has(atom.name)) res.atoms.set(atom.name, atom);
    } else if (wantLigands && !PROTEIN.has(atom.resName)) {
      const type = ELEMENT_TO_INT[atom.element];
      // Hydrogen (1) and unrecognised elements (0) are dropped by the reference.
      if (type !== undefined && type !== 1) ligandAtoms.push({ ...atom, type });
    }
  }

  // A residue becomes a model position if it has its reference atom: C-alpha
  // for protein, C1' for a nucleotide. That is `macromolecule_reference_atoms`.
  const kept = residueOrder.filter((k) => {
    const res = residues.get(k);
    return res.atoms.has(res.nucleic ? "C1'" : "CA");
  });
  const L = kept.length;

  const X = new Float32Array(L * 12);
  // Every atom the reference knows about, for LigandMPNN's side-chain context.
  // Missing atoms stay at the origin with mask 0, as `parse_PDB` leaves them.
  const xyz37 = new Float32Array(L * 37 * 3);
  const xyz37Mask = new Float32Array(L * 37);
  // NA-MPNN's 16 shared backbone slots, and which polymer each residue is.
  const X16 = new Float32Array(L * NA_ATOMS.length * 3);
  const X16Mask = new Float32Array(L * NA_ATOMS.length);
  const polytype = new Int32Array(L);
  const isRNA = new Uint8Array(L);
  const mask = new Float32Array(L);
  const S = new Int32Array(L);
  const resSeq = new Int32Array(L);
  const chainLabels = new Int32Array(L);
  const chainIds = [];
  const iCodes = [];
  const resNames = [];

  const chainList = [];
  const chainIndex = new Map();
  for (const key of kept) {
    const chain = residues.get(key).meta.chain;
    if (!chainIndex.has(chain)) {
      chainIndex.set(chain, chainList.length);
      chainList.push(chain);
    }
  }

  kept.forEach((key, i) => {
    const res = residues.get(key);
    mask[i] = fillSlots(res, BACKBONE, X, null, i);
    fillSlots(res, ATOM37, xyz37, xyz37Mask, i);
    const letter = THREE_TO_ONE[res.meta.resName] ?? "X";
    S[i] = "ACDEFGHIKLMNPQRSTVWYX".indexOf(letter);

    if (nucleicAsResidues && res.nucleic) {
      // The model reads X16, never X -- but the cartoon renderer traces X's
      // C-alpha slot, so give a nucleotide the nearest equivalents and it draws
      // like anything else. Purely cosmetic; nothing downstream of the model
      // sees these.
      fillSlots(res, NUCLEIC_TRACE, X, null, i);
    }

    if (nucleicAsResidues) {
      fillSlots(res, NA_ATOMS, X16, X16Mask, i);
      // Polymer type is decided by which backbone is complete, not by the
      // residue name -- `parse_PDB`'s default branch. RNA is subtracted out of
      // the DNA test because RNA carries every DNA backbone atom as well.
      const has = (names) => names.every((n) => res.atoms.has(n));
      const rna = has(RNA_BACKBONE);
      const dna = !rna && has(DNA_BACKBONE);
      const pp = has(PROTEIN_BACKBONE);
      polytype[i] = pp ? POLYTYPE.PP : rna ? POLYTYPE.RNA : dna ? POLYTYPE.DNA : POLYTYPE.UNK;
      isRNA[i] = res.atoms.has("O2'") ? 1 : 0;
      // `mask = protein_mask + dna_mask + rna_mask`: a residue with no complete
      // backbone of any kind is masked out entirely.
      mask[i] = polytype[i] === POLYTYPE.UNK ? 0 : 1;
      // NA-MPNN indexes residues in its own order, and with the default shared
      // tokens an RNA base is stored as the corresponding DNA one.
      const unknown = polytype[i] === POLYTYPE.DNA ? NA_UNKNOWN.DNA
        : polytype[i] === POLYTYPE.RNA ? NA_UNKNOWN.RNA : NA_UNKNOWN.PP;
      const token = NA_RESTYPE_TO_INT[res.meta.resName];
      S[i] = token === undefined ? unknown : (NA_RNA_TO_DNA.get(token) ?? token);
    }

    resSeq[i] = res.meta.resSeq;
    chainIds.push(res.meta.chain);
    chainLabels[i] = chainIndex.get(res.meta.chain);
    iCodes.push(res.meta.iCode);
    resNames.push(res.meta.resName);
  });

  const ligandXyz = new Float32Array(ligandAtoms.length * 3);
  const ligandType = new Int32Array(ligandAtoms.length);
  const ligandMask = new Float32Array(ligandAtoms.length).fill(1);
  ligandAtoms.forEach((atom, i) => {
    ligandXyz[i * 3] = atom.x;
    ligandXyz[i * 3 + 1] = atom.y;
    ligandXyz[i * 3 + 2] = atom.z;
    ligandType[i] = atom.type;
  });

  return {
    X, mask, S, L, xyz37, xyz37Mask,
    // Only populated under `nucleicAsResidues`, where `S` and `sequence` also
    // switch to NA-MPNN's 33-letter alphabet -- the structure object belongs to
    // one model family at a time, so the caller re-parses when it switches.
    X16, X16Mask, polytype, isRNA, nucleicAsResidues,
    residueIdx: renumber(resSeq),
    chainLabels, chainIds, resSeq, iCodes, resNames, chainList,
    ligandXyz, ligandType, ligandMask,
    ligandNames: ligandAtoms.map((a) => a.name),
    ligandElements: ligandAtoms.map((a) => a.element),
    ligandResidues: ligandAtoms.map((a) => `${a.resName} ${a.chain}${a.resSeq}`),
    sequence: nucleicAsResidues
      ? naDisplaySequence(S, isRNA)
      : [...S].map((v) => "ACDEFGHIKLMNPQRSTVWYX"[v]).join(""),
  };
}

/**
 * Residue numbering used by the relative-position encoding.
 *
 * Repeated residue numbers (insertion codes) are pushed apart by a running
 * counter that is never reset, exactly as `featurize` does.
 */
export function renumber(resSeq) {
  const out = new Int32Array(resSeq.length);
  let count = 0;
  let prev = -100000;
  for (let i = 0; i < resSeq.length; i++) {
    if (prev === resSeq[i]) count += 1;
    out[i] = resSeq[i] + count;
    prev = resSeq[i];
  }
  return out;
}

/** Fetch a structure from the RCSB PDB, preferring mmCIF. */
export async function fetchPDB(id, { signal } = {}) {
  const code = id.trim().toUpperCase();
  const urls = code.length === 4
    ? [`https://files.rcsb.org/download/${code}.cif`,
       `https://files.rcsb.org/download/${code}.pdb`]
    : [`https://alphafold.ebi.ac.uk/files/AF-${code}-F1-model_v4.pdb`];
  let lastError;
  for (const url of urls) {
    try {
      const response = await fetch(url, { signal });
      if (response.ok) return await response.text();
      lastError = new Error(`${response.status} ${url}`);
    } catch (err) {
      lastError = err;
    }
  }
  throw lastError ?? new Error(`could not fetch ${id}`);
}
