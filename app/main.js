// Page controller: owns the DOM, the selection state, and the worker.

import { ALPHABET, THREE_TO_ONE } from "../mpnn/constants.js";
import {
  naDisplaySequence, naDisplayToken, NA_ALPHABET, NA_DNA_TO_RNA, NA_NUCLEOTIDES,
  NA_RESTYPES, NA_RNA_TO_DNA, POLYTYPE,
} from "../mpnn/na.js";
import { fetchPDB, structureFromText } from "../mpnn/pdb.js";
import { elementRgb, Viewer, hexToRgb, orbit, spectrumRgb } from "./viewer.js";
import { AA_COLORS, Logo } from "./logo.js";
import {
  buildBias as buildBiasFrom, clearAll, clearOverrides, createConstraints,
  overriddenPositions,
} from "./constraints.js";
// The table needs the whole surface, and passing it a namespace keeps that one
// dependency visible rather than threading nine functions through its options.
import * as constraints from "./constraints.js";
import { ConstraintTable } from "./constrainttable.js";
import { SequenceView } from "./seqview.js";
import {
  acrossChains, addGroup, chainGroups, clearTies, createTies, groupByPosition,
  parsePositions, removeGroup, toEngineSymmetry, validateTies,
} from "./ties.js";

const WEIGHTS_BASE = new URL("../weights", import.meta.url).href;

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Worker plumbing
// ---------------------------------------------------------------------------

const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
const pending = new Map();
let nextId = 1;
let onProgress = null;

worker.onmessage = (event) => {
  const message = event.data;
  if (message.type === "progress") {
    if (onProgress) onProgress(message);
    return;
  }
  const entry = pending.get(message.id);
  if (!entry) return;
  pending.delete(message.id);
  if (message.ok) entry.resolve(message);
  else entry.reject(new Error(message.error));
};

function call(type, payload = {}, transfer = []) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, type, ...payload }, transfer);
  });
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  structure: null,
  models: [],
  modelName: null,
  modelType: null,
  /** 1 = design this position, 0 = keep it. */
  designMask: null,
  /**
   * What the next edit applies to, as position indices.
   *
   * Deliberately not the same thing as `designMask`. They used to be one
   * object, which meant "select these thirteen residues and bias them" could
   * not be said without also making them designed, and biasing a subset of the
   * designed positions could not be said at all. Selecting is now pointing;
   * designing is a property you set on what you have pointed at.
   */
  selection: new Set(),
  /**
   * Amino-acid bias and omissions; see `app/constraints.js`.
   *
   * Sized for the widest alphabet and keyed by *that* alphabet's order, not
   * the current model's. Both matter: a short array reads `undefined` past its
   * end, writing that into a Float32Array gives NaN, and a NaN bias loses every
   * comparison in the sampler -- which showed up as NA-MPNN quietly designing
   * amino acids into an RNA chain. And a model-relative index means something
   * different after a family switch, which silently turned "omit C" into
   * "omit R".
   */
  constraints: createConstraints(),
  /** Tied position groups; see `app/ties.js`. */
  ties: createTies(),
  designs: [],
  /** Monotonic, so a row keeps its number when the table is re-sorted. */
  nextDesignId: 1,
  /** Which column the results table is sorted by, and which way. */
  sort: { key: "score", dir: 1 },
  activeDesign: -1,
  profile: null,
  scorePerPosition: null,
  hover: -1,
  encodedFor: null,
  /** Bumped on every structure load so a stale encode cannot be reused. */
  structureId: 0,
  /** 0 soluble, 1 interface, 2 buried. Only the membrane models read it. */
  membraneLabels: null,
  membraneVersion: 0,
  /** The text the current structure was parsed from, so a change of model
   *  family can re-parse it -- NA-MPNN turns nucleic acids into positions. */
  structureText: null,
  structureLabel: "",
};

/**
 * The current alphabet. NA-MPNN has 33 letters, everything else 21.
 *
 * Keyed on the parsed structure, not on `state.modelType`: the structure is
 * re-parsed the moment the model family changes, whereas `modelType` only
 * catches up once the weights have downloaded. Getting this wrong made
 * `buildBias` omit every nucleotide, which quietly forced an RNA design to
 * come out as protein.
 */
function alphabet() {
  return state.structure?.nucleicAsResidues ? NA_ALPHABET : ALPHABET;
}
function numLetters() {
  return alphabet().length;
}
/** The selected model's family, before its weights have finished downloading. */
function modelType() {
  return $("model-select").selectedOptions[0]?.dataset.type;
}
/** True when the selected model treats nucleic acids as model positions. */
function wantsNucleic() {
  return (modelType() ?? state.modelType) === "na_mpnn";
}
/**
 * True when the selected model's encoder reads heteroatoms.
 *
 * One statement of it, because five things follow from it -- whether the
 * viewer draws a ligand, whether the sequence track shows ligand tokens,
 * whether the atom-context and side-chain controls apply, and whether "near
 * ligand" means anything -- and they have to agree. They were separate
 * comparisons against the same string, plus two booleans cached in different
 * places from it.
 */
function readsLigands() {
  return modelType() === "ligand_mpnn";
}
/**
 * One residue's letter, in the current alphabet.
 *
 * NA-MPNN stores an RNA base as the corresponding DNA token -- that is what
 * `--na_shared_tokens` means -- so a uracil is held as DT and has to be turned
 * back into a "u" for display, or an RNA chain reads as though it contained
 * thymine. The reference decides by the presence of an O2', not by the token.
 */
function displayLetter(i, v) {
  const s = state.structure;
  if (s?.nucleicAsResidues) return NA_ALPHABET[naDisplayToken(v, s.isRNA[i])] ?? "X";
  return alphabet()[v] ?? "X";
}

/** The span of NA-MPNN's nucleotide tokens, for telling them from amino acids. */
const NA_FIRST_NUCLEOTIDE = NA_RESTYPES.indexOf("DA");
const NA_LAST_NUCLEOTIDE = NA_RESTYPES.indexOf("RX");

/** One-letter code -> the three-letter name, inverted from the parser's table. */
const ONE_TO_THREE = Object.fromEntries(
  Object.entries(THREE_TO_ONE).map(([three, one]) => [one, three]),
);

/**
 * The three-letter name of what is *displayed* at a position.
 *
 * The sequence viewer takes residue names, not letters -- it does its
 * own three-to-one conversion and, more importantly, sniffs protein/DNA/RNA
 * from the names to decide which table to use. So this has to hand it the name
 * of the residue as shown, which for a painted design is the designed one and
 * not the native one.
 *
 * The nucleic case matters: NA-MPNN stores an RNA uracil as the DT token, and
 * handing "DT" to a viewer that decides chain type by name would make it call
 * an RNA chain DNA and print T. `isRNA` is what the reference uses to decide,
 * so convert first.
 */
function displayName(i, v) {
  const s = state.structure;
  if (s?.nucleicAsResidues) return NA_RESTYPES[naDisplayToken(v, s.isRNA[i])] ?? "UNK";
  return ONE_TO_THREE[ALPHABET[v]] ?? "UNK";
}

/** Hand `text` to the browser as a file download. */
function download(text, type, filename) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

/** The same, over a whole sequence. */
function showSequence(S) {
  const s = state.structure;
  return s?.nucleicAsResidues
    ? naDisplaySequence(S, s.isRNA)
    : [...S].map((v) => alphabet()[v] ?? "X").join("");
}

// Monotonic token for in-flight encodes. Changing the model and loading a new
// structure in quick succession queues two encodes; only the newest may claim
// the cache, or the page ends up designing against the previous structure.
let encodeToken = 0;

/**
 * Encodes run one at a time, and clicks coalesce into the last one.
 *
 * With side-chain context on the selection is an encoder input, so every
 * residue click needs a re-encode -- 8.2 s of worker time at L = 574. Posting
 * one per click queued them all: ten clicks blocked the worker for ~80 s and
 * Design waited behind every one of them.
 *
 * Two things fix it. `scheduleEncode` is a trailing debounce, so a burst of
 * clicks posts once. `encodeChain` serialises what does get posted, so by the
 * time a queued call runs, `encodedFor` usually already matches the current
 * selection and it returns without touching the worker.
 *
 * Cancelling from the worker side would need it to see the newest generation
 * while blocked inside `Model.encode`, which means shared memory -- and
 * `SharedArrayBuffer` needs COOP/COEP headers GitHub Pages does not send. So
 * the coalescing lives here, where it costs nothing.
 */
let encodeChain = Promise.resolve(false);
let encodeTimer = null;
const ENCODE_DEBOUNCE_MS = 350;

const viewer = new Viewer($("viewer"));
const logo = new Logo($("logo"));
const constraintTable = new ConstraintTable({
  getState: () => state,
  alphabet,
  letters: biasLetters,
  letterAt: (i) => displayLetter(i, state.structure.S[i]),
  c: constraints,
  tiesByPosition: () => groupByPosition(state.ties),
  onChange: () => {
    renderConstraints();
    redraw();
    // The design mask is an encoder input under side-chain context.
    if ($("use-side-chains").checked && readsLigands() && state.structure) scheduleEncode();
  },
});

/** The constraints pane, plus the two buttons that live beside it. */
function renderConstraints() {
  constraintTable.render();
  $("bias-clear-overrides").hidden = overriddenPositions(state.constraints).size === 0;
}

const track = new SequenceView({
  getState: () => state,
  colourFor: (i) => colourFor(i),
  nameAt: (i) => {
    const seq = activeSequence();
    return displayName(i, seq ? seq[i] : state.structure.S[i]);
  },
  /**
   * The polymer type the viewer spaces and groups on.
   *
   * Deliberately *not* `polytype`. That is the model's classification and it
   * calls a residue UNK when its backbone is incomplete -- which every
   * 5'-terminal nucleotide is, having no phosphate. Feeding that through made
   * the first base of each DNA strand type as protein, and the viewer then
   * correctly inserted its polymer-type-change spacer between it and the rest
   * of the strand: a gap in the display where the numbering is contiguous
   * (3HDD chains C and D). What the model sees is untouched; this is only how
   * the residue is drawn.
   */
  typeAt: (i) => {
    const s = state.structure;
    if (!s?.nucleicAsResidues) return "P";
    if (s.isRNA[i]) return "R";
    const v = s.S[i];
    return v >= NA_FIRST_NUCLEOTIDE && v <= NA_LAST_NUCLEOTIDE ? "D" : "P";
  },
  chainColour: (chain) => {
    const s = state.structure;
    const i = s ? s.chainIds.indexOf(chain) : -1;
    return hexToRgb(CHAIN_COLORS[(i < 0 ? 0 : s.chainLabels[i]) % CHAIN_COLORS.length]);
  },
  ligandColour: (a) => elementRgb(state.structure.ligandElements[a]),
  // Only LigandMPNN's encoder reads heteroatoms, so only it shows them.
  showLigands: readsLigands,
  onSelectionChange: () => refreshSelection(),
  onHoverChange: () => redraw(),
});

const CHAIN_COLORS = [
  "#38bdf8", "#f472b6", "#4ade80", "#fbbf24", "#a78bfa",
  "#fb923c", "#22d3ee", "#f87171", "#a3e635", "#c084fc",
];


// ---------------------------------------------------------------------------
// Model list
// ---------------------------------------------------------------------------

const MODEL_LABELS = {
  protein_mpnn: "ProteinMPNN",
  soluble_mpnn: "SolubleMPNN",
  ligand_mpnn: "LigandMPNN",
  per_residue_label_membrane_mpnn: "MembraneMPNN (per residue)",
  global_label_membrane_mpnn: "MembraneMPNN (global)",
  na_mpnn: "NA-MPNN",
};

const MODEL_NOTES = {
  protein_mpnn: "The original. Backbone only.",
  soluble_mpnn: "Trained without membrane proteins; avoids hydrophobic surfaces.",
  ligand_mpnn: "Sees heteroatoms — ligands, cofactors, metals, nucleic acids.",
  per_residue_label_membrane_mpnn: "Takes a per-residue buried/interface/soluble label.",
  global_label_membrane_mpnn: "Takes one label for the whole chain.",
  na_mpnn: "Designs RNA and protein–DNA together; nucleic acids become "
    + "positions rather than ligand context.",
};

async function loadModelList() {
  const response = await fetch(`${WEIGHTS_BASE}/models.json`);
  state.models = await response.json();
  const select = $("model-select");
  select.innerHTML = "";

  const byType = new Map();
  for (const model of state.models) {
    if (!byType.has(model.model_type)) byType.set(model.model_type, []);
    byType.get(model.model_type).push(model);
  }
  for (const [type, group] of byType) {
    const optgroup = document.createElement("optgroup");
    optgroup.label = MODEL_LABELS[type] ?? type;
    for (const model of group) {
      const option = document.createElement("option");
      option.value = model.name;
      option.dataset.type = type;
      const noise = model.noise_level.toFixed(2);
      // NA-MPNN's checkpoint records no training noise, so saying "0.00 Å"
      // would be a claim rather than a blank.
      const size = `${(model.bytes / 1e6).toFixed(1)} MB`;
      option.textContent = type === "na_mpnn"
        ? `${model.name}  —  ${size}`
        : `${model.name}  —  ${noise} Å training noise, ${size}`;
      optgroup.appendChild(option);
    }
    select.appendChild(optgroup);
  }
  select.value = "proteinmpnn_v_48_020";
  updateModelHint();
}

function updateModelHint() {
  const option = $("model-select").selectedOptions[0];
  if (!option) return;
  const type = option.dataset.type;
  const model = state.models.find((m) => m.name === option.value);
  $("model-hint").textContent = `${MODEL_NOTES[type] ?? ""} `
    + `k=${model.k_neighbors} neighbours`
    + (model.atom_context_num ? `, ${model.atom_context_num} ligand atoms per residue` : "");
  // Only LigandMPNN reads heteroatoms; every other family's encoder never
  // looks at them, so showing a cofactor would imply context that is not there.
  viewer.showLigand = readsLigands();
  if (state.structure) redraw();
  $("atom-context-row").hidden = !readsLigands();
  $("side-chain-row").hidden = !readsLigands();
  refreshAffordances();
  $("membrane-global-row").hidden = type !== "global_label_membrane_mpnn";
  $("membrane-perres-row").hidden = type !== "per_residue_label_membrane_mpnn";
}

// ---------------------------------------------------------------------------
// Structure loading
// ---------------------------------------------------------------------------

/**
 * The one status line.
 *
 * There used to be five -- load, model, design, score, kernel -- permanently on
 * screen, of which four were usually stale. The worker is serial, so there is
 * only ever one thing happening and only ever one thing worth saying. `kind`
 * goes on a data attribute rather than into `className`, which the old helper
 * overwrote wholesale and so destroyed any other class on the element.
 */
function setStatus(text, kind = "") {
  const el = $("status");
  el.textContent = text;
  if (kind) el.dataset.kind = kind;
  else delete el.dataset.kind;
}

async function loadStructureText(text, label) {
  let structure;
  const nucleic = wantsNucleic();
  try {
    structure = structureFromText(text, { nucleicAsResidues: nucleic });
  } catch (error) {
    setStatus(`Could not parse: ${error.message}`, "error");
    return;
  }
  if (structure.L === 0) {
    setStatus(nucleic
      ? "No protein or nucleic-acid residues found."
      : "No protein residues with a C-alpha found.", "error");
    return;
  }

  // Kept so a change of model family can re-parse: NA-MPNN turns nucleic acids
  // into model positions, which changes L and the alphabet.
  state.structureText = text;
  state.structureLabel = label;
  state.structure = structure;
  state.structureId += 1;
  state.designMask = new Float32Array(structure.L).fill(1);
  state.selection = new Set(Array.from({ length: structure.L }, (_, i) => i));
  state.membraneLabels = new Int32Array(structure.L);
  clearOverrides(state.constraints);
  state.membraneVersion += 1;
  state.designs = [];
  state.activeDesign = -1;
  state.profile = null;
  state.encodedFor = null;

  viewer.setStructure(structure);
  $("color-mode").value = structure.chainList.length > 1 ? "chain" : "rainbow";
  clearTies(state.ties);
  renderTies();
  // The inspector is one scrolling column and the constraints are the part you
  // come back to; fold away the two you set once.
  $("group-structure").open = false;
  $("group-model").open = false;
  renderConstraints();
  renderSequenceTrack();
  renderResults();
  refreshAffordances();
  redraw();

  const ligand = structure.ligandType.length;
  let composition = "";
  if (structure.nucleicAsResidues) {
    const n = { pp: 0, dna: 0, rna: 0, unk: 0 };
    for (let i = 0; i < structure.L; i++) {
      const p = structure.polytype[i];
      if (p === POLYTYPE.PP) n.pp++;
      else if (p === POLYTYPE.DNA) n.dna++;
      else if (p === POLYTYPE.RNA) n.rna++;
      else n.unk++;
    }
    composition = " ("
      + [[n.pp, "protein"], [n.dna, "DNA"], [n.rna, "RNA"], [n.unk, "unknown"]]
        .filter(([c]) => c).map(([c, name]) => `${c} ${name}`).join(", ")
      + ")";
  }
  setStatus(`${label}: ${structure.L} residues${composition}, ${structure.chainList.length} chain(s)`
    + (ligand ? `, ${ligand} ligand/heteroatom atoms` : ", no heteroatoms"),
  );
  if (ligand && !$("model-select").value.startsWith("ligandmpnn")) {
    setStatus("This structure has heteroatoms — LigandMPNN will use them.", "");
  }
  await ensureEncoded();
}

async function fetchStructure(id) {
  setStatus(`Fetching ${id}…`, "busy");
  try {
    const text = await fetchPDB(id);
    await loadStructureText(text, id.toUpperCase());
  } catch (error) {
    setStatus(`Fetch failed: ${error.message}`, "error");
  }
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

/**
 * Encode now, after anything already queued. Awaiting this is what Design,
 * Score and Profile do, so it also flushes a pending debounce rather than
 * running against a selection the user has already moved past.
 */
function ensureEncoded() {
  if (encodeTimer !== null) {
    clearTimeout(encodeTimer);
    encodeTimer = null;
  }
  const next = encodeChain.then(runEncode);
  // The chain must survive a rejection, or every later encode inherits it.
  // Callers still see `next` itself, so a failure is not swallowed twice.
  encodeChain = next.then(() => {}, () => {});
  return next;
}

/** Encode once the page has been quiet for a moment. Not awaited. */
function scheduleEncode() {
  if (encodeTimer !== null) clearTimeout(encodeTimer);
  encodeTimer = setTimeout(() => {
    encodeTimer = null;
    ensureEncoded();
  }, ENCODE_DEBOUNCE_MS);
}

async function runEncode() {
  const structure = state.structure;
  if (!structure) return false;
  const name = $("model-select").value;
  const useAtomContext = $("use-atom-context").checked;
  // Side-chain context reads the fixed residues' side chains, so unlike every
  // other input the encoding depends on the selection -- changing it has to
  // invalidate the cache.
  const useSideChains = $("use-side-chains").checked && readsLigands();
  const selection = useSideChains ? state.designMask.join("") : "";
  const key = `${name}|${useAtomContext}|${useSideChains}|${selection}`
    + `|${state.structureId}|${state.membraneVersion}`;
  if (state.encodedFor === key) return true;
  const token = ++encodeToken;

  try {
    if (state.modelName !== name) {
      setStatus(`Loading ${name}…`, "busy");
      showProgress(0);
      onProgress = (message) => {
        if (message.stage === "download" && message.total) {
          showProgress(message.received / message.total);
        }
      };
      const info = await call("load", { name, baseUrl: WEIGHTS_BASE });
      onProgress = null;
      hideProgress();
      if (token !== encodeToken) return false;
      state.modelName = name;
      state.modelType = info.modelType;
      // Only worth saying when it is bad: the SIMD path is the expected one and
      // saying so every time is a line that never changes.
      $("kernel-status").hidden = info.simd;
      $("kernel-status").textContent = info.simd ? ""
        : "No WebAssembly SIMD in this browser, so this is the JavaScript kernel "
          + "— expect roughly 5x slower.";
    }

    setStatus("Encoding structure…", "busy");
    const t0 = performance.now();
    const info = await call("encode", {
      inputs: {
        X: structure.X, mask: structure.mask,
        residueIdx: structure.residueIdx, chainLabels: structure.chainLabels,
        ligandXyz: structure.ligandXyz, ligandType: structure.ligandType,
        ligandMask: structure.ligandMask,
        membraneLabels: state.membraneLabels ? Array.from(state.membraneLabels) : null,
        ...(structure.nucleicAsResidues
          ? {
            X16: structure.X16,
            X16Mask: structure.X16Mask,
            polytype: Array.from(structure.polytype),
          }
          : {}),
        useAtomContext,
        useSideChains,
        ...(useSideChains
          ? {
            xyz37: structure.xyz37,
            xyz37Mask: structure.xyz37Mask,
            chainMask: Array.from(state.designMask),
          }
          : {}),
      },
    });
    // A newer request has superseded us; its encoding is the one the worker
    // now holds, so do not claim the cache or report readiness.
    if (token !== encodeToken) return false;
    state.encodedFor = key;
    setStatus(`${state.modelName} ready — encoded ${info.L} residues in `
      + `${(info.ms / 1000).toFixed(2)} s (${((performance.now() - t0) / 1000).toFixed(2)} s total)`,
    );
    $("design-btn").disabled = false;
    $("profile-btn").disabled = false;
    $("score-btn").disabled = false;
    return true;
  } catch (error) {
    onProgress = null;
    hideProgress();
    if (token === encodeToken) setStatus(`Failed: ${error.message}`, "error");
    return false;
  }
}

function showProgress(fraction) {
  $("progress").hidden = false;
  $("progress-bar").style.width = `${Math.round(fraction * 100)}%`;
}

function hideProgress() {
  $("progress").hidden = true;
  $("progress-bar").style.width = "0%";
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

/**
 * Show only the controls that can currently do something.
 *
 * Three colour modes read data that may not exist yet, "Near ligand" needs
 * heteroatoms, chain tying needs chains, and the results panel is an empty box
 * until a design run fills it. Leaving them all visible made the page look far
 * busier than the number of decisions actually on offer, and picking an inert
 * one did nothing with no explanation.
 */
function refreshAffordances() {
  const s = state.structure;
  const type = modelType();
  const show = (sel, on) => {
    const el = typeof sel === "string" ? $(sel) : sel;
    if (el) el.hidden = !on;
  };
  const option = (value) => $("color-mode").querySelector(`option[value="${value}"]`);

  show(option("confidence"), Boolean(state.profile));
  show(option("score"), Boolean(state.scorePerPosition));
  show(option("membrane"), type === "per_residue_label_membrane_mpnn"
    || type === "global_label_membrane_mpnn");
  show("select-interface", Boolean(s?.ligandType.length) && readsLigands());
  // Always visible: in a fixed-pane shell, hiding a pane leaves a hole rather
  // than reflowing, and the table has a perfectly good empty state. This also
  // retires the bug where Clear left the panel up showing that empty state.

  // A hidden option stays selected if it was already chosen, which would leave
  // the viewer painting from data that is gone.
  const mode = $("color-mode");
  if (mode.selectedOptions[0]?.hidden) {
    mode.value = s && s.chainList.length > 1 ? "chain" : "rainbow";
    redraw();
  }
}

function refreshSelection() {
  track.invalidate();
  renderConstraints();
  renderSequenceTrack();
  if (state.profile) renderLogo();
  redraw();
  // With side-chain context on, the selection is an encoder input. Everywhere
  // else it is only read at sampling time.
  if ($("use-side-chains").checked && !$("side-chain-row").hidden && state.structure) {
    scheduleEncode();
  }
}

/**
 * Residues with any backbone atom within `cutoff` of a heteroatom.
 *
 * `atoms` narrows it to one ligand -- what clicking a token in the track does.
 * Omitted, it is every heteroatom in the structure, which is the toolbar
 * button.
 */
function nearLigand(cutoff = 6.0, atoms = null) {
  const s = state.structure;
  const hits = new Set();
  if (!s || !s.ligandType.length) return hits;
  const pool = atoms ?? Array.from({ length: s.ligandType.length }, (_, a) => a);
  for (let i = 0; i < s.L; i++) {
    const cx = s.X[i * 12 + 3];
    const cy = s.X[i * 12 + 4];
    const cz = s.X[i * 12 + 5];
    for (const a of pool) {
      const dx = cx - s.ligandXyz[a * 3];
      const dy = cy - s.ligandXyz[a * 3 + 1];
      const dz = cz - s.ligandXyz[a * 3 + 2];
      if (dx * dx + dy * dy + dz * dz <= cutoff * cutoff) {
        hits.add(i);
        break;
      }
    }
  }
  return hits;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function activeSequence() {
  if (state.activeDesign >= 0) return state.designs[state.activeDesign].S;
  return state.structure ? Array.from(state.structure.S) : null;
}

/** Linear blend between two rgb triples. */
function lerpRgb(a, b, t) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

/**
 * Colour and weight for one residue.
 *
 * `dim` is the renderer's own mechanism for pushing something back, so fixed
 * residues are dimmed through it rather than by pre-blending the colour --
 * which keeps depth shading and selection dimming from compounding into mud.
 */
function colourFor(i) {
  const s = state.structure;
  const dim = state.designMask[i] > 0 ? 1 : 0.42;
  const mode = $("color-mode").value;
  let rgb;
  switch (mode) {
    case "design":
      rgb = state.designMask[i] ? [56, 189, 248] : [100, 116, 139];
      return { rgb, dim: 1 };
    case "confidence": {
      if (!state.profile) return { rgb: [100, 116, 139], dim };
      // 1 - normalised entropy over the 20 amino acids.
      const t = 1 - Math.min(state.profile.entropy[i] / Math.log(20), 1);
      rgb = lerpRgb([30, 58, 138], [251, 191, 36], t);
      break;
    }
    case "identity": {
      const seq = activeSequence();
      if (!seq) { rgb = [100, 116, 139]; break; }
      rgb = seq[i] === s.S[i] ? [74, 222, 128] : [248, 113, 113];
      break;
    }
    case "rainbow":
      rgb = spectrumRgb(i / Math.max(s.L - 1, 1));
      break;
    case "score": {
      if (!state.scorePerPosition) return { rgb: [100, 116, 139], dim };
      // 0 to ~3 nats covers everything from confident to badly out of place.
      const t = Math.min(state.scorePerPosition[i] / 3, 1);
      rgb = lerpRgb([74, 222, 128], [248, 113, 113], t);
      break;
    }
    case "membrane": {
      const label = state.membraneLabels ? state.membraneLabels[i] : 0;
      rgb = [[100, 116, 139], [251, 191, 36], [56, 189, 248]][label] ?? [100, 116, 139];
      return { rgb, dim: 1 };
    }
    case "chain":
    default:
      rgb = hexToRgb(CHAIN_COLORS[s.chainLabels[i] % CHAIN_COLORS.length]);
  }
  return { rgb, dim };
}

function redraw() {
  if (!state.structure) return;
  viewer.colourAt = colourFor;
  viewer.highlight = state.hover;
  viewer.draw();
  if (state.profile && logo.hover !== state.hover) {
    logo.hover = state.hover;
    logo.draw();
  }
  drawTrack();
}

/**
 * Hand the structure to the sequence viewer.
 *
 * Whether heteroatoms appear as ligand tokens follows the same rule the 3D
 * view uses for drawing them: only LigandMPNN's encoder reads them, and a
 * token for something the model cannot see would misrepresent the input.
 */
function renderSequenceTrack() {
  track.build();
}

/** Colours or selection changed; the structure did not. */
function drawTrack() {
  if (!state.structure) return;
  track.refresh();
}

/**
 * The results table.
 *
 * Columns rather than one template string per row, so scores line up and can be
 * sorted; and the sequence is not among them, because the sequence track is
 * where a sequence is read and it already follows the selected design.
 */
function renderResults() {
  const wrap = $("results");
  wrap.innerHTML = "";
  const has = state.designs.length > 0;
  $("copy-fasta").disabled = !has;
  $("download-fasta").disabled = !has;
  $("clear-results").disabled = !has;
  if (!has) {
    wrap.innerHTML = '<p class="empty">Designs will appear here. '
      + "Click one to paint it onto the structure.</p>";
    return;
  }

  const chains = state.structure.chainList;
  const columns = [
    { key: "id", label: "#", get: (d) => d.id, fmt: (d) => `#${d.id}` },
    { key: "score", label: "score", get: (d) => d.score, fmt: (d) => d.score.toFixed(3) },
    {
      key: "identity",
      label: "recovery",
      title: "identity to the input sequence, over the designed positions only",
      get: (d) => d.identity,
      fmt: (d) => `${(d.identity * 100).toFixed(0)}%`,
    },
    ...(chains.length > 1 ? chains.map((chain) => ({
      key: `chain:${chain}`,
      label: chain,
      title: `recovery within chain ${chain}`,
      get: (d) => d.chainIdentity.get(chain) ?? -1,
      fmt: (d) => {
        const value = d.chainIdentity.get(chain);
        return value === null || value === undefined ? "—" : `${(value * 100).toFixed(0)}%`;
      },
    })) : []),
    {
      key: "designed",
      label: "designed",
      title: "how many positions this run was allowed to change",
      get: (d) => d.designed,
      fmt: (d) => String(d.designed),
    },
    {
      key: "seed",
      label: "seed",
      title: "what makes this row reproducible",
      get: (d) => d.seed,
      fmt: (d) => String(d.seed),
    },
  ];

  const table = document.createElement("table");
  table.className = "results-table";

  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const column of columns) {
    const th = document.createElement("th");
    th.textContent = column.label;
    if (column.title) th.title = column.title;
    if (state.sort.key === column.key) {
      th.dataset.sorted = state.sort.dir > 0 ? "asc" : "desc";
    }
    th.onclick = () => {
      state.sort = state.sort.key === column.key
        ? { key: column.key, dir: -state.sort.dir }
        : { key: column.key, dir: 1 };
      renderResults();
    };
    headRow.appendChild(th);
  }
  head.appendChild(headRow);
  table.appendChild(head);

  const sorter = columns.find((c) => c.key === state.sort.key) ?? columns[1];
  const rows = [...state.designs].sort(
    (a, b) => (sorter.get(a) - sorter.get(b)) * state.sort.dir,
  );

  const body = document.createElement("tbody");
  for (const design of rows) {
    const tr = document.createElement("tr");
    const index = state.designs.indexOf(design);
    if (index === state.activeDesign) tr.className = "active";
    // The sequence is not a column -- the track is where it is read -- but it
    // belongs somewhere reachable: on hover, and as a handle for tests.
    tr.title = design.seq;
    tr.dataset.seq = design.seq;
    for (const column of columns) {
      const td = document.createElement("td");
      td.textContent = column.fmt(design);
      tr.appendChild(td);
    }
    tr.onclick = () => {
      state.activeDesign = state.activeDesign === index ? -1 : index;
      renderResults();
      renderSequenceTrack();
      redraw();
    };
    body.appendChild(tr);
  }
  table.appendChild(body);
  wrap.appendChild(table);
}

function renderLogo() {
  if (!state.profile) return;
  const s = state.structure;
  logo.setAlphabet(alphabet(), biasLetters());
  logo.readTheme(document.body);
  logo.isDesigned = (i) => state.designMask[i] > 0;
  logo.setData({
    probs: state.profile.probs,
    entropy: state.profile.entropy,
    L: s.L,
    native: s.S,
    resSeq: s.resSeq,
    chainIds: s.chainIds,
  });
  logo.hover = state.hover;
  logo.draw();
}

function topAAs(probs, i, n = 3) {
  const order = [];
  const V = numLetters();
  for (let v = 0; v < V; v++) order.push([alphabet()[v], probs[i * V + v]]);
  order.sort((a, b) => b[1] - a[1]);
  return order.slice(0, n).map(([aa, p]) => `${aa} ${(p * 100).toFixed(0)}%`).join(", ");
}

// ---------------------------------------------------------------------------
// Amino-acid bias grid
// ---------------------------------------------------------------------------

/** The selection, in ascending order. */
function selectedPositions() {
  return [...state.selection].sort((a, b) => a - b);
}

/** Mark every selected position as designed, or as kept. */
function setDesigned(on) {
  for (const i of state.selection) state.designMask[i] = on ? 1 : 0;
  state.encodedFor = null;
  refreshSelection();
}

/**
 * Letters the editor offers and the sampler may draw.
 *
 * The 20 amino acids everywhere, plus NA-MPNN's DNA tokens -- which stand in
 * for the RNA bases too, since its `--na_shared_tokens` default aliases them.
 * Everything else (UNK, the legacy RNA letters, MAS, PAD) is omitted, matching
 * the reference's default `--omit_AA`.
 */
function biasLetters() {
  if (!state.structure?.nucleicAsResidues) return [...Array(20).keys()];
  return [...Array(20).keys(), ...NA_NUCLEOTIDES];
}

/** Expand the per-letter bias into the [L, numLetters] array the model wants. */
function buildBias() {
  return buildBiasFrom(state.constraints, {
    L: state.structure.L,
    alphabet: alphabet(),
    letters: biasLetters(),
  });
}

/** Render the tie group list, with whatever the engine would quietly drop. */
function renderTies() {
  const wrap = $("tie-list");
  wrap.innerHTML = "";
  const s = state.structure;
  const { groups } = state.ties;
  $("tie-chains").hidden = (s?.chainList.length ?? 0) < 2;

  for (const group of groups) {
    const row = document.createElement("div");
    row.className = "tie-row";

    const name = document.createElement("span");
    name.className = "tie-label";
    name.textContent = group.label;

    const what = document.createElement("span");
    what.className = "tie-what";
    what.textContent = `${group.positions.length} positions`;
    what.title = group.positions
      .map((p) => (s ? `${s.chainIds[p]}${s.resSeq[p]}` : p))
      .join(" ");

    const mode = document.createElement("select");
    for (const value of ["average", "sum"]) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      mode.appendChild(option);
    }
    mode.value = group.mode;
    mode.onchange = () => { group.mode = mode.value; renderTies(); };

    const drop = document.createElement("button");
    drop.textContent = "\u2715";
    drop.title = "remove this group";
    drop.onclick = () => { removeGroup(state.ties, group.id); renderTies(); redraw(); };

    row.append(name, what, mode, drop);
    wrap.appendChild(row);
  }

  const notes = [];
  if (!groups.length) notes.push("No tied positions.");
  if (s) {
    const { conflicts, outOfRange } = validateTies(state.ties, s.L);
    for (const c of conflicts) {
      notes.push(`${s.chainIds[c.pos]}${s.resSeq[c.pos]} is in `
        + `${[c.keeps, ...c.loses].map((g) => g.label).join(" and ")} — `
        + `${c.keeps.label} wins, the rest drop it.`);
    }
    for (const o of outOfRange) {
      notes.push(`${o.group.label} has ${o.positions.length} position(s) outside the structure.`);
    }
  }
  $("tie-note").textContent = notes.join(" ");
}

/** Add a group and report why not, rather than silently doing nothing. */
function tie(positions, opts) {
  const result = addGroup(state.ties, positions, opts);
  if (!result.ok) {
    setStatus(result.reason, "error");
    return;
  }
  renderTies();
  redraw();
}

// ---------------------------------------------------------------------------
// Running the model
// ---------------------------------------------------------------------------

async function runDesign() {
  if (!await ensureEncoded()) return;
  const button = $("design-btn");
  button.disabled = true;
  setStatus("Designing…", "busy");

  try {
    const seed = $("random-seed").checked
      ? (Math.random() * 2 ** 31) | 0
      : parseInt($("seed").value, 10) || 0;
    const result = await call("design", {
      batch: parseInt($("batch").value, 10),
      temperature: parseFloat($("temperature").value),
      S: Array.from(state.structure.S),
      chainMask: Array.from(state.designMask),
      bias: Array.from(buildBias()),
      symmetry: toEngineSymmetry(state.ties, state.structure.L),
      seed,
    });

    const native = state.structure.S;
    const chains = state.structure.chainList;
    for (let b = 0; b < result.S.length; b++) {
      const S = result.S[b];
      // Recovery over the *designed* positions only -- a kept position matches
      // by construction and would flatter the number. Per chain as well as
      // overall, because "62% overall" hides one chain having gone nowhere.
      let same = 0;
      let counted = 0;
      const perChain = new Map(chains.map((c) => [c, { same: 0, counted: 0 }]));
      for (let i = 0; i < S.length; i++) {
        if (!state.designMask[i]) continue;
        counted++;
        const hit = S[i] === native[i];
        if (hit) same++;
        const row = perChain.get(state.structure.chainIds[i]);
        row.counted++;
        if (hit) row.same++;
      }
      state.designs.push({
        // Stable for the life of the page. The row number used to be a position
        // in an array re-sorted on every run, so "#3" meant a different design
        // after the second run than after the first.
        id: state.nextDesignId++,
        S,
        // Rendered here rather than in the worker: `sequenceToString` only
        // knows the 21-letter alphabet, and FASTA has to carry the RNA letters.
        seq: showSequence(S),
        score: result.scores[b],
        identity: counted ? same / counted : 0,
        chainIdentity: new Map([...perChain].map(
          ([c, r]) => [c, r.counted ? r.same / r.counted : null],
        )),
        designed: counted,
        seed,
      });
    }
    renderResults();
    refreshAffordances();
    setStatus(`${result.S.length} sequences in ${(result.ms / 1000).toFixed(2)} s `
      + `(${(result.ms / result.S.length).toFixed(0)} ms each), seed ${seed}`,
    );
  } catch (error) {
    setStatus(`Failed: ${error.message}`, "error");
  } finally {
    button.disabled = false;
  }
}

async function runProfile() {
  if (!await ensureEncoded()) return;
  const button = $("profile-btn");
  button.disabled = true;
  const mode = $("profile-mode").value;
  setStatus("Computing profile…", "busy");
  if (mode === "exact") showProgress(0);
  onProgress = (message) => {
    if (message.stage === "profile") showProgress(message.received / message.total);
  };

  try {
    const seq = activeSequence();
    const result = await call("profile", {
      S: seq ? Array.from(seq) : null,
      mode: mode === "exact" ? "order" : mode,
      exact: mode === "exact",
    });
    const probs = new Float32Array(result.probs);
    const L = state.structure.L;
    // Over the model's own alphabet: NA-MPNN's profile is [L, 33], and reading
    // it at stride 21 silently mixes neighbouring positions together.
    const V = numLetters();
    const letters = biasLetters();
    const entropy = new Float32Array(L);
    for (let i = 0; i < L; i++) {
      let h = 0;
      let z = 0;
      for (const v of letters) z += probs[i * V + v];
      for (const v of letters) {
        const p = probs[i * V + v] / (z || 1);
        if (p > 0) h -= p * Math.log(p);
      }
      entropy[i] = h;
    }
    state.profile = { probs, entropy };
    showTab("profile");
    refreshAffordances();
    $("profile-hint").textContent = MODE_TEXT[mode] ?? "";
    renderLogo();
    redraw();
    setStatus(`Profile in ${(result.ms / 1000).toFixed(2)} s`);
  } catch (error) {
    setStatus(`Failed: ${error.message}`, "error");
  } finally {
    onProgress = null;
    hideProgress();
    button.disabled = false;
  }
}

/** Parse a pasted sequence: one letter per residue, separators ignored. */
function parseSequence(text, L) {
  // NA-MPNN's alphabet is case sensitive -- lower case is a nucleotide -- so
  // only the protein models may upper-case the input.
  const na = Boolean(state.structure?.nucleicAsResidues);
  const letters = (na ? text : text.toUpperCase()).replace(/[^A-Za-z]/g, "");
  if (letters.length !== L) {
    throw new Error(`expected ${L} residues, got ${letters.length}`);
  }
  const allowed = new Set(biasLetters());
  const S = new Int32Array(L);
  for (let i = 0; i < L; i++) {
    let v = alphabet().indexOf(letters[i]);
    // Under shared tokens an RNA base is stored as the DNA one, so accept both
    // spellings of the same base.
    if (na && !allowed.has(v)) {
      const dna = NA_RNA_TO_DNA.get(v);
      if (dna !== undefined) v = dna;
    }
    if (v < 0 || !allowed.has(v)) {
      throw new Error(`unusable residue "${letters[i]}" at position ${i + 1}`);
    }
    S[i] = v;
  }
  return S;
}

async function runScore() {
  if (!await ensureEncoded()) return;
  const button = $("score-btn");
  button.disabled = true;
  let S;
  try {
    S = parseSequence($("score-seq").value, state.structure.L);
  } catch (error) {
    setStatus(error.message, "error");
    button.disabled = false;
    return;
  }
  const mode = $("score-mode").value;
  setStatus("Scoring…", "busy");
  showProgress(0);
  onProgress = (message) => {
    if (message.stage === "score") showProgress(message.received / message.total);
  };
  try {
    const result = await call("score", {
      S: Array.from(S),
      mode,
      orders: parseInt($("score-orders").value, 10) || 8,
      chainMask: Array.from(state.designMask),
      seed: 0,
    });
    state.scorePerPosition = new Float32Array(result.perPosition);
    refreshAffordances();
    const native = state.structure.S;
    let same = 0;
    for (let i = 0; i < S.length; i++) if (S[i] === native[i]) same++;
    const spread = result.sd === null
      ? ""
      : ` ± ${result.sd.toFixed(4)} over ${result.orders} orders`;
    setStatus(`nll ${result.mean.toFixed(4)}${spread}, `
      + `${((same / S.length) * 100).toFixed(0)}% identical to the input structure's sequence `
      + `(${(result.ms / 1000).toFixed(2)} s)`,
    );
    $("color-mode").value = "score";
    redraw();
  } catch (error) {
    setStatus(`Failed: ${error.message}`, "error");
  } finally {
    onProgress = null;
    hideProgress();
    button.disabled = false;
  }
}

const MODE_TEXT = {
  none: "No position sees any amino acid — this is what the backbone alone implies. "
    + "One decoder pass.",
  "all-but-self": "Pseudo-likelihood: every position is scored as if it were decoded last, "
    + "seeing all the others and none of itself, in a single pass (ar_mask = 1 − I). "
    + "Cheap and stable — see the README for how it compares with the L-pass version.",
  order: "The true autoregressive likelihood, which depends on the decoding order, so it is "
    + "averaged over that many random ones. This is the number the sampler reports for its own "
    + "designs, so it is the like-for-like comparison against them.",
  exact: "One decoder pass per position, each putting that position last. L times the cost, and "
    + "still order-dependent: the other L−1 positions have to be decoded in some order too.",
};

function describeScoreMode() {
  const mode = $("score-mode").value;
  $("score-orders-row").hidden = mode !== "order";
  $("score-hint").textContent = `${MODE_TEXT[mode]} Lower is better. Paste one letter per `
    + "residue; slashes and whitespace are ignored, so a multi-chain sequence can be pasted as "
    + "it is displayed.";
}

function fastaText() {
  const name = $("pdb-id").value.trim() || "design";
  return state.designs.map((d) =>
    `>${name}_${d.id} score=${d.score.toFixed(4)} identity=${d.identity.toFixed(3)}`
    + ` seed=${d.seed}\n${d.seq}`,
  ).join("\n") + "\n";
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

$("fetch-btn").onclick = () => {
  const id = $("pdb-id").value.trim();
  if (id) fetchStructure(id);
};

$("pdb-id").onkeydown = (event) => {
  if (event.key === "Enter") $("fetch-btn").click();
};

for (const button of document.querySelectorAll("[data-example]")) {
  button.onclick = () => {
    $("pdb-id").value = button.dataset.example;
    if (button.dataset.example === "1STP" || button.dataset.example === "4KT0") {
      $("model-select").value = "ligandmpnn_v_32_010_25";
      updateModelHint();
    }
    fetchStructure(button.dataset.example);
  };
}

$("file-input").onchange = async (event) => {
  const file = event.target.files[0];
  if (file) await loadStructureText(await file.text(), file.name);
};

const drop = $("file-drop");
drop.ondragover = (event) => {
  event.preventDefault();
  drop.classList.add("over");
};
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = async (event) => {
  event.preventDefault();
  drop.classList.remove("over");
  const file = event.dataTransfer.files[0];
  if (file) await loadStructureText(await file.text(), file.name);
};

$("model-select").onchange = async () => {
  updateModelHint();
  state.encodedFor = null;
  if (!state.structure) return;
  // NA-MPNN makes nucleic acids into model positions, so switching to or from
  // it changes L, the alphabet and the selection -- the structure has to be
  // read again rather than reinterpreted.
  if (state.structureText && wantsNucleic() !== state.structure.nucleicAsResidues) {
    await loadStructureText(state.structureText, state.structureLabel);
    return;
  }
  // Which heteroatoms the track shows depends on the family, so the item list
  // has to be rebuilt and not merely redrawn.
  renderSequenceTrack();
  ensureEncoded();
};

$("use-atom-context").onchange = () => {
  state.encodedFor = null;
  if (state.structure) ensureEncoded();
};

$("use-side-chains").onchange = () => {
  state.encodedFor = null;
  if (state.structure) ensureEncoded();
};

$("tie-selection").onclick = () => tie(selectedPositions());

$("tie-across").onclick = () => {
  const s = state.structure;
  if (!s) return;
  const { groups, unmatched } = acrossChains(s, selectedPositions());
  for (const group of groups) addGroup(state.ties, group, { source: "manual" });
  renderTies();
  redraw();
  setStatus(`${groups.length} group(s) tied across chains`
    + (unmatched.length ? `; ${unmatched.length} selected position(s) had no counterpart` : ""));
};

$("tie-chains").onclick = () => {
  const { groups, note } = chainGroups(state.structure);
  if (!groups) {
    setStatus(note, "error");
    return;
  }
  for (const group of groups) addGroup(state.ties, group, { source: "homo-oligomer" });
  renderTies();
  redraw();
  setStatus(note);
};

$("tie-add-text").onclick = () => {
  const s = state.structure;
  if (!s) return;
  const { positions, unresolved } = parsePositions(s, $("tie-text").value);
  if (unresolved.length) {
    setStatus(`Could not resolve: ${unresolved.join(", ")}`, "error");
    return;
  }
  tie(positions);
  $("tie-text").value = "";
};

$("temperature").oninput = (event) => {
  $("temperature-out").textContent = Number(event.target.value).toFixed(2);
};
$("batch").oninput = (event) => {
  $("batch-out").textContent = event.target.value;
};

$("color-mode").onchange = redraw;
$("profile-mode").onchange = () => {
  if (state.profile) runProfile();
};

$("select-all").onclick = () => {
  for (let i = 0; i < state.structure.L; i++) state.selection.add(i);
  refreshSelection();
};
$("select-none").onclick = () => {
  state.selection.clear();
  refreshSelection();
};
$("select-invert").onclick = () => {
  for (let i = 0; i < state.structure.L; i++) {
    if (state.selection.has(i)) state.selection.delete(i);
    else state.selection.add(i);
  }
  refreshSelection();
};
/**
 * Give the structure the whole window.
 *
 * The renderer fits the model into `min(width, height)` of its canvas, so on a
 * wide screen height is the only dimension that makes it bigger -- and the
 * other two rows have already given up what they can spare. This borrows the
 * rest for as long as you want it.
 */
$("expand-structure").onclick = (event) => {
  const on = document.body.classList.toggle("focus-structure");
  event.currentTarget.setAttribute("aria-pressed", String(on));
  redraw();
};

$("mark-design").onclick = () => setDesigned(true);
$("mark-keep").onclick = () => setDesigned(false);

$("select-interface").onclick = () => {
  const hits = nearLigand();
  if (!hits.size) {
    setStatus("No heteroatoms in this structure.", "error");
    return;
  }
  state.selection.clear();
  for (const i of hits) state.selection.add(i);
  refreshSelection();
};

function paintMembrane(label) {
  if (!state.structure) return;
  for (let i = 0; i < state.structure.L; i++) {
    if (state.selection.has(i)) state.membraneLabels[i] = label;
  }
  state.membraneVersion += 1;
  state.encodedFor = null;
  $("color-mode").value = "membrane";
  refreshSelection();
  scheduleEncode();
}

$("mem-soluble").onclick = () => paintMembrane(0);
$("mem-interface").onclick = () => paintMembrane(1);
$("mem-buried").onclick = () => paintMembrane(2);
$("mem-reset").onclick = () => {
  state.membraneLabels.fill(0);
  state.membraneVersion += 1;
  state.encodedFor = null;
  refreshSelection();
  scheduleEncode();
};

$("membrane-global").onchange = (event) => {
  state.membraneLabels.fill(Number(event.target.value));
  state.membraneVersion += 1;
  state.encodedFor = null;
  ensureEncoded();
};

$("score-btn").onclick = runScore;
$("score-mode").onchange = describeScoreMode;
describeScoreMode();
$("score-native").onclick = () => {
  if (state.structure) $("score-seq").value = state.structure.sequence;
};

$("design-btn").onclick = runDesign;
$("profile-btn").onclick = runProfile;
$("bias-reset").onclick = () => {
  clearAll(state.constraints);
  renderConstraints();
};
$("bias-clear-overrides").onclick = () => {
  clearOverrides(state.constraints);
  renderConstraints();
};
$("clear-results").onclick = () => {
  state.designs = [];
  state.activeDesign = -1;
  renderResults();
  // Without this the panel stayed on screen showing its own empty placeholder.
  refreshAffordances();
  renderSequenceTrack();
  redraw();
};
$("copy-fasta").onclick = () => navigator.clipboard.writeText(fastaText());
$("download-fasta").onclick = () => download(
  fastaText(), "text/plain", `${$("pdb-id").value.trim() || "designs"}.fasta`,
);

// --- viewer interaction ---------------------------------------------------

const canvas = $("viewer");
const tooltip = $("tooltip");

function localPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return [event.clientX - rect.left, event.clientY - rect.top];
}

// Shift-drag is box select. These listeners are registered BEFORE orbit()'s so
// that, at the target, they run first and can claim the gesture with
// stopImmediatePropagation -- otherwise the camera would rotate underneath the
// selection rectangle.
let boxing = null;

canvas.addEventListener("pointerdown", (event) => {
  if (!event.shiftKey) return;
  event.stopImmediatePropagation();
  event.preventDefault();
  try {
    canvas.setPointerCapture(event.pointerId);
  } catch { /* not capturable */ }
  const p = localPoint(event);
  boxing = { from: p, to: p, subtract: event.altKey };
  viewer.box = boxing;
});

canvas.addEventListener("pointermove", (event) => {
  if (!boxing) return;
  event.stopImmediatePropagation();
  boxing.to = localPoint(event);
  viewer.box = boxing;
  redraw();
});

function endBox(event) {
  if (!boxing) return;
  event.stopImmediatePropagation();
  const hits = viewer.pickBox(boxing.from, boxing.to);
  for (const i of hits) {
    if (boxing.subtract) state.selection.delete(i);
    else state.selection.add(i);
  }
  boxing = null;
  viewer.box = null;
  refreshSelection();
}

canvas.addEventListener("pointerup", endBox);
canvas.addEventListener("pointercancel", endBox);

orbit(canvas, viewer.camera, redraw, {
  zoomMin: 0.4,
  zoomMax: 12,
  onClick: (event) => {
    const i = viewer.pick(localPoint(event));
    if (i < 0) return;
    if (state.selection.has(i)) state.selection.delete(i);
    else state.selection.add(i);
    refreshSelection();
  },
  onReset: () => viewer.resetCamera(),
});

// Hover, but only when nothing is being dragged.
canvas.addEventListener("pointermove", (event) => {
  if (boxing || event.buttons !== 0 || !state.structure) return;
  const point = localPoint(event);
  const i = viewer.pick(point);
  if (i === state.hover) return;
  state.hover = i;
  if (i >= 0) {
    const s = state.structure;
    const ss = { H: "helix", E: "strand", T: "turn", C: "loop" }[viewer.sec[i]] ?? "loop";
    tooltip.hidden = false;
    tooltip.textContent = `${s.resNames[i]} ${s.chainIds[i]}${s.resSeq[i]}${s.iCodes[i]}  ${ss}\n`
      + `${state.designMask[i] ? "designed" : "kept"}`
      + `${state.selection.has(i) ? " · selected" : ""}`
      + (state.profile ? `\n${topAAs(state.profile.probs, i)}` : "");
    tooltip.style.left = `${Math.min(point[0] + 14, canvas.clientWidth - 190)}px`;
    tooltip.style.top = `${point[1] + 14}px`;
  } else {
    tooltip.hidden = true;
  }
  redraw();
});

canvas.addEventListener("pointerleave", () => {
  tooltip.hidden = true;
  state.hover = -1;
  redraw();
});

// Panes change size without the window doing so -- the results table appearing,
// the inspector scrolling, a tab switching -- and all three canvases size
// themselves from their pane, so watching the window is not enough.
const observeResize = (el, run) => {
  if (!el) return;
  let queued = false;
  new ResizeObserver(() => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      run();
    });
  }).observe(el);
};
observeResize($("viewer").parentElement, () => redraw());
// The panel, not the view inside it: the panel is sized by the grid, so
// rebuilding cannot change its height and there is no feedback loop. Watching
// the view would also mean the first build measures a wrap that has not been
// laid out yet, which came out as 6px and clipped the track to four chains.
observeResize($("sequence-panel"), () => renderSequenceTrack());
observeResize($("pane-profile"), () => { if (state.profile) renderLogo(); });
window.addEventListener("resize", () => {
  redraw();
  drawTrack();
});
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);

// --- logo interaction ------------------------------------------------------

const logoCanvas = $("logo");
const logoTooltip = $("logo-tooltip");

logoCanvas.addEventListener("pointermove", (event) => {
  const rect = logoCanvas.getBoundingClientRect();
  const i = logo.pick(event.clientX - rect.left);
  if (i !== state.hover) {
    state.hover = i;
    redraw();
  }
  if (i >= 0) {
    logoTooltip.hidden = false;
    logoTooltip.textContent = logo.describe(i);
    logoTooltip.style.left = `${event.clientX - rect.left + 12}px`;
    logoTooltip.style.top = "6px";
  } else {
    logoTooltip.hidden = true;
  }
});

logoCanvas.addEventListener("pointerleave", () => {
  logoTooltip.hidden = true;
  state.hover = -1;
  redraw();
});

logoCanvas.addEventListener("click", (event) => {
  const rect = logoCanvas.getBoundingClientRect();
  const i = logo.pick(event.clientX - rect.left);
  if (i < 0) return;
  if (state.selection.has(i)) state.selection.delete(i);
  else state.selection.add(i);
  refreshSelection();
});

/** The bottom-right pane holds two read-only analyses; only one at a time. */
function showTab(which) {
  for (const name of ["profile", "score"]) {
    const on = name === which;
    $(`tab-${name}`).setAttribute("aria-selected", String(on));
    $(`pane-${name}`).hidden = !on;
  }
}
$("tab-profile").onclick = () => showTab("profile");
$("tab-score").onclick = () => showTab("score");

$("profile-view").onchange = (event) => {
  logo.mode = event.target.value;
  renderLogo();
};

$("logo-csv").onclick = () => {
  if (!state.profile) return;
  download(logo.toCsv(), "text/csv", `${state.structureLabel || "profile"}-profile.csv`);
};

$("logo-wider").onclick = () => {
  logo.columnWidth = Math.min(logo.columnWidth + 4, 48);
  renderLogo();
};
$("logo-narrower").onclick = () => {
  logo.columnWidth = Math.max(logo.columnWidth - 4, 5);
  renderLogo();
};

// --- boot -----------------------------------------------------------------

renderConstraints();
renderTies();
loadModelList().then(() => {
  const params = new URLSearchParams(location.search);
  const pdb = params.get("pdb");
  if (params.get("model")) {
    $("model-select").value = params.get("model");
    updateModelHint();
  }
  if (pdb) {
    $("pdb-id").value = pdb;
    fetchStructure(pdb);
  }
});
