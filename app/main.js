// Page controller: owns the DOM, the selection state, and the worker.

import { ALPHABET } from "../mpnn/constants.js";
import {
  naDisplaySequence, NA_ALPHABET, NA_DNA_TO_RNA, NA_NUCLEOTIDES, NA_RNA_TO_DNA, POLYTYPE,
} from "../mpnn/na.js";
import { fetchPDB, structureFromText } from "../mpnn/pdb.js";
import { Viewer, hexToRgb, orbit, spectrumRgb } from "./viewer.js";
import { AA_COLORS, Logo } from "./logo.js";

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
   * Per-letter bias, sized for the widest alphabet (NA-MPNN's 33).
   *
   * It must not be sized to the *current* model: a short array reads
   * `undefined` past its end, writing that into a Float32Array gives NaN, and
   * a NaN bias loses every comparison in the sampler -- which showed up as
   * NA-MPNN quietly designing amino acids into an RNA chain.
   */
  bias: new Float32Array(33),
  omitted: new Set(),
  /**
   * Position -> Float32Array of bias values that replace the global table
   * there, and Position -> Set of omitted amino acids. Sparse: only positions
   * the user actually touched appear.
   */
  biasOverrides: new Map(),
  omitOverrides: new Map(),
  designs: [],
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
/** True when the selected model treats nucleic acids as model positions. */
function wantsNucleic() {
  return ($("model-select").selectedOptions[0]?.dataset.type ?? state.modelType) === "na_mpnn";
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
  if (s?.nucleicAsResidues && s.isRNA[i]) {
    return NA_ALPHABET[NA_DNA_TO_RNA.get(v) ?? v] ?? "X";
  }
  return alphabet()[v] ?? "X";
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

const viewer = new Viewer($("viewer"));
const logo = new Logo($("logo"));

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
  $("atom-context-row").hidden = type !== "ligand_mpnn";
  $("side-chain-row").hidden = type !== "ligand_mpnn";
  $("membrane-global-row").hidden = type !== "global_label_membrane_mpnn";
  $("membrane-perres-row").hidden = type !== "per_residue_label_membrane_mpnn";
}

// ---------------------------------------------------------------------------
// Structure loading
// ---------------------------------------------------------------------------

function setStatus(id, text, kind = "") {
  const el = $(id);
  el.textContent = text;
  el.className = `status ${kind}`;
}

async function loadStructureText(text, label) {
  let structure;
  const nucleic = wantsNucleic();
  try {
    structure = structureFromText(text, { nucleicAsResidues: nucleic });
  } catch (error) {
    setStatus("load-status", `Could not parse: ${error.message}`, "error");
    return;
  }
  if (structure.L === 0) {
    setStatus("load-status", nucleic
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
  state.membraneLabels = new Int32Array(structure.L);
  state.biasOverrides.clear();
  state.omitOverrides.clear();
  state.membraneVersion += 1;
  state.designs = [];
  state.activeDesign = -1;
  state.profile = null;
  state.encodedFor = null;
  $("profile-panel").hidden = true;

  viewer.setStructure(structure);
  $("color-mode").value = structure.chainList.length > 1 ? "chain" : "rainbow";
  refreshHomoOligomer();
  renderChainToggles();
  renderSequenceTrack();
  renderResults();
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
  setStatus(
    "load-status",
    `${label}: ${structure.L} residues${composition}, ${structure.chainList.length} chain(s)`
    + (ligand ? `, ${ligand} ligand/heteroatom atoms` : ", no heteroatoms"),
  );
  if (ligand && !$("model-select").value.startsWith("ligandmpnn")) {
    setStatus("model-status", "This structure has heteroatoms — LigandMPNN will use them.", "");
  }
  await ensureEncoded();
}

async function fetchStructure(id) {
  setStatus("load-status", `Fetching ${id}…`, "busy");
  try {
    const text = await fetchPDB(id);
    await loadStructureText(text, id.toUpperCase());
  } catch (error) {
    setStatus("load-status", `Fetch failed: ${error.message}`, "error");
  }
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

async function ensureEncoded() {
  const structure = state.structure;
  if (!structure) return false;
  const name = $("model-select").value;
  const useAtomContext = $("use-atom-context").checked;
  // Side-chain context reads the fixed residues' side chains, so unlike every
  // other input the encoding depends on the selection -- changing it has to
  // invalidate the cache.
  const type = $("model-select").selectedOptions[0]?.dataset.type;
  const useSideChains = $("use-side-chains").checked && type === "ligand_mpnn";
  const selection = useSideChains ? state.designMask.join("") : "";
  const key = `${name}|${useAtomContext}|${useSideChains}|${selection}`
    + `|${state.structureId}|${state.membraneVersion}`;
  if (state.encodedFor === key) return true;
  const token = ++encodeToken;

  try {
    if (state.modelName !== name) {
      setStatus("model-status", `Loading ${name}…`, "busy");
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
      $("kernel-status").textContent = info.simd
        ? "Running on the WebAssembly SIMD kernel."
        : "Running on the JavaScript kernel — this browser has no WebAssembly SIMD, "
          + "so expect roughly 5x slower.";
    }

    setStatus("model-status", "Encoding structure…", "busy");
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
    setStatus(
      "model-status",
      `${state.modelName} ready — encoded ${info.L} residues in `
      + `${(info.ms / 1000).toFixed(2)} s (${((performance.now() - t0) / 1000).toFixed(2)} s total)`,
    );
    $("design-btn").disabled = false;
    $("profile-btn").disabled = false;
    $("score-btn").disabled = false;
    return true;
  } catch (error) {
    onProgress = null;
    hideProgress();
    if (token === encodeToken) setStatus("model-status", `Failed: ${error.message}`, "error");
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

function renderChainToggles() {
  const wrap = $("chain-toggles");
  wrap.innerHTML = "";
  if (!state.structure) return;
  for (const chain of state.structure.chainList) {
    const button = document.createElement("button");
    button.textContent = `chain ${chain}`;
    button.onclick = () => {
      const positions = [];
      for (let i = 0; i < state.structure.L; i++) {
        if (state.structure.chainIds[i] === chain) positions.push(i);
      }
      const allOn = positions.every((i) => state.designMask[i] === 1);
      for (const i of positions) state.designMask[i] = allOn ? 0 : 1;
      refreshSelection();
    };
    wrap.appendChild(button);
  }
}

function refreshSelection() {
  if ($("bias-scope").value === "selected") renderBiasGrid();
  renderSequenceTrack();
  if (state.profile) renderLogo();
  redraw();
  // With side-chain context on, the selection is an encoder input. Everywhere
  // else it is only read at sampling time.
  if ($("use-side-chains").checked && !$("side-chain-row").hidden && state.structure) {
    ensureEncoded();
  }
}

/** Residues with any backbone atom within `cutoff` of a heteroatom. */
function nearLigand(cutoff = 6.0) {
  const s = state.structure;
  const hits = new Set();
  if (!s || !s.ligandType.length) return hits;
  for (let i = 0; i < s.L; i++) {
    const cx = s.X[i * 12 + 3];
    const cy = s.X[i * 12 + 4];
    const cz = s.X[i * 12 + 5];
    for (let a = 0; a < s.ligandType.length; a++) {
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
}

function renderSequenceTrack() {
  const track = $("sequence-track");
  track.innerHTML = "";
  const s = state.structure;
  if (!s) return;
  const seq = activeSequence();
  let lastChain = null;

  for (let i = 0; i < s.L; i++) {
    if (s.chainIds[i] !== lastChain) {
      const label = document.createElement("span");
      label.className = "chain-label";
      label.textContent = `${lastChain === null ? "" : " "}${s.chainIds[i]}:`;
      track.appendChild(label);
      lastChain = s.chainIds[i];
    }
    const span = document.createElement("span");
    span.className = "res " + (state.designMask[i] ? "designed" : "fixed");
    if (seq && seq[i] !== s.S[i]) span.classList.add("changed");
    span.textContent = displayLetter(i, seq ? seq[i] : s.S[i]);
    span.dataset.i = i;
    span.title = `${s.resNames[i]} ${s.chainIds[i]}${s.resSeq[i]}${s.iCodes[i]}`;
    track.appendChild(span);
  }

  track.onclick = (event) => {
    const i = event.target?.dataset?.i;
    if (i === undefined) return;
    state.designMask[+i] = state.designMask[+i] ? 0 : 1;
    refreshSelection();
  };
  track.onmousemove = (event) => {
    const i = event.target?.dataset?.i;
    state.hover = i === undefined ? -1 : +i;
    redraw();
  };
  track.onmouseleave = () => {
    state.hover = -1;
    redraw();
  };
}

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

  const native = state.structure.S;
  state.designs.forEach((design, index) => {
    const row = document.createElement("div");
    row.className = "design" + (index === state.activeDesign ? " active" : "");

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `#${index + 1}  nll ${design.score.toFixed(3)}  `
      + `id ${(design.identity * 100).toFixed(0)}%`;

    const seq = document.createElement("div");
    seq.className = "seq";
    let html = "";
    for (let i = 0; i < design.S.length; i++) {
      const same = design.S[i] === native[i];
      html += `<span class="${same ? "same" : "diff"}">${displayLetter(i, design.S[i])}</span>`;
    }
    seq.innerHTML = html;

    row.appendChild(meta);
    row.appendChild(seq);
    row.onclick = () => {
      state.activeDesign = state.activeDesign === index ? -1 : index;
      renderResults();
      renderSequenceTrack();
      redraw();
    };
    wrap.appendChild(row);
  });
}

// ---------------------------------------------------------------------------
// Sequence logo
// ---------------------------------------------------------------------------

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

function selectedPositions() {
  const out = [];
  if (!state.structure) return out;
  for (let i = 0; i < state.structure.L; i++) if (state.designMask[i] > 0) out.push(i);
  return out;
}

/**
 * The bias currently shown in the grid.
 *
 * In "selected" scope this is the value shared by every selected position, or
 * the global value where they disagree -- editing then writes to all of them,
 * which is the behaviour that makes a mixed selection usable.
 */
function shownBias(v) {
  if ($("bias-scope").value === "global") {
    return { value: state.bias[v], omitted: state.omitted.has(v), override: false };
  }
  const positions = selectedPositions();
  if (!positions.length) {
    return { value: state.bias[v], omitted: state.omitted.has(v), override: false };
  }
  const first = state.biasOverrides.get(positions[0]);
  const value = first ? first[v] : state.bias[v];
  const omitted = (state.omitOverrides.get(positions[0]) ?? state.omitted).has(v);
  const override = positions.some((p) => state.biasOverrides.has(p) || state.omitOverrides.has(p));
  return { value, omitted, override };
}

function writeBias(v, value) {
  if ($("bias-scope").value === "global") {
    state.bias[v] = value;
    return;
  }
  for (const p of selectedPositions()) {
    if (!state.biasOverrides.has(p)) state.biasOverrides.set(p, Float32Array.from(state.bias));
    state.biasOverrides.get(p)[v] = value;
  }
}

function toggleOmit(v) {
  if ($("bias-scope").value === "global") {
    if (state.omitted.has(v)) state.omitted.delete(v);
    else state.omitted.add(v);
    return;
  }
  const positions = selectedPositions();
  const turningOn = !(state.omitOverrides.get(positions[0]) ?? state.omitted).has(v);
  for (const p of positions) {
    if (!state.omitOverrides.has(p)) state.omitOverrides.set(p, new Set(state.omitted));
    if (turningOn) state.omitOverrides.get(p).add(v);
    else state.omitOverrides.get(p).delete(v);
  }
}

/**
 * Letters the bias grid offers and the sampler may draw.
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

function renderBiasGrid() {
  const wrap = $("aa-bias");
  wrap.innerHTML = "";
  const scoped = $("bias-scope").value === "selected";
  const nSelected = selectedPositions().length;

  for (const v of biasLetters()) {
    const aa = alphabet()[v];
    const shown = shownBias(v);
    const cell = document.createElement("div");
    cell.className = "cell" + (shown.omitted ? " omitted" : "")
      + (scoped && shown.override ? " override" : "");

    const letter = document.createElement("span");
    letter.className = "letter";
    letter.textContent = aa;
    letter.style.color = AA_COLORS[aa];
    letter.title = "click to omit";
    letter.onclick = () => {
      toggleOmit(v);
      renderBiasGrid();
    };

    const input = document.createElement("input");
    input.type = "number";
    input.step = "0.5";
    input.value = String(shown.value);
    input.disabled = scoped && nSelected === 0;
    input.oninput = () => {
      writeBias(v, parseFloat(input.value) || 0);
      renderBiasGrid();
    };

    cell.appendChild(letter);
    cell.appendChild(input);
    wrap.appendChild(cell);
  }

  const overrides = new Set([...state.biasOverrides.keys(), ...state.omitOverrides.keys()]);
  $("bias-clear-overrides").hidden = overrides.size === 0;
  $("bias-summary").textContent = scoped
    ? `Editing ${nSelected} selected position(s). ${overrides.size} position(s) carry an override.`
    : (overrides.size
      ? `${overrides.size} position(s) carry an override, which wins over these values.`
      : "");
}

/** Expand the per-letter bias into the [L, numLetters] array the model wants. */
function buildBias() {
  const L = state.structure.L;
  const V = numLetters();
  const allowed = new Set(biasLetters());
  const bias = new Float32Array(L * V);
  for (let i = 0; i < L; i++) {
    const local = state.biasOverrides.get(i);
    const omit = state.omitOverrides.get(i) ?? state.omitted;
    for (let v = 0; v < V; v++) {
      // Anything outside the offered letters is omitted outright: "X" for the
      // protein models, and for NA-MPNN also the legacy RNA tokens and the
      // MAS/PAD placeholders that never name a real residue.
      bias[i * V + v] = !allowed.has(v) || omit.has(v)
        ? -1e9
        : (local ? local[v] : state.bias[v]);
    }
  }
  return bias;
}

/**
 * Tie every chain to every other, LigandMPNN's `--homo_oligomer`.
 *
 * The reference matches residues by *number*, not by position in the chain, so
 * a complex whose chains share a numbering ties correctly even when one of them
 * has a gap. When the numbering does not line up at all -- chain B continuing
 * where A left off, say -- that finds nothing, so equal-length chains fall back
 * to tying by position. Which one ran is reported, because the two disagree
 * exactly when it matters.
 *
 * Weights are 1/chains, so a group's members contribute the mean of their
 * logits rather than the sum.
 *
 * @returns {{groups: {pos: number, weight: number}[][] | null, note: string}}
 */
function homoOligomerGroups() {
  const s = state.structure;
  const chains = s?.chainList ?? [];
  if (chains.length < 2) {
    return { groups: null, note: "Needs at least two chains." };
  }
  const weight = 1 / chains.length;

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
    const note = `${byNumber.length} group(s) of ${chains.length}, matched by residue number`
      + (unmatched ? `; ${unmatched} residue(s) of chain ${chains[0]} have no counterpart` : "");
    return { groups: byNumber.map((g) => g.map((pos) => ({ pos, weight }))), note };
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
  const byPosition = perChain[0].map((_, i) => perChain.map((c) => ({ pos: c[i], weight })));
  return {
    groups: byPosition,
    note: `${byPosition.length} group(s) of ${chains.length}, matched by position — the chains `
      + "share no residue numbers, so this assumes they are aligned end to end.",
  };
}

function parseSymmetry() {
  if ($("homo-oligomer").checked) return homoOligomerGroups().groups;
  const text = $("symmetry").value.trim();
  if (!text) return null;
  const groups = [];
  for (const chunk of text.split(",")) {
    const positions = chunk.split("+")
      .map((p) => parseInt(p.trim(), 10) - 1)
      .filter((p) => Number.isInteger(p) && p >= 0 && p < state.structure.L);
    if (positions.length > 1) groups.push(positions);
  }
  return groups.length ? groups : null;
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

async function runDesign() {
  if (!await ensureEncoded()) return;
  const button = $("design-btn");
  button.disabled = true;
  setStatus("design-status", "Designing…", "busy");

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
      symmetry: parseSymmetry(),
      seed,
    });

    const native = state.structure.S;
    for (let b = 0; b < result.S.length; b++) {
      const S = result.S[b];
      let same = 0;
      let counted = 0;
      for (let i = 0; i < S.length; i++) {
        if (!state.designMask[i]) continue;
        counted++;
        if (S[i] === native[i]) same++;
      }
      state.designs.push({
        S,
        // Rendered here rather than in the worker: `sequenceToString` only
        // knows the 21-letter alphabet, and FASTA has to carry the RNA letters.
        seq: showSequence(S),
        score: result.scores[b],
        identity: counted ? same / counted : 0,
        seed,
      });
    }
    state.designs.sort((a, b) => a.score - b.score);
    renderResults();
    setStatus(
      "design-status",
      `${result.S.length} sequences in ${(result.ms / 1000).toFixed(2)} s `
      + `(${(result.ms / result.S.length).toFixed(0)} ms each), seed ${seed}`,
    );
  } catch (error) {
    setStatus("design-status", `Failed: ${error.message}`, "error");
  } finally {
    button.disabled = false;
  }
}

async function runProfile() {
  if (!await ensureEncoded()) return;
  const button = $("profile-btn");
  button.disabled = true;
  const mode = $("profile-mode").value;
  setStatus("design-status", "Computing profile…", "busy");
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
    $("profile-panel").hidden = false;
    $("profile-hint").textContent = MODE_TEXT[mode] ?? "";
    renderLogo();
    redraw();
    setStatus("design-status", `Profile in ${(result.ms / 1000).toFixed(2)} s`);
  } catch (error) {
    setStatus("design-status", `Failed: ${error.message}`, "error");
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
    setStatus("score-status", error.message, "error");
    button.disabled = false;
    return;
  }
  const mode = $("score-mode").value;
  setStatus("score-status", "Scoring…", "busy");
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
    const native = state.structure.S;
    let same = 0;
    for (let i = 0; i < S.length; i++) if (S[i] === native[i]) same++;
    const spread = result.sd === null
      ? ""
      : ` ± ${result.sd.toFixed(4)} over ${result.orders} orders`;
    setStatus(
      "score-status",
      `nll ${result.mean.toFixed(4)}${spread}, `
      + `${((same / S.length) * 100).toFixed(0)}% identical to the input structure's sequence `
      + `(${(result.ms / 1000).toFixed(2)} s)`,
    );
    $("color-mode").value = "score";
    redraw();
  } catch (error) {
    setStatus("score-status", `Failed: ${error.message}`, "error");
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
  return state.designs.map((d, i) =>
    `>${name}_${i + 1} score=${d.score.toFixed(4)} identity=${d.identity.toFixed(3)}\n${d.seq}`,
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

function refreshHomoOligomer() {
  const on = $("homo-oligomer").checked;
  $("symmetry").disabled = on;
  $("homo-summary").textContent = !on ? ""
    : state.structure ? homoOligomerGroups().note
      : "Load a structure first.";
}
$("homo-oligomer").onchange = refreshHomoOligomer;

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
  state.designMask.fill(1);
  refreshSelection();
};
$("select-none").onclick = () => {
  state.designMask.fill(0);
  refreshSelection();
};
$("select-invert").onclick = () => {
  for (let i = 0; i < state.designMask.length; i++) {
    state.designMask[i] = state.designMask[i] ? 0 : 1;
  }
  refreshSelection();
};
$("select-interface").onclick = () => {
  const hits = nearLigand();
  if (!hits.size) {
    setStatus("load-status", "No heteroatoms in this structure.", "error");
    return;
  }
  state.designMask.fill(0);
  for (const i of hits) state.designMask[i] = 1;
  refreshSelection();
};

function paintMembrane(label) {
  if (!state.structure) return;
  for (let i = 0; i < state.structure.L; i++) {
    if (state.designMask[i] > 0) state.membraneLabels[i] = label;
  }
  state.membraneVersion += 1;
  state.encodedFor = null;
  $("color-mode").value = "membrane";
  refreshSelection();
  ensureEncoded();
}

$("mem-soluble").onclick = () => paintMembrane(0);
$("mem-interface").onclick = () => paintMembrane(1);
$("mem-buried").onclick = () => paintMembrane(2);
$("mem-reset").onclick = () => {
  state.membraneLabels.fill(0);
  state.membraneVersion += 1;
  state.encodedFor = null;
  refreshSelection();
  ensureEncoded();
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
$("bias-scope").onchange = renderBiasGrid;
$("bias-reset").onclick = () => {
  state.bias.fill(0);
  state.omitted.clear();
  state.biasOverrides.clear();
  state.omitOverrides.clear();
  renderBiasGrid();
};
$("bias-clear-overrides").onclick = () => {
  state.biasOverrides.clear();
  state.omitOverrides.clear();
  renderBiasGrid();
};
// Resolved against the *current* alphabet: NA-MPNN orders amino acids
// ARNDC..., so ALPHABET's index for "C" is arginine's index over there.
const omitShortcut = (letter) => () => {
  toggleOmit(alphabet().indexOf(letter));
  renderBiasGrid();
};
$("omit-cys").onclick = omitShortcut("C");
$("omit-met").onclick = omitShortcut("M");

$("clear-results").onclick = () => {
  state.designs = [];
  state.activeDesign = -1;
  renderResults();
  renderSequenceTrack();
  redraw();
};
$("copy-fasta").onclick = () => navigator.clipboard.writeText(fastaText());
$("download-fasta").onclick = () => {
  const blob = new Blob([fastaText()], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${$("pdb-id").value.trim() || "designs"}.fasta`;
  a.click();
  URL.revokeObjectURL(a.href);
};

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
  for (const i of hits) state.designMask[i] = boxing.subtract ? 0 : 1;
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
    state.designMask[i] = state.designMask[i] ? 0 : 1;
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
      + `${state.designMask[i] ? "designed" : "fixed"}`
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

window.addEventListener("resize", redraw);
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
  state.designMask[i] = state.designMask[i] ? 0 : 1;
  refreshSelection();
});

$("logo-wider").onclick = () => {
  logo.columnWidth = Math.min(logo.columnWidth + 4, 48);
  renderLogo();
};
$("logo-narrower").onclick = () => {
  logo.columnWidth = Math.max(logo.columnWidth - 4, 5);
  renderLogo();
};

// --- boot -----------------------------------------------------------------

renderBiasGrid();
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
