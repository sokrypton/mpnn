// Page controller: owns the DOM, the selection state, and the worker.

import { ALPHABET } from "../mpnn/constants.js";
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
  bias: new Float32Array(21),
  omitted: new Set(),
  designs: [],
  activeDesign: -1,
  profile: null,
  hover: -1,
  encodedFor: null,
  /** Bumped on every structure load so a stale encode cannot be reused. */
  structureId: 0,
  /** 0 soluble, 1 interface, 2 buried. Only the membrane models read it. */
  membraneLabels: null,
  membraneVersion: 0,
};

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
};

const MODEL_NOTES = {
  protein_mpnn: "The original. Backbone only.",
  soluble_mpnn: "Trained without membrane proteins; avoids hydrophobic surfaces.",
  ligand_mpnn: "Sees heteroatoms — ligands, cofactors, metals, nucleic acids.",
  per_residue_label_membrane_mpnn: "Takes a per-residue buried/interface/soluble label.",
  global_label_membrane_mpnn: "Takes one label for the whole chain.",
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
      option.textContent = `${model.name}  —  ${noise} Å training noise, ${(model.bytes / 1e6).toFixed(1)} MB`;
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
  try {
    structure = structureFromText(text);
  } catch (error) {
    setStatus("load-status", `Could not parse: ${error.message}`, "error");
    return;
  }
  if (structure.L === 0) {
    setStatus("load-status", "No protein residues with a C-alpha found.", "error");
    return;
  }

  state.structure = structure;
  state.structureId += 1;
  state.designMask = new Float32Array(structure.L).fill(1);
  state.membraneLabels = new Int32Array(structure.L);
  state.membraneVersion += 1;
  state.designs = [];
  state.activeDesign = -1;
  state.profile = null;
  state.encodedFor = null;
  $("profile-panel").hidden = true;

  viewer.setStructure(structure);
  $("color-mode").value = structure.chainList.length > 1 ? "chain" : "rainbow";
  renderChainToggles();
  renderSequenceTrack();
  renderResults();
  redraw();

  const ligand = structure.ligandType.length;
  setStatus(
    "load-status",
    `${label}: ${structure.L} residues, ${structure.chainList.length} chain(s)`
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
  const key = `${name}|${useAtomContext}|${state.structureId}|${state.membraneVersion}`;
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
        useAtomContext,
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
  renderSequenceTrack();
  if (state.profile) renderLogo();
  redraw();
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
    span.textContent = ALPHABET[seq ? seq[i] : s.S[i]] ?? "X";
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
      html += `<span class="${same ? "same" : "diff"}">${ALPHABET[design.S[i]]}</span>`;
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
  for (let v = 0; v < 20; v++) order.push([ALPHABET[v], probs[i * 21 + v]]);
  order.sort((a, b) => b[1] - a[1]);
  return order.slice(0, n).map(([aa, p]) => `${aa} ${(p * 100).toFixed(0)}%`).join(", ");
}

// ---------------------------------------------------------------------------
// Amino-acid bias grid
// ---------------------------------------------------------------------------

function renderBiasGrid() {
  const wrap = $("aa-bias");
  wrap.innerHTML = "";
  for (let v = 0; v < 20; v++) {
    const aa = ALPHABET[v];
    const cell = document.createElement("div");
    cell.className = "cell" + (state.omitted.has(v) ? " omitted" : "");

    const letter = document.createElement("span");
    letter.className = "letter";
    letter.textContent = aa;
    letter.style.color = AA_COLORS[aa];
    letter.title = "click to omit";
    letter.onclick = () => {
      if (state.omitted.has(v)) state.omitted.delete(v);
      else state.omitted.add(v);
      renderBiasGrid();
    };

    const input = document.createElement("input");
    input.type = "number";
    input.step = "0.5";
    input.value = String(state.bias[v]);
    input.oninput = () => {
      state.bias[v] = parseFloat(input.value) || 0;
    };

    cell.appendChild(letter);
    cell.appendChild(input);
    wrap.appendChild(cell);
  }
}

/** Expand the per-amino-acid bias into the [L, 21] array the model wants. */
function buildBias() {
  const L = state.structure.L;
  const bias = new Float32Array(L * 21);
  for (let i = 0; i < L; i++) {
    for (let v = 0; v < 20; v++) {
      bias[i * 21 + v] = state.omitted.has(v) ? -1e9 : state.bias[v];
    }
    bias[i * 21 + 20] = -1e9; // never emit X
  }
  return bias;
}

function parseSymmetry() {
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
    for (let b = 0; b < result.seqs.length; b++) {
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
        seq: result.seqs[b],
        score: result.scores[b],
        identity: counted ? same / counted : 0,
        seed,
      });
    }
    state.designs.sort((a, b) => a.score - b.score);
    renderResults();
    setStatus(
      "design-status",
      `${result.seqs.length} sequences in ${(result.ms / 1000).toFixed(2)} s `
      + `(${(result.ms / result.seqs.length).toFixed(0)} ms each), seed ${seed}`,
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
    const entropy = new Float32Array(L);
    for (let i = 0; i < L; i++) {
      let h = 0;
      let z = 0;
      for (let v = 0; v < 20; v++) z += probs[i * 21 + v];
      for (let v = 0; v < 20; v++) {
        const p = probs[i * 21 + v] / (z || 1);
        if (p > 0) h -= p * Math.log(p);
      }
      entropy[i] = h;
    }
    state.profile = { probs, entropy };
    $("profile-panel").hidden = false;
    $("profile-hint").textContent = describeProfileMode(mode);
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

function describeProfileMode(mode) {
  if (mode === "none") {
    return "No position sees any amino acid — this is what the backbone alone implies. "
      + "One decoder pass, exact.";
  }
  if (mode === "all-but-self") {
    return "Every position sees all the others in one pass. Fast, but because the decoder is "
      + "three layers deep a residue's own identity leaks back through two-hop paths, so treat "
      + "this as an approximation of the conditional profile.";
  }
  return "Each position is decoded last in its own pass, so it genuinely sees every other "
    + "residue and nothing of itself. Exact, and costs one decoder pass per position.";
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

$("model-select").onchange = () => {
  updateModelHint();
  state.encodedFor = null;
  if (state.structure) ensureEncoded();
};

$("use-atom-context").onchange = () => {
  state.encodedFor = null;
  if (state.structure) ensureEncoded();
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

$("design-btn").onclick = runDesign;
$("profile-btn").onclick = runProfile;
$("bias-reset").onclick = () => {
  state.bias.fill(0);
  state.omitted.clear();
  renderBiasGrid();
};
$("omit-cys").onclick = () => {
  state.omitted.add(ALPHABET.indexOf("C"));
  renderBiasGrid();
};
$("omit-met").onclick = () => {
  state.omitted.add(ALPHABET.indexOf("M"));
  renderBiasGrid();
};

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
