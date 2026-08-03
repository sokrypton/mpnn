// The constraints pane: a letter editor whose target is stated, over a table of
// the positions it will affect.
//
// What this replaces was a grid of twenty number boxes and a dropdown reading
// "applies to: every position / selected positions only". The dropdown was the
// problem: the same twenty boxes meant two different things depending on it, and
// once you had set a per-position override there was nothing anywhere on the
// page that showed you where it was. So the table exists to make the answer to
// "what is set, and where" visible, and the editor states its own target
// instead of asking you to choose one.
//
// Rows are windowed. `All` on 6VXX is 2916 rows and the pane is 200px tall;
// rendering the lot would be ~18k elements rebuilt on every selection change.
// A window of what fits plus a pad row above and below makes the cost of a
// refresh independent of L.

const ROW_H = 22;
const OVERSCAN = 6;

/** What the table is showing. `selected` also includes anything constrained. */
const FILTERS = [
  ["selected", "Selected"],
  ["constrained", "Constrained"],
  ["all", "All"],
];

export class ConstraintTable {
  /**
   * @param {object} deps
   * @param {() => object} deps.getState
   * @param {() => string} deps.alphabet current model's alphabet
   * @param {() => number[]} deps.letters model indices the sampler may draw
   * @param {(i: number) => string} deps.letterAt residue letter for display
   * @param {object} deps.c the constraints module's namespace
   * @param {() => Map<number, object[]>} deps.tiesByPosition
   * @param {() => void} deps.onChange after any edit
   */
  constructor(deps) {
    this.deps = deps;
    this.filter = "selected";
    this.rows = new Int32Array(0);
    this.built = false;
  }

  /** Positions to show, given the filter. */
  _visible() {
    const { getState, c } = this.deps;
    const state = getState();
    const s = state.structure;
    if (!s) return new Int32Array(0);
    if (this.filter === "all") {
      return Int32Array.from({ length: s.L }, (_, i) => i);
    }
    const constrained = c.overriddenPositions(state.constraints);
    if (this.filter === "constrained") {
      return Int32Array.from([...constrained].sort((a, b) => a - b));
    }
    // Selected, *plus* anything constrained -- so an empty selection shows the
    // constraints rather than an empty pane, which was the shape of the old
    // dead end where editing with nothing selected silently did nothing.
    const set = new Set([...state.selection, ...constrained]);
    return Int32Array.from([...set].sort((a, b) => a - b));
  }

  /** Build the static chrome once. */
  _build(host) {
    host.innerHTML = "";

    const head = document.createElement("div");
    head.className = "ctable-head";

    this.targetLine = document.createElement("p");
    this.targetLine.className = "ctable-target";

    const filters = document.createElement("div");
    filters.className = "seg";
    this.filterButtons = new Map();
    for (const [key, label] of FILTERS) {
      const button = document.createElement("button");
      button.textContent = label;
      button.onclick = () => { this.filter = key; this.render(); };
      filters.appendChild(button);
      this.filterButtons.set(key, button);
    }

    head.append(this.targetLine, filters);

    this.editor = document.createElement("div");
    this.editor.className = "aa-editor";

    this.scroll = document.createElement("div");
    this.scroll.className = "ctable-scroll";
    this.table = document.createElement("table");
    this.table.className = "ctable";
    this.thead = document.createElement("thead");
    this.tbody = document.createElement("tbody");
    this.table.append(this.thead, this.tbody);
    this.scroll.appendChild(this.table);
    this.scroll.onscroll = () => this._renderRows();

    this.empty = document.createElement("p");
    this.empty.className = "empty";

    host.append(head, this.editor, this.scroll, this.empty);
    this.built = true;
  }

  render() {
    const host = this.host ?? (this.host = document.getElementById("constraints"));
    if (!host) return;
    if (!this.built) this._build(host);

    const state = this.deps.getState();
    for (const [key, button] of this.filterButtons) {
      button.setAttribute("aria-pressed", String(key === this.filter));
    }
    if (!state.structure) {
      this.targetLine.textContent = "";
      this.editor.innerHTML = "";
      this.tbody.innerHTML = "";
      this.empty.textContent = "Load a structure.";
      this.empty.hidden = false;
      return;
    }

    this._renderEditor();
    this.rows = this._visible();
    this._renderHead();
    this._renderRows();

    this.empty.hidden = this.rows.length > 0;
    if (!this.rows.length) {
      this.empty.textContent = this.filter === "constrained"
        ? "Nothing is constrained yet."
        : "Nothing selected. Select residues in the structure or the sequence track.";
    }
  }

  /**
   * The letter editor, and the one sentence saying what it will change.
   *
   * Cells are built once per (target, alphabet) and thereafter only updated, and
   * an element containing the focused input is left alone -- the old grid
   * rebuilt itself on every keystroke and so ate the caret.
   */
  _renderEditor() {
    const { getState, alphabet, letters, c } = this.deps;
    const state = getState();
    const target = state.selection.size ? [...state.selection] : null;
    const ab = alphabet();
    const list = letters();

    this.targetLine.textContent = target
      ? `Editing ${target.length} selected position${target.length === 1 ? "" : "s"}`
      : "Editing the default for every position";

    const signature = `${ab}|${list.join(",")}`;
    if (this._editorSignature !== signature) {
      this.editor.innerHTML = "";
      this._cells = new Map();
      for (const v of list) {
        const cell = document.createElement("div");
        cell.className = "cell";
        const letter = document.createElement("span");
        letter.className = "letter";
        letter.textContent = ab[v];
        letter.title = "click to omit or allow";
        const input = document.createElement("input");
        input.type = "number";
        input.step = "0.5";
        cell.append(letter, input);
        this.editor.appendChild(cell);
        this._cells.set(v, { cell, letter, input });
      }
      this._editorSignature = signature;
    }

    for (const v of list) {
      const parts = this._cells.get(v);
      if (!parts) continue;
      const ci = c.canonOf(ab, v);
      const shown = c.resolveOver(state.constraints, target, ci);
      parts.cell.className = "cell"
        + (shown.omit === true ? " omitted" : "")
        + (shown.omit === "mixed" ? " omit-mixed" : "")
        + (shown.overridden !== "none" ? " override" : "");
      parts.letter.onclick = () => {
        c.setOmit(state.constraints, target, ci, shown.omit !== true);
        this.deps.onChange();
      };
      if (parts.input !== document.activeElement) {
        parts.input.value = shown.mixed ? "" : String(shown.value);
        parts.input.placeholder = shown.mixed ? "mixed" : "";
      }
      parts.input.oninput = () => {
        c.setBias(state.constraints, target, ci, parseFloat(parts.input.value) || 0);
        this.deps.onChange();
      };
    }
  }

  _renderHead() {
    if (this._headBuilt) return;
    const row = document.createElement("tr");
    for (const [label, title] of [
      ["pos", "chain and residue number"],
      ["res", "the residue in the structure"],
      ["design", "may the model change this position"],
      ["bias", "letters biased away from the default here"],
      ["omit", "letters this position may not use"],
      ["tie", "tied groups this position belongs to"],
    ]) {
      const th = document.createElement("th");
      th.textContent = label;
      th.title = title;
      row.appendChild(th);
    }
    this.thead.appendChild(row);
    this._headBuilt = true;
  }

  /** The visible slice, plus a pad row above and below to hold the scrollbar. */
  _renderRows() {
    const { getState, alphabet, letters, letterAt, c, tiesByPosition } = this.deps;
    const state = getState();
    const s = state.structure;
    if (!s) return;

    const height = this.scroll.clientHeight || ROW_H * 8;
    const first = Math.max(0, Math.floor(this.scroll.scrollTop / ROW_H) - OVERSCAN);
    const count = Math.ceil(height / ROW_H) + OVERSCAN * 2;
    const last = Math.min(this.rows.length, first + count);

    // Leave the DOM alone while a cell in it has focus, or typing in the design
    // checkbox column loses the keyboard.
    if (this._range && this._range[0] === first && this._range[1] === last
        && this.tbody.contains(document.activeElement)) return;
    this._range = [first, last];

    const ab = alphabet();
    const list = letters();
    const ties = tiesByPosition();
    const frag = document.createDocumentFragment();

    const pad = (h) => {
      const tr = document.createElement("tr");
      tr.className = "pad";
      const td = document.createElement("td");
      td.colSpan = 6;
      td.style.height = `${h}px`;
      tr.appendChild(td);
      return tr;
    };
    if (first > 0) frag.appendChild(pad(first * ROW_H));

    for (let r = first; r < last; r++) {
      const i = this.rows[r];
      const summary = c.rowSummary(state.constraints, i, ab, list);
      const tr = document.createElement("tr");
      if (state.selection.has(i)) tr.className = "selected";

      const cell = (text, className) => {
        const td = document.createElement("td");
        td.textContent = text;
        if (className) td.className = className;
        return td;
      };

      tr.appendChild(cell(`${s.chainIds[i]}${s.resSeq[i]}${s.iCodes[i]}`.trim(), "pos"));
      tr.appendChild(cell(letterAt(i), "res"));

      const designTd = document.createElement("td");
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = state.designMask[i] > 0;
      box.title = "design this position";
      box.onchange = () => {
        state.designMask[i] = box.checked ? 1 : 0;
        state.encodedFor = null;
        this.deps.onChange();
      };
      designTd.appendChild(box);
      tr.appendChild(designTd);

      const biasText = summary.biasEntries.length
        ? summary.biasEntries.slice(0, 2)
          .map((e) => `${e.letter}${e.value > 0 ? "+" : ""}${e.value}`).join(" ")
          + (summary.biasEntries.length > 2 ? ` +${summary.biasEntries.length - 2}` : "")
        : "—";
      const biasTd = cell(biasText, summary.hasOverride ? "bias override" : "bias");
      biasTd.title = summary.biasEntries.length
        ? summary.biasEntries.map((e) => `${e.letter} ${e.value}`).join(", ")
        : "";
      tr.appendChild(biasTd);

      tr.appendChild(cell(summary.omitLetters.join("") || "—", "omit"));

      const groups = ties.get(i) ?? [];
      tr.appendChild(cell(
        groups.length ? groups.map((g) => g.label).join(" ") + (groups.length > 1 ? " ⚠" : "") : "—",
        "tie",
      ));

      // Clicking a row selects that position: the table is a list of positions,
      // and the selection is what the editor above it acts on, so this is the
      // obvious meaning. It is also the only DOM handle on a position now that
      // the sequence track is a canvas, which is what `test/browser.mjs` uses.
      tr.dataset.pos = String(i);
      tr.onclick = (event) => {
        if (event.target.tagName === "INPUT") return;
        if (state.selection.has(i)) state.selection.delete(i);
        else state.selection.add(i);
        this.deps.onChange();
      };
      frag.appendChild(tr);
    }

    if (last < this.rows.length) frag.appendChild(pad((this.rows.length - last) * ROW_H));

    this.tbody.innerHTML = "";
    this.tbody.appendChild(frag);
  }
}
