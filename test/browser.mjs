// End-to-end smoke test: serve the repo, drive the real page in Chromium,
// load a structure from disk, design sequences, and compute a profile.
//
//   node test/browser.mjs [--headed] [--shot out.png]

import { createServer } from "node:http";
import { existsSync } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const TYPES = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".css": "text/css", ".json": "application/json", ".mpnn": "application/octet-stream",
  ".pdb": "text/plain", ".cif": "text/plain", ".png": "image/png",
  ".wasm": "application/wasm",
};

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://localhost");
    const path = join(ROOT, normalize(decodeURIComponent(url.pathname)).replace(/^(\.\.[/\\])+/, ""));
    const target = url.pathname.endsWith("/") ? join(path, "index.html") : path;
    const body = await readFile(target);
    res.writeHead(200, {
      "content-type": TYPES[extname(target)] ?? "application/octet-stream",
      "content-length": body.length,
    });
    res.end(body);
  } catch {
    res.writeHead(404).end("not found");
  }
});

await new Promise((resolve) => server.listen(0, resolve));
const base = `http://127.0.0.1:${server.address().port}`;
console.log(`serving ${ROOT} at ${base}`);

const headed = process.argv.includes("--headed");
const shotIndex = process.argv.indexOf("--shot");
const shotPath = shotIndex >= 0 ? process.argv[shotIndex + 1] : null;
const structurePath = process.argv.includes("--pdb")
  ? process.argv[process.argv.indexOf("--pdb") + 1]
  : null;

// The container ships a Chromium that may not match this Playwright build's
// expected revision, so point at it explicitly rather than downloading one.
const PREINSTALLED = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";
const browser = await chromium.launch({
  headless: !headed,
  executablePath: existsSync(PREINSTALLED) ? PREINSTALLED : undefined,
});
const page = await browser.newPage({ viewport: { width: 1440, height: 1600 } });

/** Wait until the model panel reports it encoded exactly `residues` residues. */
async function waitReady(residues, timeout) {
  await page.waitForFunction(
    (n) => new RegExp(`encoded ${n} residues`).test(
      document.getElementById("model-status").textContent),
    residues,
    { timeout },
  );
}

/**
 * Wait for a *fresh* encode. `waitReady` alone returns immediately when the
 * status line still reads the previous run's success, which is exactly the
 * case whenever something re-encodes the same structure.
 */
async function waitReencode(action, residues, timeout) {
  const before = await page.textContent("#model-status");
  await action();
  await page.waitForFunction(
    ({ prev, n }) => {
      const text = document.getElementById("model-status").textContent;
      return text !== prev && new RegExp(`encoded ${n} residues`).test(text);
    },
    { prev: before, n: residues },
    { timeout },
  );
}

const problems = [];
page.on("console", (msg) => {
  if (msg.type() === "error") problems.push(`console: ${msg.text()}`);
});
page.on("pageerror", (error) => problems.push(`pageerror: ${error.message}`));

await page.goto(base, { waitUntil: "networkidle" });
console.log("page loaded");

// Feed a local structure through the file input so the test does not need the network.
const pdbPath = structurePath ?? join(ROOT, "assets", "1ubq.pdb");
await page.setInputFiles("#file-input", pdbPath);

await waitReady(76, 180000);
console.log("load :", (await page.textContent("#load-status")).trim());
console.log("model:", (await page.textContent("#model-status")).trim());
const kernel = await page.textContent("#kernel-status");
console.log("kernel:", kernel.trim());
if (!/SIMD/.test(kernel)) problems.push("the SIMD kernel did not load in the browser");

const canvasInk = await page.evaluate(() => {
  const c = document.getElementById("viewer");
  const ctx = c.getContext("2d");
  const { data } = ctx.getImageData(0, 0, c.width, c.height);
  let painted = 0;
  let x0 = Infinity, y0 = Infinity, x1 = -1, y1 = -1;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] === 0) continue;
    painted++;
    const p = i / 4;
    const x = p % c.width;
    const y = (p / c.width) | 0;
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
  }
  return {
    painted, total: data.length / 4, w: c.width, h: c.height,
    side: Math.min(c.width, c.height),
    fillW: (x1 - x0) / Math.min(c.width, c.height),
    fillH: (y1 - y0) / Math.min(c.width, c.height),
  };
});
console.log(
  `viewer: ${canvasInk.painted} of ${canvasInk.total} px painted, `
  + `bounding box fills ${(canvasInk.fillW * 100).toFixed(0)}% x `
  + `${(canvasInk.fillH * 100).toFixed(0)}% of the ${canvasInk.side}px draw box`,
);
if (canvasInk.painted < 1000) problems.push("viewer drew almost nothing");
if (Math.max(canvasInk.fillW, canvasInk.fillH) < 0.7) {
  problems.push(
    `structure only fills ${(canvasInk.fillH * 100).toFixed(0)}% of the draw box`);
}

const stillHidden = await page.evaluate(() =>
  [...document.querySelectorAll("[hidden]")]
    .filter((el) => getComputedStyle(el).display !== "none")
    .map((el) => el.id || el.className));
if (stillHidden.length) problems.push(`[hidden] not respected on: ${stillHidden.join(", ")}`);

// Design a couple of sequences.
await page.fill("#batch", "2");
await page.evaluate(() => document.getElementById("batch").dispatchEvent(new Event("input")));
await page.click("#design-btn");
await page.waitForFunction(
  () => /sequences in/.test(document.getElementById("design-status").textContent),
  { timeout: 300000 },
);
console.log("design:", (await page.textContent("#design-status")).trim());

const designs = await page.evaluate(() =>
  [...document.querySelectorAll(".design")].map((d) => ({
    meta: d.querySelector(".meta").textContent.trim(),
    seq: d.querySelector(".seq").textContent.trim(),
  })));
for (const d of designs) console.log(`   ${d.meta}  ${d.seq.slice(0, 60)}`);
if (designs.length !== 2) problems.push(`expected 2 designs, got ${designs.length}`);
if (designs.some((d) => !/^[ACDEFGHIKLMNPQRSTVWY]+$/.test(d.seq))) {
  problems.push("a design contains non-standard letters");
}

// Profile + logo.
await page.click("#profile-btn");
await page.waitForFunction(
  () => /Profile in/.test(document.getElementById("design-status").textContent),
  { timeout: 300000 },
);
console.log("profile:", (await page.textContent("#design-status")).trim());
// The logo is a canvas, so check what it painted rather than counting nodes:
// how much ink, and how many distinct columns actually got glyphs.
const logoInk = await page.evaluate(() => {
  const c = document.getElementById("logo");
  const ctx = c.getContext("2d");
  const { data } = ctx.getImageData(0, 0, c.width, c.height);
  let painted = 0;
  const cols = new Set();
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] === 0) continue;
    painted++;
    cols.add(Math.floor(((i / 4) % c.width) / 8));
  }
  return { painted, cols: cols.size, w: c.width, h: c.height };
});
console.log(`logo: ${logoInk.w}x${logoInk.h}, ${logoInk.painted} px in ${logoInk.cols} bands`);
if (logoInk.painted < 5000) problems.push("logo painted almost nothing");
if (logoInk.cols < 20) problems.push("logo has too few distinct columns");

// Hovering a logo column must drive the 3D highlight, and clicking must toggle
// the same residue the sequence track shows.
const designedBefore = await page.evaluate(() =>
  document.querySelectorAll(".res.designed").length);
await page.evaluate(() => {
  const c = document.getElementById("logo");
  const r = c.getBoundingClientRect();
  c.dispatchEvent(new PointerEvent("click", {
    clientX: r.left + 38 + 15 * 3 + 7, clientY: r.top + 20, bubbles: true,
  }));
});
const designedAfter = await page.evaluate(() =>
  document.querySelectorAll(".res.designed").length);
console.log(`logo click: ${designedBefore} -> ${designedAfter} designed`);
if (designedAfter !== designedBefore - 1) problems.push("clicking a logo column did nothing");

// Selection round-trip: clicking a residue in the track toggles it.
const before = await page.evaluate(() => document.querySelectorAll(".res.designed").length);
await page.evaluate(() => document.querySelector('.res[data-i="5"]').click());
const after = await page.evaluate(() => document.querySelectorAll(".res.designed").length);
console.log(`selection: ${before} -> ${after} designed`);
if (after !== before - 1) problems.push("clicking a residue did not toggle it");

// Colour modes must all render without throwing.
for (const mode of ["chain", "design", "confidence", "identity", "rainbow"]) {
  await page.selectOption("#color-mode", mode);
  await page.waitForTimeout(80);
}
console.log("colour modes: ok");

// --- per-position bias and sequence scoring ---------------------------------
{
  // The bias controls live inside a <details>, which starts collapsed.
  await page.evaluate(() => {
    for (const d of document.querySelectorAll("details")) d.open = true;
  });

  // Scope the bias to a selection and confirm only those positions carry it.
  await page.click("#select-none");
  await page.evaluate(() => {
    for (let i = 10; i < 20; i++) document.querySelector(`.res[data-i="${i}"]`).click();
  });
  await page.selectOption("#bias-scope", "selected");
  await page.evaluate(() => {
    const cell = [...document.querySelectorAll("#aa-bias .cell")]
      .find((c) => c.querySelector(".letter").textContent === "W");
    const input = cell.querySelector("input");
    input.value = "5";
    input.dispatchEvent(new Event("input"));
  });
  const built = await page.evaluate(() => {
    // Reach into the module's own bias builder through a design run's payload
    // would be indirect; instead read the summary it renders.
    return document.getElementById("bias-summary").textContent;
  });
  console.log("bias:", built.trim());
  if (!/10 position\(s\) carry an override/.test(built)) {
    problems.push("per-position bias override was not recorded");
  }

  // With W boosted only at 10..19, a low-temperature design should put W there
  // and (mostly) not elsewhere. Results accumulate and re-sort by score across
  // runs, so clear them first -- otherwise `.design` is whichever earlier,
  // unbiased sequence happens to score best.
  await page.click("#clear-results");
  await page.click("#select-all");
  await page.fill("#temperature", "0.1");
  await page.evaluate(() => document.getElementById("temperature")
    .dispatchEvent(new Event("input")));
  await page.fill("#batch", "1");
  await page.evaluate(() => document.getElementById("batch").dispatchEvent(new Event("input")));
  await page.click("#design-btn");
  await page.waitForFunction(
    () => /sequences in/.test(document.getElementById("design-status").textContent),
    { timeout: 300000 },
  );
  const seq = await page.evaluate(() =>
    document.querySelector(".design .seq").textContent.trim());
  const inWindow = [...seq.slice(10, 20)].filter((c) => c === "W").length;
  const outside = [...seq.slice(0, 10) + seq.slice(20)].filter((c) => c === "W").length;
  console.log(`bias: W inside biased window ${inWindow}/10, outside ${outside}/${seq.length - 10}`);
  // 8-9 of 10, not 10 -- bias is added to the logits, not a constraint, and a
  // couple of buried core positions in ubiquitin have a raw logit gap wider
  // than 5 nats. `omit` (-1e9) is the hard version; sampling.mjs checks it.
  if (inWindow < 5) problems.push("per-position bias did not steer the design");
  await page.click("#bias-clear-overrides");
  await page.selectOption("#bias-scope", "global");

  // Scoring. The default is the single-pass pseudo-likelihood, which reports no
  // spread because there is nothing to average over.
  await page.click("#score-native");
  const scoreOnce = async () => {
    await page.click("#score-btn");
    await page.waitForFunction(
      (prev) => {
        const t = document.getElementById("score-status").textContent;
        return t !== prev && /nll |Failed/.test(t);
      },
      "Scoring…", { timeout: 300000 },
    );
    return (await page.textContent("#score-status")).trim();
  };

  if (await page.isVisible("#score-orders-row")) {
    problems.push("the orders control is shown for a mode that does not average");
  }
  const plText = await scoreOnce();
  console.log("score (pseudo-likelihood):", plText);
  if (!/^nll [0-9.]+, /.test(plText)) {
    problems.push(`pseudo-likelihood scoring did not report a value: ${plText}`);
  }
  if (/±/.test(plText)) problems.push("single-pass score reported a spread");
  if (!/100% identical/.test(plText)) problems.push("native sequence not recognised as identical");

  await page.selectOption("#score-mode", "order");
  if (!await page.isVisible("#score-orders-row")) {
    problems.push("the orders control stayed hidden for autoregressive scoring");
  }
  await page.fill("#score-orders", "4");
  const arText = await scoreOnce();
  console.log("score (autoregressive):", arText);
  if (!/nll [0-9.]+ ± [0-9.]+ over 4 orders/.test(arText)) {
    problems.push(`autoregressive scoring did not report a value: ${arText}`);
  }
  await page.selectOption("#score-mode", "all-but-self");
}

// --- membrane labels must reach the encoder ---------------------------------
{
  console.log("\n-- MembraneMPNN --");
  await page.selectOption("#model-select", "per_residue_label_membrane_mpnn_v_48_020");
  await waitReady(76, 300000);
  const controlShown = await page.evaluate(() =>
    !document.getElementById("membrane-perres-row").hidden);
  if (!controlShown) problems.push("per-residue membrane control stayed hidden");

  // Label half the chain buried, and check it forces a re-encode rather than
  // silently reusing the all-soluble one.
  await page.click("#select-none");
  await page.evaluate(() => {
    for (let i = 0; i < 38; i++) document.querySelector(`.res[data-i="${i}"]`).click();
  });
  const before = await page.textContent("#model-status");
  await page.click("#mem-buried");
  await page.waitForFunction(
    (prev) => document.getElementById("model-status").textContent !== prev
      && /encoded 76 residues/.test(document.getElementById("model-status").textContent),
    before, { timeout: 300000 },
  );
  const labels = await page.evaluate(() => {
    const c = document.getElementById("viewer");
    return c.getContext("2d").getImageData(0, 0, c.width, c.height).data.length;
  });
  console.log("membrane: labelled 38 residues buried, re-encoded", labels > 0 ? "ok" : "");
  await page.click("#mem-reset");
  await waitReady(76, 300000);
  await page.click("#select-all");
}

// --- second phase: LigandMPNN on a structure with a real ligand ------------
if (!process.argv.includes("--no-ligand")) {
  console.log("\n-- LigandMPNN --");
  await page.selectOption("#model-select", "ligandmpnn_v_32_010_25");
  await page.setInputFiles("#file-input", join(ROOT, "assets", "1stp.pdb"));
  // Wait for the encoding of *this* structure, not a stale one still in flight.
  await waitReady(121, 600000);
  console.log("load :", (await page.textContent("#load-status")).trim());
  console.log("model:", (await page.textContent("#model-status")).trim());

  const ligandVisible = await page.evaluate(() =>
    !document.getElementById("atom-context-row").hidden);
  if (!ligandVisible) problems.push("ligand atom-context control stayed hidden");

  // "Near ligand" must select the binding site and nothing else.
  await page.click("#select-interface");
  const selected = await page.evaluate(() => document.querySelectorAll(".res.designed").length);
  const total = await page.evaluate(() => document.querySelectorAll(".res").length);
  console.log(`near-ligand selection: ${selected} of ${total} residues`);
  if (selected === 0 || selected >= total) problems.push("near-ligand selection looks wrong");

  // A fixed seed and a warmer temperature, so "did the sampler actually move"
  // is a deterministic question. At T = 0.1 LigandMPNN reproduces a biotin site
  // exactly, which is a real result rather than a failure -- but it makes the
  // assertion below vacuous.
  await page.fill("#batch", "1");
  await page.evaluate(() => document.getElementById("batch").dispatchEvent(new Event("input")));
  await page.fill("#temperature", "0.6");
  await page.evaluate(() => document.getElementById("temperature")
    .dispatchEvent(new Event("input")));
  await page.uncheck("#random-seed");
  await page.fill("#seed", "12345");
  await page.click("#design-btn");
  await page.waitForFunction(
    () => /sequences in/.test(document.getElementById("design-status").textContent),
    { timeout: 600000 },
  );
  console.log("design:", (await page.textContent("#design-status")).trim());
  const ligandDesign = await page.evaluate(() => ({
    meta: document.querySelector(".design .meta").textContent.trim(),
    seq: document.querySelector(".design .seq").textContent.trim(),
  }));
  console.log(`   ${ligandDesign.meta}  ${ligandDesign.seq.slice(0, 60)}`);

  // Paint the design onto the track, then confirm it only touched the
  // positions that were marked designable.
  await page.evaluate(() => document.querySelector(".design").click());
  const changed = await page.evaluate(() =>
    document.querySelectorAll(".res.changed").length);
  console.log(`residues changed: ${changed}`);

  // Side-chain context. Only the fixed residues contribute, so it must both
  // re-encode when switched on and re-encode again when the selection moves --
  // the one input that is otherwise read at sampling time only.
  if (await page.evaluate(() => document.getElementById("side-chain-row").hidden)) {
    problems.push("side-chain context control stayed hidden for LigandMPNN");
  }
  const scoreWith = async () => {
    await page.click("#score-native");
    await page.click("#score-btn");
    await page.waitForFunction(
      (prev) => {
        const t = document.getElementById("score-status").textContent;
        return t !== prev && /nll |Failed/.test(t);
      },
      "Scoring…", { timeout: 600000 },
    );
    const text = (await page.textContent("#score-status")).trim();
    return parseFloat(text.match(/nll ([0-9.]+)/)?.[1] ?? "NaN");
  };
  // The score averages over the designed positions, which here are the ~14
  // around biotin -- the setting the flag exists for.
  const nllPlain = await scoreWith();
  await waitReencode(() => page.check("#use-side-chains"), 121, 600000);
  const nllSc = await scoreWith();
  console.log(`side chains: binding-site nll ${nllPlain.toFixed(4)} -> ${nllSc.toFixed(4)}`);
  if (!(nllSc < nllPlain)) {
    problems.push(`side-chain context did not sharpen the score (${nllPlain} -> ${nllSc})`);
  }

  // Designing everything leaves nothing fixed, so there are no side chains to
  // add and the answer has to fall back to the plain one exactly. This is also
  // what catches the selection failing to invalidate the encoding: without the
  // re-encode the previous, sharper number would still be showing.
  await waitReencode(() => page.click("#select-all"), 121, 600000);
  const nllAllSc = await scoreWith();
  await waitReencode(() => page.uncheck("#use-side-chains"), 121, 600000);
  const nllAllPlain = await scoreWith();
  console.log(`side chains: nothing fixed ${nllAllSc.toFixed(4)} vs plain ${nllAllPlain.toFixed(4)}`);
  if (Math.abs(nllAllSc - nllAllPlain) > 1e-4) {
    problems.push(`with nothing fixed, side-chain context changed the answer `
      + `(${nllAllSc} vs ${nllAllPlain})`);
  }
  if (changed === 0) problems.push("design changed nothing at all");
  if (changed > selected) problems.push("design altered residues that were marked fixed");

  if (shotPath) {
    await page.screenshot({ path: shotPath.replace(/\.png$/, "-ligand.png"), fullPage: true });
  }
}

// --- NA-MPNN ----------------------------------------------------------------
{
  console.log("\n-- NA-MPNN --");
  await page.goto(base, { waitUntil: "networkidle" });
  await page.selectOption("#model-select", "na_mpnn_design");
  // 4oqu is 97 nucleotides of RNA. Selecting NA-MPNN re-parses the structure,
  // because nucleic acids are model positions for this model and ligand atoms
  // for every other one.
  await page.setInputFiles("#file-input", join(ROOT, "assets", "4oqu.pdb"));
  await waitReady(97, 600000);
  const naLoad = (await page.textContent("#load-status")).trim();
  console.log("load :", naLoad);
  const naModel = (await page.textContent("#model-status")).trim();
  console.log("model:", naModel);
  if (!/na_mpnn_design/.test(naModel)) {
    problems.push(`NA-MPNN was not the model that encoded: ${naModel}`);
  }
  if (!/97 RNA/.test(naLoad)) problems.push(`nucleic acids not read as residues: ${naLoad}`);

  // The input sequence must come back in RNA letters, not protein ones.
  const track = await page.evaluate(() =>
    [...document.querySelectorAll(".res")].map((e) => e.textContent).join(""));
  // RNA, so every letter must be an RNA base -- b/d/h/u. Seeing DNA letters
  // here would mean a uracil was being reported as a thymine.
  if (!/^[bdhuy]+$/.test(track)) {
    problems.push(`RNA sequence track is not RNA letters: ${track.slice(0, 40)}`);
  }
  console.log(`native: ${track.slice(0, 60)}`);

  await page.fill("#batch", "2");
  await page.evaluate(() => document.getElementById("batch").dispatchEvent(new Event("input")));
  await page.fill("#temperature", "0.3");
  await page.evaluate(() => document.getElementById("temperature")
    .dispatchEvent(new Event("input")));
  await page.click("#design-btn");
  await page.waitForFunction(
    () => /sequences in|Failed/.test(document.getElementById("design-status").textContent),
    { timeout: 900000 },
  );
  console.log("design:", (await page.textContent("#design-status")).trim());
  const naSeqs = await page.evaluate(() =>
    [...document.querySelectorAll(".design .seq")].map((e) => e.textContent.trim()));
  for (const seq of naSeqs) console.log(`   ${seq.slice(0, 60)}`);
  // The bug this catches: the sampler used to consider only the first 20
  // letters, so a 33-letter model could never emit a nucleotide at all.
  if (!naSeqs.length) problems.push("NA-MPNN produced no designs");
  for (const seq of naSeqs) {
    if (!/^[bdhuy]+$/.test(seq)) {
      problems.push(`NA-MPNN designed an RNA chain with non-RNA letters: ${seq.slice(0, 40)}`);
    }
  }

  // A profile exercises the 33-wide logo path. The mode selector lives inside
  // the profile panel, which stays hidden until the first profile exists.
  await page.click("#profile-btn");
  await page.waitForFunction(
    () => /Profile in|Failed/.test(document.getElementById("design-status").textContent),
    { timeout: 900000 },
  );
  console.log("profile:", (await page.textContent("#design-status")).trim());
  const logoInk = await page.evaluate(() => {
    const c = document.getElementById("logo");
    const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 3; i < d.length; i += 4) if (d[i] > 8) n++;
    return n;
  });
  console.log(`logo: ${logoInk} px painted`);
  if (logoInk === 0) problems.push("NA-MPNN profile drew an empty logo");

  // Switching back to a protein model must re-parse: 4oqu has no protein, so
  // the load has to fail cleanly rather than encode a 0-residue structure.
  await page.selectOption("#model-select", "proteinmpnn_v_48_020");
  await page.waitForFunction(
    () => /No protein residues/.test(document.getElementById("load-status").textContent),
    { timeout: 60000 },
  );
  console.log("switch back:", (await page.textContent("#load-status")).trim());
}

// --- homo-oligomer tying ----------------------------------------------------
{
  console.log("\n-- homo-oligomer --");
  // A two-chain assembly, built by translating ubiquitin rather than shipping
  // another fixture. Both copies keep their original residue numbering, which
  // is the case the reference's residue-number matching is written for.
  const source = await readFile(join(ROOT, "assets", "1ubq.pdb"), "utf8");
  const copy = source.split("\n")
    .filter((line) => line.startsWith("ATOM  "))
    .map((line) => {
      const x = (parseFloat(line.slice(30, 38)) + 40).toFixed(3).padStart(8);
      return `${line.slice(0, 21)}B${line.slice(22, 30)}${x}${line.slice(38)}`;
    });
  const dimerPath = join(tmpdir(), "mpnn-dimer.pdb");
  await writeFile(dimerPath, `${source}\n${copy.join("\n")}\nEND\n`);

  await page.goto(base, { waitUntil: "networkidle" });
  await page.setInputFiles("#file-input", dimerPath);
  await waitReady(152, 300000);
  await page.evaluate(() => {
    for (const d of document.querySelectorAll("details")) d.open = true;
  });
  await page.check("#homo-oligomer");
  const note = (await page.textContent("#homo-summary")).trim();
  console.log("homo-oligomer:", note);
  if (!/76 group\(s\) of 2, matched by residue number/.test(note)) {
    problems.push(`homo-oligomer grouping looks wrong: ${note}`);
  }
  if (!await page.evaluate(() => document.getElementById("symmetry").disabled)) {
    problems.push("the manual symmetry field stayed editable while chains were tied");
  }

  // A warm temperature, so agreement between the chains is the tying and not
  // just two confident argmaxes landing in the same place.
  await page.fill("#batch", "2");
  await page.evaluate(() => document.getElementById("batch").dispatchEvent(new Event("input")));
  await page.fill("#temperature", "0.8");
  await page.evaluate(() => document.getElementById("temperature")
    .dispatchEvent(new Event("input")));
  await page.click("#design-btn");
  await page.waitForFunction(
    () => /sequences in/.test(document.getElementById("design-status").textContent),
    { timeout: 600000 },
  );
  const tied = await page.evaluate(() =>
    [...document.querySelectorAll(".design .seq")].map((e) => e.textContent.trim()));
  for (const seq of tied) {
    const [a, b] = [seq.slice(0, 76), seq.slice(76)];
    let same = 0;
    for (let i = 0; i < 76; i++) if (a[i] === b[i]) same++;
    console.log(`   chains agree at ${same}/76  ${a.slice(0, 40)}`);
    if (same !== 76) problems.push(`tied chains disagree at ${76 - same} position(s)`);
  }

  // Untied, the same warm temperature should let them drift apart -- otherwise
  // the check above proves nothing.
  await page.uncheck("#homo-oligomer");
  await page.click("#clear-results");
  await page.click("#design-btn");
  await page.waitForFunction(
    () => /sequences in/.test(document.getElementById("design-status").textContent),
    { timeout: 600000 },
  );
  const free = await page.evaluate(() =>
    document.querySelector(".design .seq").textContent.trim());
  let drift = 0;
  for (let i = 0; i < 76; i++) if (free[i] !== free[76 + i]) drift++;
  console.log(`   untied, chains differ at ${drift}/76`);
  if (drift === 0) problems.push("untied chains came out identical, so the tying check is vacuous");
}

if (shotPath) {
  await page.goto(base, { waitUntil: "networkidle" });
  await page.setInputFiles("#file-input", pdbPath);
  await waitReady(76, 180000);
  await page.screenshot({ path: shotPath, fullPage: true });
  console.log(`screenshot -> ${shotPath}`);
}

await browser.close();
server.close();

if (problems.length) {
  console.log(`\n${problems.length} problem(s):`);
  for (const p of problems) console.log(`  - ${p}`);
  process.exit(1);
}
console.log("\nbrowser smoke test passed");
await writeFile(join(ROOT, ".last-browser-test"), new Date().toISOString());
