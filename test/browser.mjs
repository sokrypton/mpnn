// End-to-end smoke test: serve the repo, drive the real page in Chromium,
// load a structure from disk, design sequences, and compute a profile.
//
//   node test/browser.mjs [--headed] [--shot out.png]

import { createServer } from "node:http";
import { existsSync } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
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
  if (changed === 0) problems.push("design changed nothing at all");
  if (changed > selected) problems.push("design altered residues that were marked fixed");

  if (shotPath) {
    await page.screenshot({ path: shotPath.replace(/\.png$/, "-ligand.png"), fullPage: true });
  }
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
