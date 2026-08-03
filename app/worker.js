// Runs the engine off the main thread.
//
// Protocol: the page posts {id, type, ...} and gets back {id, ok, ...}, plus
// unsolicited {type: "progress"} messages while long jobs run.

import { enableAcceleration } from "../mpnn/accel.js";
import { AR, Model, perPositionNLL } from "../mpnn/model.js";
import { Weights } from "../mpnn/weights.js";

// Best effort. If the module is missing or the runtime has no SIMD this
// resolves to null and everything runs on the JS kernel instead.
const acceleration = enableAcceleration(new URL("../wasm/kernels.wasm", import.meta.url))
  .then((a) => Boolean(a))
  .catch(() => false);

/** @type {Map<string, Model>} */
const models = new Map();
let current = null;
let encoded = null;
let inputs = null;

function post(message, transfer) {
  self.postMessage(message, transfer ?? []);
}

async function loadModel(name, baseUrl) {
  if (models.has(name)) return models.get(name);
  const url = `${baseUrl}/${name}.mpnn`;
  const weights = await Weights.fetch(url, {
    onProgress: (received, total) => post({ type: "progress", stage: "download", received, total }),
  });
  const model = new Model(weights);
  models.set(name, model);
  return model;
}

function reviveInputs(raw) {
  return {
    X: new Float32Array(raw.X),
    mask: new Float32Array(raw.mask),
    residueIdx: new Int32Array(raw.residueIdx),
    chainLabels: new Int32Array(raw.chainLabels),
    ligandXyz: new Float32Array(raw.ligandXyz),
    ligandType: new Int32Array(raw.ligandType),
    ligandMask: new Float32Array(raw.ligandMask),
    membraneLabels: raw.membraneLabels ? new Int32Array(raw.membraneLabels) : undefined,
    useAtomContext: raw.useAtomContext,
    useSideChains: raw.useSideChains,
    xyz37: raw.xyz37 ? new Float32Array(raw.xyz37) : undefined,
    xyz37Mask: raw.xyz37Mask ? new Float32Array(raw.xyz37Mask) : undefined,
    chainMask: raw.chainMask ? new Float32Array(raw.chainMask) : undefined,
    X16: raw.X16 ? new Float32Array(raw.X16) : undefined,
    X16Mask: raw.X16Mask ? new Float32Array(raw.X16Mask) : undefined,
    polytype: raw.polytype ? new Int32Array(raw.polytype) : undefined,
  };
}

/** Deterministic PRNG so a given seed reproduces a design run exactly. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const handlers = {
  async load({ name, baseUrl }) {
    const simd = await acceleration;
    current = await loadModel(name, baseUrl);
    encoded = null;
    return {
      simd,
      modelType: current.modelType,
      k: current.K,
      atomContextNum: current.M,
      noiseLevel: current.w.noiseLevel,
    };
  },

  async encode({ inputs: raw }) {
    if (!current) throw new Error("no model loaded");
    inputs = reviveInputs(raw);
    const t0 = performance.now();
    encoded = current.encode(inputs);
    return { ms: performance.now() - t0, L: encoded.L, k: encoded.K };
  },

  async profile({ S, mode, exact }) {
    if (!encoded) throw new Error("nothing encoded");
    const t0 = performance.now();
    const out = current.profile(encoded, {
      S: S ? new Int32Array(S) : null,
      mode: mode ?? AR.NONE,
      exact: Boolean(exact),
      onProgress: exact
        ? (done, total) => post({ type: "progress", stage: "profile", received: done, total })
        : null,
    });
    return {
      ms: performance.now() - t0,
      logits: out.logits.buffer,
      probs: out.probs.buffer,
    };
  },

  async design({ batch, temperature, S, chainMask, bias, symmetry, seed }) {
    if (!encoded) throw new Error("nothing encoded");
    const t0 = performance.now();
    const rng = seed === undefined || seed === null ? Math.random : mulberry32(seed);
    const out = current.sample(encoded, {
      batch,
      temperature,
      S: S ? new Int32Array(S) : undefined,
      chainMask: chainMask ? new Float32Array(chainMask) : undefined,
      bias: bias ? new Float32Array(bias) : undefined,
      symmetry,
      rng,
    });
    return {
      ms: performance.now() - t0,
      seqs: out.seq,
      scores: out.score,
      S: out.S.map((s) => Array.from(s)),
    };
  },

  /**
   * Negative log-likelihood of a sequence, per position and averaged.
   *
   * The default is the pseudo-likelihood mask -- every position scored as if it
   * were decoded last, all in one pass. `AR.ORDER` is the other useful reading:
   * the true autoregressive likelihood, which depends on the decoding order and
   * so is averaged over several random ones, with the spread reported.
   */
  async score({ S, mode = AR.ALL_BUT_SELF, orders = 8, chainMask, seed }) {
    if (!encoded) throw new Error("nothing encoded");
    const L = encoded.L;
    const V = current.numLetters;
    const seq = new Int32Array(S);
    const weight = new Float32Array(L);
    for (let i = 0; i < L; i++) {
      weight[i] = encoded.mask[i] * (chainMask ? chainMask[i] : 1);
    }

    const nRuns = mode === AR.ORDER ? Math.max(1, orders) : 1;
    const rng = seed === undefined || seed === null ? Math.random : mulberry32(seed);
    const perPosition = new Float32Array(L);
    const step = new Float32Array(L);
    const order = new Int32Array(L);
    const runs = [];
    const t0 = performance.now();

    for (let r = 0; r < nRuns; r++) {
      let ar;
      if (mode === AR.ORDER) {
        for (let i = 0; i < L; i++) order[i] = i;
        for (let i = L - 1; i > 0; i--) {
          const j = Math.floor(rng() * (i + 1));
          const t = order[i];
          order[i] = order[j];
          order[j] = t;
        }
        ar = { type: AR.ORDER, order };
      } else {
        ar = { type: mode };
      }
      perPositionNLL(current.score(encoded, seq, ar), seq, L, V, step);
      let total = 0;
      let n = 0;
      for (let i = 0; i < L; i++) {
        perPosition[i] += step[i] / nRuns;
        if (weight[i] > 0) {
          total += step[i];
          n++;
        }
      }
      runs.push(n ? total / n : 0);
      if (nRuns > 1) post({ type: "progress", stage: "score", received: r + 1, total: nRuns });
    }

    const mean = runs.reduce((a, b) => a + b, 0) / runs.length;
    const sd = runs.length < 2 ? null : Math.sqrt(
      runs.reduce((a, b) => a + (b - mean) ** 2, 0) / (runs.length - 1),
    );
    return {
      ms: performance.now() - t0,
      mean,
      sd,
      mode,
      orders: nRuns,
      perPosition: perPosition.buffer,
    };
  },
};

self.onmessage = async (event) => {
  const { id, type, ...rest } = event.data;
  try {
    const handler = handlers[type];
    if (!handler) throw new Error(`unknown request ${type}`);
    const result = await handler(rest);
    const transfer = [];
    for (const value of Object.values(result ?? {})) {
      if (value instanceof ArrayBuffer) transfer.push(value);
    }
    post({ id, ok: true, ...result }, transfer);
  } catch (error) {
    post({ id, ok: false, error: String(error?.stack ?? error) });
  }
};
