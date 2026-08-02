// The message-passing blocks the model is built from.
//
// Both encoder and decoder layers compute
//
//     message = W3(gelu(W2(gelu(W1([a ‖ b ‖ ...])))))
//
// over a [rows, k, ·] edge tensor. W1's input is always a *concatenation*, and a
// Linear over a concatenation is the sum of the Linears over each block:
//
//     W1 · [a ‖ b ‖ c] = W1a · a + W1b · b + W1c · c
//
// Most of those blocks are per-node, not per-edge -- the residue's own state,
// the neighbour's state, the neighbour's amino acid. Projecting them once per
// residue and gathering onto edges replaces an [L·K, 384] × [384, 128] product
// with an [L·K, 128] × [128, 128] one plus two [L, 128] × [128, 128] ones. On
// the encoder that is ~1.6x fewer multiply-adds; on the decoder, where three of
// the four blocks are per-node and one of them never changes, it is more.
//
// `splitColumns` does the slicing once, at construction.
//
// Weights are bound by the `make*` factories; scratch comes from a shared
// Arena, so a decode loop that runs L times allocates nothing.

import {
  addInto,
  catNeighborsNodes,
  expandNodeOntoEdges,
  gelu,
  layerNorm,
  linear,
  maskRows,
  reduceMessages,
  tryFeedForward,
  tryTail2,
} from "./ops.js";
import { MESSAGE_SCALE } from "./constants.js";

/** Rows processed per pass. Keeps the [rows, K, ·] intermediates bounded. */
export const CHUNK = 96;

/**
 * Slice a Linear weight of shape [cout, nBlocks · cin] into `nBlocks`
 * contiguous [cout, cin] matrices, one per concatenated input block.
 */
export function splitColumns(weight, cout, cin, nBlocks) {
  const blocks = [];
  const stride = cin * nBlocks;
  for (let b = 0; b < nBlocks; b++) {
    const m = new Float32Array(cout * cin);
    for (let o = 0; o < cout; o++) {
      m.set(weight.subarray(o * stride + b * cin, o * stride + (b + 1) * cin), o * cin);
    }
    blocks.push(m);
  }
  return blocks;
}

/** Position-wise feed-forward: hidden -> 4·hidden -> hidden with GELU. */
function makeDense(w, arena, prefix, hidden, tag) {
  const wIn = w.linear(`${prefix}.W_in`);
  const wOut = w.linear(`${prefix}.W_out`);
  const ff = wIn.shape[0];
  return (x, out, rows) => {
    if (tryFeedForward(x, wIn.weight, wIn.bias, wOut.weight, wOut.bias,
      rows, hidden, ff, out)) return out;
    const h = arena.f32(`${tag}.ff`, rows * ff);
    linear(x, wIn.weight, wIn.bias, rows, hidden, ff, h);
    gelu(h);
    linear(h, wOut.weight, wOut.bias, rows, ff, hidden, out);
    return out;
  };
}

/** The post-W1 half of the message MLP: gelu -> W2 -> gelu -> W3. */
function makeMessageTail(w, arena, prefix, names, hidden, tag) {
  const w2 = w.linear(`${prefix}.${names[1]}`);
  const w3 = w.linear(`${prefix}.${names[2]}`);
  return (h1, rows, out) => {
    if (tryTail2(h1, w2.weight, w2.bias, w3.weight, w3.bias, rows, hidden, out)) return out;
    gelu(h1, rows * hidden);
    const h2 = arena.f32(`${tag}.h2`, rows * hidden);
    linear(h1, w2.weight, w2.bias, rows, hidden, hidden, h2);
    gelu(h2);
    linear(h2, w3.weight, w3.bias, rows, hidden, hidden, out);
    return out;
  };
}

/**
 * Encoder layer, chunked over residues.
 *
 * The node and edge updates are separate sweeps because the edge update reads
 * the *new* node states at neighbouring residues -- doing both in one pass
 * would let earlier chunks leak into later ones.
 *
 * @returns {(hV, hE, EIdx, mask, maskAttend, L) => void} mutates hV and hE
 */
export function makeEncoderLayer(w, arena, prefix, hidden, K) {
  const node = makeFusedMessage(w, arena, prefix, ["W1", "W2", "W3"], hidden, "enc");
  const edge = makeFusedMessage(w, arena, prefix, ["W11", "W12", "W13"], hidden, "enc");
  const dense = makeDense(w, arena, `${prefix}.dense`, hidden, "enc");
  const norm1 = w.norm(`${prefix}.norm1`);
  const norm2 = w.norm(`${prefix}.norm2`);
  const norm3 = w.norm(`${prefix}.norm3`);

  return (hV, hE, EIdx, mask, maskAttend, L) => {
    // --- node update (double buffered so neighbours read pre-update states) ---
    const hVNext = arena.f32("enc.hVNext", L * hidden);
    let proj = node.project(hV, L);
    for (let start = 0; start < L; start += CHUNK) {
      const rows = Math.min(CHUNK, L - start);
      const msg = arena.f32("enc.msg", rows * K * hidden);
      const dh = arena.f32("enc.dh", rows * hidden);
      const tmp = arena.f32("enc.tmp", rows * hidden);
      const out = hVNext.subarray(start * hidden, (start + rows) * hidden);

      node.messages(proj, hE, EIdx, start, rows, K, msg);
      reduceMessages(msg, maskAttend.subarray(start * K), rows, K, hidden, MESSAGE_SCALE, dh);
      addInto(dh, hV.subarray(start * hidden), rows * hidden);
      layerNorm(dh, norm1.gamma, norm1.beta, rows, hidden, out);

      dense(out, tmp, rows);
      addInto(tmp, out, rows * hidden);
      layerNorm(tmp, norm2.gamma, norm2.beta, rows, hidden, out);
      maskRows(out, mask.subarray(start), rows, hidden);
    }
    hV.set(hVNext.subarray(0, L * hidden));

    // --- edge update ---
    proj = edge.project(hV, L);
    for (let start = 0; start < L; start += CHUNK) {
      const rows = Math.min(CHUNK, L - start);
      const msg = arena.f32("enc.msg", rows * K * hidden);
      const slice = hE.subarray(start * K * hidden, (start + rows) * K * hidden);

      edge.messages(proj, hE, EIdx, start, rows, K, msg);
      addInto(msg, slice, rows * K * hidden);
      layerNorm(msg, norm3.gamma, norm3.beta, rows * K, hidden, slice);
    }
  };
}

/**
 * Encoder message MLP with W1 split across [h_V_i ‖ h_E_ij ‖ h_V_j].
 *
 * `project` does the two per-residue halves; `messages` does the per-edge half
 * and the sum, then the W2/W3 tail.
 */
function makeFusedMessage(w, arena, prefix, names, hidden, tag) {
  const w1 = w.linear(`${prefix}.${names[0]}`);
  const [Wself, Wedge, Wneighbor] = splitColumns(w1.weight, hidden, hidden, 3);
  const tail = makeMessageTail(w, arena, prefix, names, hidden, tag);

  return {
    project(hV, L) {
      const self = arena.f32(`${tag}.pself`, L * hidden);
      const neighbor = arena.f32(`${tag}.pnb`, L * hidden);
      linear(hV, Wself, w1.bias, L, hidden, hidden, self);
      linear(hV, Wneighbor, null, L, hidden, hidden, neighbor);
      return { self, neighbor };
    },

    messages(proj, hE, EIdx, start, rows, K, out) {
      const h1 = arena.f32(`${tag}.h1`, rows * K * hidden);
      linear(
        hE.subarray(start * K * hidden, (start + rows) * K * hidden),
        Wedge, null, rows * K, hidden, hidden, h1,
      );
      for (let i = 0; i < rows; i++) {
        const so = (start + i) * hidden;
        for (let k = 0; k < K; k++) {
          const to = (i * K + k) * hidden;
          const no = EIdx[(start + i) * K + k] * hidden;
          for (let d = 0; d < hidden; d++) h1[to + d] += proj.self[so + d] + proj.neighbor[no + d];
        }
      }
      return tail(h1, rows * K, out);
    },
  };
}

/**
 * Decoder layer.
 *
 * Two entry points:
 *
 *  - `layer(hV, hE, mask, maskAttend, rows, k, out)` builds the concatenation
 *    itself. Used where the edge tensor is small or irregular (ligand context).
 *  - `layer.blocks` exposes the split W1 and `layer.applyPre(...)` runs the rest,
 *    so a caller that can assemble W1's pre-activation cheaply may do so.
 */
export function makeDecoderLayer(w, arena, prefix, hidden, edgeDim, tag = "dec") {
  const cin = hidden + edgeDim;
  const nBlocks = cin / hidden;
  const w1 = w.linear(`${prefix}.W1`);
  const tail_ = makeMessageTail(w, arena, prefix, ["W1", "W2", "W3"], hidden, tag);
  const dense = makeDense(w, arena, `${prefix}.dense`, hidden, tag);
  const norm1 = w.norm(`${prefix}.norm1`);
  const norm2 = w.norm(`${prefix}.norm2`);

  const blocks = Number.isInteger(nBlocks)
    ? splitColumns(w1.weight, hidden, hidden, nBlocks)
    : null;

  /** Finish a layer given W1's pre-activation `h1` ([rows·k, hidden]). */
  function applyPre(hV, h1, mask, maskAttend, rows, k, out) {
    const msg = arena.f32(`${tag}.msg`, rows * k * hidden);
    const dh = arena.f32(`${tag}.dh`, rows * hidden);
    const tmp = arena.f32(`${tag}.tmp`, rows * hidden);

    tail_(h1, rows * k, msg);
    reduceMessages(msg, maskAttend, rows, k, hidden, MESSAGE_SCALE, dh);
    addInto(dh, hV, rows * hidden);
    layerNorm(dh, norm1.gamma, norm1.beta, rows, hidden, out);

    dense(out, tmp, rows);
    addInto(tmp, out, rows * hidden);
    layerNorm(tmp, norm2.gamma, norm2.beta, rows, hidden, out);
    if (mask !== null) maskRows(out, mask, rows, hidden);
    return out;
  }

  const layer = (hV, hE, mask, maskAttend, rows, k, out) => {
    const x = expandNodeOntoEdges(
      hV, hE, rows, k, hidden, edgeDim, arena.f32(`${tag}.wide`, rows * k * cin),
    );
    const h1 = arena.f32(`${tag}.h1`, rows * k * hidden);
    linear(x, w1.weight, w1.bias, rows * k, cin, hidden, h1);
    return applyPre(hV, h1, mask, maskAttend, rows, k, out);
  };

  /**
   * The message MLP after W1: gelu -> W2 -> gelu -> W3, over an arbitrary row
   * list. Callers that have deduplicated or compacted their rows use this
   * directly instead of going through a [rows, k] edge tensor.
   */
  function tail(h1, rows, out) {
    return makeTail(h1, rows, out);
  }
  const makeTail = tail_;

  /**
   * Everything after the neighbour sum: residual LayerNorm, feed-forward,
   * residual LayerNorm, mask.
   */
  function finish(hV, dh, maskV, rows, out) {
    const tmp = arena.f32(`${tag}.finTmp`, rows * hidden);
    addInto(dh, hV, rows * hidden);
    layerNorm(dh, norm1.gamma, norm1.beta, rows, hidden, out);
    dense(out, tmp, rows);
    addInto(tmp, out, rows * hidden);
    layerNorm(tmp, norm2.gamma, norm2.beta, rows, hidden, out);
    if (maskV !== null) maskRows(out, maskV, rows, hidden);
    return out;
  }

  layer.blocks = blocks;
  layer.bias = w1.bias;
  layer.applyPre = applyPre;
  layer.tail = tail;
  layer.finish = finish;
  layer.hidden = hidden;
  return layer;
}

export { catNeighborsNodes };
