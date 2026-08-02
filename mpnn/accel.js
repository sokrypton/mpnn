// Optional WebAssembly SIMD accelerator for the dense kernel.
//
// `mpnn/ops.js` remains the reference implementation and the fallback. This
// module is best-effort: if the module is missing, or the runtime has no SIMD,
// or anything else goes wrong, `load()` returns null and everything keeps
// working in JS.
//
// Inputs are staged into the module's linear memory rather than the whole
// engine being moved into it. That costs a copy per call, but for the shapes
// that matter -- [3648, 384] x [384, 128] -- the copy is ~7 MB against ~180
// million multiply-adds, so it is a couple of percent. Weights are the
// exception: they are uploaded once and cached by identity, because they are
// reused on every single call.

const PAGE = 65536;

/** Below this many multiply-adds the call overhead outweighs SIMD. */
const MIN_MACS = 1 << 16;

export class Accelerator {
  constructor(instance) {
    this.exports = instance.exports;
    this.memory = this.exports.memory;
    this.base = this.exports.__heap_base.value ?? this.exports.__heap_base;
    this.brk = align(this.base);
    /** Float32Array -> byte offset of its uploaded copy. */
    this.uploaded = new WeakMap();
    this.uploadedEnd = this.brk;
    this.scratch = this.brk;
    this._view = null;
    this.calls = 0;
  }

  get f32() {
    // A grow() detaches every existing view, so re-derive lazily.
    if (this._view === null || this._view.buffer !== this.memory.buffer) {
      this._view = new Float32Array(this.memory.buffer);
    }
    return this._view;
  }

  _ensure(bytes) {
    const needed = bytes - this.memory.buffer.byteLength;
    if (needed > 0) {
      this.memory.grow(Math.ceil(needed / PAGE));
      this._view = null;
    }
  }

  /** Copy a weight matrix in once and remember where it went. */
  _upload(array) {
    let offset = this.uploaded.get(array);
    if (offset !== undefined) return offset;
    offset = align(this.uploadedEnd);
    this._ensure(offset + array.byteLength);
    this.f32.set(array, offset >> 2);
    this.uploaded.set(array, offset);
    this.uploadedEnd = offset + array.byteLength;
    this.scratch = align(this.uploadedEnd);
    return offset;
  }

  /** Bump-allocate `bytes` of staging space after the uploaded weights. */
  _stage(cursor, bytes) {
    const at = align(cursor);
    this._ensure(at + bytes);
    return { at, next: at + bytes };
  }

  /**
   * gelu -> W2 -> gelu -> W3 over [n, hidden], all inside this memory.
   *
   * Exported as one call rather than four because GELU is memory bound: staged
   * on its own it would spend as long being copied as being computed.
   *
   * @returns {boolean} true if it ran here
   */
  tail2(h1, w2, b2, w3, b3, n, hidden, out) {
    if (n * hidden * hidden < MIN_MACS) return false;
    const w2Ptr = this._upload(w2);
    const b2Ptr = b2 === null ? 0 : this._upload(b2);
    const w3Ptr = this._upload(w3);
    const b3Ptr = b3 === null ? 0 : this._upload(b3);

    const bytes = n * hidden * 4;
    let s = this._stage(this.scratch, bytes);
    const h1Ptr = s.at;
    s = this._stage(s.next, bytes);
    const scratchPtr = s.at;
    s = this._stage(s.next, bytes);
    const outPtr = s.at;

    const mem = this.f32;
    mem.set(h1.length === n * hidden ? h1 : h1.subarray(0, n * hidden), h1Ptr >> 2);
    this.exports.tail2_f32(h1Ptr, w2Ptr, b2Ptr, w3Ptr, b3Ptr, n, hidden, scratchPtr, outPtr);
    out.set(this.f32.subarray(outPtr >> 2, (outPtr >> 2) + n * hidden), 0);
    this.calls++;
    return true;
  }

  /** W_in -> gelu -> W_out over [n, hidden] with an `ff`-wide middle. */
  ff(x, wIn, bIn, wOut, bOut, n, hidden, ff, out) {
    if (n * hidden * ff < MIN_MACS) return false;
    const wInPtr = this._upload(wIn);
    const bInPtr = bIn === null ? 0 : this._upload(bIn);
    const wOutPtr = this._upload(wOut);
    const bOutPtr = bOut === null ? 0 : this._upload(bOut);

    let s = this._stage(this.scratch, n * hidden * 4);
    const xPtr = s.at;
    s = this._stage(s.next, n * ff * 4);
    const scratchPtr = s.at;
    s = this._stage(s.next, n * hidden * 4);
    const outPtr = s.at;

    const mem = this.f32;
    mem.set(x.length === n * hidden ? x : x.subarray(0, n * hidden), xPtr >> 2);
    this.exports.ff_f32(xPtr, wInPtr, bInPtr, wOutPtr, bOutPtr,
      n, hidden, ff, scratchPtr, outPtr);
    out.set(this.f32.subarray(outPtr >> 2, (outPtr >> 2) + n * hidden), 0);
    this.calls++;
    return true;
  }

  /**
   * @returns {boolean} true if the call ran here, false to fall back to JS
   */
  linear(x, w, b, n, cin, cout, out) {
    if (n * cin * cout < MIN_MACS) return false;

    const wPtr = this._upload(w);
    const bPtr = b === null ? 0 : this._upload(b);
    // Weight uploads move `scratch`, so lay the staging buffers out after it.
    const xPtr = align(this.scratch);
    const outPtr = align(xPtr + n * cin * 4);
    this._ensure(outPtr + n * cout * 4);

    const mem = this.f32;
    mem.set(x.length === n * cin ? x : x.subarray(0, n * cin), xPtr >> 2);
    this.exports.linear_f32(xPtr, wPtr, bPtr, n, cin, cout, outPtr);
    const result = mem.subarray(outPtr >> 2, (outPtr >> 2) + n * cout);
    if (out.length === n * cout) out.set(result);
    else out.set(result, 0);
    this.calls++;
    return true;
  }
}

function align(x) {
  return (x + 15) & ~15;
}

/**
 * Instantiate the kernel module.
 *
 * There is no feature probe. A runtime without SIMD fails to validate a module
 * containing v128 instructions, so instantiation is the probe.
 *
 * @param {string|URL|ArrayBuffer|Response} source
 * @returns {Promise<Accelerator|null>}
 */
export async function loadAccelerator(source) {
  try {
    let instance;
    if (source instanceof ArrayBuffer) {
      instance = (await WebAssembly.instantiate(source, {})).instance;
    } else if (typeof Response !== "undefined" && source instanceof Response) {
      instance = (await WebAssembly.instantiateStreaming(source, {})).instance;
    } else {
      const response = await fetch(source);
      if (!response.ok) return null;
      try {
        instance = (await WebAssembly.instantiateStreaming(response.clone(), {})).instance;
      } catch {
        // Wrong MIME type, most likely; fall back to buffering.
        instance = (await WebAssembly.instantiate(await response.arrayBuffer(), {})).instance;
      }
    }
    if (!instance.exports.linear_f32) return null;
    return new Accelerator(instance);
  } catch {
    return null;
  }
}

/**
 * Load the kernels and install them into `ops.linear`.
 *
 * Returns the accelerator, or null if it could not be used -- in which case
 * nothing changes and the JS kernel keeps serving every call.
 */
export async function enableAcceleration(source) {
  const { useAccelerator } = await import("./ops.js");
  const accel = await loadAccelerator(source);
  if (accel) useAccelerator(accel);
  return accel;
}
