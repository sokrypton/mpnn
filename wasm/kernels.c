// SIMD kernels for the hot inner loops.
//
// Built for wasm32 with -msimd128. The JS in mpnn/ops.js stays the reference
// implementation and the fallback: this file has to agree with it, and the
// parity suite is run against both.
//
//   clang --target=wasm32 -O3 -msimd128 -nostdlib -ffreestanding \
//     -Wl,--no-entry -Wl,--export-dynamic -Wl,--initial-memory=…  \
//     -o wasm/kernels.wasm wasm/kernels.c
//
// Everything works on a single linear memory that JS bump-allocates. Callers
// stage inputs into it and read outputs back out; weights are uploaded once and
// kept, since they are reused on every call.

#include <wasm_simd128.h>

#define EXPORT __attribute__((visibility("default")))

/** Horizontal sum of a f32x4. */
static inline float hsum(v128_t v) {
  return wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1)
       + wasm_f32x4_extract_lane(v, 2) + wasm_f32x4_extract_lane(v, 3);
}

/**
 * out[i, o] = dot(x[i], w[o]) + b[o]
 *
 * x is [n, cin], w is [cout, cin] row major (PyTorch's Linear layout), out is
 * [n, cout]. Four input rows share each pass over the weight matrix, which is
 * what keeps the kernel off the memory bus: without it the whole of w streams
 * from L2 once per row.
 *
 * `b` may be null.
 */
EXPORT void linear_f32(const float *x, const float *w, const float *b,
                       int n, int cin, int cout, float *out) {
  const int tail = cin & 3;
  const int body = cin - tail;
  int i = 0;

  for (; i + 4 <= n; i += 4) {
    const float *x0 = x + (i + 0) * cin;
    const float *x1 = x + (i + 1) * cin;
    const float *x2 = x + (i + 2) * cin;
    const float *x3 = x + (i + 3) * cin;
    float *o0 = out + (i + 0) * cout;
    float *o1 = out + (i + 1) * cout;
    float *o2 = out + (i + 2) * cout;
    float *o3 = out + (i + 3) * cout;

    for (int o = 0; o < cout; o++) {
      const float *wo = w + o * cin;
      v128_t a0 = wasm_f32x4_const_splat(0.0f);
      v128_t a1 = a0, a2 = a0, a3 = a0;
      for (int k = 0; k < body; k += 4) {
        v128_t wv = wasm_v128_load(wo + k);
        a0 = wasm_f32x4_add(a0, wasm_f32x4_mul(wasm_v128_load(x0 + k), wv));
        a1 = wasm_f32x4_add(a1, wasm_f32x4_mul(wasm_v128_load(x1 + k), wv));
        a2 = wasm_f32x4_add(a2, wasm_f32x4_mul(wasm_v128_load(x2 + k), wv));
        a3 = wasm_f32x4_add(a3, wasm_f32x4_mul(wasm_v128_load(x3 + k), wv));
      }
      float s0 = hsum(a0), s1 = hsum(a1), s2 = hsum(a2), s3 = hsum(a3);
      for (int k = body; k < cin; k++) {
        float wv = wo[k];
        s0 += x0[k] * wv;
        s1 += x1[k] * wv;
        s2 += x2[k] * wv;
        s3 += x3[k] * wv;
      }
      float bv = b ? b[o] : 0.0f;
      o0[o] = s0 + bv;
      o1[o] = s1 + bv;
      o2[o] = s2 + bv;
      o3[o] = s3 + bv;
    }
  }

  for (; i < n; i++) {
    const float *xi = x + i * cin;
    float *oi = out + i * cout;
    for (int o = 0; o < cout; o++) {
      const float *wo = w + o * cin;
      v128_t acc = wasm_f32x4_const_splat(0.0f);
      for (int k = 0; k < body; k += 4) {
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(xi + k),
                                                 wasm_v128_load(wo + k)));
      }
      float s = hsum(acc);
      for (int k = body; k < cin; k++) s += xi[k] * wo[k];
      oi[o] = s + (b ? b[o] : 0.0f);
    }
  }
}

// --- GELU ------------------------------------------------------------------
//
// Once the matmul is nine times faster, GELU is the next bottleneck. It cannot
// be exported on its own and still pay: it is memory bound, so staging its
// array in and back out would cost about what the arithmetic does. So the
// entry points below are coarser -- a whole message MLP, a whole feed-forward
// -- and the activation never leaves this memory.
//
// exp is computed by range reduction to [-ln2/2, ln2/2] plus a degree-6 series,
// which is accurate to about 1e-7 relative. That is float32 epsilon, and the
// parity suite runs against both this and the JS path.

static inline v128_t exp_f32x4(v128_t r) {
  const v128_t LOG2E = wasm_f32x4_const_splat(1.44269504088896341f);
  const v128_t LN2 = wasm_f32x4_const_splat(0.6931471805599453f);
  const v128_t one = wasm_f32x4_const_splat(1.0f);

  v128_t n = wasm_f32x4_nearest(wasm_f32x4_mul(r, LOG2E));
  v128_t f = wasm_f32x4_sub(r, wasm_f32x4_mul(n, LN2));

  v128_t p = wasm_f32x4_add(one, wasm_f32x4_mul(f, wasm_f32x4_const_splat(1.0f / 6.0f)));
  p = wasm_f32x4_add(one, wasm_f32x4_mul(wasm_f32x4_mul(f, wasm_f32x4_const_splat(0.2f)), p));
  p = wasm_f32x4_add(one, wasm_f32x4_mul(wasm_f32x4_mul(f, wasm_f32x4_const_splat(0.25f)), p));
  p = wasm_f32x4_add(one, wasm_f32x4_mul(wasm_f32x4_mul(f, wasm_f32x4_const_splat(1.0f / 3.0f)), p));
  p = wasm_f32x4_add(one, wasm_f32x4_mul(wasm_f32x4_mul(f, wasm_f32x4_const_splat(0.5f)), p));
  p = wasm_f32x4_add(one, wasm_f32x4_mul(f, p));

  // 2^n by assembling the exponent field; clamped so a very negative argument
  // flushes to zero rather than wrapping into a denormal.
  v128_t ni = wasm_i32x4_trunc_sat_f32x4(n);
  ni = wasm_i32x4_max(ni, wasm_i32x4_splat(-127));
  v128_t scale = wasm_i32x4_shl(wasm_i32x4_add(ni, wasm_i32x4_splat(127)), 23);
  return wasm_f32x4_mul(p, scale);
}

static const float ERF_P = 0.3275911f;
static const float ERF_A0 = 0.254829592f;
static const float ERF_A1 = -0.284496736f;
static const float ERF_A2 = 1.421413741f;
static const float ERF_A3 = -1.453152027f;
static const float ERF_A4 = 1.061405429f;

/** Exact (erf-based) GELU, in place. Same A&S 7.1.26 form as ops.js. */
EXPORT void gelu_f32(float *a, int n) {
  const v128_t one = wasm_f32x4_const_splat(1.0f);
  const v128_t half = wasm_f32x4_const_splat(0.5f);
  const v128_t isqrt2 = wasm_f32x4_const_splat(0.70710678118654752f);
  const v128_t signmask = wasm_i32x4_const_splat((int)0x80000000);
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    v128_t v = wasm_v128_load(a + i);
    v128_t x = wasm_f32x4_mul(v, isqrt2);
    v128_t ax = wasm_f32x4_abs(x);
    v128_t t = wasm_f32x4_div(one, wasm_f32x4_add(one, wasm_f32x4_mul(
        wasm_f32x4_const_splat(ERF_P), ax)));
    v128_t poly = wasm_f32x4_const_splat(ERF_A4);
    poly = wasm_f32x4_add(wasm_f32x4_mul(poly, t), wasm_f32x4_const_splat(ERF_A3));
    poly = wasm_f32x4_add(wasm_f32x4_mul(poly, t), wasm_f32x4_const_splat(ERF_A2));
    poly = wasm_f32x4_add(wasm_f32x4_mul(poly, t), wasm_f32x4_const_splat(ERF_A1));
    poly = wasm_f32x4_add(wasm_f32x4_mul(poly, t), wasm_f32x4_const_splat(ERF_A0));
    poly = wasm_f32x4_mul(poly, t);
    v128_t e = wasm_f32x4_sub(one, wasm_f32x4_mul(
        poly, exp_f32x4(wasm_f32x4_neg(wasm_f32x4_mul(ax, ax)))));
    // erf is odd, and e is non-negative, so copying x's sign bit suffices.
    v128_t er = wasm_v128_or(e, wasm_v128_and(x, signmask));
    wasm_v128_store(a + i, wasm_f32x4_mul(wasm_f32x4_mul(half, v),
                                          wasm_f32x4_add(one, er)));
  }
  for (; i < n; i++) {
    float v = a[i];
    float x = v * 0.70710678118654752f;
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = x < 0.0f ? -x : x;
    float t = 1.0f / (1.0f + ERF_P * ax);
    float poly = ((((ERF_A4 * t + ERF_A3) * t + ERF_A2) * t + ERF_A1) * t + ERF_A0) * t;
    v128_t ev = exp_f32x4(wasm_f32x4_splat(-ax * ax));
    float e = 1.0f - poly * wasm_f32x4_extract_lane(ev, 0);
    a[i] = 0.5f * v * (1.0f + sign * e);
  }
}

/**
 * gelu -> W2 -> gelu -> W3, the back half of every message MLP.
 *
 * `scratch` must have room for n * hidden floats. Keeping the whole chain here
 * means one copy in and one out instead of five.
 */
EXPORT void tail2_f32(float *h1, const float *w2, const float *b2,
                      const float *w3, const float *b3,
                      int n, int hidden, float *scratch, float *out) {
  gelu_f32(h1, n * hidden);
  linear_f32(h1, w2, b2, n, hidden, hidden, scratch);
  gelu_f32(scratch, n * hidden);
  linear_f32(scratch, w3, b3, n, hidden, hidden, out);
}

/**
 * W_in -> gelu -> W_out, the position-wise feed-forward.
 *
 * `scratch` must have room for n * ff floats.
 */
EXPORT void ff_f32(const float *x, const float *wIn, const float *bIn,
                   const float *wOut, const float *bOut,
                   int n, int hidden, int ff, float *scratch, float *out) {
  linear_f32(x, wIn, bIn, n, hidden, ff, scratch);
  gelu_f32(scratch, n * ff);
  linear_f32(scratch, wOut, bOut, n, ff, hidden, out);
}

/** Row-wise LayerNorm over the trailing axis of length c. */
EXPORT void layernorm_f32(const float *x, const float *gamma, const float *beta,
                          int n, int c, float *out, float eps) {
  const int tail = c & 3;
  const int body = c - tail;
  for (int i = 0; i < n; i++) {
    const float *xi = x + i * c;
    float *oi = out + i * c;

    v128_t sum = wasm_f32x4_const_splat(0.0f);
    for (int k = 0; k < body; k += 4) sum = wasm_f32x4_add(sum, wasm_v128_load(xi + k));
    float mean = hsum(sum);
    for (int k = body; k < c; k++) mean += xi[k];
    mean /= (float)c;

    v128_t mv = wasm_f32x4_splat(mean);
    v128_t vsum = wasm_f32x4_const_splat(0.0f);
    for (int k = 0; k < body; k += 4) {
      v128_t d = wasm_f32x4_sub(wasm_v128_load(xi + k), mv);
      vsum = wasm_f32x4_add(vsum, wasm_f32x4_mul(d, d));
    }
    float var = hsum(vsum);
    for (int k = body; k < c; k++) {
      float d = xi[k] - mean;
      var += d * d;
    }
    var /= (float)c;

    float inv = 1.0f / __builtin_sqrtf(var + eps);
    v128_t iv = wasm_f32x4_splat(inv);
    for (int k = 0; k < body; k += 4) {
      v128_t d = wasm_f32x4_sub(wasm_v128_load(xi + k), mv);
      v128_t r = wasm_f32x4_add(
          wasm_f32x4_mul(wasm_f32x4_mul(d, iv), wasm_v128_load(gamma + k)),
          wasm_v128_load(beta + k));
      wasm_v128_store(oi + k, r);
    }
    for (int k = body; k < c; k++) oi[k] = (xi[k] - mean) * inv * gamma[k] + beta[k];
  }
}


// --- coarse blocks -----------------------------------------------------------
//
// Same reasoning as tail2_f32: the pieces between the matmuls -- the masked
// neighbour sum, the residual adds, the LayerNorms, the output mask -- are all
// memory bound, so exporting them individually would spend as long staging as
// computing. Exported as whole blocks, they ride along with the matmul that is
// already here and the intermediates never leave this memory.

static inline void add_into(float *a, const float *b, int n) {
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    wasm_v128_store(a + i, wasm_f32x4_add(wasm_v128_load(a + i), wasm_v128_load(b + i)));
  }
  for (; i < n; i++) a[i] += b[i];
}

/** dh[r] = (1/scale) * sum_j maskAttend[r, j] * msg[r, j, :]; mask may be null. */
static void reduce_scaled(const float *msg, const float *maskAttend,
                          int rows, int k, int hidden, float scale, float *dh) {
  const v128_t inv = wasm_f32x4_splat(1.0f / scale);
  for (int i = 0; i < rows; i++) {
    float *d = dh + i * hidden;
    for (int c = 0; c < hidden; c += 4) wasm_v128_store(d + c, wasm_f32x4_const_splat(0.0f));
    for (int j = 0; j < k; j++) {
      const float m = maskAttend ? maskAttend[i * k + j] : 1.0f;
      if (m == 0.0f) continue;
      const float *src = msg + ((long)i * k + j) * hidden;
      if (m == 1.0f) {
        for (int c = 0; c < hidden; c += 4) {
          wasm_v128_store(d + c, wasm_f32x4_add(wasm_v128_load(d + c), wasm_v128_load(src + c)));
        }
      } else {
        const v128_t mv = wasm_f32x4_splat(m);
        for (int c = 0; c < hidden; c += 4) {
          wasm_v128_store(d + c, wasm_f32x4_add(wasm_v128_load(d + c),
                                                wasm_f32x4_mul(mv, wasm_v128_load(src + c))));
        }
      }
    }
    for (int c = 0; c < hidden; c += 4) {
      wasm_v128_store(d + c, wasm_f32x4_mul(wasm_v128_load(d + c), inv));
    }
  }
}

/**
 * A whole node update, given W1's pre-activation.
 *
 * gelu -> W2 -> gelu -> W3 -> masked neighbour sum -> residual LayerNorm ->
 * feed-forward -> residual LayerNorm -> mask. This is the encoder's node half
 * and the decoder's `applyPre` both; they were always the same sequence.
 *
 * `scratch` needs rows*(k*hidden + 2*hidden + ff) floats. `maskAttend` and
 * `maskV` may be null.
 */
EXPORT void message_block_f32(
    float *h1, const float *maskAttend, const float *hV,
    const float *w2, const float *b2, const float *w3, const float *b3,
    const float *g1, const float *c1,
    const float *wIn, const float *bIn, const float *wOut, const float *bOut,
    const float *g2, const float *c2, const float *maskV,
    int rows, int k, int hidden, int ff, float scale,
    float *scratch, float *out) {
  float *msg = scratch;
  float *dh = msg + (long)rows * k * hidden;
  float *tmp = dh + (long)rows * hidden;
  float *ffbuf = tmp + (long)rows * hidden;

  gelu_f32(h1, rows * k * hidden);
  linear_f32(h1, w2, b2, rows * k, hidden, hidden, msg);
  gelu_f32(msg, rows * k * hidden);
  // h1 is spent by now, so W3 writes over it.
  linear_f32(msg, w3, b3, rows * k, hidden, hidden, h1);

  reduce_scaled(h1, maskAttend, rows, k, hidden, scale, dh);
  add_into(dh, hV, rows * hidden);
  layernorm_f32(dh, g1, c1, rows, hidden, out, 1e-5f);

  linear_f32(out, wIn, bIn, rows, hidden, ff, ffbuf);
  gelu_f32(ffbuf, rows * ff);
  linear_f32(ffbuf, wOut, bOut, rows, ff, hidden, tmp);
  add_into(tmp, out, rows * hidden);
  layernorm_f32(tmp, g2, c2, rows, hidden, out, 1e-5f);

  if (maskV) {
    for (int i = 0; i < rows; i++) {
      const v128_t m = wasm_f32x4_splat(maskV[i]);
      float *o = out + (long)i * hidden;
      for (int c = 0; c < hidden; c += 4) {
        wasm_v128_store(o + c, wasm_f32x4_mul(wasm_v128_load(o + c), m));
      }
    }
  }
}

/**
 * The encoder's edge half: gelu -> W2 -> gelu -> W3, then a residual LayerNorm
 * against the incoming edge state.
 *
 * `scratch` needs n*hidden floats. `hE` and `out` may be the same buffer.
 */
EXPORT void edge_block_f32(float *h1, const float *hE,
                           const float *w2, const float *b2,
                           const float *w3, const float *b3,
                           const float *g, const float *c,
                           int n, int hidden, float *scratch, float *out) {
  gelu_f32(h1, n * hidden);
  linear_f32(h1, w2, b2, n, hidden, hidden, scratch);
  gelu_f32(scratch, n * hidden);
  linear_f32(scratch, w3, b3, n, hidden, hidden, h1);
  add_into(h1, hE, n * hidden);
  layernorm_f32(h1, g, c, n, hidden, out, 1e-5f);
}
