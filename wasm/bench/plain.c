// Baselines for wasm/bench: the loop anyone writes first, with no
// intrinsics. Built twice, once plain and once with -ffast-math, to show
// how much of the gap is LLVM refusing to reassociate a float reduction.
#define EXPORT __attribute__((visibility("default")))
// No intrinsics: the same loop anyone would write first, in C.
EXPORT void linear_plain(const float *x, const float *w, const float *b,
                         int n, int cin, int cout, float *out) {
  for (int i = 0; i < n; i++)
    for (int o = 0; o < cout; o++) {
      float acc = 0.0f;
      for (int k = 0; k < cin; k++) acc += x[i * cin + k] * w[o * cin + k];
      out[i * cout + o] = acc + (b ? b[o] : 0.0f);
    }
}
