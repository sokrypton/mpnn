# Does the kernel's language matter?

Short answer: no, and the thing that *does* matter is not the language.

```
./wasm/bench/build.sh && node wasm/bench/run.mjs
```

Five builds of the same matmul, GFLOP/s, on the shapes the engine actually
issues (the blocked JS kernel in `mpnn/ops.js` does ~2.4 for scale):

| shape | C ptr+SIMD | Rust ptr+SIMD | Rust safe | Rust safe blocked | C plain | C plain `-ffast-math` |
| --- | --- | --- | --- | --- | --- | --- |
| [3648,384]×[384,128] | 20.4 | 21.7 | 3.3 | 3.3 | 3.2 | 12.0 |
| [3648,128]×[128,128] | 12.1 | 18.1 | 4.0 | 4.1 | 3.8 | 12.1 |
| [3648,512]×[512,128] | 21.0 | 22.1 | 3.2 | 3.2 | 3.0 | 12.0 |
| [3025,148]×[148,128] | 18.3 | 20.1 | 4.0 | 4.2 | 3.7 | 12.2 |
| [608,128]×[128,512] | 20.6 | 20.3 | 4.1 | 4.1 | 3.6 | 12.7 |

Three things fall out.

**C and Rust are the same kernel.** Both go through LLVM, and
`core::arch::wasm32` maps to the same builtins as `wasm_simd128.h`, so written
the same way they emit the same code. Rust is a percent or two ahead here,
which is noise plus the odd scheduling difference. There is no speedup
available by switching languages.

**Safety is not what costs you.** Safe Rust with bounds-checked slices (3.2-4.2)
lands slightly *ahead* of the equivalent plain C loop (3.0-3.8) — LLVM hoists
the slice bounds out of the loop. Anyone reaching for `unsafe` to make this
faster is solving the wrong problem.

**What costs you is the float reduction.** LLVM will not vectorise
`acc += x[k] * w[k]` on its own, because float addition is not associative and
reordering it changes the answer. Grant permission and plain C jumps from 3.2 to
12.0. Write the intrinsics by hand and it goes to 20+, the rest coming from
blocking four input rows against each pass over the weight matrix so `w` is
loaded once per four rows instead of once per row.

So the shipped kernel reassociates the sum too — four SIMD lanes accumulate
separately. The difference from `-ffast-math` is that it is *scoped*: the matmul
gets it because there it is safe and worth 4x, while the `exp` polynomial next
to it keeps strict evaluation. Parity against PyTorch is unaffected either way,
at ~1e-6.

## Why clang, then

Only that `clang` and `wasm-ld` ship with LLVM and were already present, so the
build is one command in a 12-line shell script with no package manager, no
lockfile, and no extra target to install. Rust would work exactly as well. If
you switch, keep the explicit `core::arch::wasm32` intrinsics — a safe idiomatic
port would be 5-6x slower and barely ahead of the JavaScript it replaced.


# One-hot times a matrix, or a gather?

```
node wasm/bench/onehot.mjs
```

ColabDesign's `EmbedToken` turns every embedding lookup into a dense product:

```python
if jnp.issubdtype(arr.dtype, jnp.integer):
    one_hot = jax.nn.one_hot(arr, self.vocab_size)
else:
    one_hot = arr                       # already a soft distribution
return jnp.tensordot(one_hot, self.embeddings, 1)
```

That is the right call there for two reasons, and only one of them is speed.
The `else` branch is the real one: `arr` may be a *continuous* distribution over
amino acids, because ColabDesign optimises sequences by gradient descent and you
cannot backpropagate through an integer index. The dense form also happens to
suit a TPU, where a gather is slow and a matmul is nearly free.

Neither applies to this engine. It only runs inference, so nothing needs to be
differentiable, and it runs on one wasm thread, where a gather is a memory read
and a matmul is multiply-adds you pay for. LigandMPNN's atom-type encoding makes
the gap plain: the one-hot is 147 wide — `[type 120 | group 19 | period 8]` —
with exactly **three** non-zeros, so 98% of the products are against zero.

Over L·M = 3025 residue-atom slots:

| form | time |
| --- | --- |
| one-hot @ matrix, the ColabDesign form | 6.31 ms |
| three column reads plus the bias | 1.30 ms |
| deduplicate by element first, then gather rows | **0.25 ms** |

So the engine goes the other way in three places: the atom-type encodings
(147→64 and 147→128) and the relative-position encoding (66→16) are all column
reads rather than products, and the atom-type table is then built once per
*element* (120 rows) instead of once per residue-atom slot.

The inverse is not available anywhere that matters. The remaining gathers are
neighbour lookups, `pc[EIdx[i, k]]`; expressing one as a matmul would need an
`[L·K, L]` selection matrix, which is L times *more* work, not less. Those cost
about 10% of runtime now and the way to reduce them is to move them into wasm
alongside the matmul, not to turn them into matmuls.

One thing to keep in mind if the page ever grows gradient-based design — a
ColabDesign-style hallucination or binder mode over soft sequences — the dense
form stops being a choice and becomes a requirement, at least for `W_s`.
