//! Same kernel, three ways, to see what the language actually costs.
#![no_std]
#![allow(clippy::missing_safety_doc)]

use core::arch::wasm32::*;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { core::arch::wasm32::unreachable() }

#[inline(always)]
fn hsum(v: v128) -> f32 {
    f32x4_extract_lane::<0>(v) + f32x4_extract_lane::<1>(v)
        + f32x4_extract_lane::<2>(v) + f32x4_extract_lane::<3>(v)
}

/// 1. Raw pointers + explicit SIMD -- a transliteration of the C.
#[no_mangle]
pub unsafe extern "C" fn linear_ptr(
    x: *const f32, w: *const f32, b: *const f32,
    n: i32, cin: i32, cout: i32, out: *mut f32,
) {
    let (n, cin, cout) = (n as isize, cin as isize, cout as isize);
    let body = cin & !3;
    let mut i: isize = 0;
    while i + 4 <= n {
        let x0 = x.offset(i * cin);
        let x1 = x.offset((i + 1) * cin);
        let x2 = x.offset((i + 2) * cin);
        let x3 = x.offset((i + 3) * cin);
        for o in 0..cout {
            let wo = w.offset(o * cin);
            let (mut a0, mut a1, mut a2, mut a3) =
                (f32x4_splat(0.0), f32x4_splat(0.0), f32x4_splat(0.0), f32x4_splat(0.0));
            let mut k: isize = 0;
            while k < body {
                let wv = v128_load(wo.offset(k) as *const v128);
                a0 = f32x4_add(a0, f32x4_mul(v128_load(x0.offset(k) as *const v128), wv));
                a1 = f32x4_add(a1, f32x4_mul(v128_load(x1.offset(k) as *const v128), wv));
                a2 = f32x4_add(a2, f32x4_mul(v128_load(x2.offset(k) as *const v128), wv));
                a3 = f32x4_add(a3, f32x4_mul(v128_load(x3.offset(k) as *const v128), wv));
                k += 4;
            }
            let (mut s0, mut s1, mut s2, mut s3) = (hsum(a0), hsum(a1), hsum(a2), hsum(a3));
            while k < cin {
                let wv = *wo.offset(k);
                s0 += *x0.offset(k) * wv;
                s1 += *x1.offset(k) * wv;
                s2 += *x2.offset(k) * wv;
                s3 += *x3.offset(k) * wv;
                k += 1;
            }
            let bv = if b.is_null() { 0.0 } else { *b.offset(o) };
            *out.offset(i * cout + o) = s0 + bv;
            *out.offset((i + 1) * cout + o) = s1 + bv;
            *out.offset((i + 2) * cout + o) = s2 + bv;
            *out.offset((i + 3) * cout + o) = s3 + bv;
        }
        i += 4;
    }
    while i < n {
        for o in 0..cout {
            let mut s = 0.0f32;
            for k in 0..cin { s += *x.offset(i * cin + k) * *w.offset(o * cin + k); }
            *out.offset(i * cout + o) = s + if b.is_null() { 0.0 } else { *b.offset(o) };
        }
        i += 1;
    }
}

/// 2. Safe slices, bounds-checked indexing, no explicit SIMD -- what you get if
///    you just write the loop and trust the autovectoriser.
#[no_mangle]
pub unsafe extern "C" fn linear_safe(
    x: *const f32, w: *const f32, b: *const f32,
    n: i32, cin: i32, cout: i32, out: *mut f32,
) {
    let (n, cin, cout) = (n as usize, cin as usize, cout as usize);
    let x = core::slice::from_raw_parts(x, n * cin);
    let w = core::slice::from_raw_parts(w, cout * cin);
    let bs = if b.is_null() { None } else { Some(core::slice::from_raw_parts(b, cout)) };
    let out = core::slice::from_raw_parts_mut(out, n * cout);

    for i in 0..n {
        let xr = &x[i * cin..(i + 1) * cin];
        for o in 0..cout {
            let wr = &w[o * cin..(o + 1) * cin];
            let mut acc = 0.0f32;
            for k in 0..cin { acc += xr[k] * wr[k]; }
            out[i * cout + o] = acc + bs.map_or(0.0, |v| v[o]);
        }
    }
}

/// 3. Safe slices with the same 4-row blocking, iterator-based inner loop.
#[no_mangle]
pub unsafe extern "C" fn linear_safe_blocked(
    x: *const f32, w: *const f32, b: *const f32,
    n: i32, cin: i32, cout: i32, out: *mut f32,
) {
    let (n, cin, cout) = (n as usize, cin as usize, cout as usize);
    let x = core::slice::from_raw_parts(x, n * cin);
    let w = core::slice::from_raw_parts(w, cout * cin);
    let bs = if b.is_null() { None } else { Some(core::slice::from_raw_parts(b, cout)) };
    let out = core::slice::from_raw_parts_mut(out, n * cout);

    let mut i = 0;
    while i + 4 <= n {
        for o in 0..cout {
            let wr = &w[o * cin..(o + 1) * cin];
            let bv = bs.map_or(0.0, |v| v[o]);
            let mut a = [0.0f32; 4];
            for r in 0..4 {
                let xr = &x[(i + r) * cin..(i + r + 1) * cin];
                a[r] = xr.iter().zip(wr.iter()).map(|(p, q)| p * q).sum();
            }
            for r in 0..4 { out[(i + r) * cout + o] = a[r] + bv; }
        }
        i += 4;
    }
    while i < n {
        for o in 0..cout {
            let wr = &w[o * cin..(o + 1) * cin];
            let xr = &x[i * cin..(i + 1) * cin];
            out[i * cout + o] = xr.iter().zip(wr.iter()).map(|(p, q)| p * q).sum::<f32>()
                + bs.map_or(0.0, |v| v[o]);
        }
        i += 1;
    }
}
