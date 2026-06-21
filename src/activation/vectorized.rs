// SPDX-FileCopyrightText: 2026 Marcin Jędrasik
// SPDX-License-Identifier: MIT

//! SIMD vectorisation helpers shared by the activation functions.
//!
//! All activation functions expose a scalar inner form and a vector inner form. The drivers in this
//! module walk an input slice in chunks of [`LANES`] elements, applying the vector form to each full
//! chunk and the scalar form to the remaining tail. This keeps the per-function code focused on the
//! mathematics rather than the chunking bookkeeping.
//!
//! `portable_simd` provides arithmetic, comparisons and `select` directly on `Simd` vectors. For the
//! transcendental functions that it does not expose (`sin`, `atan`, `log`) we fall back to lane-wise
//! evaluation via [`map_lanes`], which is correct and branch-free at the vector level but does not
//! exploit intra-lane parallelism for those specific functions. The exponential function — which
//! underpins the most common SharpNeat sigmoids — is given a dedicated vectorised implementation in
//! [`vexp`] so that `Logistic`, `LogisticSteep`, `TanH`, `ScaledELU` and `Gaussian` run fully
//! vectorised.

use std::simd::{Simd, StdFloat, f64x4};

/// Number of `f64` lanes processed per SIMD vector.
///
/// Four `f64` lanes occupy 256 bits, matching the width of AVX/AVX2 on x86-64 and two NEON
/// registers on AArch64. The `portable_simd` lowering handles the mapping to native instructions.
pub const LANES: usize = 4;

/// The SIMD vector type used throughout the crate.
pub(crate) type Vf = f64x4;

/// Applies `vec_fn` to each full [`LANES`]-wide chunk of `v` and `scalar_fn` to the tail, in place.
#[inline]
pub(crate) fn apply_inplace(
    v: &mut [f64],
    mut vec_fn: impl FnMut(Vf) -> Vf,
    mut scalar_fn: impl FnMut(f64) -> f64,
) {
    let mut i = 0;
    while i + LANES <= v.len() {
        let chunk = Simd::from_slice(&v[i..i + LANES]);
        let result = vec_fn(chunk);
        result.copy_to_slice(&mut v[i..i + LANES]);
        i += LANES;
    }
    while i < v.len() {
        v[i] = scalar_fn(v[i]);
        i += 1;
    }
}

/// Applies `vec_fn` / `scalar_fn` reading from `src` and writing to `dst`.
///
/// `src` and `dst` must have equal length.
#[inline]
pub(crate) fn apply_into(
    src: &[f64],
    dst: &mut [f64],
    mut vec_fn: impl FnMut(Vf) -> Vf,
    mut scalar_fn: impl FnMut(f64) -> f64,
) {
    assert_eq!(src.len(), dst.len(), "src and dst must have equal length");
    let mut i = 0;
    while i + LANES <= src.len() {
        let chunk = Simd::from_slice(&src[i..i + LANES]);
        let result = vec_fn(chunk);
        result.copy_to_slice(&mut dst[i..i + LANES]);
        i += LANES;
    }
    while i < src.len() {
        dst[i] = scalar_fn(src[i]);
        i += 1;
    }
}

/// Lane-wise fallback for transcendental functions not provided by `portable_simd`.
#[inline]
pub(crate) fn map_lanes(x: Vf, f: impl Fn(f64) -> f64) -> Vf {
    let a = x.to_array();
    Simd::from_array([f(a[0]), f(a[1]), f(a[2]), f(a[3])])
}

/// SIMD vectorised exponential function.
///
/// Uses the standard range reduction `e^x = 2^k * e^r` where `x = k*ln2 + r` and
/// `r ∈ [-ln2/2, ln2/2]`. The reduced argument `r` is approximated by a degree-9 Taylor
/// polynomial evaluated in Horner form. On the reduced interval the truncation error is below
/// 1e-11 relative, so the result matches `f64::exp` to well within 1e-9 across the input range —
/// enough to be indistinguishable from the reference for neural-network inference.
///
/// The implementation is pure arithmetic on `Vf` and uses no `unsafe`.
#[inline]
pub(crate) fn vexp(x: Vf) -> Vf {
    // ln(2) and its reciprocal.
    const LN2: f64 = std::f64::consts::LN_2;
    const INV_LN2: f64 = 1.0 / LN2;

    // k = round(x / ln2), carried as a float vector.
    let kf = (x * Simd::splat(INV_LN2)).round();
    // Reduced argument r = x - k * ln2, lying in [-ln2/2, ln2/2].
    let r = x - kf * Simd::splat(LN2);

    // Degree-9 Horner evaluation of 1 + r + r²/2! + ... + r⁹/9!.
    let one = Simd::splat(1.0);
    let mut p = one;
    p = one + r / Simd::splat(9.0) * p;
    p = one + r / Simd::splat(8.0) * p;
    p = one + r / Simd::splat(7.0) * p;
    p = one + r / Simd::splat(6.0) * p;
    p = one + r / Simd::splat(5.0) * p;
    p = one + r / Simd::splat(4.0) * p;
    p = one + r / Simd::splat(3.0) * p;
    p = one + r / Simd::splat(2.0) * p;
    let poly = one + r * p;

    // Scale by 2^k. `f64::powi` is exact for integer exponents and overflows to infinity for
    // very large inputs, matching the behaviour of `f64::exp`.
    let k = kf.to_array();
    let poly_arr = poly.to_array();
    Simd::from_array([
        poly_arr[0] * 2f64.powi(k[0] as i32),
        poly_arr[1] * 2f64.powi(k[1] as i32),
        poly_arr[2] * 2f64.powi(k[2] as i32),
        poly_arr[3] * 2f64.powi(k[3] as i32),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vexp_matches_std_exp_within_tolerance() {
        for &x in &[
            -20.0, -5.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 5.0, 20.0, 100.0,
        ] {
            let v = vexp(Simd::splat(x));
            let got = v[0];
            let want = f64::exp(x);
            let rel = ((got - want).abs() / want.max(1e-300)).min(1.0);
            assert!(rel < 1e-9, "vexp({x}) = {got}, std exp = {want}, rel={rel}");
        }
    }

    #[test]
    fn apply_inplace_processes_full_and_tail() {
        let mut v = (0..10).map(|i| i as f64).collect::<Vec<_>>();
        apply_inplace(&mut v, |x| x + Simd::splat(1.0), |x| x + 1.0);
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn apply_into_matches_inplace() {
        let src: Vec<f64> = (0..9).map(|i| i as f64 * 0.1).collect();
        let mut a = src.clone();
        let mut b = vec![0.0; src.len()];
        apply_inplace(&mut a, |x| x * Simd::splat(2.0), |x| x * 2.0);
        apply_into(&src, &mut b, |x| x * Simd::splat(2.0), |x| x * 2.0);
        assert_eq!(a, b);
    }
}
