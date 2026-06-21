//! Implementations of the standard SharpNeat activation functions.
//!
//! Each function is a small module exposing a `scalar(f64) -> f64` inner and a `vec(Vf) -> Vf` inner,
//! together with `apply_inplace` / `apply_into` wrappers that route them through the SIMD drivers in
//! [`super::vectorized`]. The arithmetic functions are fully vectorised. The exponential based
//! sigmoids use [`super::vectorized::vexp`]. The remaining transcendentals (`atan`, `sin`, `log`)
//! fall back to lane-wise evaluation via [`super::vectorized::map_lanes`].
//!
//! The function mathematics — including the magic constants — are taken directly from the C#
//! implementations in `src/SharpNeat/NeuralNets/ActivationFunctions/` (and the `Cppn` sub-namespace)
//! so that networks loaded from `.net` files behave as they do in SharpNeat.

use super::vectorized::{
    Vf, apply_inplace as vec_apply_inplace, apply_into as vec_apply_into, map_lanes, vexp,
};
use std::simd::prelude::*;
use std::simd::{Simd, StdFloat};

// ---------------------------------------------------------------------------
// Logistic
// ---------------------------------------------------------------------------

pub(super) mod logistic {
    use super::*;

    #[inline]
    pub fn scalar(x: f64) -> f64 {
        1.0 / (1.0 + f64::exp(-x))
    }

    #[inline]
    pub fn vec(x: Vf) -> Vf {
        let one = Simd::splat(1.0);
        one / (one + vexp(-x))
    }

    pub fn apply_inplace(v: &mut [f64]) {
        vec_apply_inplace(v, vec, scalar);
    }

    pub fn apply_into(src: &[f64], dst: &mut [f64]) {
        vec_apply_into(src, dst, vec, scalar);
    }
}

// ---------------------------------------------------------------------------
// LogisticSteep — logistic with a steepened slope (factor -4.9).
// ---------------------------------------------------------------------------

pub(super) mod logistic_steep {
    use super::*;

    const K: f64 = -4.9;

    #[inline]
    pub fn scalar(x: f64) -> f64 {
        1.0 / (1.0 + f64::exp(K * x))
    }

    #[inline]
    pub fn vec(x: Vf) -> Vf {
        let k = Simd::splat(K);
        let one = Simd::splat(1.0);
        one / (one + vexp(k * x))
    }

    pub fn apply_inplace(v: &mut [f64]) {
        vec_apply_inplace(v, vec, scalar);
    }

    pub fn apply_into(src: &[f64], dst: &mut [f64]) {
        vec_apply_into(src, dst, vec, scalar);
    }
}

// ---------------------------------------------------------------------------
// TanH
// ---------------------------------------------------------------------------

pub(super) mod tanh {
    use super::*;

    #[inline]
    pub fn scalar(x: f64) -> f64 {
        f64::tanh(x)
    }

    /// tanh(x) = 2·σ(2x) − 1, evaluated with the vectorised exp. This is numerically stable and
    /// avoids the per-lane `f64::tanh` call on the hot path.
    #[inline]
    pub fn vec(x: Vf) -> Vf {
        let two = Simd::splat(2.0);
        let one = Simd::splat(1.0);
        let sig = one / (one + vexp(-two * x));
        two * sig - one
    }

    pub fn apply_inplace(v: &mut [f64]) {
        vec_apply_inplace(v, vec, scalar);
    }

    pub fn apply_into(src: &[f64], dst: &mut [f64]) {
        vec_apply_into(src, dst, vec, scalar);
    }
}

// ---------------------------------------------------------------------------
// ReLU
// ---------------------------------------------------------------------------

pub(super) mod relu {
    use super::*;

    #[inline]
    pub fn scalar(x: f64) -> f64 {
        if x < 0.0 { 0.0 } else { x }
    }

    #[inline]
    pub fn vec(x: Vf) -> Vf {
        x.simd_max(Simd::splat(0.0))
    }

    pub fn apply_inplace(v: &mut [f64]) {
        vec_apply_inplace(v, vec, scalar);
    }

    pub fn apply_into(src: &[f64], dst: &mut [f64]) {
        vec_apply_into(src, dst, vec, scalar);
    }
}

// ---------------------------------------------------------------------------
// LeakyReLU — slope of 0.001 for negative inputs.
// ---------------------------------------------------------------------------

pub(super) mod leaky_relu {
    use super::*;

    const A: f64 = 0.001;

    #[inline]
    pub fn scalar(x: f64) -> f64 {
        if x < 0.0 { x * A } else { x }
    }

    #[inline]
    pub fn vec(x: Vf) -> Vf {
        let a = Simd::splat(A);
        let neg = x.simd_lt(Simd::splat(0.0));
        neg.select(a * x, x)
    }

    pub fn apply_inplace(v: &mut [f64]) {
        vec_apply_inplace(v, vec, scalar);
    }

    pub fn apply_into(src: &[f64], dst: &mut [f64]) {
        vec_apply_into(src, dst, vec, scalar);
    }
}

// ---------------------------------------------------------------------------
// LeakyReLUShifted — shifted on the x-axis so that x=0 maps to y≈0.5.
// ---------------------------------------------------------------------------

pub(super) mod leaky_relu_shifted {
    use super::*;

    const A: f64 = 0.001;
    const OFFSET: f64 = 0.5;

    #[inline]
    pub fn scalar(mut x: f64) -> f64 {
        x += OFFSET;
        if x < 0.0 { x * A } else { x }
    }

    #[inline]
    pub fn vec(x: Vf) -> Vf {
        let shifted = x + Simd::splat(OFFSET);
        let a = Simd::splat(A);
        let neg = shifted.simd_lt(Simd::splat(0.0));
        neg.select(a * shifted, shifted)
    }

    pub fn apply_inplace(v: &mut [f64]) {
        vec_apply_inplace(v, vec, scalar);
    }

    pub fn apply_into(src: &[f64], dst: &mut [f64]) {
        vec_apply_into(src, dst, vec, scalar);
    }
}

// ---------------------------------------------------------------------------
// MaxMinusOne — clamps the lower bound at -1.
// ---------------------------------------------------------------------------

pub(super) mod max_minus_one {
    use super::*;

    #[inline]
    pub fn scalar(x: f64) -> f64 {
        if x < -1.0 { -1.0 } else { x }
    }

    #[inline]
    pub fn vec(x: Vf) -> Vf {
        x.simd_max(Simd::splat(-1.0))
    }

    pub fn apply_inplace(v: &mut [f64]) {
        vec_apply_inplace(v, vec, scalar);
    }

    pub fn apply_into(src: &[f64], dst: &mut [f64]) {
        vec_apply_into(src, dst, vec, scalar);
    }
}

// ---------------------------------------------------------------------------
// NullFn — returns zero regardless of input.
// ---------------------------------------------------------------------------

pub(super) mod null_fn {
    pub fn apply_inplace(v: &mut [f64]) {
        v.fill(0.0);
    }

    pub fn apply_into(_src: &[f64], dst: &mut [f64]) {
        dst.fill(0.0);
    }
}

// ---------------------------------------------------------------------------
// ScaledELU (SELU).
// ---------------------------------------------------------------------------

pub(super) mod scaled_elu {
    use super::*;

    const ALPHA: f64 = 1.6732632423543773;
    const SCALE: f64 = 1.0507009873554805;

    #[inline]
    pub fn scalar(x: f64) -> f64 {
        if x >= 0.0 {
            SCALE * x
        } else {
            SCALE * (ALPHA * f64::exp(x) - ALPHA)
        }
    }

    #[inline]
    pub fn vec(x: Vf) -> Vf {
        let scale = Simd::splat(SCALE);
        let alpha = Simd::splat(ALPHA);
        let pos = x.simd_ge(Simd::splat(0.0));
        let neg_branch = alpha * vexp(x) - alpha;
        pos.select(x, neg_branch) * scale
    }

    pub fn apply_inplace(v: &mut [f64]) {
        vec_apply_inplace(v, vec, scalar);
    }

    pub fn apply_into(src: &[f64], dst: &mut [f64]) {
        vec_apply_into(src, dst, vec, scalar);
    }
}

// ---------------------------------------------------------------------------
// SReLU — S-shaped rectified linear unit.
// ---------------------------------------------------------------------------

pub(super) mod srelu {
    use super::*;

    const TL: f64 = 0.001;
    const TR: f64 = 0.999;
    const A: f64 = 0.00001;

    #[inline]
    pub fn scalar(x: f64) -> f64 {
        if x > TL && x < TR {
            x
        } else if x <= TL {
            TL + (x - TL) * A
        } else {
            TR + (x - TR) * A
        }
    }

    #[inline]
    pub fn vec(x: Vf) -> Vf {
        let tl = Simd::splat(TL);
        let tr = Simd::splat(TR);
        let a = Simd::splat(A);
        let mid = x.simd_gt(tl) & x.simd_lt(tr);
        let left = x.simd_le(tl);
        // right = !mid & !left
        let left_val = tl + (x - tl) * a;
        let right_val = tr + (x - tr) * a;
        // First pick between left_val (left region) and right_val; then pick x for the mid region.
        let boundary = left.select(left_val, right_val);
        mid.select(x, boundary)
    }

    pub fn apply_inplace(v: &mut [f64]) {
        vec_apply_inplace(v, vec, scalar);
    }

    pub fn apply_into(src: &[f64], dst: &mut [f64]) {
        vec_apply_into(src, dst, vec, scalar);
    }
}

// ---------------------------------------------------------------------------
// SReLUShifted — SReLU shifted so that x=0 maps to y≈0.5.
// ---------------------------------------------------------------------------

pub(super) mod srelu_shifted {
    use super::*;

    const TL: f64 = 0.001;
    const TR: f64 = 0.999;
    const A: f64 = 0.00001;
    const OFFSET: f64 = 0.5;

    #[inline]
    pub fn scalar(x: f64) -> f64 {
        let s = x + OFFSET;
        if s > TL && s < TR {
            s
        } else if s <= TL {
            TL + (s - TL) * A
        } else {
            TR + (s - TR) * A
        }
    }

    #[inline]
    pub fn vec(x: Vf) -> Vf {
        let s = x + Simd::splat(OFFSET);
        let tl = Simd::splat(TL);
        let tr = Simd::splat(TR);
        let a = Simd::splat(A);
        let mid = s.simd_gt(tl) & s.simd_lt(tr);
        let left = s.simd_le(tl);
        let left_val = tl + (s - tl) * a;
        let right_val = tr + (s - tr) * a;
        let boundary = left.select(left_val, right_val);
        mid.select(s, boundary)
    }

    pub fn apply_inplace(v: &mut [f64]) {
        vec_apply_inplace(v, vec, scalar);
    }

    pub fn apply_into(src: &[f64], dst: &mut [f64]) {
        vec_apply_into(src, dst, vec, scalar);
    }
}

// ---------------------------------------------------------------------------
// SoftSignSteep — softsign sigmoid with a steepened slope.
// ---------------------------------------------------------------------------

pub(super) mod softsign_steep {
    use super::*;

    const HALF: f64 = 0.5;
    const TWO: f64 = 2.0;
    const POINT_TWO: f64 = 0.2;

    #[inline]
    pub fn scalar(x: f64) -> f64 {
        HALF + x / (TWO * (POINT_TWO + x.abs()))
    }

    #[inline]
    pub fn vec(x: Vf) -> Vf {
        let half = Simd::splat(HALF);
        let two = Simd::splat(TWO);
        let pt = Simd::splat(POINT_TWO);
        half + x / (two * (pt + x.abs()))
    }

    pub fn apply_inplace(v: &mut [f64]) {
        vec_apply_inplace(v, vec, scalar);
    }

    pub fn apply_into(src: &[f64], dst: &mut [f64]) {
        vec_apply_into(src, dst, vec, scalar);
    }
}

// ---------------------------------------------------------------------------
// PolynomialApproximantSteep — fast logistic approximation avoiding exp.
// ---------------------------------------------------------------------------

pub(super) mod polynomial_approximant_steep {
    use super::*;

    const A: f64 = 4.9;
    const B: f64 = 0.555;
    const C: f64 = 0.143;

    #[inline]
    pub fn scalar(x: f64) -> f64 {
        let x = x * A;
        let x2 = x * x;
        let e = 1.0 + x.abs() + x2 * B + x2 * x2 * C;
        let f = if x > 0.0 { 1.0 / e } else { e };
        1.0 / (1.0 + f)
    }

    #[inline]
    pub fn vec(x: Vf) -> Vf {
        let a = Simd::splat(A);
        let b = Simd::splat(B);
        let c = Simd::splat(C);
        let one = Simd::splat(1.0);

        let x = x * a;
        let x2 = x * x;
        let e = one + x.abs() + x2 * b + x2 * x2 * c;
        let pos = x.simd_gt(Simd::splat(0.0));
        let f = pos.select(one / e, e);
        one / (one + f)
    }

    pub fn apply_inplace(v: &mut [f64]) {
        vec_apply_inplace(v, vec, scalar);
    }

    pub fn apply_into(src: &[f64], dst: &mut [f64]) {
        vec_apply_into(src, dst, vec, scalar);
    }
}

// ---------------------------------------------------------------------------
// QuadraticSigmoid — two sub-sections of y = x^2 with leaky-relu tails.
// ---------------------------------------------------------------------------

pub(super) mod quadratic_sigmoid {
    use super::*;

    const HALF: f64 = 0.5;
    const T: f64 = 0.999;
    const A: f64 = 0.00001;

    #[inline]
    pub fn scalar(x: f64) -> f64 {
        let mut y = x;
        let mut sign = 1.0;
        if y < 0.0 {
            y = -y;
            sign = -1.0;
        }
        if y < T {
            y = T - (y - T) * (y - T);
        } else {
            y = T + (y - T) * A;
        }
        (y * sign * HALF) + HALF
    }

    #[inline]
    pub fn vec(x: Vf) -> Vf {
        let half = Simd::splat(HALF);
        let t = Simd::splat(T);
        let a = Simd::splat(A);
        let zero = Simd::splat(0.0);

        let neg = x.simd_lt(zero);
        let y = x.abs(); // |x|
        let sign = neg.select(Simd::splat(-1.0), Simd::splat(1.0));
        let in_curve = y.simd_lt(t);
        let curve = t - (y - t) * (y - t);
        let tail = t + (y - t) * a;
        let y = in_curve.select(curve, tail);
        (y * sign * half) + half
    }

    pub fn apply_inplace(v: &mut [f64]) {
        vec_apply_inplace(v, vec, scalar);
    }

    pub fn apply_into(src: &[f64], dst: &mut [f64]) {
        vec_apply_into(src, dst, vec, scalar);
    }
}

// ---------------------------------------------------------------------------
// ArcTan
// ---------------------------------------------------------------------------

pub(super) mod arctan {
    use super::*;

    #[inline]
    pub fn scalar(x: f64) -> f64 {
        f64::atan(x)
    }

    #[inline]
    pub fn vec(x: Vf) -> Vf {
        map_lanes(x, f64::atan)
    }

    pub fn apply_inplace(v: &mut [f64]) {
        vec_apply_inplace(v, vec, scalar);
    }

    pub fn apply_into(src: &[f64], dst: &mut [f64]) {
        vec_apply_into(src, dst, vec, scalar);
    }
}

// ---------------------------------------------------------------------------
// ArcSinH — scaled inverse hyperbolic sine.
// ---------------------------------------------------------------------------

pub(super) mod arcsinh {
    use super::*;

    const K: f64 = 1.2567348023993685;
    const HALF: f64 = 0.5;

    #[inline]
    fn asinh_scalar(x: f64) -> f64 {
        // Exact definition of arcsinh; faster than f64::asinh on the targets SharpNeat benchmarks.
        f64::ln(x + (x * x + 1.0).sqrt())
    }

    #[inline]
    pub fn scalar(x: f64) -> f64 {
        K * ((asinh_scalar(x) + 1.0) * HALF)
    }

    #[inline]
    pub fn vec(x: Vf) -> Vf {
        let k = Simd::splat(K);
        let half = Simd::splat(HALF);
        let one = Simd::splat(1.0);
        // sqrt is a portable_simd intrinsic; log is not, so finish lane-wise.
        let inner = x + (x * x + one).sqrt();
        let asinh_v = map_lanes(inner, f64::ln);
        k * ((asinh_v + one) * half)
    }

    pub fn apply_inplace(v: &mut [f64]) {
        vec_apply_inplace(v, vec, scalar);
    }

    pub fn apply_into(src: &[f64], dst: &mut [f64]) {
        vec_apply_into(src, dst, vec, scalar);
    }
}

// ---------------------------------------------------------------------------
// Sine — CPPN sine with doubled period.
// ---------------------------------------------------------------------------

pub(super) mod sine {
    use super::*;

    const TWO: f64 = 2.0;

    #[inline]
    pub fn scalar(x: f64) -> f64 {
        f64::sin(TWO * x)
    }

    #[inline]
    pub fn vec(x: Vf) -> Vf {
        map_lanes(x * Simd::splat(TWO), f64::sin)
    }

    pub fn apply_inplace(v: &mut [f64]) {
        vec_apply_inplace(v, vec, scalar);
    }

    pub fn apply_into(src: &[f64], dst: &mut [f64]) {
        vec_apply_into(src, dst, vec, scalar);
    }
}

// ---------------------------------------------------------------------------
// Gaussian — CPPN gaussian, peak at x=0, tails → 0.
// ---------------------------------------------------------------------------

pub(super) mod gaussian {
    use super::*;

    const TWO_POINT_FIVE: f64 = 2.5;

    #[inline]
    pub fn scalar(x: f64) -> f64 {
        f64::exp(-((x * TWO_POINT_FIVE).powi(2)))
    }

    #[inline]
    pub fn vec(x: Vf) -> Vf {
        let s = x * Simd::splat(TWO_POINT_FIVE);
        vexp(-(s * s))
    }

    pub fn apply_inplace(v: &mut [f64]) {
        vec_apply_inplace(v, vec, scalar);
    }

    pub fn apply_into(src: &[f64], dst: &mut [f64]) {
        vec_apply_into(src, dst, vec, scalar);
    }
}

// ---------------------------------------------------------------------------
// Unit-struct activation function types.
//
// Each is a zero-sized marker that implements `Activation` by delegating to the matching inner
// module above. Because they are ZSTs and the trait methods are `&self`, a generic
// `NeuralNetAcyclic<A>` monomorphises to a direct call on the concrete inner functions, with the
// SIMD loops fully inlinable.
// ---------------------------------------------------------------------------

use super::Activation;

macro_rules! activation_type {
    (
        $(#[$meta:meta])*
        $name:ident, $module:ident, $code:literal
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
        pub struct $name;

        impl Activation for $name {
            #[inline]
            fn code(&self) -> &'static str {
                $code
            }

            #[inline]
            fn activate_inplace(&self, v: &mut [f64]) {
                $module::apply_inplace(v);
            }

            #[inline]
            fn activate_into(&self, src: &[f64], dst: &mut [f64]) {
                $module::apply_into(src, dst);
            }
        }
    };
}

activation_type! {
    /// Inverse hyperbolic sine, scaled to roughly the [-1, 1] range.
    ArcSinH, arcsinh, "ArcSinH"
}
activation_type! {
    /// Inverse tangent.
    ArcTan, arctan, "ArcTan"
}
activation_type! {
    /// Leaky rectified linear unit (slope 0.001 for negative inputs).
    LeakyReLU, leaky_relu, "LeakyReLU"
}
activation_type! {
    /// Leaky ReLU shifted so that x=0 maps to y≈0.5.
    LeakyReLUShifted, leaky_relu_shifted, "LeakyReLUShifted"
}
activation_type! {
    /// The logistic sigmoid `1 / (1 + e^-x)`.
    Logistic, logistic, "Logistic"
}
activation_type! {
    /// The logistic sigmoid with a steepened slope (`-4.9 * x`).
    LogisticSteep, logistic_steep, "LogisticSteep"
}
activation_type! {
    /// `max(-1, x)`.
    MaxMinusOne, max_minus_one, "MaxMinusOne"
}
activation_type! {
    /// Always returns zero.
    NullFn, null_fn, "NullFn"
}
activation_type! {
    /// A close polynomial approximation of the steepened logistic sigmoid.
    PolynomialApproximantSteep, polynomial_approximant_steep, "PolynomialApproximantSteep"
}
activation_type! {
    /// A sigmoid formed by two sub-sections of `y = x^2` with leaky-relu tails.
    QuadraticSigmoid, quadratic_sigmoid, "QuadraticSigmoid"
}
activation_type! {
    /// Rectified linear unit.
    ReLU, relu, "ReLU"
}
activation_type! {
    /// Scaled exponential linear unit (SELU).
    ScaledELU, scaled_elu, "ScaledELU"
}
activation_type! {
    /// Softsign sigmoid with a steepened slope.
    SoftSignSteep, softsign_steep, "SoftSignSteep"
}
activation_type! {
    /// S-shaped rectified linear unit.
    SReLU, srelu, "SReLU"
}
activation_type! {
    /// SReLU shifted so that x=0 maps to y≈0.5.
    SReLUShifted, srelu_shifted, "SReLUShifted"
}
activation_type! {
    /// Hyperbolic tangent.
    TanH, tanh, "TanH"
}
activation_type! {
    /// CPPN sine with a doubled period.
    Sine, sine, "Sine"
}
activation_type! {
    /// CPPN gaussian, peak at x=0, tails tend to 0.
    Gaussian, gaussian, "Gaussian"
}
