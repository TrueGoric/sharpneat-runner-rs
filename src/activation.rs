// SPDX-FileCopyrightText: 2026 Marcin Jędrasik
// SPDX-License-Identifier: MIT

//! Neuron activation functions for SharpNeat networks.
//!
//! The primary abstraction is the [`Activation`] trait. Each SharpNeat activation function is a
//! zero-sized unit struct that implements [`Activation`] (e.g. [`Logistic`], [`ReLU`], [`TanH`]),
//! so neural nets generic over `A: Activation` monomorphise to a single specialised code path with
//! no virtual dispatch. The [`ActivationFn`] enum also implements [`Activation`]; it is the
//! runtime-dispatch adapter used when the function is chosen at runtime (e.g. from a `.net` file's
//! function code string).
//!
//! The set of functions and their constants mirror the C# classes in
//! `src/SharpNeat/NeuralNets/ActivationFunctions/` and its `Cppn` sub-namespace.

mod functions;
mod vectorized;

use std::fmt;

pub use functions::{
    ArcSinH, ArcTan, Gaussian, LeakyReLU, LeakyReLUShifted, Logistic, LogisticSteep, MaxMinusOne,
    NullFn, PolynomialApproximantSteep, QuadraticSigmoid, ReLU, SReLU, SReLUShifted, ScaledELU,
    Sine, SoftSignSteep, TanH,
};
pub use vectorized::LANES;

/// A neuron activation function usable by a [`NeuralNet`](crate::net::NeuralNet).
///
/// Implementations are zero-sized unit structs ([`Logistic`], [`ReLU`], …) so that a generic
/// `NeuralNetAcyclic<A>` monomorphises to code that calls the function directly, with the SIMD
/// inner loops inlined. The [`ActivationFn`] enum also implements this trait for runtime dispatch
/// (e.g. when the function is read from a `.net` file at runtime).
///
/// The supertraits keep activation functions trivially storable, copyable and sendable: they are
/// stateless values, so `Copy` + `Debug` + `Send + Sync + 'static` impose no real burden.
pub trait Activation: Copy + fmt::Debug + Send + Sync + 'static {
    /// The function code string used in `.net` files (e.g. `"ReLU"`).
    fn code(&self) -> &'static str;

    /// Apply the function to each element of `v` in place.
    fn activate_inplace(&self, v: &mut [f64]);

    /// Apply the function reading from `src` and writing to `dst`.
    ///
    /// `src` and `dst` must have equal length. This is the form used by the cyclic network, which
    /// keeps pre-activation and post-activation signals in separate arrays.
    fn activate_into(&self, src: &[f64], dst: &mut [f64]);
}

/// Runtime-dispatched activation function: one variant per SharpNeat function code.
///
/// Implements [`Activation`] by `match`-ing on the variant, so it can be used anywhere a generic
/// `A: Activation` is expected. Use this when the function is not known until runtime (e.g. it is
/// read from a `.net` file). When the function is known at compile time, prefer the concrete unit
/// structs ([`Logistic`], [`ReLU`], …) for a fully monomorphised, inlinable code path.
///
/// Variant names match the function codes written by SharpNeat's `NetFileWriter`, so
/// [`ActivationFn::from_code`] and [`Activation::code`] are exact inverses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationFn {
    /// Inverse hyperbolic sine, scaled to roughly the [-1, 1] range.
    ArcSinH,
    /// Inverse tangent.
    ArcTan,
    /// Leaky rectified linear unit (slope 0.001 for negative inputs).
    LeakyReLU,
    /// Leaky ReLU shifted so that x=0 maps to y≈0.5.
    LeakyReLUShifted,
    /// The logistic sigmoid `1 / (1 + e^-x)`.
    Logistic,
    /// The logistic sigmoid with a steepened slope (`-4.9 * x`).
    LogisticSteep,
    /// `max(-1, x)`.
    MaxMinusOne,
    /// Always returns zero.
    NullFn,
    /// A close polynomial approximation of the steepened logistic sigmoid.
    PolynomialApproximantSteep,
    /// A sigmoid formed by two sub-sections of `y = x^2` with leaky-relu tails.
    QuadraticSigmoid,
    /// Rectified linear unit.
    ReLU,
    /// Scaled exponential linear unit (SELU).
    ScaledELU,
    /// Softsign sigmoid with a steepened slope.
    SoftSignSteep,
    /// S-shaped rectified linear unit.
    SReLU,
    /// SReLU shifted so that x=0 maps to y≈0.5.
    SReLUShifted,
    /// Hyperbolic tangent.
    TanH,
    /// CPPN sine with a doubled period.
    Sine,
    /// CPPN gaussian, peak at x=0, tails tend to 0.
    Gaussian,
}

impl ActivationFn {
    /// Parse the function code string used in `.net` files.
    ///
    /// Returns `None` for unrecognised codes; callers typically wrap this in an error type that
    /// records the offending line.
    pub fn from_code(code: &str) -> Option<Self> {
        Some(match code {
            "ArcSinH" => Self::ArcSinH,
            "ArcTan" => Self::ArcTan,
            "LeakyReLU" => Self::LeakyReLU,
            "LeakyReLUShifted" => Self::LeakyReLUShifted,
            "Logistic" => Self::Logistic,
            "LogisticSteep" => Self::LogisticSteep,
            "MaxMinusOne" => Self::MaxMinusOne,
            "NullFn" => Self::NullFn,
            "PolynomialApproximantSteep" => Self::PolynomialApproximantSteep,
            "QuadraticSigmoid" => Self::QuadraticSigmoid,
            "ReLU" => Self::ReLU,
            "ScaledELU" => Self::ScaledELU,
            "SoftSignSteep" => Self::SoftSignSteep,
            "SReLU" => Self::SReLU,
            "SReLUShifted" => Self::SReLUShifted,
            "TanH" => Self::TanH,
            "Sine" => Self::Sine,
            "Gaussian" => Self::Gaussian,
            _ => return None,
        })
    }
}

impl Activation for ActivationFn {
    fn code(&self) -> &'static str {
        match self {
            Self::ArcSinH => "ArcSinH",
            Self::ArcTan => "ArcTan",
            Self::LeakyReLU => "LeakyReLU",
            Self::LeakyReLUShifted => "LeakyReLUShifted",
            Self::Logistic => "Logistic",
            Self::LogisticSteep => "LogisticSteep",
            Self::MaxMinusOne => "MaxMinusOne",
            Self::NullFn => "NullFn",
            Self::PolynomialApproximantSteep => "PolynomialApproximantSteep",
            Self::QuadraticSigmoid => "QuadraticSigmoid",
            Self::ReLU => "ReLU",
            Self::ScaledELU => "ScaledELU",
            Self::SoftSignSteep => "SoftSignSteep",
            Self::SReLU => "SReLU",
            Self::SReLUShifted => "SReLUShifted",
            Self::TanH => "TanH",
            Self::Sine => "Sine",
            Self::Gaussian => "Gaussian",
        }
    }

    #[inline]
    fn activate_inplace(&self, v: &mut [f64]) {
        match self {
            Self::ArcSinH => functions::arcsinh::apply_inplace(v),
            Self::ArcTan => functions::arctan::apply_inplace(v),
            Self::LeakyReLU => functions::leaky_relu::apply_inplace(v),
            Self::LeakyReLUShifted => functions::leaky_relu_shifted::apply_inplace(v),
            Self::Logistic => functions::logistic::apply_inplace(v),
            Self::LogisticSteep => functions::logistic_steep::apply_inplace(v),
            Self::MaxMinusOne => functions::max_minus_one::apply_inplace(v),
            Self::NullFn => functions::null_fn::apply_inplace(v),
            Self::PolynomialApproximantSteep => {
                functions::polynomial_approximant_steep::apply_inplace(v)
            }
            Self::QuadraticSigmoid => functions::quadratic_sigmoid::apply_inplace(v),
            Self::ReLU => functions::relu::apply_inplace(v),
            Self::ScaledELU => functions::scaled_elu::apply_inplace(v),
            Self::SoftSignSteep => functions::softsign_steep::apply_inplace(v),
            Self::SReLU => functions::srelu::apply_inplace(v),
            Self::SReLUShifted => functions::srelu_shifted::apply_inplace(v),
            Self::TanH => functions::tanh::apply_inplace(v),
            Self::Sine => functions::sine::apply_inplace(v),
            Self::Gaussian => functions::gaussian::apply_inplace(v),
        }
    }

    #[inline]
    fn activate_into(&self, src: &[f64], dst: &mut [f64]) {
        match self {
            Self::ArcSinH => functions::arcsinh::apply_into(src, dst),
            Self::ArcTan => functions::arctan::apply_into(src, dst),
            Self::LeakyReLU => functions::leaky_relu::apply_into(src, dst),
            Self::LeakyReLUShifted => functions::leaky_relu_shifted::apply_into(src, dst),
            Self::Logistic => functions::logistic::apply_into(src, dst),
            Self::LogisticSteep => functions::logistic_steep::apply_into(src, dst),
            Self::MaxMinusOne => functions::max_minus_one::apply_into(src, dst),
            Self::NullFn => functions::null_fn::apply_into(src, dst),
            Self::PolynomialApproximantSteep => {
                functions::polynomial_approximant_steep::apply_into(src, dst)
            }
            Self::QuadraticSigmoid => functions::quadratic_sigmoid::apply_into(src, dst),
            Self::ReLU => functions::relu::apply_into(src, dst),
            Self::ScaledELU => functions::scaled_elu::apply_into(src, dst),
            Self::SoftSignSteep => functions::softsign_steep::apply_into(src, dst),
            Self::SReLU => functions::srelu::apply_into(src, dst),
            Self::SReLUShifted => functions::srelu_shifted::apply_into(src, dst),
            Self::TanH => functions::tanh::apply_into(src, dst),
            Self::Sine => functions::sine::apply_into(src, dst),
            Self::Gaussian => functions::gaussian::apply_into(src, dst),
        }
    }
}

impl fmt::Display for ActivationFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_check(fn_: ActivationFn, inputs: &[f64], expected: &[f64], tol: f64) {
        let mut v = inputs.to_vec();
        fn_.activate_inplace(&mut v);
        for (i, (got, want)) in v.iter().zip(expected).enumerate() {
            assert!(
                (got - want).abs() <= tol,
                "{}[{i}] = {got}, expected {want}",
                fn_.code()
            );
        }
    }

    #[test]
    fn from_code_and_code_are_inverses() {
        for variant in [
            ActivationFn::ArcSinH,
            ActivationFn::ArcTan,
            ActivationFn::LeakyReLU,
            ActivationFn::LeakyReLUShifted,
            ActivationFn::Logistic,
            ActivationFn::LogisticSteep,
            ActivationFn::MaxMinusOne,
            ActivationFn::NullFn,
            ActivationFn::PolynomialApproximantSteep,
            ActivationFn::QuadraticSigmoid,
            ActivationFn::ReLU,
            ActivationFn::ScaledELU,
            ActivationFn::SoftSignSteep,
            ActivationFn::SReLU,
            ActivationFn::SReLUShifted,
            ActivationFn::TanH,
            ActivationFn::Sine,
            ActivationFn::Gaussian,
        ] {
            let code = variant.code();
            assert_eq!(ActivationFn::from_code(code), Some(variant));
        }
        assert_eq!(ActivationFn::from_code("Bogus"), None);
    }

    #[test]
    fn vectorised_matches_scalar_across_range() {
        let inputs: Vec<f64> = (-60..=60).map(|i| i as f64 * 0.1).collect();
        for variant in [
            ActivationFn::ArcSinH,
            ActivationFn::ArcTan,
            ActivationFn::LeakyReLU,
            ActivationFn::LeakyReLUShifted,
            ActivationFn::Logistic,
            ActivationFn::LogisticSteep,
            ActivationFn::MaxMinusOne,
            ActivationFn::NullFn,
            ActivationFn::PolynomialApproximantSteep,
            ActivationFn::QuadraticSigmoid,
            ActivationFn::ReLU,
            ActivationFn::ScaledELU,
            ActivationFn::SoftSignSteep,
            ActivationFn::SReLU,
            ActivationFn::SReLUShifted,
            ActivationFn::TanH,
            ActivationFn::Sine,
            ActivationFn::Gaussian,
        ] {
            // Compute via the SIMD path (length chosen to include a tail of 2 elements).
            let mut simd = inputs.clone();
            variant.activate_inplace(&mut simd);
            // Compute via a purely scalar path by processing one element at a time.
            let mut scalar = inputs.clone();
            for chunk in scalar.chunks_mut(1) {
                variant.activate_inplace(chunk);
            }
            for (i, (a, b)) in simd.iter().zip(&scalar).enumerate() {
                let rel = (a - b).abs() / b.abs().max(1e-12);
                assert!(
                    rel < 1e-9,
                    "{} mismatch at index {i}: simd={a}, scalar={b}, rel={rel}",
                    variant.code()
                );
            }
        }
    }

    #[test]
    fn relu_endpoints() {
        reference_check(
            ActivationFn::ReLU,
            &[-2.0, -0.0, 0.0, 3.5],
            &[0.0, 0.0, 0.0, 3.5],
            0.0,
        );
    }

    #[test]
    fn logistic_endpoints() {
        reference_check(
            ActivationFn::Logistic,
            &[0.0, 10.0, -10.0],
            &[
                0.5,
                1.0 / (1.0 + f64::exp(-10.0)),
                1.0 / (1.0 + f64::exp(10.0)),
            ],
            1e-9,
        );
    }

    #[test]
    fn logistic_steep_uses_factor_4_9() {
        let x = 0.3;
        let expected = 1.0 / (1.0 + f64::exp(-4.9 * x));
        reference_check(ActivationFn::LogisticSteep, &[x], &[expected], 1e-9);
    }

    #[test]
    fn tanh_matches_std() {
        let xs: Vec<f64> = (-10..=10).map(|i| i as f64 * 0.3).collect();
        let want: Vec<f64> = xs.iter().map(|x| x.tanh()).collect();
        reference_check(ActivationFn::TanH, &xs, &want, 1e-6);
    }

    #[test]
    fn null_fn_zeros() {
        reference_check(
            ActivationFn::NullFn,
            &[-5.0, 0.0, 7.0, 100.0],
            &[0.0, 0.0, 0.0, 0.0],
            0.0,
        );
    }

    #[test]
    fn max_minus_one_clamps() {
        reference_check(
            ActivationFn::MaxMinusOne,
            &[-5.0, -1.0, -0.5, 2.0],
            &[-1.0, -1.0, -0.5, 2.0],
            0.0,
        );
    }

    #[test]
    fn leaky_relu_negative_slope() {
        reference_check(
            ActivationFn::LeakyReLU,
            &[-10.0, -1.0, 0.0, 4.0],
            &[-0.01, -0.001, 0.0, 4.0],
            0.0,
        );
    }

    #[test]
    fn leaky_relu_shifted_offset() {
        reference_check(
            ActivationFn::LeakyReLUShifted,
            &[-0.5, 0.0, 0.5],
            &[0.0, 0.5, 1.0],
            0.0,
        );
    }

    #[test]
    fn softsign_steep_midpoint() {
        let xs: Vec<f64> = vec![0.0, 1.0, -1.0];
        let want: Vec<f64> = xs
            .iter()
            .map(|&x| 0.5 + x / (2.0 * (0.2 + x.abs())))
            .collect();
        reference_check(ActivationFn::SoftSignSteep, &xs, &want, 1e-12);
    }

    #[test]
    fn scaled_elu_matches_reference() {
        let xs: Vec<f64> = (-10..=10).map(|i| i as f64 * 0.5).collect();
        let want: Vec<f64> = xs
            .iter()
            .map(|&x| {
                if x >= 0.0 {
                    1.0507009873554805 * x
                } else {
                    1.0507009873554805 * (1.6732632423543773 * f64::exp(x) - 1.6732632423543773)
                }
            })
            .collect();
        reference_check(ActivationFn::ScaledELU, &xs, &want, 1e-9);
    }

    #[test]
    fn gaussian_peak_and_tail() {
        let want = |x: f64| f64::exp(-((x * 2.5).powi(2)));
        reference_check(
            ActivationFn::Gaussian,
            &[0.0, 1.0, -1.0, 5.0],
            &[want(0.0), want(1.0), want(-1.0), want(5.0)],
            1e-9,
        );
    }

    #[test]
    fn sine_doubled_period() {
        let want = |x: f64| (2.0 * x).sin();
        let xs = vec![0.0, 0.25, 0.5, 1.0];
        let want: Vec<f64> = xs.iter().map(|&x| want(x)).collect();
        reference_check(ActivationFn::Sine, &xs, &want, 1e-9);
    }

    #[test]
    fn into_matches_inplace() {
        let src: Vec<f64> = (0..17).map(|i| (i as f64 - 8.0) * 0.2).collect();
        for variant in [
            ActivationFn::Logistic,
            ActivationFn::ReLU,
            ActivationFn::TanH,
            ActivationFn::Sine,
            ActivationFn::Gaussian,
            ActivationFn::SReLU,
        ] {
            let mut a = src.clone();
            variant.activate_inplace(&mut a);
            let mut b = vec![0.0; src.len()];
            variant.activate_into(&src, &mut b);
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                assert!((x - y).abs() < 1e-12, "{}[{i}] {x} != {y}", variant.code());
            }
        }
    }
}
