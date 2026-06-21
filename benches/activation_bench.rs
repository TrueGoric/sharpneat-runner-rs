//! Micro-benchmarks for the activation functions.
//!
//! Zero-dependency harness: each scenario is run in a loop for a fixed wall-clock budget and the
//! mean time per call is reported. Run with `cargo bench --bench activation_bench` (release mode
//! is the default for the `bench` profile).
//!
//! Each function is benchmarked twice: once through the [`ActivationFn`] enum (runtime dispatch)
//! and once through its concrete unit struct (monomorphised). Comparing the two confirms that the
//! trait-based design introduces no performance regression.

use std::hint::black_box;
use std::time::{Duration, Instant};

use sharpneat_runner_rs::{
    Activation, ActivationFn, ArcSinH, ArcTan, Gaussian, LeakyReLU, LeakyReLUShifted, Logistic,
    LogisticSteep, MaxMinusOne, NullFn, PolynomialApproximantSteep, QuadraticSigmoid, ReLU, SReLU,
    SReLUShifted, ScaledELU, Sine, SoftSignSteep, TanH,
};

const BUFFER_LEN: usize = 1024;
const WARMUP: Duration = Duration::from_millis(200);
const MEASURE: Duration = Duration::from_secs(2);

/// Benchmark any `Activation` implementation through a shared trait bound.
fn bench_inplace<A: Activation>(name: &str, fn_: A, input: &[f64]) {
    assert!(!input.is_empty());
    let mut workspace = input.to_vec();

    // Warm up.
    let start = Instant::now();
    while start.elapsed() < WARMUP {
        fn_.activate_inplace(black_box(&mut workspace));
    }

    // Measure.
    let mut calls = 0u64;
    let start = Instant::now();
    while start.elapsed() < MEASURE {
        fn_.activate_inplace(black_box(&mut workspace));
        calls += 1;
    }
    let elapsed = start.elapsed();
    let per_call_ns = elapsed.as_nanos() as f64 / calls as f64;
    let per_element_ns = per_call_ns / input.len() as f64;
    println!(
        "{name:<42} {calls:>10} calls | {per_call_ns:>10.2} ns/call | {per_element_ns:>8.3} ns/elem"
    );
}

fn main() {
    // Two input distributions: a moderate range that exercises the linear region of most
    // sigmoids, and a wide range that hits the polynomial scaling of the vectorised exp.
    let moderate: Vec<f64> = (0..BUFFER_LEN)
        .map(|i| (i as f64 - BUFFER_LEN as f64 / 2.0) * 0.01)
        .collect();
    let wide: Vec<f64> = (0..BUFFER_LEN)
        .map(|i| (i as f64 - BUFFER_LEN as f64 / 2.0) * 0.5)
        .collect();

    println!("=== activation functions (in-place, {BUFFER_LEN} elements) ===");
    println!();

    println!("-- moderate range [-5.12, 5.12] --");
    println!();
    println!("  [runtime dispatch via ActivationFn enum]");
    for fn_ in [
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
        bench_inplace(fn_.code(), fn_, &moderate);
    }

    println!();
    println!("  [monomorphised via concrete unit struct]");
    bench_inplace("ArcSinH", ArcSinH, &moderate);
    bench_inplace("ArcTan", ArcTan, &moderate);
    bench_inplace("LeakyReLU", LeakyReLU, &moderate);
    bench_inplace("LeakyReLUShifted", LeakyReLUShifted, &moderate);
    bench_inplace("Logistic", Logistic, &moderate);
    bench_inplace("LogisticSteep", LogisticSteep, &moderate);
    bench_inplace("MaxMinusOne", MaxMinusOne, &moderate);
    bench_inplace("NullFn", NullFn, &moderate);
    bench_inplace(
        "PolynomialApproximantSteep",
        PolynomialApproximantSteep,
        &moderate,
    );
    bench_inplace("QuadraticSigmoid", QuadraticSigmoid, &moderate);
    bench_inplace("ReLU", ReLU, &moderate);
    bench_inplace("ScaledELU", ScaledELU, &moderate);
    bench_inplace("SoftSignSteep", SoftSignSteep, &moderate);
    bench_inplace("SReLU", SReLU, &moderate);
    bench_inplace("SReLUShifted", SReLUShifted, &moderate);
    bench_inplace("TanH", TanH, &moderate);
    bench_inplace("Sine", Sine, &moderate);
    bench_inplace("Gaussian", Gaussian, &moderate);

    println!();
    println!("-- wide range [-256, 256] (exercises exp scaling) --");
    println!();
    println!("  [runtime dispatch via ActivationFn enum]");
    for fn_ in [
        ActivationFn::Logistic,
        ActivationFn::LogisticSteep,
        ActivationFn::TanH,
        ActivationFn::ScaledELU,
        ActivationFn::Gaussian,
    ] {
        bench_inplace(fn_.code(), fn_, &wide);
    }

    println!();
    println!("  [monomorphised via concrete unit struct]");
    bench_inplace("Logistic", Logistic, &wide);
    bench_inplace("LogisticSteep", LogisticSteep, &wide);
    bench_inplace("TanH", TanH, &wide);
    bench_inplace("ScaledELU", ScaledELU, &wide);
    bench_inplace("Gaussian", Gaussian, &wide);
}
