//! Micro-benchmarks for the activation functions.
//!
//! Zero-dependency harness: each scenario is run in a loop for a fixed wall-clock budget and the
//! mean time per call is reported. Run with `cargo bench --bench activation_bench` (release mode
//! is the default for the `bench` profile).

use std::hint::black_box;
use std::time::{Duration, Instant};

use sharpneat_runner_rs::ActivationFn;

const BUFFER_LEN: usize = 1024;
const WARMUP: Duration = Duration::from_millis(200);
const MEASURE: Duration = Duration::from_secs(2);

fn bench_inplace(name: &str, fn_: ActivationFn, input: &[f64]) {
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
        "{name:<30} {calls:>10} calls | {per_call_ns:>10.2} ns/call | {per_element_ns:>8.3} ns/elem"
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
    for fn_ in [
        ActivationFn::Logistic,
        ActivationFn::LogisticSteep,
        ActivationFn::TanH,
        ActivationFn::ReLU,
        ActivationFn::LeakyReLU,
        ActivationFn::ScaledELU,
        ActivationFn::SoftSignSteep,
        ActivationFn::PolynomialApproximantSteep,
        ActivationFn::QuadraticSigmoid,
        ActivationFn::SReLU,
        ActivationFn::Gaussian,
        ActivationFn::Sine,
        ActivationFn::ArcTan,
        ActivationFn::ArcSinH,
        ActivationFn::NullFn,
        ActivationFn::MaxMinusOne,
        ActivationFn::LeakyReLUShifted,
        ActivationFn::SReLUShifted,
    ] {
        bench_inplace(fn_.code(), fn_, &moderate);
    }

    println!();
    println!("-- wide range [-256, 256] (exercises exp scaling) --");
    for fn_ in [
        ActivationFn::Logistic,
        ActivationFn::LogisticSteep,
        ActivationFn::TanH,
        ActivationFn::ScaledELU,
        ActivationFn::Gaussian,
    ] {
        bench_inplace(fn_.code(), fn_, &wide);
    }
}
