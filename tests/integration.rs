// SPDX-FileCopyrightText: 2026 Marcin Jędrasik
// SPDX-License-Identifier: MIT

//! End-to-end tests that load the SharpNeat `.net` fixture files, exercise the full
//! parse → build → activate pipeline, and verify file IO round-trips.

use std::path::PathBuf;

use sharpneat_runner_rs::{
    ActivationFn, Net, NeuralNet,
    io::{NetFile, NetFileError},
};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn sample(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("sample_genomes")
        .join(name)
}

#[test]
fn loads_example1_net() {
    let model = NetFile::load(fixture("example1.net")).unwrap();
    assert_eq!(model.input_count, 3);
    assert_eq!(model.output_count, 2);
    assert!(model.is_acyclic);
    assert_eq!(model.connections.len(), 12);
    assert_eq!(model.activation_fns.len(), 4);
    assert_eq!(model.activation_fns[0].code, "ReLU");
    // The per-node activation function section is ignored, as in SharpNeat.
}

#[test]
fn loads_example2_net() {
    let model = NetFile::load(fixture("example2.net")).unwrap();
    assert_eq!(model.input_count, 3);
    assert_eq!(model.output_count, 2);
    assert!(model.is_acyclic);
    assert_eq!(model.connections.len(), 6);
    // The high-precision weight must survive parsing unchanged.
    assert_eq!(model.connections[3].weight, 5.123456789);
}

#[test]
fn loads_example3_cyclic_net() {
    let model = NetFile::load(fixture("example3.net")).unwrap();
    assert!(!model.is_acyclic);
    assert_eq!(model.cycles_per_activation, 3);
    assert_eq!(model.connections.len(), 9);
}

#[test]
fn example1_runs_and_produces_two_outputs() {
    let model = NetFile::load(fixture("example1.net")).unwrap();
    let mut net = Net::from_model(&model).unwrap();
    net.inputs_mut().copy_from_slice(&[1.0, 2.0, 3.0]);
    net.activate();
    assert_eq!(net.outputs().len(), 2);
    // With ReLU as the first activation function, the outputs are non-negative.
    for &o in net.outputs() {
        assert!(o >= 0.0, "ReLU output should be non-negative, got {o}");
    }
}

#[test]
fn example1_relu_outputs_match_hand_computed_reference() {
    // The first activation function in example1.net is ReLU, applied at every node.
    let model = NetFile::load(fixture("example1.net")).unwrap();
    let mut net = Net::from_model(&model).unwrap();
    net.inputs_mut().copy_from_slice(&[1.0, 2.0, 3.0]);
    net.activate();
    // Output 0 (node 3) = ReLU(1*0.3 + 2*1.3) = ReLU(2.9) = 2.9
    // Output 1 (node 4) = ReLU(2*1.4 + 3*2.4) = ReLU(10.0) = 10.0
    assert!(
        (net.outputs()[0] - 2.9).abs() < 1e-12,
        "out0={}",
        net.outputs()[0]
    );
    assert!(
        (net.outputs()[1] - 10.0).abs() < 1e-12,
        "out1={}",
        net.outputs()[1]
    );
}

#[test]
fn example3_cyclic_runs_for_three_cycles() {
    let model = NetFile::load(fixture("example3.net")).unwrap();
    let mut net = Net::from_model(&model).unwrap();
    net.inputs_mut().copy_from_slice(&[1.0, 2.0, 3.0]);
    net.activate();
    assert_eq!(net.outputs().len(), 2);
    // The network is cyclic with feedback; after 3 cycles the result should be finite.
    for &o in net.outputs() {
        assert!(o.is_finite(), "output should be finite, got {o}");
    }
}

#[test]
fn net_file_round_trips_through_string() {
    let original = NetFile::load(fixture("example1.net")).unwrap();
    let text = NetFile::write_to_string(&original);
    let reparsed = NetFile::read_from_str(&text).unwrap();
    assert_eq!(reparsed.input_count, original.input_count);
    assert_eq!(reparsed.output_count, original.output_count);
    assert_eq!(reparsed.is_acyclic, original.is_acyclic);
    assert_eq!(reparsed.connections, original.connections);
    assert_eq!(reparsed.activation_fns, original.activation_fns);
}

#[test]
fn net_file_save_and_reload_from_disk() {
    let original = NetFile::load(fixture("example2.net")).unwrap();
    let tmp = std::env::temp_dir().join("sharpneat_runner_rs_roundtrip.net");
    NetFile::save(&tmp, &original).unwrap();
    let reparsed = NetFile::load(&tmp).unwrap();
    assert_eq!(reparsed, original);
    let _ = std::fs::remove_file(tmp);
}

#[test]
fn unknown_activation_code_in_file_is_reported() {
    let text = "1 1\n\nacyclic\n\n0 1 1.0\n\n0 NotARealFn\n";
    let model = NetFile::read_from_str(text).unwrap();
    let err = Net::from_model(&model).unwrap_err();
    assert!(matches!(err, NetFileError::UnknownActivationCode { .. }));
}

#[test]
fn activation_fn_code_round_trips_through_file() {
    let model = NetFile::load(fixture("example2.net")).unwrap();
    let net = Net::from_model(&model).unwrap();
    // The model uses ReLU; the builder should resolve it to the ReLU variant.
    // (Indirect check: a ReLU network with zero inputs yields zero outputs.)
    let mut net = net;
    net.reset();
    net.inputs_mut().fill(0.0);
    net.activate();
    assert_eq!(net.outputs(), &[0.0, 0.0]);
}

#[test]
fn cyclic_and_acyclic_models_build_to_the_right_variant() {
    let acyclic = NetFile::load(fixture("example1.net")).unwrap();
    let net = Net::from_model(&acyclic).unwrap();
    assert!(matches!(net, Net::Acyclic(_)));

    let cyclic = NetFile::load(fixture("example3.net")).unwrap();
    let net = Net::from_model(&cyclic).unwrap();
    assert!(matches!(net, Net::Cyclic(_)));
}

#[test]
fn activation_fn_from_code_covers_fixture_codes() {
    // example1.net emits ReLU, Logistic, Sine and Gaussian codes.
    for code in ["ReLU", "Logistic", "Sine", "Gaussian"] {
        assert!(ActivationFn::from_code(code).is_some(), "missing {code}");
    }
}

// ---------------------------------------------------------------------------
// sample_genomes/cf-acyclic.net — a large acyclic genome (87 inputs, 7 outputs,
// LeakyReLU).  The graph has a high max node ID (~20 k) with many unreachable
// and phantom nodes, which exercise the depth-analysis and layer-scheduling
// code paths that the small fixtures do not.
// ---------------------------------------------------------------------------

#[test]
fn loads_cf_acyclic_net() {
    let model = NetFile::load(sample("cf-acyclic.net")).unwrap();
    assert_eq!(model.input_count, 87);
    assert_eq!(model.output_count, 7);
    assert!(model.is_acyclic);
    assert_eq!(model.connections.len(), 1439);
    assert_eq!(model.activation_fns.len(), 1);
    assert_eq!(model.activation_fns[0].code, "LeakyReLU");
}

#[test]
fn cf_acyclic_builds_to_acyclic_variant() {
    let model = NetFile::load(sample("cf-acyclic.net")).unwrap();
    let net = Net::from_model(&model).unwrap();
    assert!(matches!(net, Net::Acyclic(_)));
    assert_eq!(net.input_count(), 87);
    assert_eq!(net.output_count(), 7);
}

#[test]
fn cf_acyclic_zero_inputs_give_zero_outputs() {
    // LeakyReLU(0) = 0, so with all-zero inputs every node settles to 0.
    let model = NetFile::load(sample("cf-acyclic.net")).unwrap();
    let mut net = Net::from_model(&model).unwrap();
    net.inputs_mut().fill(0.0);
    net.activate();
    assert_eq!(net.outputs(), &[0.0; 7]);
}

#[test]
fn cf_acyclic_produces_finite_outputs_with_unit_inputs() {
    let model = NetFile::load(sample("cf-acyclic.net")).unwrap();
    let mut net = Net::from_model(&model).unwrap();
    net.inputs_mut().fill(1.0);
    net.activate();
    assert_eq!(net.outputs().len(), 7);
    for (i, &o) in net.outputs().iter().enumerate() {
        assert!(o.is_finite(), "output[{i}] is not finite: {o}");
    }
}

#[test]
fn cf_acyclic_activation_is_deterministic() {
    let model = NetFile::load(sample("cf-acyclic.net")).unwrap();
    let mut net = Net::from_model(&model).unwrap();
    let inputs: Vec<f64> = (0..87).map(|i| (i as f64) * 0.01 - 0.43).collect();

    net.inputs_mut().copy_from_slice(&inputs);
    net.activate();
    let first = net.outputs().to_vec();

    // Re-run with the same inputs; acyclic nets have no persistent state.
    net.inputs_mut().copy_from_slice(&inputs);
    net.activate();
    let second = net.outputs().to_vec();

    for (i, (a, b)) in first.iter().zip(&second).enumerate() {
        assert!((a - b).abs() <= 1e-12, "output[{i}] differs: {a} vs {b}");
    }
}

#[test]
fn cf_acyclic_round_trips_through_string() {
    let original = NetFile::load(sample("cf-acyclic.net")).unwrap();
    let text = NetFile::write_to_string(&original);
    let reparsed = NetFile::read_from_str(&text).unwrap();
    assert_eq!(reparsed.input_count, original.input_count);
    assert_eq!(reparsed.output_count, original.output_count);
    assert_eq!(reparsed.is_acyclic, original.is_acyclic);
    assert_eq!(reparsed.connections, original.connections);
    assert_eq!(reparsed.activation_fns, original.activation_fns);
}

// ---------------------------------------------------------------------------
// sample_genomes/cf-cyclic.net — the cyclic counterpart (87 inputs, 7 outputs,
// 4 cycles per activation, LeakyReLU).
// ---------------------------------------------------------------------------

#[test]
fn loads_cf_cyclic_net() {
    let model = NetFile::load(sample("cf-cyclic.net")).unwrap();
    assert_eq!(model.input_count, 87);
    assert_eq!(model.output_count, 7);
    assert!(!model.is_acyclic);
    assert_eq!(model.cycles_per_activation, 4);
    assert_eq!(model.connections.len(), 1472);
    assert_eq!(model.activation_fns.len(), 1);
    assert_eq!(model.activation_fns[0].code, "LeakyReLU");
}

#[test]
fn cf_cyclic_builds_to_cyclic_variant() {
    let model = NetFile::load(sample("cf-cyclic.net")).unwrap();
    let net = Net::from_model(&model).unwrap();
    assert!(matches!(net, Net::Cyclic(_)));
    assert_eq!(net.input_count(), 87);
    assert_eq!(net.output_count(), 7);
}

#[test]
fn cf_cyclic_zero_inputs_give_zero_outputs() {
    let model = NetFile::load(sample("cf-cyclic.net")).unwrap();
    let mut net = Net::from_model(&model).unwrap();
    net.reset();
    net.inputs_mut().fill(0.0);
    net.activate();
    assert_eq!(net.outputs(), &[0.0; 7]);
}

#[test]
fn cf_cyclic_produces_finite_outputs_with_unit_inputs() {
    let model = NetFile::load(sample("cf-cyclic.net")).unwrap();
    let mut net = Net::from_model(&model).unwrap();
    net.reset();
    net.inputs_mut().fill(1.0);
    net.activate();
    assert_eq!(net.outputs().len(), 7);
    for (i, &o) in net.outputs().iter().enumerate() {
        assert!(o.is_finite(), "output[{i}] is not finite: {o}");
    }
}

#[test]
fn cf_cyclic_activation_is_deterministic_after_reset() {
    let model = NetFile::load(sample("cf-cyclic.net")).unwrap();
    let mut net = Net::from_model(&model).unwrap();
    let inputs: Vec<f64> = (0..87).map(|i| (i as f64) * 0.01 - 0.43).collect();

    net.reset();
    net.inputs_mut().copy_from_slice(&inputs);
    net.activate();
    let first = net.outputs().to_vec();

    net.reset();
    net.inputs_mut().copy_from_slice(&inputs);
    net.activate();
    let second = net.outputs().to_vec();

    for (i, (a, b)) in first.iter().zip(&second).enumerate() {
        assert!((a - b).abs() <= 1e-12, "output[{i}] differs: {a} vs {b}");
    }
}

#[test]
fn cf_cyclic_round_trips_through_string() {
    let original = NetFile::load(sample("cf-cyclic.net")).unwrap();
    let text = NetFile::write_to_string(&original);
    let reparsed = NetFile::read_from_str(&text).unwrap();
    assert_eq!(reparsed.input_count, original.input_count);
    assert_eq!(reparsed.output_count, original.output_count);
    assert_eq!(reparsed.is_acyclic, original.is_acyclic);
    assert_eq!(
        reparsed.cycles_per_activation,
        original.cycles_per_activation
    );
    assert_eq!(reparsed.connections, original.connections);
    assert_eq!(reparsed.activation_fns, original.activation_fns);
}

#[test]
fn cf_genomes_resolve_leaky_relu_activation_code() {
    assert!(ActivationFn::from_code("LeakyReLU").is_some());
}
