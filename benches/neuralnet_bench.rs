//! Micro-benchmarks for acyclic and cyclic network activation.
//!
//! Builds networks of varying size from synthetic connection lists and times
//! [`NeuralNet::activate`]. Zero-dependency harness, same approach as `activation_bench`.

use std::hint::black_box;
use std::time::{Duration, Instant};

use sharpneat_runner_rs::graph::WeightedDirectedConnection;
use sharpneat_runner_rs::graph::WeightedDirectedGraph;
use sharpneat_runner_rs::graph::acyclic::build_weighted_directed_graph_acyclic;
use sharpneat_runner_rs::{ActivationFn, NeuralNet, NeuralNetAcyclic, NeuralNetCyclic};

const WARMUP: Duration = Duration::from_millis(200);
const MEASURE: Duration = Duration::from_secs(2);

/// Generate a layered acyclic network: `inputs` input nodes, `layers` hidden layers each of
/// `width` nodes, fully connected between adjacent layers, feeding `outputs` output nodes.
fn layered_acyclic(inputs: usize, outputs: usize, layers: usize, width: usize) -> NeuralNetAcyclic {
    let mut conns = Vec::new();
    let mut prev_layer: Vec<usize> = (0..inputs).collect();
    let mut layer_start = inputs;
    for _ in 0..layers {
        let next_layer: Vec<usize> = (layer_start..layer_start + width).collect();
        for &s in &prev_layer {
            for &t in &next_layer {
                conns.push(WeightedDirectedConnection {
                    src_id: s,
                    tgt_id: t,
                    weight: 0.5,
                });
            }
        }
        layer_start += width;
        prev_layer = next_layer;
    }
    let output_ids: Vec<usize> = (layer_start..layer_start + outputs).collect();
    for &s in &prev_layer {
        for &t in &output_ids {
            conns.push(WeightedDirectedConnection {
                src_id: s,
                tgt_id: t,
                weight: 0.5,
            });
        }
    }
    let g = WeightedDirectedGraph::build(conns, inputs, outputs);
    let acyclic = build_weighted_directed_graph_acyclic(g);
    NeuralNetAcyclic::new(acyclic, ActivationFn::Logistic)
}

/// Generate a cyclic network with `width` hidden nodes in a recurrent ring plus feed-forward
/// connections from the inputs, matching the kind of topology SharpNeat evolves for cyclic nets.
fn ring_cyclic(inputs: usize, outputs: usize, width: usize) -> NeuralNetCyclic {
    let mut conns = Vec::new();
    let hidden: Vec<usize> = (inputs + outputs..inputs + outputs + width).collect();
    for s in 0..inputs {
        for &t in &hidden {
            conns.push(WeightedDirectedConnection {
                src_id: s,
                tgt_id: t,
                weight: 0.5,
            });
        }
    }
    for w in 0..width {
        let s = hidden[w];
        let t = hidden[(w + 1) % width];
        conns.push(WeightedDirectedConnection {
            src_id: s,
            tgt_id: t,
            weight: 0.25,
        });
    }
    for &s in &hidden {
        for o in 0..outputs {
            conns.push(WeightedDirectedConnection {
                src_id: s,
                tgt_id: inputs + o,
                weight: 0.5,
            });
        }
    }
    let g = WeightedDirectedGraph::build(conns, inputs, outputs);
    NeuralNetCyclic::new(g, ActivationFn::Logistic, 4)
}

/// Time `activate` on a pre-built network whose inputs have already been set.
fn bench<N: NeuralNet>(name: &str, mut net: N) {
    net.inputs_mut().fill(0.5);

    // Warm up.
    let start = Instant::now();
    while start.elapsed() < WARMUP {
        net.activate();
        black_box(net.outputs());
    }

    // Measure.
    let mut calls = 0u64;
    let start = Instant::now();
    while start.elapsed() < MEASURE {
        net.activate();
        black_box(net.outputs());
        calls += 1;
    }
    let elapsed = start.elapsed();
    let per_call_us = elapsed.as_nanos() as f64 / calls as f64 / 1000.0;
    println!("{name:<42} {calls:>8} calls | {per_call_us:>10.3} us/activation");
}

fn main() {
    println!("=== neural network activation (Logistic) ===");
    println!();

    println!("-- acyclic, layered, fully connected --");
    bench(
        "acyclic 6->16->4 (1 hidden layer)",
        layered_acyclic(6, 4, 1, 16),
    );
    bench(
        "acyclic 6->32x3->4 (3 hidden layers)",
        layered_acyclic(6, 4, 3, 32),
    );
    bench(
        "acyclic 12->64x4->8 (4 hidden layers)",
        layered_acyclic(12, 8, 4, 64),
    );

    println!();
    println!("-- cyclic, recurrent ring, 4 cycles/activation --");
    bench("cyclic 6->ring16->4 (4 cycles)", ring_cyclic(6, 4, 16));
    bench("cyclic 12->ring64->8 (4 cycles)", ring_cyclic(12, 8, 64));
}
