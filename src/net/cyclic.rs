//! Cyclic neural network activation.
//!
//! A cyclic network has no natural layer ordering, so activation is performed as a fixed number of
//! relaxation timesteps. Each timestep computes a fresh set of pre-activation values from the
//! previous timestep's post-activation values, applies the activation function to the non-input
//! nodes, and then zeroes the pre-activation accumulators ready for the next sweep. After the
//! configured number of cycles the post-activation slice holds the result.
//!
//! This mirrors SharpNeat's `NeuralNetCyclic.cs`. Input nodes are never activated — their
//! post-activation values are set by the caller and held constant across cycles. Output nodes
//! occupy `[input_count, input_count + output_count)` within the post-activation slice, so reading
//! [`NeuralNet::outputs`] immediately after [`NeuralNet::activate`] returns their relaxed state.

use crate::activation::Activation;
use crate::activation::LANES;
use crate::graph::{ConnectionIds, WeightedDirectedGraph};
use crate::net::NeuralNet;
use std::simd::Simd;

/// A cyclic SharpNeat neural network, generic over the neuron activation function.
///
/// Holds two parallel signal arrays — pre-activation and post-activation — of length
/// `total_node_count`. Inputs occupy the first `input_count` slots of the post-activation array;
/// outputs occupy the next `output_count` slots. When `A` is a concrete unit struct (e.g.
/// [`Logistic`](crate::activation::Logistic)) the activation hot path is monomorphised and
/// inlined; when `A` is [`ActivationFn`](crate::activation::ActivationFn) the function is selected
/// at runtime via a `match`.
#[derive(Debug)]
pub struct NeuralNetCyclic<A: Activation> {
    src_ids: Vec<usize>,
    tgt_ids: Vec<usize>,
    weights: Vec<f64>,
    activation_fn: A,
    /// `pre` then `post`, each of length `total_node_count`.
    activations: Vec<f64>,
    total_node_count: usize,
    input_count: usize,
    output_count: usize,
    cycles_per_activation: usize,
}

impl<A: Activation> NeuralNetCyclic<A> {
    /// Build a runnable cyclic network.
    ///
    /// `cycles_per_activation` is the number of relaxation timesteps performed per
    /// [`NeuralNet::activate`] call; it must be at least 1.
    pub fn new(
        graph: WeightedDirectedGraph,
        activation_fn: A,
        cycles_per_activation: usize,
    ) -> Self {
        assert!(
            cycles_per_activation >= 1,
            "cycles_per_activation must be at least 1"
        );
        let digraph = graph.digraph;
        let total = digraph.total_node_count;
        let ConnectionIds { src_ids, tgt_ids } = digraph.conn_ids;
        Self {
            src_ids,
            tgt_ids,
            weights: graph.weights,
            activation_fn,
            activations: vec![0.0; total * 2],
            total_node_count: total,
            input_count: digraph.input_count,
            output_count: digraph.output_count,
            cycles_per_activation,
        }
    }

    /// Number of relaxation timesteps per activation.
    pub fn cycles_per_activation(&self) -> usize {
        self.cycles_per_activation
    }

    /// The activation function applied at every non-input neuron.
    pub fn activation_fn(&self) -> &A {
        &self.activation_fn
    }
}

impl<A: Activation> NeuralNet for NeuralNetCyclic<A> {
    fn input_count(&self) -> usize {
        self.input_count
    }

    fn output_count(&self) -> usize {
        self.output_count
    }

    fn inputs_mut(&mut self) -> &mut [f64] {
        let total = self.total_node_count;
        let inp = self.input_count;
        &mut self.activations[total..total + inp]
    }

    fn outputs(&self) -> &[f64] {
        let total = self.total_node_count;
        let inp = self.input_count;
        &self.activations[total + inp..total + inp + self.output_count]
    }

    fn activate(&mut self) {
        let Self {
            src_ids,
            tgt_ids,
            weights,
            activation_fn,
            activations,
            total_node_count,
            input_count,
            cycles_per_activation,
            ..
        } = self;

        let total = *total_node_count;
        let inp = *input_count;
        let (pre, post) = activations.split_at_mut(total);
        let pre = &mut pre[..total];
        let post = &mut post[..total];

        for _ in 0..*cycles_per_activation {
            // Accumulate weighted signals from `post` into `pre` for every connection.
            let mut con_idx = 0usize;
            while con_idx + LANES <= src_ids.len() {
                let mut tmp = [0.0f64; LANES];
                for k in 0..LANES {
                    tmp[k] = post[src_ids[con_idx + k]];
                }
                let src_vals = Simd::from_array(tmp);
                let w = Simd::from_slice(&weights[con_idx..con_idx + LANES]);
                let prod = src_vals * w;
                let arr = prod.to_array();
                for k in 0..LANES {
                    let t = tgt_ids[con_idx + k];
                    pre[t] += arr[k];
                }
                con_idx += LANES;
            }
            while con_idx < src_ids.len() {
                let s = src_ids[con_idx];
                let t = tgt_ids[con_idx];
                let w = weights[con_idx];
                pre[t] = post[s].mul_add(w, pre[t]);
                con_idx += 1;
            }

            // Activate the non-input nodes, writing post-activation values back into `post`.
            // Inputs keep their caller-supplied values (they are in `post[..inp]`, not touched).
            activation_fn.activate_into(&pre[inp..total], &mut post[inp..total]);

            // Reset the pre-activation accumulators for the next sweep.
            for x in &mut pre[inp..total] {
                *x = 0.0;
            }
        }
    }

    fn reset(&mut self) {
        let total = self.total_node_count;
        let inp = self.input_count;
        let (pre, post) = self.activations.split_at_mut(total);
        for x in &mut pre[inp..total] {
            *x = 0.0;
        }
        for x in &mut post[inp..total] {
            *x = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::{Activation, ActivationFn, Logistic, ReLU};
    use crate::graph::WeightedDirectedConnection;

    /// A small cyclic graph with a feedback loop, matching SharpNeat's `example3.net`.
    /// Inputs {0,1,2}, outputs {3,4}, hidden {5,6,7}.
    fn build<A: Activation>(activation_fn: A, cycles: usize) -> NeuralNetCyclic<A> {
        let conns = vec![
            WeightedDirectedConnection {
                src_id: 0,
                tgt_id: 5,
                weight: 3.0,
            },
            WeightedDirectedConnection {
                src_id: 1,
                tgt_id: 5,
                weight: 5.0,
            },
            WeightedDirectedConnection {
                src_id: 1,
                tgt_id: 6,
                weight: 7.0,
            },
            WeightedDirectedConnection {
                src_id: 2,
                tgt_id: 6,
                weight: 11.0,
            },
            WeightedDirectedConnection {
                src_id: 5,
                tgt_id: 6,
                weight: 13.0,
            },
            WeightedDirectedConnection {
                src_id: 6,
                tgt_id: 7,
                weight: 17.0,
            },
            WeightedDirectedConnection {
                src_id: 7,
                tgt_id: 5,
                weight: 19.0,
            },
            WeightedDirectedConnection {
                src_id: 7,
                tgt_id: 3,
                weight: 23.0,
            },
            WeightedDirectedConnection {
                src_id: 6,
                tgt_id: 4,
                weight: 29.0,
            },
        ];
        let g = WeightedDirectedGraph::build(conns, 3, 2);
        NeuralNetCyclic::new(g, activation_fn, cycles)
    }

    /// A feedforward-only cyclic graph (no feedback) where a single cycle fully propagates signals
    /// to the outputs. Inputs {0,1,2}, outputs {3,4}.
    fn feedforward<A: Activation>(activation_fn: A, cycles: usize) -> NeuralNetCyclic<A> {
        let conns = vec![
            WeightedDirectedConnection {
                src_id: 0,
                tgt_id: 3,
                weight: 2.0,
            },
            WeightedDirectedConnection {
                src_id: 1,
                tgt_id: 4,
                weight: 3.0,
            },
            WeightedDirectedConnection {
                src_id: 2,
                tgt_id: 4,
                weight: 5.0,
            },
        ];
        let g = WeightedDirectedGraph::build(conns, 3, 2);
        NeuralNetCyclic::new(g, activation_fn, cycles)
    }

    #[test]
    fn counts_and_accessors() {
        let mut net = build(ActivationFn::ReLU, 1);
        assert_eq!(net.input_count(), 3);
        assert_eq!(net.output_count(), 2);
        assert_eq!(net.inputs_mut().len(), 3);
        assert_eq!(net.outputs().len(), 2);
        assert_eq!(net.cycles_per_activation(), 1);
    }

    #[test]
    fn single_cycle_matches_reference() {
        // Feedforward-only graph: one cycle suffices to propagate inputs to outputs.
        // pre[3] = 1*2 = 2; pre[4] = 2*3 + 3*5 = 21. ReLU -> (2, 21).
        let mut net = feedforward(ActivationFn::ReLU, 1);
        net.inputs_mut().copy_from_slice(&[1.0, 2.0, 3.0]);
        net.activate();
        let outs = net.outputs();
        assert!((outs[0] - 2.0).abs() < 1e-12, "out0 = {}", outs[0]);
        assert!((outs[1] - 21.0).abs() < 1e-12, "out1 = {}", outs[1]);
    }

    #[test]
    fn two_cycles_propagate_through_hidden() {
        // 0 -> 5 (w=2), 5 -> 1 (w=3). Node 1 is the output (input_count=1, so output is node 1).
        // Output 1 depends on hidden 5, so it needs a second cycle.
        let conns = vec![
            WeightedDirectedConnection {
                src_id: 0,
                tgt_id: 5,
                weight: 2.0,
            },
            WeightedDirectedConnection {
                src_id: 5,
                tgt_id: 1,
                weight: 3.0,
            },
        ];
        let g = WeightedDirectedGraph::build(conns, 1, 1);
        let mut net = NeuralNetCyclic::new(g, ActivationFn::ReLU, 2);
        net.inputs_mut()[0] = 1.0;
        net.activate();
        // Cycle 1: pre[5]=2, pre[1]=0 (post[5] starts 0). post[5]=2, post[1]=0.
        // Cycle 2: pre[5]=2, pre[1]=post[5]*3=6. post[1]=6.
        assert!(
            (net.outputs()[0] - 6.0).abs() < 1e-12,
            "out0 = {}",
            net.outputs()[0]
        );
    }

    #[test]
    fn zero_inputs_give_zero_outputs_with_relu() {
        let mut net = build(ActivationFn::ReLU, 3);
        net.inputs_mut().fill(0.0);
        net.reset();
        net.activate();
        assert_eq!(net.outputs(), &[0.0, 0.0]);
    }

    #[test]
    fn reset_clears_state_between_activations() {
        let mut net = build(ActivationFn::ReLU, 2);
        net.inputs_mut().copy_from_slice(&[1.0, 1.0, 1.0]);
        net.activate();
        // Without reset the post-activation state persists; with reset it is cleared.
        net.reset();
        net.inputs_mut().fill(0.0);
        net.activate();
        assert_eq!(net.outputs(), &[0.0, 0.0]);
    }

    #[test]
    fn more_cycles_can_change_outputs() {
        // Logistic squashes values into (0,1); extra cycles let feedback propagate. The two-cycle
        // result should differ from the one-cycle result for this feedback-rich graph.
        let mut a = build(ActivationFn::Logistic, 1);
        a.inputs_mut().copy_from_slice(&[0.5, 0.5, 0.5]);
        a.activate();
        let oa = a.outputs().to_vec();

        let mut b = build(ActivationFn::Logistic, 5);
        b.inputs_mut().copy_from_slice(&[0.5, 0.5, 0.5]);
        b.activate();
        let ob = b.outputs().to_vec();

        assert!(
            (oa[0] - ob[0]).abs() > 1e-6 || (oa[1] - ob[1]).abs() > 1e-6,
            "cycles should change output: {oa:?} vs {ob:?}"
        );
    }

    #[test]
    fn concrete_activation_type_matches_enum_dispatch() {
        let inputs = [1.0, 2.0, 3.0];

        let mut enum_net = feedforward(ActivationFn::ReLU, 1);
        enum_net.inputs_mut().copy_from_slice(&inputs);
        enum_net.activate();

        let mut concrete_net = feedforward(ReLU, 1);
        concrete_net.inputs_mut().copy_from_slice(&inputs);
        concrete_net.activate();

        assert_eq!(enum_net.outputs(), concrete_net.outputs());
    }

    #[test]
    fn concrete_logistic_type_works() {
        let mut net = feedforward(Logistic, 1);
        net.inputs_mut().copy_from_slice(&[1.0, 2.0, 3.0]);
        net.activate();
        // pre[3] = 2; pre[4] = 21. Logistic -> (1/(1+e^-2), 1/(1+e^-21)).
        let expected = [1.0 / (1.0 + f64::exp(-2.0)), 1.0 / (1.0 + f64::exp(-21.0))];
        for (i, (g, e)) in net.outputs().iter().zip(expected).enumerate() {
            assert!((g - e).abs() < 1e-9, "output[{i}] = {g}, expected {e}");
        }
    }

    #[test]
    fn activation_fn_accessor_returns_the_function() {
        let net = feedforward(ReLU, 1);
        assert_eq!(net.activation_fn().code(), "ReLU");
    }
}
