//! Acyclic neural network activation.
//!
//! The network is activated one layer at a time. For each layer the connections whose *source*
//! node is in that layer are processed in order: each connection's source post-activation value is
//! multiplied by its weight and accumulated into its target node's pre-activation slot. Once all of
//! a layer's connections have been processed, the activation function is applied to the *next*
//! layer's nodes — those are exactly the nodes whose every incoming connection has now been
//! accounted for. The schedule comes from the [`LayerInfo`](crate::graph::acyclic::LayerInfo)
//! array built in [`crate::graph::acyclic`].
//!
//! Vectorisation follows SharpNeat's `NeuralNetAcyclic.cs`: connections are processed in chunks of
//! [`LANES`](crate::activation::LANES); the source signals and weights are loaded into SIMD
//! vectors and multiplied together, then the four results are scattered back into the targets'
//! accumulators one at a time (the targets are generally not contiguous).

use crate::activation::Activation;
use crate::activation::LANES;
use crate::graph::acyclic::{LayerInfo, WeightedDirectedGraphAcyclic};
use crate::net::NeuralNet;
use std::simd::Simd;

/// An acyclic SharpNeat neural network, generic over the neuron activation function.
///
/// Construct with [`NeuralNetAcyclic::new`] from a [`WeightedDirectedGraphAcyclic`] and any
/// [`Activation`] function. When `A` is a concrete unit struct (e.g. [`Logistic`](crate::activation::Logistic))
/// the activation hot path is monomorphised and inlined; when `A` is
/// [`ActivationFn`](crate::activation::ActivationFn) the function is selected at runtime via a
/// `match`. Inputs are written into the first `input_count` slots of the activation buffer via
/// [`NeuralNet::inputs_mut`]; after [`NeuralNet::activate`] the outputs are available in a
/// separate contiguous slice via [`NeuralNet::outputs`].
#[derive(Debug)]
pub struct NeuralNetAcyclic<A: Activation> {
    src_ids: Vec<usize>,
    tgt_ids: Vec<usize>,
    weights: Vec<f64>,
    layer_info: Vec<LayerInfo>,
    activation_fn: A,
    /// Holds `total_node_count` activation values followed by `output_count` output slots.
    working_arr: Vec<f64>,
    total_node_count: usize,
    input_count: usize,
    output_count: usize,
    /// For each output index, the index in `working_arr[..total_node_count]` holding that output's
    /// post-activation value.
    output_node_idx: Vec<usize>,
}

impl<A: Activation> NeuralNetAcyclic<A> {
    /// Build a runnable acyclic network from a weighted acyclic graph and an activation function.
    pub fn new(graph: WeightedDirectedGraphAcyclic, activation_fn: A) -> Self {
        let digraph = graph.digraph;
        let total = digraph.total_node_count;
        let output_count = digraph.output_count;
        Self {
            src_ids: digraph.conn_ids.src_ids,
            tgt_ids: digraph.conn_ids.tgt_ids,
            weights: graph.weights,
            layer_info: digraph.layer_array,
            activation_fn,
            working_arr: vec![0.0; total + output_count],
            total_node_count: total,
            input_count: digraph.input_count,
            output_count,
            output_node_idx: digraph.output_node_idx_arr,
        }
    }

    /// Number of hidden + output layers, i.e. the number of activation passes per `activate`.
    pub fn layer_count(&self) -> usize {
        self.layer_info.len()
    }

    /// The activation function applied at every non-input neuron.
    pub fn activation_fn(&self) -> &A {
        &self.activation_fn
    }
}

impl<A: Activation> NeuralNet for NeuralNetAcyclic<A> {
    fn input_count(&self) -> usize {
        self.input_count
    }

    fn output_count(&self) -> usize {
        self.output_count
    }

    fn inputs_mut(&mut self) -> &mut [f64] {
        &mut self.working_arr[..self.input_count]
    }

    fn outputs(&self) -> &[f64] {
        let start = self.total_node_count;
        &self.working_arr[start..start + self.output_count]
    }

    fn activate(&mut self) {
        let Self {
            src_ids,
            tgt_ids,
            weights,
            layer_info,
            activation_fn,
            working_arr,
            total_node_count,
            input_count,
            output_node_idx,
            ..
        } = self;

        let total = *total_node_count;
        let inp = *input_count;
        let (activations, outputs) = working_arr.split_at_mut(total);
        let activations = &mut activations[..total];

        // Reset hidden and output pre-activation accumulators. Input slots are preserved (they hold
        // the caller's input values) and the output segment is overwritten below.
        for a in &mut activations[inp..] {
            *a = 0.0;
        }

        let mut con_idx = 0usize;
        let mut node_idx = inp;

        // Process every layer except the last (the last layer holds only nodes; its connections
        // were handled by the previous layer's pass).
        for layer_idx in 0..layer_info.len().saturating_sub(1) {
            let end_con = layer_info[layer_idx].end_connection_idx;

            // Vectorised chunk: gather LANES source signals, SIMD-multiply by weights, scatter.
            while con_idx + LANES <= end_con {
                let mut tmp = [0.0f64; LANES];
                for k in 0..LANES {
                    tmp[k] = activations[src_ids[con_idx + k]];
                }
                let src_vals = Simd::from_array(tmp);
                let w = Simd::from_slice(&weights[con_idx..con_idx + LANES]);
                let prod = src_vals * w;
                let arr = prod.to_array();
                for k in 0..LANES {
                    let t = tgt_ids[con_idx + k];
                    activations[t] += arr[k];
                }
                con_idx += LANES;
            }

            // Scalar tail with fused multiply-add.
            while con_idx < end_con {
                let s = src_ids[con_idx];
                let t = tgt_ids[con_idx];
                let w = weights[con_idx];
                activations[t] = activations[s].mul_add(w, activations[t]);
                con_idx += 1;
            }

            // Activate the next layer's nodes — their incoming connections are all processed.
            let next_end = layer_info[layer_idx + 1].end_node_idx;
            activation_fn.activate_inplace(&mut activations[node_idx..next_end]);
            node_idx = next_end;
        }

        // Gather the (possibly scattered) output node signals into the contiguous output segment.
        for (i, &nidx) in output_node_idx.iter().enumerate() {
            outputs[i] = activations[nidx];
        }
    }

    fn reset(&mut self) {
        // Acyclic activation fully overwrites the hidden/output state each call, so there is no
        // persistent state to clear. Inputs are owned by the caller and left untouched.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::{Activation, ActivationFn, Logistic, ReLU};
    use crate::graph::WeightedDirectedConnection;
    use crate::graph::WeightedDirectedGraph;
    use crate::graph::acyclic::build_weighted_directed_graph_acyclic;

    /// Build a small acyclic net: 3 inputs, 2 outputs, one hidden layer of 4 nodes.
    /// Topology from SharpNeat's `example1.net` (all sources are inputs, so a single hidden layer).
    fn build<A: Activation>(activation_fn: A) -> NeuralNetAcyclic<A> {
        let conns = vec![
            WeightedDirectedConnection {
                src_id: 0,
                tgt_id: 5,
                weight: 0.5,
            },
            WeightedDirectedConnection {
                src_id: 0,
                tgt_id: 7,
                weight: 0.7,
            },
            WeightedDirectedConnection {
                src_id: 0,
                tgt_id: 3,
                weight: 0.3,
            },
            WeightedDirectedConnection {
                src_id: 1,
                tgt_id: 5,
                weight: 1.5,
            },
            WeightedDirectedConnection {
                src_id: 1,
                tgt_id: 7,
                weight: 1.7,
            },
            WeightedDirectedConnection {
                src_id: 1,
                tgt_id: 3,
                weight: 1.3,
            },
            WeightedDirectedConnection {
                src_id: 1,
                tgt_id: 6,
                weight: 1.6,
            },
            WeightedDirectedConnection {
                src_id: 1,
                tgt_id: 8,
                weight: 1.8,
            },
            WeightedDirectedConnection {
                src_id: 1,
                tgt_id: 4,
                weight: 1.4,
            },
            WeightedDirectedConnection {
                src_id: 2,
                tgt_id: 6,
                weight: 2.6,
            },
            WeightedDirectedConnection {
                src_id: 2,
                tgt_id: 8,
                weight: 2.8,
            },
            WeightedDirectedConnection {
                src_id: 2,
                tgt_id: 4,
                weight: 2.4,
            },
        ];
        let g = WeightedDirectedGraph::build(conns, 3, 2);
        let acyclic = build_weighted_directed_graph_acyclic(g);
        NeuralNetAcyclic::new(acyclic, activation_fn)
    }

    #[test]
    fn input_output_counts() {
        let mut net = build(ActivationFn::ReLU);
        assert_eq!(net.input_count(), 3);
        assert_eq!(net.output_count(), 2);
        assert_eq!(net.inputs_mut().len(), 3);
        assert_eq!(net.outputs().len(), 2);
    }

    #[test]
    fn relu_zero_inputs_give_zero_outputs() {
        let mut net = build(ActivationFn::ReLU);
        net.inputs_mut().fill(0.0);
        net.activate();
        assert_eq!(net.outputs(), &[0.0, 0.0]);
    }

    #[test]
    fn activate_matches_reference_relu() {
        // Hand-computed reference: each hidden/output node's pre-activation is the sum of
        // (input * weight) over its incoming connections, then ReLU.
        let mut net = build(ActivationFn::ReLU);
        net.inputs_mut().copy_from_slice(&[1.0, 2.0, 3.0]);
        net.activate();

        // Inputs 1,2,3. Node 3 (output 0): 1*0.3 + 2*1.3 = 2.9 -> ReLU 2.9
        // Node 4 (output 1): 2*1.4 + 3*2.4 = 2.8 + 7.2 = 10.0 -> ReLU 10.0
        let expected = [2.9f64, 10.0];
        let got = net.outputs();
        for (i, (g, e)) in got.iter().zip(expected).enumerate() {
            assert!((g - e).abs() < 1e-12, "output[{i}] = {g}, expected {e}");
        }
    }

    #[test]
    fn activate_matches_reference_logistic() {
        let mut net = build(ActivationFn::Logistic);
        net.inputs_mut().copy_from_slice(&[1.0, 2.0, 3.0]);
        net.activate();

        let pre_3 = 1.0 * 0.3 + 2.0 * 1.3; // 2.9
        let pre_4 = 2.0 * 1.4 + 3.0 * 2.4; // 10.0
        let expected = [
            1.0 / (1.0 + f64::exp(-pre_3)),
            1.0 / (1.0 + f64::exp(-pre_4)),
        ];
        let got = net.outputs();
        for (i, (g, e)) in got.iter().zip(expected).enumerate() {
            assert!((g - e).abs() < 1e-9, "output[{i}] = {g}, expected {e}");
        }
    }

    #[test]
    fn deeper_network_uses_two_activation_passes() {
        // 0 -> 3 -> 4, inputs {0,1,2}, outputs {3,4} (output 4 depends on hidden 3).
        let conns = vec![
            WeightedDirectedConnection {
                src_id: 0,
                tgt_id: 3,
                weight: 2.0,
            },
            WeightedDirectedConnection {
                src_id: 3,
                tgt_id: 4,
                weight: 0.5,
            },
        ];
        let g = WeightedDirectedGraph::build(conns, 3, 2);
        let acyclic = build_weighted_directed_graph_acyclic(g);
        let mut net = NeuralNetAcyclic::new(acyclic, ActivationFn::ReLU);
        // 3 layers => 2 activation passes (layer 0 -> layer 1 -> layer 2).
        assert_eq!(net.layer_count(), 3);
        net.inputs_mut().copy_from_slice(&[5.0, 0.0, 0.0]);
        net.activate();
        // Node 3: ReLU(5*2) = 10. Node 4: ReLU(10*0.5) = 5.
        // Output 0 is old node 3, output 1 is old node 4.
        let outs = net.outputs();
        let (o0, o1) = (outs[0], outs[1]);
        // One of the outputs is 10, the other 5 (order depends on which output index maps where).
        let mut got = vec![o0, o1];
        got.sort_by(|a, b| a.total_cmp(b));
        assert!((got[0] - 5.0).abs() < 1e-12, "got {got:?}");
        assert!((got[1] - 10.0).abs() < 1e-12, "got {got:?}");
    }

    #[test]
    fn reset_is_safe_to_call() {
        let mut net = build(ActivationFn::ReLU);
        net.inputs_mut().fill(1.0);
        net.activate();
        net.reset(); // should be a no-op and not panic
        net.activate();
        assert_eq!(net.outputs().len(), 2);
    }

    #[test]
    fn concrete_activation_type_matches_enum_dispatch() {
        // The same network built with the concrete `ReLU` unit struct must produce identical
        // outputs to the runtime-dispatched `ActivationFn::ReLU` variant.
        let inputs = [1.0, 2.0, 3.0];

        let mut enum_net = build(ActivationFn::ReLU);
        enum_net.inputs_mut().copy_from_slice(&inputs);
        enum_net.activate();

        let mut concrete_net = build(ReLU);
        concrete_net.inputs_mut().copy_from_slice(&inputs);
        concrete_net.activate();

        assert_eq!(enum_net.outputs(), concrete_net.outputs());
    }

    #[test]
    fn concrete_logistic_type_works() {
        let mut net = build(Logistic);
        net.inputs_mut().copy_from_slice(&[1.0, 2.0, 3.0]);
        net.activate();
        let pre_3 = 1.0 * 0.3 + 2.0 * 1.3;
        let pre_4 = 2.0 * 1.4 + 3.0 * 2.4;
        let expected = [
            1.0 / (1.0 + f64::exp(-pre_3)),
            1.0 / (1.0 + f64::exp(-pre_4)),
        ];
        for (i, (g, e)) in net.outputs().iter().zip(expected).enumerate() {
            assert!((g - e).abs() < 1e-9, "output[{i}] = {g}, expected {e}");
        }
    }

    #[test]
    fn activation_fn_accessor_returns_the_function() {
        let net = build(ReLU);
        assert_eq!(net.activation_fn().code(), "ReLU");
    }
}
