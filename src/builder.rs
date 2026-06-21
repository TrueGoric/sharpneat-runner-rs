//! High-level convenience: build a runnable [`Net`] from a [`NetFileModel`](crate::io::NetFileModel).
//!
//! This module bridges the parsed file format and the runtime types. It maps the model's
//! activation function code to an [`ActivationFn`], constructs the appropriate weighted directed
//! graph, and — for acyclic networks — runs the depth analysis and layer scheduling. The result is
//! a single [`Net`] enum that can be activated without the caller caring whether the underlying
//! network is cyclic or acyclic.

use crate::activation::ActivationFn;
use crate::graph::acyclic::build_weighted_directed_graph_acyclic;
use crate::graph::{WeightedDirectedConnection, WeightedDirectedGraph};
use crate::io::{NetFileError, NetFileModel};
use crate::net::{NeuralNet, NeuralNetAcyclic, NeuralNetCyclic};

/// A runnable network, either acyclic or cyclic.
///
/// Both variants use [`ActivationFn`](crate::activation::ActivationFn) (runtime dispatch) because
/// the function is selected from the `.net` file's string code at runtime. When the activation
/// function is known at compile time, construct [`NeuralNetAcyclic`] / [`NeuralNetCyclic`]
/// directly with a concrete [`Activation`](crate::activation::Activation) type for a monomorphised
/// hot path.
///
/// Construct with [`build_from_model`] (or [`Net::from_model`]) from a [`NetFileModel`]. Both
/// variants implement [`NeuralNet`]; this enum forwards through to the active variant so callers
/// can work with a single concrete type.
#[derive(Debug)]
pub enum Net {
    Acyclic(NeuralNetAcyclic<ActivationFn>),
    Cyclic(NeuralNetCyclic<ActivationFn>),
}

impl Net {
    /// Build a [`Net`] from a parsed `.net` file model.
    pub fn from_model(model: &NetFileModel) -> Result<Self, NetFileError> {
        build_from_model(model)
    }
}

impl NeuralNet for Net {
    fn input_count(&self) -> usize {
        match self {
            Self::Acyclic(n) => n.input_count(),
            Self::Cyclic(n) => n.input_count(),
        }
    }

    fn output_count(&self) -> usize {
        match self {
            Self::Acyclic(n) => n.output_count(),
            Self::Cyclic(n) => n.output_count(),
        }
    }

    fn inputs_mut(&mut self) -> &mut [f64] {
        match self {
            Self::Acyclic(n) => n.inputs_mut(),
            Self::Cyclic(n) => n.inputs_mut(),
        }
    }

    fn outputs(&self) -> &[f64] {
        match self {
            Self::Acyclic(n) => n.outputs(),
            Self::Cyclic(n) => n.outputs(),
        }
    }

    fn activate(&mut self) {
        match self {
            Self::Acyclic(n) => n.activate(),
            Self::Cyclic(n) => n.activate(),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Acyclic(n) => n.reset(),
            Self::Cyclic(n) => n.reset(),
        }
    }
}

/// Build a runnable network from a parsed `.net` file model.
///
/// The first activation function (ID 0) determines the neuron function used at every non-input
/// node, matching SharpNeat's current behaviour. Acyclic models are scheduled into layers;
/// cyclic models use the model's `cycles_per_activation`.
pub fn build_from_model(model: &NetFileModel) -> Result<Net, NetFileError> {
    let code = model
        .activation_fns
        .first()
        .expect("NetFileModel guarantees at least one activation function")
        .code
        .as_str();
    let activation_fn =
        ActivationFn::from_code(code).ok_or_else(|| NetFileError::UnknownActivationCode {
            code: code.to_string(),
        })?;

    let connections: Vec<WeightedDirectedConnection> = model
        .connections
        .iter()
        .map(|c| WeightedDirectedConnection {
            src_id: c.source_id,
            tgt_id: c.target_id,
            weight: c.weight,
        })
        .collect();

    let weighted = WeightedDirectedGraph::build(connections, model.input_count, model.output_count);

    let net = if model.is_acyclic {
        let acyclic = build_weighted_directed_graph_acyclic(weighted);
        Net::Acyclic(NeuralNetAcyclic::new(acyclic, activation_fn))
    } else {
        Net::Cyclic(NeuralNetCyclic::new(
            weighted,
            activation_fn,
            model.cycles_per_activation,
        ))
    };
    Ok(net)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{ActivationFnLine, ConnectionLine};

    fn example1_connections() -> Vec<ConnectionLine> {
        vec![
            ConnectionLine::new(0, 5, 0.5),
            ConnectionLine::new(0, 7, 0.7),
            ConnectionLine::new(0, 3, 0.3),
            ConnectionLine::new(1, 5, 1.5),
            ConnectionLine::new(1, 7, 1.7),
            ConnectionLine::new(1, 3, 1.3),
            ConnectionLine::new(1, 6, 1.6),
            ConnectionLine::new(1, 8, 1.8),
            ConnectionLine::new(1, 4, 1.4),
            ConnectionLine::new(2, 6, 2.6),
            ConnectionLine::new(2, 8, 2.8),
            ConnectionLine::new(2, 4, 2.4),
        ]
    }

    #[test]
    fn builds_acyclic_net_from_model() {
        let model = NetFileModel::create_acyclic(
            3,
            2,
            example1_connections(),
            vec![ActivationFnLine::new(0, "ReLU")],
        )
        .unwrap();
        let mut net = build_from_model(&model).unwrap();
        assert!(matches!(net, Net::Acyclic(_)));
        net.inputs_mut().copy_from_slice(&[1.0, 2.0, 3.0]);
        net.activate();
        // ReLU; output 0 = max(0, 1*0.3 + 2*1.3) = 2.9; output 1 = max(0, 2*1.4 + 3*2.4) = 10.0.
        let outs = net.outputs();
        assert!((outs[0] - 2.9).abs() < 1e-12, "out0 = {}", outs[0]);
        assert!((outs[1] - 10.0).abs() < 1e-12, "out1 = {}", outs[1]);
    }

    #[test]
    fn unknown_activation_code_is_rejected() {
        let model = NetFileModel::create_acyclic(
            1,
            1,
            vec![ConnectionLine::new(0, 1, 1.0)],
            vec![ActivationFnLine::new(0, "BogusFn")],
        )
        .unwrap();
        let err = build_from_model(&model).unwrap_err();
        assert!(matches!(err, NetFileError::UnknownActivationCode { .. }));
    }

    #[test]
    fn builds_cyclic_net_from_model() {
        // Feedforward-only cyclic graph: one cycle propagates inputs to outputs.
        // pre[3] = 1*2 = 2; pre[4] = 2*3 + 3*5 = 21. ReLU -> (2, 21).
        let model = NetFileModel::create_cyclic(
            3,
            2,
            1,
            vec![
                ConnectionLine::new(0, 3, 2.0),
                ConnectionLine::new(1, 4, 3.0),
                ConnectionLine::new(2, 4, 5.0),
            ],
            vec![ActivationFnLine::new(0, "ReLU")],
        )
        .unwrap();
        let mut net = build_from_model(&model).unwrap();
        assert!(matches!(net, Net::Cyclic(_)));
        net.inputs_mut().copy_from_slice(&[1.0, 2.0, 3.0]);
        net.activate();
        let outs = net.outputs();
        assert!((outs[0] - 2.0).abs() < 1e-12, "out0 = {}", outs[0]);
        assert!((outs[1] - 21.0).abs() < 1e-12, "out1 = {}", outs[1]);
    }
}
