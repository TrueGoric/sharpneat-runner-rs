//! Inference-only runtime for neural networks trained by [SharpNeat](https://github.com/colgreen/sharpneat).
//!
//! SharpNeat is a C# genetic algorithm library that evolves neural network topologies. This crate
//! implements the *inference* side only: it loads a trained network from SharpNeat's `.net` file
//! format and runs forward (activation) passes over it. No training functionality is provided.
//!
//! # What is supported
//!
//! - **Acyclic networks** — activated layer-by-layer using a depth schedule computed from the graph
//!   topology (see [`net::NeuralNetAcyclic`]).
//! - **Cyclic networks** — activated by a fixed number of relaxation timesteps per call (see
//!   [`net::NeuralNetCyclic`]).
//! - **Activation functions** — the standard SharpNeat neuron activation functions, exposed as a
//!   trait ([`Activation`]) with one zero-sized unit struct per function ([`Logistic`], [`ReLU`],
//!   …) plus an [`ActivationFn`] enum for runtime dispatch. SIMD vectorised hot paths via
//!   `portable_simd` (see [`activation`]).
//! - **Net file IO** — parsing and writing the human readable `.net` format produced by SharpNeat's
//!   `NetFile.Load` / `NetFile.Save` (see [`io::NetFile`]).
//!
//! # Quick start
//!
//! Load a `.net` file (runtime-dispatched activation function):
//!
//! ```
//! use sharpneat_runner_rs::{Net, NeuralNet, io::NetFile};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let src = "3 2\n\nacyclic\n\n0 3 0.5\n1 3 -0.5\n2 4 1.0\n1 4 0.25\n\n0 Logistic\n";
//! let model = NetFile::read_from_str(src)?;
//! let mut net = Net::from_model(&model)?;
//!
//! net.inputs_mut().copy_from_slice(&[1.0, 0.5, -0.5]);
//! net.activate();
//! let outputs = net.outputs();
//! assert_eq!(outputs.len(), 2);
//! # Ok(())
//! # }
//! ```
//!
//! Or construct a network with a concrete, monomorphised activation function:
//!
//! ```
//! use sharpneat_runner_rs::{
//!     Activation, Logistic, NeuralNet, NeuralNetAcyclic,
//!     graph::{WeightedDirectedConnection, WeightedDirectedGraph},
//!     graph::acyclic::build_weighted_directed_graph_acyclic,
//! };
//!
//! let conns = vec![
//!     WeightedDirectedConnection { src_id: 0, tgt_id: 2, weight: 0.5 },
//!     WeightedDirectedConnection { src_id: 1, tgt_id: 2, weight: -0.5 },
//!     WeightedDirectedConnection { src_id: 2, tgt_id: 3, weight: 1.0 },
//! ];
//! let graph = build_weighted_directed_graph_acyclic(
//!     WeightedDirectedGraph::build(conns, 2, 1),
//! );
//! let mut net = NeuralNetAcyclic::new(graph, Logistic);
//! net.inputs_mut().copy_from_slice(&[1.0, 1.0]);
//! net.activate();
//! assert_eq!(net.outputs().len(), 1);
//! assert_eq!(Logistic.code(), "Logistic");
//! ```
//!
//! # Design notes
//!
//! - All computations use `f64`, matching SharpNeat's double-precision net files.
//! - SIMD lanes are fixed at four `f64` values (256-bit on x86-64). The `portable_simd` feature
//!   requires a nightly compiler, pinned via `rust-toolchain.toml`.
//! - No `unsafe` code is used anywhere in the crate.

#![feature(portable_simd)]

pub mod activation;
pub mod builder;
pub mod graph;
pub mod io;
pub mod net;

pub use activation::{
    Activation, ActivationFn, ArcSinH, ArcTan, Gaussian, LeakyReLU, LeakyReLUShifted, Logistic,
    LogisticSteep, MaxMinusOne, NullFn, PolynomialApproximantSteep, QuadraticSigmoid, ReLU, SReLU,
    SReLUShifted, ScaledELU, Sine, SoftSignSteep, TanH,
};
pub use builder::{Net, build_from_model};
pub use io::{NetFile, NetFileError, NetFileModel};
pub use net::{NeuralNet, NeuralNetAcyclic, NeuralNetCyclic};
