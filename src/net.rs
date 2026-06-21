//! Neural network runtime: the [`NeuralNet`] trait and its two implementations.
//!
//! - [`NeuralNetAcyclic`] activates a depth-scheduled acyclic graph in a single forward sweep.
//! - [`NeuralNetCyclic`] relaxes a cyclic graph for a fixed number of timesteps per activation.
//!
//! Both implementations mirror the vectorised activation loops in SharpNeat's
//! `NeuralNets/Vectorized/NeuralNet{Acyclic,Cyclic}.cs`: connection signals are gathered into a
//! SIMD vector, multiplied by a vector of weights, and scattered back onto the target nodes'
//! pre-activation accumulators. The scatter is scalar (targets are arbitrary), exactly as in the
//! C# reference.

pub mod acyclic;
pub mod cyclic;

pub use acyclic::NeuralNetAcyclic;
pub use cyclic::NeuralNetCyclic;

/// A feed-forward activation interface for a neural network.
///
/// Inputs and outputs are `f64` slices exposed through mutable and shared accessors so the caller
/// can set inputs, trigger [`activate`](Self::activate), and read the resulting outputs.
pub trait NeuralNet {
    /// Number of input nodes.
    fn input_count(&self) -> usize;
    /// Number of output nodes.
    fn output_count(&self) -> usize;

    /// Mutable access to the input signal slot, length `input_count`.
    fn inputs_mut(&mut self) -> &mut [f64];
    /// Shared access to the output signal slot, length `output_count`.
    fn outputs(&self) -> &[f64];

    /// Run one forward pass: read inputs, compute outputs.
    fn activate(&mut self);

    /// Clear any internal state held between activations.
    ///
    /// Acyclic networks have no persistent state and treat this as a no-op; cyclic networks zero
    /// their pre- and post-activation buffers for the hidden and output nodes.
    fn reset(&mut self);
}
