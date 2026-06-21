# sharpneat-runner-rs

Inference-only runtime for neural networks trained by [SharpNeat](https://github.com/colgreen/sharpneat).

SharpNeat is a C# genetic algorithm library that evolves neural network topologies. This crate
implements the **inference** side only: it loads a trained network from SharpNeat's `.net` file
format and runs forward (activation) passes. No training functionality is provided.

## Features

- **Acyclic networks** — activated layer-by-layer using a depth schedule computed from the graph
  topology.
- **Cyclic networks** — activated by a fixed number of relaxation timesteps per call.
- **18 activation functions** — the full standard SharpNeat set plus CPPN functions (Sine, Gaussian),
  each SIMD-vectorised via `portable_simd` (4 × `f64` lanes).
- **Trait-based generics** — neural nets are generic over `A: Activation`. Use a concrete unit
  struct (`Logistic`, `ReLU`, …) for a monomorphised, inlined hot path, or the `ActivationFn` enum
  for runtime dispatch when the function is read from a file.
- **Net file IO** — parsing and writing the human-readable `.net` format produced by SharpNeat's
  `NetFile.Load` / `NetFile.Save`.
- **No `unsafe` code anywhere.** No external dependencies (std only).

## Quick start

### Load and run a `.net` file

```rust
use sharpneat_runner_rs::{Net, NeuralNet, io::NetFile};

let model = NetFile::load("mynet.net")?;
let mut net = Net::from_model(&model)?;

net.inputs_mut().copy_from_slice(&[1.0, 0.5, -0.5]);
net.activate();
let outputs = net.outputs();
```

### Construct a network with a concrete activation function

When the activation function is known at compile time, use the concrete unit struct for a fully
monomorphised code path:

```rust
use sharpneat_runner_rs::{
    Logistic, NeuralNet, NeuralNetAcyclic,
    graph::{WeightedDirectedConnection, WeightedDirectedGraph},
    graph::acyclic::build_weighted_directed_graph_acyclic,
};

let conns = vec![
    WeightedDirectedConnection { src_id: 0, tgt_id: 2, weight: 0.5 },
    WeightedDirectedConnection { src_id: 1, tgt_id: 2, weight: -0.5 },
    WeightedDirectedConnection { src_id: 2, tgt_id: 3, weight: 1.0 },
];
let graph = build_weighted_directed_graph_acyclic(
    WeightedDirectedGraph::build(conns, 2, 1),
);
let mut net = NeuralNetAcyclic::new(graph, Logistic);
net.inputs_mut().copy_from_slice(&[1.0, 1.0]);
net.activate();
```

### Write a function generic over the activation function

```rust
use sharpneat_runner_rs::{Activation, NeuralNet};

fn evaluate<A: Activation>(net: &mut impl NeuralNet, _fn: A) -> Vec<f64> {
    net.activate();
    net.outputs().to_vec()
}
```

## Architecture

```
src/
├── activation/       Activation trait, unit-struct types, ActivationFn enum, SIMD inner functions
│   ├── functions.rs  Per-function scalar/vector implementations + unit struct + Activation impl
│   └── vectorized.rs SIMD drivers (apply_inplace/apply_into), shared vexp, map_lanes
├── graph/            Directed graph representations
│   └── acyclic.rs    Depth analysis (iterative DFS), layer scheduling, LayerInfo
├── io/               .net file format reader/writer + in-memory model
│   ├── model.rs      NetFileModel, ConnectionLine, ActivationFnLine, NetFileError
│   ├── reader.rs     Line-for-line port of SharpNeat's NetFileReader
│   └── writer.rs     Serialiser matching SharpNeat's NetFileWriter
├── net/              Neural network runtime
│   ├── acyclic.rs    NeuralNetAcyclic<A> — layer-by-layer sweep
│   └── cyclic.rs     NeuralNetCyclic<A> — fixed relaxation timesteps
├── builder.rs        Net enum + build_from_model — glue between IO and runtime
└── lib.rs            Public API and re-exports
```

### Activation functions

| Function | Code | Type | Notes |
|---|---|---|---|
| Logistic | `Logistic` | Sigmoid | `1 / (1 + e^-x)` |
| LogisticSteep | `LogisticSteep` | Sigmoid | steepened slope (`-4.9x`) |
| TanH | `TanH` | Sigmoid | `tanh(x)`, via vectorised `vexp` |
| SoftSignSteep | `SoftSignSteep` | Sigmoid | softsign with steepened slope |
| PolynomialApproximantSteep | `PolynomialApproximantSteep` | Sigmoid | fast exp-free logistic approximation |
| QuadraticSigmoid | `QuadraticSigmoid` | Sigmoid | two `x²` sub-sections with leaky tails |
| ReLU | `ReLU` | Piecewise linear | `max(0, x)` |
| LeakyReLU | `LeakyReLU` | Piecewise linear | slope 0.001 for negative inputs |
| LeakyReLUShifted | `LeakyReLUShifted` | Piecewise linear | shifted so x=0 → y≈0.5 |
| SReLU | `SReLU` | Piecewise linear | S-shaped rectified linear unit |
| SReLUShifted | `SReLUShifted` | Piecewise linear | SReLU shifted to x=0 → y≈0.5 |
| MaxMinusOne | `MaxMinusOne` | Piecewise linear | `max(-1, x)` |
| ScaledELU | `ScaledELU` | Piecewise linear | SELU (self-normalising) |
| NullFn | `NullFn` | Constant | always returns 0 |
| ArcTan | `ArcTan` | Other | `atan(x)` |
| ArcSinH | `ArcSinH` | Other | scaled inverse hyperbolic sine |
| Sine | `Sine` | CPPN | `sin(2x)` |
| Gaussian | `Gaussian` | CPPN | `exp(-(2.5x)²)` |

Each function exists as both a concrete unit struct (e.g. `Logistic`) and an `ActivationFn` enum
variant (e.g. `ActivationFn::Logistic`). Both implement the `Activation` trait. The unit structs
are zero-sized, so storing one in a generic `NeuralNetAcyclic<A>` costs no memory and the compiler
monomorphises and inlines the activation calls.

## Benchmarks

Run with:

```
cargo bench --bench activation_bench
cargo bench --bench neuralnet_bench
```

Benchmarks are zero-dependency harnesses (`std::time::Instant` + `std::hint::black_box`) that warm
up for ~200 ms then measure for ~2 s per scenario. They are not run by `cargo test`.

Each scenario is benchmarked twice — through the `ActivationFn` enum (runtime dispatch, as used by
the `.net` file loader) and through the concrete unit struct (monomorphised) — to confirm that the
trait-based generic design introduces no performance regression.

### Activation functions (1024 elements, in-place)

| Function | Enum dispatch (ns/elem) | Monomorphised (ns/elem) |
|---|---|---|
| Logistic | 6.69 | 6.71 |
| LogisticSteep | 7.32 | 7.65 |
| TanH | 6.47 | 6.53 |
| ReLU | 0.152 | 0.156 |
| LeakyReLU | 0.345 | 0.348 |
| ScaledELU | 14.20 | 14.21 |
| SoftSignSteep | 0.622 | 0.630 |
| PolynomialApproximantSteep | 1.25 | 1.25 |
| QuadraticSigmoid | 0.778 | 0.803 |
| SReLU | 0.773 | 0.770 |
| Gaussian | 8.21 | 7.98 |
| Sine | 7.45 | 7.39 |
| ArcTan | 7.29 | 7.15 |
| ArcSinH | 5.28 | 5.50 |
| NullFn | 0.090 | 0.092 |
| MaxMinusOne | 0.152 | 0.154 |

The two dispatch strategies are within measurement noise (<3% difference) across all functions.
The `match` in the enum path executes once per `activate_inplace` call (i.e. once per 1024-element
buffer), so its overhead is negligible.

### Network activation (Logistic)

| Network | Enum dispatch (µs) | Monomorphised (µs) |
|---|---|---|
| acyclic 6→16→4 (1 hidden layer) | 0.339 | 0.344 |
| acyclic 6→32×3→4 (3 hidden layers) | 2.93 | 2.87 |
| acyclic 12→64×4→8 (4 hidden layers) | 14.16 | 14.10 |
| cyclic 6→ring16→4 (4 cycles) | 1.31 | 1.29 |
| cyclic 12→ring64→8 (4 cycles) | 7.55 | 6.98 |

The monomorphised path is equal or slightly faster than enum dispatch, as expected — the compiler
can inline the activation function across layer boundaries when the concrete type is known.

## Development

Requires nightly Rust (pinned via `rust-toolchain.toml` for `portable_simd`):

```
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Integration tests load fixture `.net` files from `tests/fixtures/` (copied from SharpNeat's test
data). Run a single test with `cargo test <name>`.

## License

MIT, matching SharpNeat.
