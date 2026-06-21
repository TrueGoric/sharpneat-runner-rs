# AGENTS.md

Work in this repo as an experienced software engineer who favors maintainable, readable, testable, and elegant code: small well-named functions, minimal public APIs, tests that pin down behavior rather than implementation.

## Project facts

- **Package name differs from the directory.** The Cargo package is `sharpneat-runner-rs`; the repo directory is `sharpneat-rust-runner` (word order swapped). Use the manifest name in `cargo` commands; in `use` declarations the hyphens become underscores (`sharpneat_runner_rs`).
- **Library crate only — no binary.** There is no `src/main.rs`; `cargo run` fails with "a bin target must be available". Verify with `cargo build` / `cargo test`.
- **Nightly Rust is required and pinned.** `rust-toolchain.toml` forces the `nightly` channel because the crate uses `#![feature(portable_simd)]` for SIMD-vectorised activation functions and network activation loops. `cargo` / `rustc` invoked in this repo automatically use that toolchain; there is no need to pass `+nightly`. The pin also requests the `rustfmt` and `clippy` components.
- **Rust edition 2024.** Edition 2024 changes some rules from 2021 (e.g. `gen` is a reserved keyword, `unsafe extern` blocks are required); confirm against edition 2024 notes before assuming older idioms.
- **No external dependencies.** The library and its benchmarks use only `std`. Benchmarks are custom zero-dependency harnesses (`std::time::Instant` + `std::hint::black_box`), not `criterion`.
- **No `unsafe` anywhere.** Do not add `unsafe` blocks; the SIMD code path is written entirely with safe `portable_simd` APIs. If a perf-critical path seems to require `unsafe`, reconsider the algorithm first.

## What the crate does

Inference-only runtime for neural networks trained by [SharpNeat](https://github.com/colgreen/sharpneat). It loads SharpNeat's `.net` file format and runs forward (activation) passes. **No training.** Modules under `src/`:

- `activation` — `Activation` trait (the primary abstraction), zero-sized unit struct types (`Logistic`, `ReLU`, `TanH`, …) implementing it, and the `ActivationFn` enum (runtime-dispatch adapter that also implements `Activation`). SIMD-vectorised scalar/vector inner functions in `activation/functions.rs`, shared drivers + vectorised `vexp` in `activation/vectorized.rs`. SIMD lanes are fixed at 4 × `f64` (`activation::LANES`).
- `graph` — `DirectedGraph`, `ConnectionIds`, `WeightedDirectedGraph` and the `graph::acyclic` submodule: depth analysis (`calculate_node_depths`), layer scheduling (`LayerInfo`, `build_weighted_directed_graph_acyclic`). The depth-analysis traversal is iterative (explicit stack) and is a direct port of SharpNeat's `AcyclicGraphDepthAnalysis` / `DirectedGraphAcyclicBuilderUtils`; be careful editing `advance_or_pop_top` — its "peek, then mutate top" ordering is load-bearing.
- `net` — `NeuralNet` trait plus `NeuralNetAcyclic<A: Activation>` (layer-by-layer sweep) and `NeuralNetCyclic<A: Activation>` (fixed relaxation timesteps). Both are generic over the activation function so call sites monomorphise. They mirror the vectorised connection loops in SharpNeat's `NeuralNet{Acyclic,Cyclic}.cs`: gather source signals into a `Simd`, multiply by a weight vector, scatter-accumulate into targets scalarly.
- `io` — `NetFile` load/save plus `NetFileModel`, `ConnectionLine`, `ActivationFnLine`, `NetFileError`. The reader in `io/reader.rs` is a line-for-line port of SharpNeat's `NetFileReader`: sections are blank-line separated, `#` comments may appear anywhere, and the per-node activation function section emitted by newer SharpNeat is intentionally ignored.
- `builder` — `Net` enum and `build_from_model`, the glue that turns a parsed `NetFileModel` into a runnable `NeuralNetAcyclic<ActivationFn>` or `NeuralNetCyclic<ActivationFn>` (runtime dispatch, since the function is read from the file at runtime).

## Verification

Run before considering work done:

```
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Clippy is this repo's lint/typecheck; the `-D warnings` flag treats warnings as failures. Run a single test with `cargo test <name>`. Integration tests live in `tests/integration.rs` and read fixture `.net` files from `tests/fixtures/` (copied from SharpNeat's test data).

### Benchmarks

```
cargo bench --bench activation_bench
cargo bench --bench neuralnet_bench
```

Each bench is a standalone `main()` (`harness = false`) that warms up for ~200 ms then measures for ~2 s per scenario, printing ns/call. They are not run by `cargo test`. `cargo bench` uses the release profile automatically; there is no `criterion`.

## Conventions worth preserving

- **`Activation` is the primary abstraction, not the enum.** Activation functions are zero-sized unit structs implementing the `Activation` trait; neural nets are generic over `A: Activation` so call sites monomorphise. The `ActivationFn` enum also implements `Activation` and is the runtime-dispatch adapter for file loading — keep it that way. Do not add trait objects (`dyn Activation`); the design is built around static dispatch.
- **Vectorised and scalar activation inner functions must agree to within ~1e-9.** The shared `vexp` (degree-9 Horner on a reduced argument) is accurate to ~1e-11; if you add an exp-based function, route its vector form through `vexp` (or `map_lanes` for rare transcendentals like `sin`/`atan`/`log`) and add a case to the `vectorised_matches_scalar_across_range` test. Add a unit struct + `Activation` impl via the `activation_type!` macro in `functions.rs`.
- **Acyclic node IDs are remapped by depth** during `build_weighted_directed_graph_acyclic`; input nodes keep their IDs, hidden/output nodes are stably sorted by depth. Connection weights are permuted in lockstep with the connection sort — if you touch `sort_connections_with_weights` or `apply_permutation`, keep them aligned.
- **Cyclic inputs are held in the post-activation slice** (`post[..input_count]`) and are never overwritten by activation; outputs live at `post[input_count..input_count+output_count]`. `reset` only clears the non-input portions.
- **Test tolerances**: use `<=` (not `<`) when comparing floats against a zero tolerance, so exact matches on ReLU/NullFn etc. pass.
