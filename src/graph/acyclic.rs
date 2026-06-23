// SPDX-FileCopyrightText: 2026 Marcin Jędrasik
// SPDX-License-Identifier: MIT

//! Acyclic graph construction: depth analysis and layer scheduling.
//!
//! An acyclic SharpNeat network is activated one layer at a time. The layer (depth) of a node is
//! the longest path, in hops, from any input node to that node; input nodes are at depth 0. The
//! algorithm mirrors `AcyclicGraphDepthAnalysis.cs` and `DirectedGraphAcyclicBuilderUtils.cs` in
//! SharpNeat:
//!
//! 1. Calculate each node's depth via an iterative depth-first traversal (no recursion, so the
//!    call stack cannot overflow on deep graphs).
//! 2. Sort non-input nodes by depth (stably), assigning each node a new contiguous ID so that IDs
//!    increase with depth. Input nodes keep their original IDs.
//! 3. Remap the connection endpoints to the new IDs.
//! 4. Sort the connections by `(src, tgt)` again — now in layer order — and build a [`LayerInfo`]
//!    array recording the exclusive end index of each layer's nodes and connections.
//!
//! The resulting [`DirectedGraphAcyclic`] lets the activator sweep the connections once, in order,
//! accumulating weighted signals onto target nodes and applying the activation function to a layer
//! only after every connection feeding into it has been processed.

use super::{ConnectionIds, DirectedGraph, WeightedDirectedGraph};

/// The exclusive end bounds of one layer's nodes and connections.
///
/// A layer's connections are those whose *source* node sits in that layer. `end_node_idx` /
/// `end_connection_idx` are exclusive upper bounds, so layer `L` covers nodes
/// `[prev_end_node_idx, end_node_idx)` and connections
/// `[prev_end_connection_idx, end_connection_idx)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerInfo {
    pub end_node_idx: usize,
    pub end_connection_idx: usize,
}

/// Per-node depths and the overall graph depth, the output of [`calculate_node_depths`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphDepthInfo {
    /// Number of layers, i.e. `max(depth) + 1`.
    pub graph_depth: usize,
    /// Depth of each node, indexed by the original node ID.
    pub node_depth: Vec<usize>,
}

/// Computes the depth of every node in an acyclic directed graph.
///
/// Input nodes are at depth 0. The graph **must** be acyclic: a cycle causes this function to
/// visit each node at most once (via the `visited at greater-or-equal depth` skip) and therefore
/// return without exploring the cycle, leaving those nodes at depth 0. Callers that cannot
/// guarantee acyclicity should validate the graph beforehand.
///
/// The traversal is iterative, using an explicit stack of `(connection_index, depth)` frames, and
/// avoids recursion so deep topologies cannot overflow the call stack.
pub fn calculate_node_depths(digraph: &DirectedGraph) -> GraphDepthInfo {
    let total = digraph.total_node_count;
    let mut node_depth = vec![0usize; total];
    let mut stack: Vec<StackFrame> = Vec::with_capacity(16);

    // Seed the stack with the first outgoing connection of every input node.
    for src in 0..digraph.input_count {
        if let Some(conn_idx) = digraph.first_connection_index(src) {
            stack.push(StackFrame { conn_idx, depth: 1 });
        }
    }

    let src_ids = &digraph.conn_ids.src_ids;
    let tgt_ids = &digraph.conn_ids.tgt_ids;

    while let Some(frame) = stack.last().copied() {
        // Prepare the current top frame for its *next* visit: advance it to the next sibling
        // connection (same source) whose target is not yet better-visited, or pop it if there is
        // no such sibling. The snapshot `frame` still describes the connection to process now.
        advance_or_pop_top(&mut stack, src_ids, tgt_ids, &node_depth, frame);

        let child = tgt_ids[frame.conn_idx];
        // Skip if the child already has a depth at least as great as the one we'd assign — it has
        // been (or will be) reached via a longer-or-equal path.
        if node_depth[child] >= frame.depth {
            continue;
        }
        node_depth[child] = frame.depth;

        // Descend into the child's first outgoing connection, if any.
        if let Some(next_conn) = digraph.first_connection_index(child) {
            stack.push(StackFrame {
                conn_idx: next_conn,
                depth: frame.depth + 1,
            });
        }
    }

    let graph_depth = node_depth.iter().copied().max().map(|m| m + 1).unwrap_or(1);
    GraphDepthInfo {
        graph_depth,
        node_depth,
    }
}

#[derive(Clone, Copy)]
struct StackFrame {
    conn_idx: usize,
    depth: usize,
}

/// Re-points the current top of stack at the next connection from the same source that leads to a
/// not-yet-better-visited target; pops the top if there is no such connection.
///
/// This is the tail-call-elimination step from SharpNeat's `MoveForward`. The caller snapshots the
/// top frame as `current` *before* calling this, so mutating (or popping) the top here does not
/// affect the connection the caller is about to process.
fn advance_or_pop_top(
    stack: &mut Vec<StackFrame>,
    src_ids: &[usize],
    tgt_ids: &[usize],
    node_depth: &[usize],
    current: StackFrame,
) {
    let current_src = src_ids[current.conn_idx];
    let depth = current.depth;
    let mut i = current.conn_idx + 1;
    while i < src_ids.len() && src_ids[i] == current_src {
        if node_depth[tgt_ids[i]] < depth {
            if let Some(top) = stack.last_mut() {
                top.conn_idx = i;
            }
            return;
        }
        i += 1;
    }
    stack.pop();
}

/// An acyclic directed graph annotated with its layer schedule.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectedGraphAcyclic {
    pub input_count: usize,
    pub output_count: usize,
    pub total_node_count: usize,
    pub conn_ids: ConnectionIds,
    /// One entry per layer, from depth 0 (inputs) to the deepest layer.
    pub layer_array: Vec<LayerInfo>,
    /// For each output (by output index), the new node index of that output after the depth sort.
    pub output_node_idx_arr: Vec<usize>,
}

/// A [`DirectedGraphAcyclic`] with a parallel array of connection weights.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedDirectedGraphAcyclic {
    pub digraph: DirectedGraphAcyclic,
    pub weights: Vec<f64>,
}

/// Builds a [`WeightedDirectedGraphAcyclic`] from a weighted directed graph.
///
/// The input graph must be acyclic. The connections are re-ordered by source layer and the node
/// IDs are re-arranged so that non-input nodes appear in non-decreasing depth order.
pub fn build_weighted_directed_graph_acyclic(
    weighted: WeightedDirectedGraph,
) -> WeightedDirectedGraphAcyclic {
    let digraph = weighted.digraph;
    let input_count = digraph.input_count;
    let output_count = digraph.output_count;
    let total = digraph.total_node_count;
    let conn_len = digraph.conn_ids.len();

    // 1. Depth analysis (node_depth indexed by old node ID).
    let GraphDepthInfo {
        graph_depth,
        mut node_depth,
    } = calculate_node_depths(&digraph);

    // 2. Sort non-input nodes by depth (stable), tracking the old ID at each new position.
    let mut node_ids: Vec<usize> = (0..total).collect();
    sort_non_inputs_by_depth(&mut node_ids, &mut node_depth, input_count);

    // Invert: new ID for each old ID.
    let mut new_id_by_old = vec![0usize; total];
    for (new_idx, &old_id) in node_ids.iter().enumerate() {
        new_id_by_old[old_id] = new_idx;
    }

    // 3. Remap connection endpoints to the new IDs.
    let mut src_ids = digraph.conn_ids.src_ids;
    let mut tgt_ids = digraph.conn_ids.tgt_ids;
    let mut weights = weighted.weights;
    for i in 0..conn_len {
        src_ids[i] = new_id_by_old[src_ids[i]];
        tgt_ids[i] = new_id_by_old[tgt_ids[i]];
    }

    // 4. Stable sort connections by (new src, new tgt), keeping weights aligned.
    sort_connections_with_weights(&mut src_ids, &mut tgt_ids, &mut weights);

    // 5. Output node indices in the new ID space.
    let output_node_idx_arr: Vec<usize> = (input_count..input_count + output_count)
        .map(|old_id| new_id_by_old[old_id])
        .collect();

    // 6. Build the LayerInfo array by scanning nodes and connections in depth order.
    let layer_array = build_layer_array(&node_depth, &src_ids, graph_depth, total);

    let acyclic = DirectedGraphAcyclic {
        input_count,
        output_count,
        total_node_count: total,
        conn_ids: ConnectionIds { src_ids, tgt_ids },
        layer_array,
        output_node_idx_arr,
    };
    WeightedDirectedGraphAcyclic {
        digraph: acyclic,
        weights,
    }
}

/// Stable-sort the non-input portion of `node_ids` (and the matching `node_depth` slice) by depth.
///
/// Input nodes (`[0, input_count)`) stay in place at depth 0.
fn sort_non_inputs_by_depth(node_ids: &mut [usize], node_depth: &mut [usize], input_count: usize) {
    // Pair (depth, old_id) for the non-input range, sort stably by depth, write back.
    let mut entries: Vec<(usize, usize)> = (input_count..node_ids.len())
        .map(|i| (node_depth[i], node_ids[i]))
        .collect();
    entries.sort_by_key(|&(d, _)| d);
    for (k, (d, old_id)) in entries.into_iter().enumerate() {
        let j = input_count + k;
        node_depth[j] = d;
        node_ids[j] = old_id;
    }
}

/// Sort `src_ids`, `tgt_ids` and `weights` together by `(src, tgt)` using a stable sort.
fn sort_connections_with_weights(
    src_ids: &mut [usize],
    tgt_ids: &mut [usize],
    weights: &mut [f64],
) {
    debug_assert_eq!(src_ids.len(), tgt_ids.len());
    debug_assert_eq!(src_ids.len(), weights.len());
    let n = src_ids.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        src_ids[a]
            .cmp(&src_ids[b])
            .then(tgt_ids[a].cmp(&tgt_ids[b]))
    });
    apply_permutation(src_ids, &idx);
    apply_permutation(tgt_ids, &idx);
    apply_permutation(weights, &idx);
}

fn apply_permutation<T: Clone>(slice: &mut [T], idx: &[usize]) {
    let original: Vec<T> = slice.to_vec();
    for (i, &p) in idx.iter().enumerate() {
        slice[i] = original[p].clone();
    }
}

/// Build the [`LayerInfo`] array by scanning nodes and connections in (now sorted) depth order.
fn build_layer_array(
    node_depth: &[usize],
    src_ids: &[usize],
    graph_depth: usize,
    total: usize,
) -> Vec<LayerInfo> {
    let mut layers = Vec::with_capacity(graph_depth);
    let mut node_idx = 0usize;
    let mut conn_idx = 0usize;
    for curr_depth in 0..graph_depth {
        // Advance past nodes at the current depth.
        while node_idx < total && node_depth[node_idx] == curr_depth {
            node_idx += 1;
        }
        // Advance past connections whose source is at the current depth.
        while conn_idx < src_ids.len() && node_depth[src_ids[conn_idx]] == curr_depth {
            conn_idx += 1;
        }
        layers.push(LayerInfo {
            end_node_idx: node_idx,
            end_connection_idx: conn_idx,
        });
    }
    // Layer 0 (depth 0) holds the input nodes plus any non-input nodes that were not reached by
    // the depth analysis — either because they have no incoming path from an input, or because they
    // are phantom IDs (referenced nowhere but covered by `total_node_count`). These extra nodes
    // keep pre-activation 0.0 and, being in the same layer as the inputs, are never activated before
    // their (possibly empty) outgoing connections are processed; this matches SharpNeat's behaviour,
    // where unreachable nodes contribute zero to downstream targets.
    layers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::WeightedDirectedConnection;

    /// A small acyclic graph:
    /// ```text
    /// inputs 0,1,2 -> hidden 5,6,7,8 -> outputs 3,4
    /// ```
    /// Matches SharpNeat's `example1.net` topology (without per-node activation functions).
    fn example1_graph() -> WeightedDirectedGraph {
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
        WeightedDirectedGraph::build(conns, 3, 2)
    }

    #[test]
    fn depths_match_example1() {
        // Every non-input node is one hop from an input, so all hidden + output nodes sit at depth 1.
        let g = example1_graph();
        let info = calculate_node_depths(&g.digraph);
        assert_eq!(info.graph_depth, 2);
        assert_eq!(&info.node_depth[..3], &[0, 0, 0]); // inputs
        for d in &info.node_depth[3..9] {
            assert_eq!(*d, 1);
        }
    }

    #[test]
    fn acyclic_builder_produces_two_layers() {
        let g = example1_graph();
        let acyclic = build_weighted_directed_graph_acyclic(g);
        assert_eq!(acyclic.digraph.layer_array.len(), 2);
        let layer0 = acyclic.digraph.layer_array[0];
        let layer1 = acyclic.digraph.layer_array[1];
        // Layer 0 holds the 3 input nodes and all 12 connections (all sources are inputs).
        assert_eq!(layer0.end_node_idx, 3);
        assert_eq!(layer0.end_connection_idx, 12);
        // Layer 1 holds the remaining 6 nodes and no connections.
        assert_eq!(layer1.end_node_idx, 9);
        assert_eq!(layer1.end_connection_idx, 12);
    }

    #[test]
    fn acyclic_builder_preserves_output_indices() {
        let g = example1_graph();
        let acyclic = build_weighted_directed_graph_acyclic(g);
        // Two outputs; both end up at depth 1, after the inputs.
        assert_eq!(acyclic.digraph.output_node_idx_arr.len(), 2);
        for &idx in &acyclic.digraph.output_node_idx_arr {
            assert!((3..9).contains(&idx), "output idx {idx} out of range");
        }
    }

    #[test]
    fn deeper_graph_has_three_layers() {
        // 0 -> 3 -> 4, with inputs {0,1,2} and outputs {4} (1-based output_count handling).
        // Node 3 is depth 1, node 4 is depth 2.
        let conns = vec![
            WeightedDirectedConnection {
                src_id: 0,
                tgt_id: 3,
                weight: 1.0,
            },
            WeightedDirectedConnection {
                src_id: 3,
                tgt_id: 4,
                weight: 1.0,
            },
        ];
        let g = WeightedDirectedGraph::build(conns, 3, 2);
        let info = calculate_node_depths(&g.digraph);
        assert_eq!(info.node_depth, vec![0, 0, 0, 1, 2]);
        assert_eq!(info.graph_depth, 3);

        let acyclic = build_weighted_directed_graph_acyclic(g);
        assert_eq!(acyclic.digraph.layer_array.len(), 3);
        // Connections sorted by source layer: 0->3 (layer0), 3->4 (layer1).
        assert_eq!(acyclic.digraph.layer_array[0].end_connection_idx, 1);
        assert_eq!(acyclic.digraph.layer_array[1].end_connection_idx, 2);
        assert_eq!(acyclic.digraph.layer_array[2].end_connection_idx, 2);
    }

    #[test]
    fn connections_remain_sorted_by_src_then_tgt() {
        let g = example1_graph();
        let acyclic = build_weighted_directed_graph_acyclic(g);
        let srcs = &acyclic.digraph.conn_ids.src_ids;
        for w in srcs.windows(2) {
            assert!(w[0] <= w[1], "src ids not sorted: {srcs:?}");
        }
    }

    #[test]
    fn weights_track_connections_through_sort() {
        let g = example1_graph();
        let acyclic = build_weighted_directed_graph_acyclic(g);
        // Every weight in example1 is distinct, so the rebuilt weights must be a permutation of
        // the originals — i.e. no weight became detached from its connection during the sort.
        let mut original: Vec<f64> =
            vec![0.5, 0.7, 0.3, 1.5, 1.7, 1.3, 1.6, 1.8, 1.4, 2.6, 2.8, 2.4];
        let mut rebuilt: Vec<f64> = acyclic.weights.clone();
        original.sort_by(|a, b| a.total_cmp(b));
        rebuilt.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(original, rebuilt);
    }

    #[test]
    fn layer_zero_includes_unreachable_and_phantom_nodes() {
        // A graph whose max referenced ID (100) is much larger than input_count + output_count,
        // creating many phantom IDs in [0, 100) that are never referenced by any connection but
        // are covered by `total_node_count`. Node 5 has an outgoing edge but no incoming path from
        // any input, so it is unreachable and stays at depth 0. Both the phantom IDs and node 5
        // end up in layer 0 alongside the inputs.
        //
        // inputs {0,1,2}, output {3}, unreachable hidden {5}, reachable hidden {100}.
        // 0 -> 3 (w=1.0); 5 -> 3 (w=2.0); 0 -> 100 (w=0.5).
        let conns = vec![
            WeightedDirectedConnection {
                src_id: 0,
                tgt_id: 3,
                weight: 1.0,
            },
            WeightedDirectedConnection {
                src_id: 5,
                tgt_id: 3,
                weight: 2.0,
            },
            WeightedDirectedConnection {
                src_id: 0,
                tgt_id: 100,
                weight: 0.5,
            },
        ];
        let g = WeightedDirectedGraph::build(conns, 3, 1);
        // total_node_count = max(3+1, 101) = 101 (the edge to node 100 inflates it).
        assert_eq!(g.digraph.total_node_count, 101);
        let acyclic = build_weighted_directed_graph_acyclic(g);
        // Layer 0 contains the 3 inputs plus all unreachable/phantom non-input nodes (IDs 4..99
        // and node 5), so it is much larger than `input_count`.
        assert!(
            acyclic.digraph.layer_array[0].end_node_idx > 3,
            "layer 0 should contain unreachable + phantom nodes, got end_node_idx = {}",
            acyclic.digraph.layer_array[0].end_node_idx
        );
        // The build must not panic and the output index must be valid.
        assert_eq!(acyclic.digraph.output_node_idx_arr.len(), 1);
    }

    #[test]
    fn unreachable_node_contributes_zero_to_acyclic_output() {
        // 0 -> 3 (w=1), 5 -> 3 (w=2). Node 5 is unreachable (no incoming edges).
        // With input 0 = 5.0: output = LeakyReLU(5*1 + 0*2) = 5.0 (node 5 contributes 0).
        use crate::activation::ActivationFn;
        use crate::net::NeuralNet;
        use crate::net::NeuralNetAcyclic;

        let conns = vec![
            WeightedDirectedConnection {
                src_id: 0,
                tgt_id: 3,
                weight: 1.0,
            },
            WeightedDirectedConnection {
                src_id: 5,
                tgt_id: 3,
                weight: 2.0,
            },
        ];
        let g = WeightedDirectedGraph::build(conns, 3, 1);
        let acyclic = build_weighted_directed_graph_acyclic(g);
        let mut net = NeuralNetAcyclic::new(acyclic, ActivationFn::LeakyReLU);
        net.inputs_mut().copy_from_slice(&[5.0, 0.0, 0.0]);
        net.activate();
        // Unreachable node 5 has post-activation 0.0; its weight-2 edge adds 0.0.
        assert!(
            (net.outputs()[0] - 5.0).abs() <= 1e-12,
            "got {}",
            net.outputs()[0]
        );
    }
}
