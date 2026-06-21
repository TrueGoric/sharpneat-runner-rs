// SPDX-FileCopyrightText: 2026 Marcin Jędrasik
// SPDX-License-Identifier: MIT

//! Directed graph representations used to build neural networks.
//!
//! A SharpNeat network is a weighted directed graph. Node IDs are contiguous zero-based indexes
//! where inputs occupy `[0, input_count)` and outputs `[input_count, input_count + output_count)`.
//! The remaining IDs are hidden nodes.
//!
//! The acyclic variant lives in [`acyclic`]: it adds a depth-based layer schedule so the network
//! can be activated one layer at a time.

pub mod acyclic;

use std::cmp::Ordering;

/// A directed connection between two node IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectedConnection {
    pub src_id: usize,
    pub tgt_id: usize,
}

/// A directed connection with an associated weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightedDirectedConnection {
    pub src_id: usize,
    pub tgt_id: usize,
    pub weight: f64,
}

impl WeightedDirectedConnection {
    /// Order by `(src_id, tgt_id)`, the order SharpNeat uses for its connection arrays.
    pub fn connection_order(&self, other: &Self) -> Ordering {
        self.src_id
            .cmp(&other.src_id)
            .then(self.tgt_id.cmp(&other.tgt_id))
    }
}

/// Parallel arrays of source and target node IDs for the connections of a graph.
///
/// Connections are kept sorted by `(src_id, tgt_id)` so that all outgoing connections of a node
/// occupy a contiguous slice, which the graph exposes via
/// [`DirectedGraph::first_connection_index`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectionIds {
    pub src_ids: Vec<usize>,
    pub tgt_ids: Vec<usize>,
}

impl ConnectionIds {
    pub fn len(&self) -> usize {
        self.src_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.src_ids.is_empty()
    }
}

/// A directed graph with input, output and hidden nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectedGraph {
    pub input_count: usize,
    pub output_count: usize,
    pub total_node_count: usize,
    pub conn_ids: ConnectionIds,
    /// For each source node index, the index of its first outgoing connection, or `usize::MAX`
    /// when the node has no outgoing connections.
    conn_idx_by_src: Vec<usize>,
}

impl DirectedGraph {
    /// Index of the first connection whose source is `src_node_idx`, or `None` if there is none.
    pub fn first_connection_index(&self, src_node_idx: usize) -> Option<usize> {
        let i = *self.conn_idx_by_src.get(src_node_idx)?;
        if i == usize::MAX { None } else { Some(i) }
    }

    /// Build a [`DirectedGraph`] from already-sorted connection ID arrays.
    ///
    /// The caller must guarantee that `src_ids` and `tgt_ids` have equal length and are sorted by
    /// `(src, tgt)`, and that `total_node_count` covers every referenced ID.
    pub(crate) fn from_sorted_ids(
        input_count: usize,
        output_count: usize,
        total_node_count: usize,
        conn_ids: ConnectionIds,
    ) -> Self {
        let conn_idx_by_src = compile_first_connection_lookup(total_node_count, &conn_ids.src_ids);
        Self {
            input_count,
            output_count,
            total_node_count,
            conn_ids,
            conn_idx_by_src,
        }
    }
}

/// A directed graph with a parallel array of connection weights.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedDirectedGraph {
    pub digraph: DirectedGraph,
    pub weights: Vec<f64>,
}

impl WeightedDirectedGraph {
    /// Build a weighted directed graph from an arbitrary set of connections.
    ///
    /// The connections are sorted in place by `(src_id, tgt_id)` and split into parallel ID and
    /// weight arrays. `total_node_count` is taken as the largest of `input_count + output_count`
    /// and one past the largest referenced node ID, so the net file format's contiguous ID scheme
    /// is handled correctly even when hidden nodes exist.
    pub fn build(
        mut connections: Vec<WeightedDirectedConnection>,
        input_count: usize,
        output_count: usize,
    ) -> Self {
        connections.sort_by(|a, b| a.connection_order(b));

        let max_id = connections
            .iter()
            .flat_map(|c| [c.src_id, c.tgt_id])
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        let total_node_count = (input_count + output_count).max(max_id);

        let n = connections.len();
        let mut src_ids = Vec::with_capacity(n);
        let mut tgt_ids = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        for c in connections {
            src_ids.push(c.src_id);
            tgt_ids.push(c.tgt_id);
            weights.push(c.weight);
        }

        let digraph = DirectedGraph::from_sorted_ids(
            input_count,
            output_count,
            total_node_count,
            ConnectionIds { src_ids, tgt_ids },
        );
        Self { digraph, weights }
    }
}

/// Build the per-source-node lookup of the first connection index.
///
/// `src_ids` must be sorted non-decreasingly. Nodes with no outgoing connection receive
/// `usize::MAX`.
fn compile_first_connection_lookup(total_node_count: usize, src_ids: &[usize]) -> Vec<usize> {
    let mut lookup = vec![usize::MAX; total_node_count];
    if src_ids.is_empty() {
        return lookup;
    }
    let mut current_src = src_ids[0];
    lookup[current_src] = 0;
    for (i, &s) in src_ids.iter().enumerate().skip(1) {
        if s != current_src {
            current_src = s;
            lookup[s] = i;
        }
    }
    lookup
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conns() -> Vec<WeightedDirectedConnection> {
        vec![
            WeightedDirectedConnection {
                src_id: 2,
                tgt_id: 4,
                weight: 2.4,
            },
            WeightedDirectedConnection {
                src_id: 0,
                tgt_id: 5,
                weight: 0.5,
            },
            WeightedDirectedConnection {
                src_id: 1,
                tgt_id: 3,
                weight: 1.3,
            },
            WeightedDirectedConnection {
                src_id: 0,
                tgt_id: 3,
                weight: 0.3,
            },
        ]
    }

    #[test]
    fn builder_sorts_by_src_then_tgt() {
        let g = WeightedDirectedGraph::build(conns(), 3, 2);
        assert_eq!(g.digraph.conn_ids.src_ids, vec![0, 0, 1, 2]);
        assert_eq!(g.digraph.conn_ids.tgt_ids, vec![3, 5, 3, 4]);
        assert_eq!(g.weights, vec![0.3, 0.5, 1.3, 2.4]);
    }

    #[test]
    fn total_node_count_covers_hidden_nodes() {
        let g = WeightedDirectedGraph::build(conns(), 3, 2);
        // max id referenced is 5, so total = max(3+2, 6) = 6.
        assert_eq!(g.digraph.total_node_count, 6);
    }

    #[test]
    fn first_connection_index_lookup() {
        let g = WeightedDirectedGraph::build(conns(), 3, 2);
        assert_eq!(g.digraph.first_connection_index(0), Some(0));
        assert_eq!(g.digraph.first_connection_index(1), Some(2));
        assert_eq!(g.digraph.first_connection_index(2), Some(3));
        assert_eq!(g.digraph.first_connection_index(3), None); // hidden node 3 has no outputs
        assert_eq!(g.digraph.first_connection_index(4), None);
        assert_eq!(g.digraph.first_connection_index(5), None);
    }

    #[test]
    fn empty_connections_yield_node_counts_only() {
        let g = WeightedDirectedGraph::build(vec![], 3, 2);
        assert!(g.digraph.conn_ids.is_empty());
        assert_eq!(g.digraph.total_node_count, 5);
        assert_eq!(g.digraph.first_connection_index(0), None);
    }
}
