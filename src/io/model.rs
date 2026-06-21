// SPDX-FileCopyrightText: 2026 Marcin Jędrasik
// SPDX-License-Identifier: MIT

//! In-memory representation of a `.net` file and the error type returned by the reader/writer.

use std::fmt;

/// One connection line: `source target weight`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConnectionLine {
    pub source_id: usize,
    pub target_id: usize,
    pub weight: f64,
}

impl ConnectionLine {
    pub fn new(source_id: usize, target_id: usize, weight: f64) -> Self {
        Self {
            source_id,
            target_id,
            weight,
        }
    }
}

/// One activation function line: `id code`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationFnLine {
    pub id: usize,
    pub code: String,
}

impl ActivationFnLine {
    pub fn new(id: usize, code: impl Into<String>) -> Self {
        Self {
            id,
            code: code.into(),
        }
    }
}

/// Object model for the `net` file format.
///
/// Fields mirror SharpNeat's `NetFileModel`. Construction validates the same invariants as the C#
/// constructor: input count is non-negative, output count is at least one, cyclic networks specify
/// a positive `cycles_per_activation`, and at least one activation function is defined with the
/// first having ID 0.
#[derive(Debug, Clone, PartialEq)]
pub struct NetFileModel {
    pub input_count: usize,
    pub output_count: usize,
    pub is_acyclic: bool,
    pub cycles_per_activation: usize,
    pub connections: Vec<ConnectionLine>,
    pub activation_fns: Vec<ActivationFnLine>,
}

impl NetFileModel {
    /// Construct an acyclic model.
    pub fn create_acyclic(
        input_count: usize,
        output_count: usize,
        connections: Vec<ConnectionLine>,
        activation_fns: Vec<ActivationFnLine>,
    ) -> Result<Self, NetFileError> {
        Self::new(
            input_count,
            output_count,
            true,
            0,
            connections,
            activation_fns,
        )
    }

    /// Construct a cyclic model.
    pub fn create_cyclic(
        input_count: usize,
        output_count: usize,
        cycles_per_activation: usize,
        connections: Vec<ConnectionLine>,
        activation_fns: Vec<ActivationFnLine>,
    ) -> Result<Self, NetFileError> {
        Self::new(
            input_count,
            output_count,
            false,
            cycles_per_activation,
            connections,
            activation_fns,
        )
    }

    pub(crate) fn new(
        input_count: usize,
        output_count: usize,
        is_acyclic: bool,
        cycles_per_activation: usize,
        connections: Vec<ConnectionLine>,
        activation_fns: Vec<ActivationFnLine>,
    ) -> Result<Self, NetFileError> {
        if activation_fns.is_empty() {
            return Err(NetFileError::NoActivationFunction);
        }
        if activation_fns[0].id != 0 {
            return Err(NetFileError::FirstActivationFunctionIdNotZero {
                id: activation_fns[0].id,
            });
        }
        if !is_acyclic && cycles_per_activation < 1 {
            return Err(NetFileError::ModelValidation(
                "cycles_per_activation must be at least 1 for cyclic networks".into(),
            ));
        }
        Ok(Self {
            input_count,
            output_count,
            is_acyclic,
            cycles_per_activation,
            connections,
            activation_fns,
        })
    }
}

/// Errors raised while parsing or writing a `.net` file.
///
/// `line` fields are 1-based physical line numbers, to help locate the problem in the source file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetFileError {
    /// The file ended before a required non-empty line was read.
    UnexpectedEof { line: usize },
    /// A section terminator (blank line or end of file) was expected but more data was found.
    ExpectedEndOfSection { line: usize },
    /// The `input_count output_count` header was malformed.
    InvalidNodeCounts { line: usize, reason: &'static str },
    /// The `acyclic` / `cyclic N` indicator was malformed.
    InvalidCyclicIndicator { line: usize, reason: &'static str },
    /// A connection line was malformed or referenced an input node as its target.
    InvalidConnection { line: usize, reason: &'static str },
    /// An activation function line was malformed or had an out-of-sequence ID.
    InvalidActivationFunction { line: usize, reason: &'static str },
    /// No activation function was defined.
    NoActivationFunction,
    /// The first activation function did not have ID 0.
    FirstActivationFunctionIdNotZero { id: usize },
    /// A constructed model violated one of the structural invariants.
    ModelValidation(String),
    /// An activation function code in the file is not known to this crate.
    UnknownActivationCode { code: String },
    /// An I/O error occurred while reading or writing the underlying stream.
    Io(String),
}

impl fmt::Display for NetFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof { line } => write!(f, "unexpected end of file at line {line}"),
            Self::ExpectedEndOfSection { line } => {
                write!(f, "expected end of section at line {line}")
            }
            Self::InvalidNodeCounts { line, reason } => {
                write!(f, "invalid node counts at line {line}: {reason}")
            }
            Self::InvalidCyclicIndicator { line, reason } => {
                write!(
                    f,
                    "invalid cyclic/acyclic indicator at line {line}: {reason}"
                )
            }
            Self::InvalidConnection { line, reason } => {
                write!(f, "invalid connection at line {line}: {reason}")
            }
            Self::InvalidActivationFunction { line, reason } => {
                write!(f, "invalid activation function at line {line}: {reason}")
            }
            Self::NoActivationFunction => write!(f, "no activation function defined"),
            Self::FirstActivationFunctionIdNotZero { id } => {
                write!(f, "first activation function must have ID 0, got {id}")
            }
            Self::ModelValidation(msg) => write!(f, "invalid model: {msg}"),
            Self::UnknownActivationCode { code } => {
                write!(f, "unknown activation function code: {code}")
            }
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl std::error::Error for NetFileError {}

impl From<std::io::Error> for NetFileError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acyclic_model_validates() {
        let m = NetFileModel::create_acyclic(
            3,
            2,
            vec![ConnectionLine::new(0, 3, 0.5)],
            vec![ActivationFnLine::new(0, "ReLU")],
        )
        .unwrap();
        assert!(m.is_acyclic);
        assert_eq!(m.cycles_per_activation, 0);
    }

    #[test]
    fn cyclic_model_requires_positive_cycles() {
        let err = NetFileModel::create_cyclic(
            1,
            1,
            0,
            vec![ConnectionLine::new(0, 1, 1.0)],
            vec![ActivationFnLine::new(0, "ReLU")],
        )
        .unwrap_err();
        assert!(matches!(err, NetFileError::ModelValidation(_)));
    }

    #[test]
    fn model_requires_at_least_one_activation_function() {
        let err = NetFileModel::create_acyclic(1, 1, vec![], vec![]).unwrap_err();
        assert_eq!(err, NetFileError::NoActivationFunction);
    }

    #[test]
    fn first_activation_function_must_have_id_zero() {
        let err = NetFileModel::create_acyclic(
            1,
            1,
            vec![ConnectionLine::new(0, 1, 1.0)],
            vec![ActivationFnLine::new(1, "ReLU")],
        )
        .unwrap_err();
        assert_eq!(
            err,
            NetFileError::FirstActivationFunctionIdNotZero { id: 1 }
        );
    }
}
