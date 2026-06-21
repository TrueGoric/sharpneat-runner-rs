// SPDX-FileCopyrightText: 2026 Marcin Jędrasik
// SPDX-License-Identifier: MIT

//! Parser for SharpNeat's `.net` file format.
//!
//! The format is a sequence of blank-line-separated sections:
//!
//! 1. `input_count output_count`
//! 2. `acyclic` or `cyclic <cycles_per_activation>`
//! 3. Zero or more `source target weight` connection lines
//! 4. One or more `<id> <code>` activation function lines (IDs sequential from 0)
//!
//! Lines starting with `#` are comments and may appear anywhere; they are skipped without
//! terminating a section. Trailing content after the activation function section is ignored, which
//! allows the per-node activation function section emitted by newer SharpNeat versions to pass
//! through unchanged.
//!
//! The implementation mirrors `NetFileReader.cs` line for line, including its validation rules.

use super::model::{ActivationFnLine, ConnectionLine, NetFileError, NetFileModel};

/// A cursor over the physical lines of a text that skips comment lines transparently.
///
/// `next` yields `Some(Ok(line))` for each non-comment line (including blank lines, which act as
/// section terminators) and `None` at end of file. The reported `line` is the 1-based physical
/// line number, which keeps error messages aligned with the source file even when comments are
/// skipped.
struct LineCursor<'a> {
    lines: std::str::Lines<'a>,
    line_idx: usize,
}

impl<'a> LineCursor<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            lines: text.lines(),
            line_idx: 0,
        }
    }

    /// The most recent physical line number advanced past (1-based).
    fn current_line(&self) -> usize {
        self.line_idx
    }

    /// Read the next non-comment line, or `None` at end of file.
    fn next_line(&mut self) -> Option<&'a str> {
        for line in self.lines.by_ref() {
            self.line_idx += 1;
            if line.starts_with('#') {
                continue;
            }
            return Some(line);
        }
        None
    }

    /// Read the next non-empty, non-comment line. Returns an error if the file ends first.
    fn read_non_empty(&mut self) -> Result<&'a str, NetFileError> {
        while let Some(line) = self.next_line() {
            if !line.trim().is_empty() {
                return Ok(line);
            }
        }
        Err(NetFileError::UnexpectedEof {
            line: self.current_line(),
        })
    }

    /// Expect the next line to be empty or end of file (a section terminator).
    fn read_end_of_section(&mut self) -> Result<(), NetFileError> {
        match self.next_line() {
            None => Ok(()),
            Some(line) if line.trim().is_empty() => Ok(()),
            Some(_) => Err(NetFileError::ExpectedEndOfSection {
                line: self.current_line(),
            }),
        }
    }
}

/// Parse a `.net` file from its text contents.
pub fn parse(text: &str) -> Result<NetFileModel, NetFileError> {
    let mut cursor = LineCursor::new(text);

    let (input_count, output_count) = read_input_output_counts(&mut cursor)?;
    let (is_acyclic, cycles_per_activation) = read_cyclic_indicator(&mut cursor)?;
    let connections = read_connections(&mut cursor, input_count)?;
    let activation_fns = read_activation_functions(&mut cursor)?;

    NetFileModel::new(
        input_count,
        output_count,
        is_acyclic,
        cycles_per_activation,
        connections,
        activation_fns,
    )
}

fn read_input_output_counts(cursor: &mut LineCursor<'_>) -> Result<(usize, usize), NetFileError> {
    let line = cursor.read_non_empty()?;
    let line_no = cursor.current_line();
    let fields = split_fields(line);
    if fields.len() != 2 {
        return Err(NetFileError::InvalidNodeCounts {
            line: line_no,
            reason: "expected two whitespace-separated integers",
        });
    }
    let input_count = parse_usize(fields[0]).ok_or(NetFileError::InvalidNodeCounts {
        line: line_no,
        reason: "invalid input count",
    })?;
    let output_count = parse_usize(fields[1]).ok_or(NetFileError::InvalidNodeCounts {
        line: line_no,
        reason: "invalid output count",
    })?;
    if output_count < 1 {
        return Err(NetFileError::InvalidNodeCounts {
            line: line_no,
            reason: "output count must be at least 1",
        });
    }
    cursor.read_end_of_section()?;
    Ok((input_count, output_count))
}

fn read_cyclic_indicator(cursor: &mut LineCursor<'_>) -> Result<(bool, usize), NetFileError> {
    let line = cursor.read_non_empty()?;
    let line_no = cursor.current_line();
    let fields = split_fields(line);
    if fields.is_empty() || fields.len() > 2 {
        return Err(NetFileError::InvalidCyclicIndicator {
            line: line_no,
            reason: "expected 'acyclic' or 'cyclic <cycles>'",
        });
    }
    let is_acyclic = match fields[0] {
        "acyclic" => true,
        "cyclic" => false,
        _ => {
            return Err(NetFileError::InvalidCyclicIndicator {
                line: line_no,
                reason: "expected 'acyclic' or 'cyclic'",
            });
        }
    };
    let cycles = if is_acyclic {
        if fields.len() != 1 {
            return Err(NetFileError::InvalidCyclicIndicator {
                line: line_no,
                reason: "'acyclic' takes no extra fields",
            });
        }
        0
    } else {
        if fields.len() != 2 {
            return Err(NetFileError::InvalidCyclicIndicator {
                line: line_no,
                reason: "'cyclic' requires a cycles-per-activation integer",
            });
        }
        parse_usize(fields[1]).ok_or(NetFileError::InvalidCyclicIndicator {
            line: line_no,
            reason: "invalid cycles-per-activation",
        })?
    };
    cursor.read_end_of_section()?;
    Ok((is_acyclic, cycles))
}

fn read_connections(
    cursor: &mut LineCursor<'_>,
    input_count: usize,
) -> Result<Vec<ConnectionLine>, NetFileError> {
    let mut connections = Vec::new();
    while let Some(line) = cursor.next_line() {
        if line.trim().is_empty() {
            break;
        }
        let line_no = cursor.current_line();
        let fields = split_fields(line);
        if fields.len() != 3 {
            return Err(NetFileError::InvalidConnection {
                line: line_no,
                reason: "expected 'source target weight'",
            });
        }
        let src = parse_usize(fields[0]).ok_or(NetFileError::InvalidConnection {
            line: line_no,
            reason: "invalid source ID",
        })?;
        let tgt = parse_usize(fields[1]).ok_or(NetFileError::InvalidConnection {
            line: line_no,
            reason: "invalid target ID",
        })?;
        let weight = parse_weight(fields[2]).ok_or(NetFileError::InvalidConnection {
            line: line_no,
            reason: "invalid weight",
        })?;
        if tgt < input_count {
            return Err(NetFileError::InvalidConnection {
                line: line_no,
                reason: "target cannot be an input node",
            });
        }
        connections.push(ConnectionLine::new(src, tgt, weight));
    }
    Ok(connections)
}

fn read_activation_functions(
    cursor: &mut LineCursor<'_>,
) -> Result<Vec<ActivationFnLine>, NetFileError> {
    let mut fns = Vec::new();
    let mut expected_id = 0usize;
    while let Some(line) = cursor.next_line() {
        if line.trim().is_empty() {
            break;
        }
        let line_no = cursor.current_line();
        let fields = split_fields(line);
        if fields.len() != 2 {
            return Err(NetFileError::InvalidActivationFunction {
                line: line_no,
                reason: "expected '<id> <code>'",
            });
        }
        let id = parse_usize(fields[0]).ok_or(NetFileError::InvalidActivationFunction {
            line: line_no,
            reason: "invalid function ID",
        })?;
        if id != expected_id {
            return Err(NetFileError::InvalidActivationFunction {
                line: line_no,
                reason: "function IDs must be sequential starting at 0",
            });
        }
        fns.push(ActivationFnLine::new(id, fields[1]));
        expected_id += 1;
    }
    Ok(fns)
}

/// Split a line on ASCII whitespace, dropping empty fields.
fn split_fields(line: &str) -> Vec<&str> {
    line.split([' ', '\t']).filter(|s| !s.is_empty()).collect()
}

/// Parse a non-negative integer. Rejects empty strings and stray characters.
fn parse_usize(s: &str) -> Option<usize> {
    if s.is_empty() {
        return None;
    }
    let mut value = 0usize;
    for byte in s.bytes() {
        if !byte.is_ascii_digit() {
            return None;
        }
        value = value.checked_mul(10)?.checked_add((byte - b'0') as usize)?;
    }
    Some(value)
}

/// Parse a connection weight, accepting an optional leading sign, decimal point and exponent.
///
/// Rejects `inf` and `nan` so they cannot sneak in as weights, matching the restricted
/// `NumberStyles` used by SharpNeat's `TryParseDouble`.
fn parse_weight(s: &str) -> Option<f64> {
    let trimmed = s.trim();
    if trimmed.eq_ignore_ascii_case("inf")
        || trimmed.eq_ignore_ascii_case("infinity")
        || trimmed.eq_ignore_ascii_case("nan")
    {
        return None;
    }
    trimmed.parse::<f64>().ok().filter(|f| f.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE1: &str = "\
# Input and output node counts.
3 2

# Cyclic/acyclic indicator.
acyclic

# Connections (source target weight).
0 5 0.5
0 7 0.7
0 3 0.3
1 5 1.5
1 7 1.7
1 3 1.3
1 6 1.6
1 8 1.8
1 4 1.4
2 6 2.6
2 8 2.8
2 4 2.4

# Activation functions (functionId functionCode).
0 ReLU
1 Logistic
2 Sine
3 Gaussian

# Per node activation function (nodeId, functionId).
3 0
4 0
";

    #[test]
    fn parses_example1() {
        let model = parse(EXAMPLE1).unwrap();
        assert_eq!(model.input_count, 3);
        assert_eq!(model.output_count, 2);
        assert!(model.is_acyclic);
        assert_eq!(model.cycles_per_activation, 0);
        assert_eq!(model.connections.len(), 12);
        assert_eq!(model.connections[0], ConnectionLine::new(0, 5, 0.5));
        assert_eq!(model.connections[11], ConnectionLine::new(2, 4, 2.4));
        assert_eq!(model.activation_fns.len(), 4);
        assert_eq!(model.activation_fns[0], ActivationFnLine::new(0, "ReLU"));
        assert_eq!(
            model.activation_fns[3],
            ActivationFnLine::new(3, "Gaussian")
        );
    }

    #[test]
    fn parses_cyclic_with_cycles() {
        let text = "3 2\n\ncyclic 5\n\n0 3 1.0\n\n0 ReLU\n";
        let model = parse(text).unwrap();
        assert!(!model.is_acyclic);
        assert_eq!(model.cycles_per_activation, 5);
    }

    #[test]
    fn rejects_target_on_input_node() {
        let text = "3 2\n\nacyclic\n\n0 0 1.0\n\n0 ReLU\n";
        let err = parse(text).unwrap_err();
        assert!(matches!(err, NetFileError::InvalidConnection { .. }));
    }

    #[test]
    fn rejects_non_sequential_activation_ids() {
        let text = "1 1\n\nacyclic\n\n0 1 1.0\n\n0 ReLU\n2 Logistic\n";
        let err = parse(text).unwrap_err();
        assert!(matches!(
            err,
            NetFileError::InvalidActivationFunction { .. }
        ));
    }

    #[test]
    fn rejects_nan_weight() {
        let text = "1 1\n\nacyclic\n\n0 1 nan\n\n0 ReLU\n";
        assert!(parse(text).is_err());
    }

    #[test]
    fn parses_exponential_weights() {
        let text = "1 1\n\nacyclic\n\n0 1 1.5e2\n\n0 ReLU\n";
        let model = parse(text).unwrap();
        assert_eq!(model.connections[0].weight, 150.0);
    }

    #[test]
    fn tabs_are_valid_separators() {
        let text = "1\t1\n\nacyclic\n\n0\t1\t0.5\n\n0\tReLU\n";
        let model = parse(text).unwrap();
        assert_eq!(model.connections[0].weight, 0.5);
    }

    #[test]
    fn missing_activation_section_is_error() {
        let text = "1 1\n\nacyclic\n\n0 1 1.0\n\n";
        let err = parse(text).unwrap_err();
        assert_eq!(err, NetFileError::NoActivationFunction);
    }

    #[test]
    fn empty_file_is_error() {
        assert!(matches!(parse(""), Err(NetFileError::UnexpectedEof { .. })));
    }
}
