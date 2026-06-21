//! Serialiser for SharpNeat's `.net` file format.
//!
//! Produces output that round-trips through [`super::reader::parse`]. The layout, section headers
//! and tab separators match SharpNeat's `NetFileWriter.cs`.

use super::model::{ActivationFnLine, ConnectionLine, NetFileModel};

/// Render `model` as `.net` file text, appending it to `out`.
pub fn write(model: &NetFileModel, out: &mut String) {
    // Section 1: node counts.
    out.push_str("# Input and output node counts.\n");
    out.push_str(&format!(
        "{}\t{}\n\n",
        model.input_count, model.output_count
    ));

    // Section 2: cyclic/acyclic indicator.
    out.push_str("# Cyclic/acyclic indicator.\n");
    if model.is_acyclic {
        out.push_str("acyclic\n\n");
    } else {
        out.push_str(&format!("cyclic\t{}\n\n", model.cycles_per_activation));
    }

    // Section 3: connections.
    out.push_str("# Connections (source target weight).\n");
    for ConnectionLine {
        source_id,
        target_id,
        weight,
    } in &model.connections
    {
        // `{}` renders f64 with the shortest round-trippable representation, matching C#'s `:R`.
        out.push_str(&format!("{source_id}\t{target_id}\t{weight}\n"));
    }
    out.push('\n');

    // Section 4: activation functions.
    out.push_str("# Activation functions (functionId functionCode).\n");
    for ActivationFnLine { id, code } in &model.activation_fns {
        out.push_str(&format!("{id}\t{code}\n"));
    }
    out.push('\n');
}

#[cfg(test)]
mod tests {
    use super::super::reader::parse;
    use super::*;

    fn sample_model() -> NetFileModel {
        NetFileModel::create_acyclic(
            3,
            2,
            vec![
                ConnectionLine::new(0, 5, 0.5),
                ConnectionLine::new(0, 7, 0.7),
                ConnectionLine::new(2, 4, 2.4),
            ],
            vec![
                ActivationFnLine::new(0, "ReLU"),
                ActivationFnLine::new(1, "Logistic"),
            ],
        )
        .unwrap()
    }

    #[test]
    fn round_trip_preserves_data() {
        let original = sample_model();
        let mut text = String::new();
        write(&original, &mut text);
        let reparsed = parse(&text).unwrap();
        assert_eq!(reparsed.input_count, original.input_count);
        assert_eq!(reparsed.output_count, original.output_count);
        assert_eq!(reparsed.is_acyclic, original.is_acyclic);
        assert_eq!(reparsed.connections, original.connections);
        assert_eq!(reparsed.activation_fns, original.activation_fns);
    }

    #[test]
    fn cyclic_model_writes_cycles() {
        let model = NetFileModel::create_cyclic(
            3,
            2,
            3,
            vec![ConnectionLine::new(0, 5, 3.0)],
            vec![ActivationFnLine::new(0, "ReLU")],
        )
        .unwrap();
        let mut text = String::new();
        write(&model, &mut text);
        assert!(text.contains("cyclic\t3"));
        let reparsed = parse(&text).unwrap();
        assert!(!reparsed.is_acyclic);
        assert_eq!(reparsed.cycles_per_activation, 3);
    }

    #[test]
    fn weight_round_trips_at_full_precision() {
        let tricky = 5.123456789012345;
        let model = NetFileModel::create_acyclic(
            1,
            1,
            vec![ConnectionLine::new(0, 1, tricky)],
            vec![ActivationFnLine::new(0, "ReLU")],
        )
        .unwrap();
        let mut text = String::new();
        write(&model, &mut text);
        let reparsed = parse(&text).unwrap();
        assert_eq!(reparsed.connections[0].weight, tricky);
    }
}
