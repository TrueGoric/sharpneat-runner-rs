//! Loading and saving of SharpNeat `.net` files.
//!
//! The text-based parser in [`reader`] and serialiser in [`writer`] operate on strings. The
//! [`NetFile`] helper exposes convenience functions for reading from and writing to files and
//! streams, analogous to SharpNeat's `NetFile.Load` / `NetFile.Save`.

mod model;
mod reader;
mod writer;

pub use model::{ActivationFnLine, ConnectionLine, NetFileError, NetFileModel};
pub use reader::parse as parse_netfile;
pub use writer::write as write_netfile;

use std::fs;
use std::io::{Read, Write};
use std::path::Path;

/// Convenience helpers for loading and saving `.net` files, mirroring SharpNeat's `NetFile` class.
pub struct NetFile;

impl NetFile {
    /// Load a [`NetFileModel`] from a file at `path`.
    pub fn load(path: impl AsRef<Path>) -> Result<NetFileModel, NetFileError> {
        let text = fs::read_to_string(path).map_err(NetFileError::from)?;
        Self::read_from_str(&text)
    }

    /// Parse a [`NetFileModel`] from an in-memory string.
    pub fn read_from_str(text: &str) -> Result<NetFileModel, NetFileError> {
        reader::parse(text)
    }

    /// Read a [`NetFileModel`] from any readable source. The entire contents are buffered into
    /// memory first, matching the line-oriented structure of the format.
    pub fn load_stream(mut reader: impl Read) -> Result<NetFileModel, NetFileError> {
        let mut text = String::new();
        reader
            .read_to_string(&mut text)
            .map_err(NetFileError::from)?;
        Self::read_from_str(&text)
    }

    /// Serialise `model` and write it to a file at `path`, replacing any existing contents.
    pub fn save(path: impl AsRef<Path>, model: &NetFileModel) -> Result<(), NetFileError> {
        let text = Self::write_to_string(model);
        fs::write(path, text).map_err(NetFileError::from)
    }

    /// Serialise `model` to any writable sink.
    pub fn save_stream(mut writer: impl Write, model: &NetFileModel) -> Result<(), NetFileError> {
        let text = Self::write_to_string(model);
        writer
            .write_all(text.as_bytes())
            .map_err(NetFileError::from)
    }

    /// Serialise `model` to an in-memory string.
    pub fn write_to_string(model: &NetFileModel) -> String {
        let mut out = String::new();
        writer::write(model, &mut out);
        out
    }
}
