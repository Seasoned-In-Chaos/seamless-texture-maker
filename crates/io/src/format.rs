//! File formats, sample depths and the metadata kept alongside an image.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{IoError, Result};

/// A file format SEAMS can read and write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileFormat {
    #[default]
    Png,
    Jpeg,
    Tiff,
    Tga,
    Exr,
}

impl FileFormat {
    pub const ALL: [FileFormat; 5] =
        [FileFormat::Png, FileFormat::Jpeg, FileFormat::Tiff, FileFormat::Tga, FileFormat::Exr];

    /// Extensions accepted when opening files.
    pub const EXTENSIONS: [&'static str; 7] = ["png", "jpg", "jpeg", "tif", "tiff", "tga", "exr"];

    pub fn label(self) -> &'static str {
        match self {
            FileFormat::Png => "PNG",
            FileFormat::Jpeg => "JPG",
            FileFormat::Tiff => "TIFF",
            FileFormat::Tga => "TGA",
            FileFormat::Exr => "EXR",
        }
    }

    /// The extension used when SEAMS names a file itself.
    pub fn extension(self) -> &'static str {
        match self {
            FileFormat::Png => "png",
            FileFormat::Jpeg => "jpg",
            FileFormat::Tiff => "tiff",
            FileFormat::Tga => "tga",
            FileFormat::Exr => "exr",
        }
    }

    pub fn from_extension(extension: &str) -> Option<Self> {
        match extension.to_ascii_lowercase().as_str() {
            "png" => Some(FileFormat::Png),
            "jpg" | "jpeg" => Some(FileFormat::Jpeg),
            "tif" | "tiff" => Some(FileFormat::Tiff),
            "tga" => Some(FileFormat::Tga),
            "exr" => Some(FileFormat::Exr),
            _ => None,
        }
    }

    /// The format a path's extension names.
    pub fn from_path(path: &Path) -> Result<Self> {
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or_default();
        Self::from_extension(extension)
            .ok_or_else(|| IoError::UnsupportedFormat(extension.to_owned()))
    }

    /// Whether the format can store 16-bit integer samples.
    pub fn supports_sixteen_bit(self) -> bool {
        matches!(self, FileFormat::Png | FileFormat::Tiff)
    }

    pub fn supports_alpha(self) -> bool {
        !matches!(self, FileFormat::Jpeg)
    }
}

/// How many bits each sample holds in a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BitDepth {
    #[default]
    Eight,
    Sixteen,
    Float,
}

impl BitDepth {
    pub fn label(self) -> &'static str {
        match self {
            BitDepth::Eight => "8-bit",
            BitDepth::Sixteen => "16-bit",
            BitDepth::Float => "32-bit float",
        }
    }
}

/// What is known about an image file besides its pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageMeta {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub bit_depth: BitDepth,
    pub format: FileFormat,
    /// Horizontal and vertical resolution in dots per inch, when recorded.
    pub dpi: Option<(f32, f32)>,
    pub file_size: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_lookup_is_case_insensitive() {
        assert_eq!(FileFormat::from_extension("JPEG"), Some(FileFormat::Jpeg));
        assert_eq!(FileFormat::from_extension("Tif"), Some(FileFormat::Tiff));
        assert_eq!(FileFormat::from_extension("bmp"), None);
    }

    #[test]
    fn every_format_round_trips_through_its_extension() {
        for format in FileFormat::ALL {
            assert_eq!(FileFormat::from_extension(format.extension()), Some(format));
        }
    }

    #[test]
    fn unsupported_path_reports_the_extension() {
        let error = FileFormat::from_path(Path::new("texture.webp")).unwrap_err();

        assert!(matches!(error, IoError::UnsupportedFormat(ext) if ext == "webp"));
    }

    #[test]
    fn path_without_extension_is_unsupported() {
        assert!(FileFormat::from_path(Path::new("texture")).is_err());
    }
}
