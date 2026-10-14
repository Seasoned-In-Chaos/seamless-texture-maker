//! Naming helpers for exported files.

use std::path::{Path, PathBuf};

use crate::format::FileFormat;

const FALLBACK_STEM: &str = "texture";
const INVALID_CHARS: [char; 9] = ['<', '>', ':', '"', '/', '\\', '|', '?', '*'];

/// A file-system-safe version of `stem`, or a fallback when nothing is left.
pub fn sanitize_file_stem(stem: &str) -> String {
    let replaced: String = stem
        .trim()
        .chars()
        .map(|c| {
            if INVALID_CHARS.contains(&c) || c.is_control() || c.is_whitespace() { '_' } else { c }
        })
        .collect();
    let mut collapsed = String::with_capacity(replaced.len());
    for c in replaced.chars() {
        if c != '_' || !collapsed.ends_with('_') {
            collapsed.push(c);
        }
    }
    let trimmed = collapsed.trim_matches(|c| matches!(c, ' ' | '.' | '_'));
    if trimmed.is_empty() { FALLBACK_STEM.to_owned() } else { trimmed.to_owned() }
}

/// Where an export should go by default: beside the source, with `suffix`
/// added to its name and the extension of `format`.
pub fn output_path(source: Option<&Path>, suffix: &str, format: FileFormat) -> PathBuf {
    let stem = source.and_then(Path::file_stem).and_then(|s| s.to_str()).unwrap_or(FALLBACK_STEM);
    let name = format!("{}{suffix}.{}", sanitize_file_stem(stem), format.extension());
    match source.and_then(Path::parent) {
        Some(folder) => folder.join(name),
        None => PathBuf::from(name),
    }
}

/// A short human-readable size such as `715.2 KB`.
pub fn format_file_size(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    let size = bytes as f64;
    if size < KB {
        format!("{bytes} B")
    } else if size < KB * KB {
        format!("{:.1} KB", size / KB)
    } else {
        format!("{:.1} MB", size / (KB * KB))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_replaces_reserved_characters_and_collapses_runs() {
        assert_eq!(sanitize_file_stem("rock: wall?*  v2"), "rock_wall_v2");
        assert_eq!(sanitize_file_stem("  .hidden. "), "hidden");
    }

    #[test]
    fn sanitize_falls_back_when_nothing_usable_remains() {
        assert_eq!(sanitize_file_stem("???"), "texture");
        assert_eq!(sanitize_file_stem(""), "texture");
    }

    #[test]
    fn output_path_sits_beside_the_source() {
        let source = Path::new("textures").join("rock.jpg");

        let path = output_path(Some(&source), "_seamless", FileFormat::Png);

        assert_eq!(path, Path::new("textures").join("rock_seamless.png"));
    }

    #[test]
    fn output_path_without_a_source_uses_the_fallback_name() {
        let path = output_path(None, "_normal", FileFormat::Tiff);

        assert_eq!(path, PathBuf::from("texture_normal.tiff"));
    }

    #[test]
    fn file_sizes_use_sensible_units() {
        assert_eq!(format_file_size(512), "512 B");
        assert_eq!(format_file_size(732_396), "715.2 KB");
        assert_eq!(format_file_size(5 * 1024 * 1024), "5.0 MB");
    }
}
