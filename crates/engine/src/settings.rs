//! Settings remembered between runs.

use std::path::{Path, PathBuf};

use seams_io::FileFormat;
use serde::{Deserialize, Serialize};

use crate::model::{Params, SaveMode, Stage};

const APP_FOLDER: &str = "SeamlessTextureMaker";
const SETTINGS_FILE: &str = "settings.json";
const MAX_RECENT_FILES: usize = 10;
const MIN_TILES: u8 = 2;
const MAX_TILES: u8 = 4;

/// How the canvas is shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ViewSettings {
    /// Repeats per side in the tiled view, `2..=4`.
    pub tiles: u8,
    /// Draw lines on tile borders in the tiled view.
    pub show_guides: bool,
}

impl Default for ViewSettings {
    fn default() -> Self {
        Self { tiles: MIN_TILES, show_guides: true }
    }
}

/// Size of the main window when it was last closed.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WindowSettings {
    pub width: f32,
    pub height: f32,
    pub maximized: bool,
}

/// Everything persisted to the settings file.
///
/// Unknown fields are ignored and missing ones take their defaults, so the
/// file stays readable across versions.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub params: Params,
    pub stage: Stage,
    pub export_format: FileFormat,
    pub save_mode: SaveMode,
    pub last_directory: Option<PathBuf>,
    pub recent_files: Vec<PathBuf>,
    pub view: ViewSettings,
    pub window: Option<WindowSettings>,
}

impl Settings {
    /// Where the settings file lives, when the platform has a config folder.
    pub fn default_path() -> Option<PathBuf> {
        app_data_dir().map(|dir| dir.join(SETTINGS_FILE))
    }

    /// Reads settings, falling back to defaults when the file is missing or
    /// unreadable. A corrupt file must never stop the app from starting.
    pub fn load(path: &Path) -> Self {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Self::default(),
            Err(error) => {
                log::warn!("could not read settings {}: {error}", path.display());
                return Self::default();
            }
        };
        match serde_json::from_str::<Self>(&text) {
            Ok(settings) => settings.sanitized(),
            Err(error) => {
                log::warn!("ignoring invalid settings {}: {error}", path.display());
                Self::default()
            }
        }
    }

    /// Writes settings through a temporary file, so a crash mid-write cannot
    /// leave a truncated file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        let temp = path.with_extension("json.tmp");
        std::fs::write(&temp, json)?;
        std::fs::rename(&temp, path)
    }

    /// Moves `path` to the front of the recent list and remembers its folder.
    pub fn remember_file(&mut self, path: &Path) {
        self.recent_files.retain(|recent| recent != path);
        self.recent_files.insert(0, path.to_owned());
        self.recent_files.truncate(MAX_RECENT_FILES);
        self.last_directory = path.parent().map(Path::to_owned);
    }

    /// Forces values that came from disk back into their valid ranges.
    fn sanitized(mut self) -> Self {
        self.params.delight = self.params.delight.sanitized();
        self.params.seamless = self.params.seamless.sanitized();
        self.view.tiles = self.view.tiles.clamp(MIN_TILES, MAX_TILES);
        self.recent_files.truncate(MAX_RECENT_FILES);
        self.last_directory = self.last_directory.filter(|dir| !dir.as_os_str().is_empty());
        self.window = self.window.filter(|w| w.width.is_finite() && w.height.is_finite());
        self
    }
}

/// The per-user folder for settings and logs.
pub fn app_data_dir() -> Option<PathBuf> {
    let base = std::env::var_os("APPDATA")
        .or_else(|| std::env::var_os("XDG_CONFIG_HOME"))
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    Some(base.join(APP_FOLDER))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_gives_defaults() {
        let dir = tempfile::tempdir().unwrap();

        assert_eq!(Settings::load(&dir.path().join("none.json")), Settings::default());
    }

    #[test]
    fn corrupt_file_gives_defaults_instead_of_failing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(&path, "{ not json at all").unwrap();

        assert_eq!(Settings::load(&path), Settings::default());
    }

    #[test]
    fn settings_round_trip_through_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("settings.json");
        let mut settings = Settings {
            export_format: FileFormat::Tiff,
            window: Some(WindowSettings { width: 1400.0, height: 900.0, maximized: true }),
            ..Settings::default()
        };
        settings.params.seamless.overlap.x = 0.33;
        settings.view.tiles = 3;

        settings.save(&path).unwrap();

        assert_eq!(Settings::load(&path), settings);
        assert!(!path.with_extension("json.tmp").exists());
    }

    #[test]
    fn out_of_range_values_from_disk_are_clamped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let json = r#"{"view":{"tiles":99},"params":{"seamless":{"overlap":{"x":7.0}}},
                       "last_directory":""}"#;
        std::fs::write(&path, json).unwrap();

        let settings = Settings::load(&path);

        assert_eq!(settings.view.tiles, MAX_TILES);
        assert_eq!(settings.params.seamless.overlap.x, 0.5);
        assert_eq!(settings.last_directory, None);
    }

    #[test]
    fn settings_written_by_the_previous_python_app_still_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let legacy = r#"{"blend_strength":0.5,"export_format":"png","save_mode":"new_file",
                         "last_directory":"","last_file":"","recent_files":[],
                         "active_tool":"home","window_width":1200,"window_height":800}"#;
        std::fs::write(&path, legacy).unwrap();

        let settings = Settings::load(&path);

        assert_eq!(settings.export_format, FileFormat::Png);
        assert_eq!(settings.save_mode, SaveMode::NewFile);
    }

    #[test]
    fn remember_file_deduplicates_and_caps_the_list() {
        let mut settings = Settings::default();
        for i in 0..15 {
            settings.remember_file(&PathBuf::from(format!("dir/file{i}.png")));
        }

        settings.remember_file(&PathBuf::from("dir/file12.png"));

        assert_eq!(settings.recent_files.len(), MAX_RECENT_FILES);
        assert_eq!(settings.recent_files[0], PathBuf::from("dir/file12.png"));
        assert_eq!(settings.recent_files.iter().filter(|p| p.ends_with("file12.png")).count(), 1);
        assert_eq!(settings.last_directory, Some(PathBuf::from("dir")));
    }
}
