//! Everything the user can do, as data.
//!
//! Menus, keyboard shortcuts and buttons all produce an [`Action`]; the app
//! executes them in one place. The command table below is the single source
//! for menu labels, shortcuts and the shortcuts dialog.

use std::path::PathBuf;

use egui::{Key, KeyboardShortcut, Modifiers};
use seams_engine::Stage;

use crate::canvas::ViewMode;

#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    Open,
    OpenPath(PathBuf),
    /// Quick save of the base colour, following the save mode setting.
    Save,
    SaveAs,
    ExportChannel,
    /// Replace the source file, after the user confirmed.
    OverwriteSource,
    /// Switch quick save between writing a new file and replacing the source.
    ToggleSaveMode,
    Exit,
    Undo,
    Redo,
    ApplyDelight,
    /// Open or close the Crop tool.
    ToggleCrop,
    /// Open the Crop tool in perspective mode.
    PerspectiveCrop,
    ApplyCrop,
    CancelCrop,
    SetStage(Stage),
    SetViewMode(ViewMode),
    ToggleGuides,
    /// Switch the canvas between the pan tool and normal pointing.
    TogglePanTool,
    /// Switch the canvas between the zoom tool and normal pointing.
    ToggleZoomTool,
    ZoomIn,
    ZoomOut,
    FitView,
    ActualSize,
    ShowShortcuts,
    ShowAbout,
}

/// A menu entry with its optional shortcut.
pub struct Command {
    pub action: Action,
    pub label: &'static str,
    pub shortcut: Option<KeyboardShortcut>,
}

const fn key(logical_key: Key) -> Option<KeyboardShortcut> {
    Some(KeyboardShortcut::new(Modifiers::NONE, logical_key))
}

const fn ctrl(logical_key: Key) -> Option<KeyboardShortcut> {
    Some(KeyboardShortcut::new(Modifiers::COMMAND, logical_key))
}

const fn ctrl_shift(logical_key: Key) -> Option<KeyboardShortcut> {
    Some(KeyboardShortcut::new(Modifiers::COMMAND.plus(Modifiers::SHIFT), logical_key))
}

const fn command(
    action: Action,
    label: &'static str,
    shortcut: Option<KeyboardShortcut>,
) -> Command {
    Command { action, label, shortcut }
}

pub const OPEN: Command = command(Action::Open, "Open texture…", ctrl(Key::O));
pub const SAVE: Command = command(Action::Save, "Save base color", ctrl(Key::S));
pub const SAVE_AS: Command = command(Action::SaveAs, "Save base color as…", ctrl_shift(Key::S));
pub const EXPORT: Command = command(Action::ExportChannel, "Export selected map…", ctrl(Key::E));
pub const SAVE_MODE: Command =
    command(Action::ToggleSaveMode, "Quick save replaces the original", None);
pub const EXIT: Command = command(Action::Exit, "Exit", ctrl(Key::Q));
pub const UNDO: Command = command(Action::Undo, "Undo", ctrl(Key::Z));
pub const REDO: Command = command(Action::Redo, "Redo", ctrl_shift(Key::Z));
pub const APPLY_DELIGHT: Command = command(Action::ApplyDelight, "Apply delight", None);
pub const CROP: Command = command(Action::ToggleCrop, "Crop", key(Key::C));
pub const PERSPECTIVE_CROP: Command = command(
    Action::PerspectiveCrop,
    "Perspective crop",
    Some(KeyboardShortcut::new(Modifiers::SHIFT, Key::C)),
);
pub const STAGE_DELIGHT: Command =
    command(Action::SetStage(Stage::Delight), "Delight", key(Key::Num1));
pub const STAGE_SEAMLESS: Command =
    command(Action::SetStage(Stage::Seamless), "Seamless", key(Key::Num2));
pub const STAGE_MATERIAL: Command =
    command(Action::SetStage(Stage::Material), "Material", key(Key::Num3));
pub const ZOOM_IN: Command = command(Action::ZoomIn, "Zoom in", ctrl(Key::Equals));
pub const ZOOM_OUT: Command = command(Action::ZoomOut, "Zoom out", ctrl(Key::Minus));
pub const FIT_VIEW: Command = command(Action::FitView, "Fit to window", ctrl(Key::Num0));
pub const ACTUAL_SIZE: Command = command(Action::ActualSize, "Actual size", ctrl(Key::Num1));
pub const TOGGLE_GUIDES: Command = command(Action::ToggleGuides, "Tile guides", ctrl(Key::G));
pub const PAN_TOOL: Command = command(Action::TogglePanTool, "Pan tool", key(Key::H));
pub const ZOOM_TOOL: Command = command(Action::ToggleZoomTool, "Zoom tool", key(Key::Z));
pub const SHORTCUTS: Command = command(Action::ShowShortcuts, "Keyboard shortcuts", key(Key::F1));
pub const ABOUT: Command = command(Action::ShowAbout, "About SEAMS", None);

/// Every command with a shortcut, in the order the shortcuts dialog lists them.
///
/// A shortcut also matches with extra Shift held, so `Ctrl+Shift+S` comes
/// before `Ctrl+S`, and modified shortcuts come before bare keys, so `Ctrl+1`
/// is matched before `1`. The order follows Photoshop's menus.
pub const WITH_SHORTCUTS: [&Command; 20] = [
    &OPEN,
    &SAVE_AS,
    &SAVE,
    &EXPORT,
    &EXIT,
    &REDO,
    &UNDO,
    &ZOOM_IN,
    &ZOOM_OUT,
    &FIT_VIEW,
    &ACTUAL_SIZE,
    &TOGGLE_GUIDES,
    &PAN_TOOL,
    &ZOOM_TOOL,
    &PERSPECTIVE_CROP,
    &CROP,
    &STAGE_DELIGHT,
    &STAGE_SEAMLESS,
    &STAGE_MATERIAL,
    &SHORTCUTS,
];

/// Second keys for commands, matched like their main shortcut but not listed
/// in the shortcuts dialog: `Ctrl+Y` redo (the Windows convention) and
/// `Ctrl++` zoom in (`=` with Shift held on most layouts).
pub const ALIASES: [(KeyboardShortcut, Action); 2] = [
    (KeyboardShortcut::new(Modifiers::COMMAND, Key::Y), Action::Redo),
    (KeyboardShortcut::new(Modifiers::COMMAND, Key::Plus), Action::ZoomIn),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_listed_command_has_a_unique_shortcut() {
        let shortcuts: Vec<KeyboardShortcut> =
            WITH_SHORTCUTS.iter().filter_map(|command| command.shortcut).collect();

        assert_eq!(shortcuts.len(), WITH_SHORTCUTS.len());
        for (i, a) in shortcuts.iter().enumerate() {
            assert!(shortcuts.iter().skip(i + 1).all(|b| a != b), "duplicate {a:?}");
        }
    }

    #[test]
    fn modified_shortcuts_are_matched_before_bare_keys() {
        let position = |command: &Command| {
            WITH_SHORTCUTS.iter().position(|c| c.label == command.label).unwrap()
        };

        assert!(position(&ACTUAL_SIZE) < position(&STAGE_DELIGHT));
        assert!(position(&SAVE_AS) < position(&SAVE));
        assert!(position(&REDO) < position(&UNDO));
        assert!(position(&PERSPECTIVE_CROP) < position(&CROP));
    }

    #[test]
    fn aliases_do_not_take_a_listed_shortcut() {
        let listed: Vec<KeyboardShortcut> =
            WITH_SHORTCUTS.iter().filter_map(|command| command.shortcut).collect();

        for (alias, _) in &ALIASES {
            assert!(!listed.contains(alias), "{alias:?} is already listed");
        }
    }
}
