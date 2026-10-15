//! The menu bar.

use egui::{Button, MenuBar, Ui};
use seams_engine::{Channel, SaveMode, Stage};

use crate::actions::{self, Action, Command};
use crate::app::SeamsApp;
use crate::canvas::{NavTool, ViewMode};

/// A menu entry for `command`; queues its action when chosen.
fn item(app: &mut SeamsApp, ui: &mut Ui, command: &Command, enabled: bool) {
    let mut button = Button::new(command.label);
    if let Some(shortcut) = &command.shortcut {
        button = button.shortcut_text(ui.ctx().format_shortcut(shortcut));
    }
    if ui.add_enabled(enabled, button).clicked() {
        app.queue(command.action.clone());
        ui.close();
    }
}

/// A menu entry that shows which of several options is active.
fn choice(app: &mut SeamsApp, ui: &mut Ui, command: &Command, selected: bool, enabled: bool) {
    let mut button = Button::selectable(selected, command.label);
    if let Some(shortcut) = &command.shortcut {
        button = button.shortcut_text(ui.ctx().format_shortcut(shortcut));
    }
    if ui.add_enabled(enabled, button).clicked() {
        app.queue(command.action.clone());
        ui.close();
    }
}

fn file_menu(app: &mut SeamsApp, ui: &mut Ui) {
    let can_save = app.session.can_export(Channel::BaseColor);
    let can_export = app.session.can_export(app.channel);
    item(app, ui, &actions::OPEN, true);
    ui.menu_button("Open recent", |ui| recent_files(app, ui));
    ui.separator();
    item(app, ui, &actions::SAVE, can_save);
    item(app, ui, &actions::SAVE_AS, can_save);
    item(app, ui, &actions::EXPORT, can_export);
    ui.separator();
    let overwrites = app.settings.save_mode == SaveMode::Overwrite;
    choice(app, ui, &actions::SAVE_MODE, overwrites, true);
    ui.separator();
    item(app, ui, &actions::EXIT, true);
}

fn recent_files(app: &mut SeamsApp, ui: &mut Ui) {
    let recent: Vec<_> =
        app.settings.recent_files.iter().filter(|path| path.is_file()).cloned().collect();
    if recent.is_empty() {
        ui.add_enabled(false, Button::new("No recent textures"));
        return;
    }
    for path in recent {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if ui.button(name).on_hover_text(path.display().to_string()).clicked() {
            app.queue(Action::OpenPath(path));
            ui.close();
        }
    }
}

fn edit_menu(app: &mut SeamsApp, ui: &mut Ui) {
    let (can_undo, can_redo) = (app.session.can_undo(), app.session.can_redo());
    let can_apply = app.session.can_apply_delight();
    item(app, ui, &actions::UNDO, can_undo);
    item(app, ui, &actions::REDO, can_redo);
    ui.separator();
    let (has_document, cropping) = (app.has_document(), app.crop.is_some());
    choice(app, ui, &actions::CROP, cropping, has_document);
    item(app, ui, &actions::PERSPECTIVE_CROP, has_document);
    let retouching = app.retouch.is_some();
    choice(app, ui, &actions::RETOUCH, retouching, has_document);
    let can_bake = app.session.can_bake_seamless();
    item(app, ui, &actions::BAKE_SEAMLESS, can_bake);
    item(app, ui, &actions::APPLY_DELIGHT, can_apply);
}

fn view_menu(app: &mut SeamsApp, ui: &mut Ui) {
    let has_document = app.has_document();
    let stage = app.session.stage();
    choice(app, ui, &actions::STAGE_DELIGHT, stage == Stage::Delight, has_document);
    choice(app, ui, &actions::STAGE_SEAMLESS, stage == Stage::Seamless, has_document);
    choice(app, ui, &actions::STAGE_MATERIAL, stage == Stage::Material, has_document);
    ui.separator();
    for (mode, label) in ViewMode::ALL {
        let selected = app.canvas.mode == mode;
        if ui.add_enabled(has_document, Button::selectable(selected, label)).clicked() {
            app.queue(Action::SetViewMode(mode));
            ui.close();
        }
    }
    let guides = app.settings.view.show_guides;
    choice(app, ui, &actions::TOGGLE_GUIDES, guides, has_document);
    let checking = app.tiling.on;
    choice(app, ui, &actions::TILING_CHECK, checking, has_document);
    let tool = app.canvas.tool;
    choice(app, ui, &actions::PAN_TOOL, tool == NavTool::Pan, has_document);
    choice(app, ui, &actions::ZOOM_TOOL, tool == NavTool::Zoom, has_document);
    ui.separator();
    item(app, ui, &actions::ZOOM_IN, has_document);
    item(app, ui, &actions::ZOOM_OUT, has_document);
    item(app, ui, &actions::FIT_VIEW, has_document);
    item(app, ui, &actions::ACTUAL_SIZE, has_document);
}

pub fn show(app: &mut SeamsApp, ui: &mut Ui) {
    MenuBar::new().ui(ui, |ui| {
        ui.menu_button("File", |ui| file_menu(app, ui));
        ui.menu_button("Edit", |ui| edit_menu(app, ui));
        ui.menu_button("View", |ui| view_menu(app, ui));
        ui.menu_button("Help", |ui| {
            item(app, ui, &actions::SHORTCUTS, true);
            item(app, ui, &actions::ABOUT, true);
        });
    });
}
