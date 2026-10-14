//! Modal dialogs: keyboard shortcuts, about, and the overwrite confirmation.

use std::path::Path;

use egui::{Context, Frame, Grid, Id, Margin, Modal, RichText, Stroke, Ui};

use crate::actions::{Action, WITH_SHORTCUTS};
use crate::app::{Dialog, SeamsApp};
use crate::theme::{BODY_SIZE, TOKENS, semibold};
use crate::widgets;

const DIALOG_WIDTH: f32 = 360.0;
const DIALOG_PADDING: i8 = 20;
const REPOSITORY_URL: &str = env!("CARGO_PKG_REPOSITORY");
const MOUSE_HINTS: [(&str, &str); 6] = [
    ("Scroll", "Zoom"),
    ("Shift / Alt + scroll", "Scroll sideways / up and down"),
    ("Middle drag, or Space + drag", "Pan"),
    ("Drag with the pan tool (H)", "Pan"),
    ("Double-click", "Fit to window"),
    ("Drag the divider", "Move the split"),
];

fn title(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text).font(semibold(BODY_SIZE + 2.0)));
    ui.add_space(6.0);
}

fn close_button(ui: &mut Ui) -> bool {
    ui.add_space(10.0);
    ui.button("Close").clicked()
}

fn shortcuts(ui: &mut Ui) -> bool {
    title(ui, "Keyboard shortcuts");
    Grid::new("shortcuts").num_columns(2).spacing([32.0, 6.0]).show(ui, |ui| {
        for command in WITH_SHORTCUTS {
            let Some(shortcut) = &command.shortcut else { continue };
            ui.label(command.label.trim_end_matches('…'));
            ui.label(RichText::new(ui.ctx().format_shortcut(shortcut)).color(TOKENS.text_dim));
            ui.end_row();
        }
        for (gesture, effect) in MOUSE_HINTS {
            ui.label(effect);
            ui.label(RichText::new(gesture).color(TOKENS.text_dim));
            ui.end_row();
        }
    });
    close_button(ui)
}

fn about(ui: &mut Ui, preview: &str) -> bool {
    title(ui, "SEAMS");
    ui.label(format!("Version {}", env!("CARGO_PKG_VERSION")));
    ui.label("Seamless texture and PBR map studio.");
    ui.add_space(6.0);
    widgets::hint(ui, preview);
    widgets::hint(ui, &format!("By {}. MIT license.", env!("CARGO_PKG_AUTHORS")));
    widgets::hint(ui, "Map generation follows the Materialize and NormalMap-Online conventions.");
    ui.add_space(6.0);
    ui.hyperlink_to("Project page and issue tracker", REPOSITORY_URL);
    close_button(ui)
}

/// Returns `Some(true)` to replace the file, `Some(false)` to cancel.
fn confirm_overwrite(ui: &mut Ui, path: &Path) -> Option<bool> {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    title(ui, &format!("Replace {name}?"));
    ui.label("The original file will be replaced with the processed base color.");
    widgets::hint(ui, "This cannot be undone.");
    ui.add_space(10.0);
    let mut decision = None;
    ui.horizontal(|ui| {
        if ui.button("Cancel").clicked() {
            decision = Some(false);
        }
        if ui.button(RichText::new("Replace file").color(TOKENS.danger)).clicked() {
            decision = Some(true);
        }
    });
    decision
}

pub fn show(app: &mut SeamsApp, ctx: &Context) {
    if app.dialog == Dialog::None {
        return;
    }
    let dialog = app.dialog.clone();
    let preview = app.preview_description();
    let frame = Frame::NONE
        .fill(TOKENS.popup)
        .stroke(Stroke::new(1.0, TOKENS.outline))
        .corner_radius(TOKENS.radius_lg)
        .inner_margin(Margin::same(DIALOG_PADDING));
    let modal = Modal::new(Id::new("dialog")).frame(frame).backdrop_color(TOKENS.scrim);
    let response = modal.show(ctx, |ui| {
        ui.set_width(DIALOG_WIDTH);
        match &dialog {
            Dialog::None => false,
            Dialog::Shortcuts => shortcuts(ui),
            Dialog::About => about(ui, &preview),
            Dialog::ConfirmOverwrite(path) => match confirm_overwrite(ui, path) {
                Some(true) => {
                    app.queue(Action::OverwriteSource);
                    true
                }
                Some(false) => true,
                None => false,
            },
        }
    });
    if response.inner || response.should_close() {
        app.dialog = Dialog::None;
    }
}
