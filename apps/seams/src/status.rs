//! The status bar: document facts, the latest message and current activity.

use egui::{Align, Layout, RichText, Ui};
use seams_engine::Activity;
use seams_io::format_file_size;

use crate::app::SeamsApp;
use crate::theme::{SMALL_SIZE, TOKENS};

fn dim(text: impl Into<String>) -> RichText {
    RichText::new(text).size(SMALL_SIZE).color(TOKENS.text_dim)
}

/// A one-line description of the open file.
fn document_summary(app: &SeamsApp) -> Option<String> {
    let document = app.session.document()?;
    let meta = document.meta();
    // The size of the base colour as it is now: a crop changes it.
    let base = document.base();
    Some(format!(
        "{}  ·  {} × {} px  ·  {}  ·  {}",
        document.file_name(),
        base.width(),
        base.height(),
        meta.bit_depth.label(),
        format_file_size(meta.file_size),
    ))
}

fn activity_label(activity: Activity) -> Option<&'static str> {
    if activity.loading {
        Some("Opening…")
    } else if activity.saving {
        Some("Saving…")
    } else if activity.rendering {
        Some("Rendering…")
    } else if activity.generating_maps {
        Some("Generating maps…")
    } else {
        None
    }
}

pub fn show(app: &mut SeamsApp, ui: &mut Ui) {
    ui.horizontal_centered(|ui| {
        match document_summary(app) {
            Some(summary) => ui.label(dim(summary)),
            None => ui.label(dim("No texture open")),
        };
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if let Some(label) = activity_label(app.session.activity()) {
                ui.label(dim(label));
                ui.add(egui::Spinner::new().size(12.0));
                ui.separator();
            }
            if let Some(notice) = &app.notice {
                let color = if notice.is_error { TOKENS.danger } else { TOKENS.text };
                ui.label(RichText::new(&notice.text).size(SMALL_SIZE).color(color));
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_most_blocking_activity_is_reported_first() {
        let everything =
            Activity { loading: true, rendering: true, generating_maps: true, saving: true };
        let background = Activity { rendering: true, generating_maps: true, ..Activity::default() };

        assert_eq!(activity_label(everything), Some("Opening…"));
        assert_eq!(activity_label(background), Some("Rendering…"));
        assert_eq!(activity_label(Activity::default()), None);
    }
}
