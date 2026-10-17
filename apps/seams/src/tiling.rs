//! The tiling check: analyses the result off the UI thread and draws what
//! it found over the canvas. Seam lines show red, repeated content orange.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

use egui::{Color32, ColorImage, Context, TextureHandle, TextureId, TextureOptions};
use seams_core::Image;
use seams_core::tiling::{TilingReport, analyse};

const SEAM_COLOR: [u8; 3] = [255, 72, 72];
const REPEAT_COLOR: [u8; 3] = [255, 168, 32];
/// Strongest opacity of the overlay, so the texture stays visible under it.
const SEAM_ALPHA: f32 = 220.0;
const REPEAT_ALPHA: f32 = 130.0;

#[derive(Default)]
pub struct TilingCheck {
    pub on: bool,
    /// The image the report and overlay describe.
    analysed: Option<Arc<Image>>,
    report: Option<TilingReport>,
    pending: Option<(Arc<Image>, Receiver<TilingReport>)>,
    overlay: Option<TextureHandle>,
}

impl TilingCheck {
    pub fn toggle(&mut self) {
        self.on = !self.on;
    }

    /// Picks up a finished analysis, and starts one when `image` is not the
    /// one analysed or being analysed.
    pub fn update(&mut self, ctx: &Context, image: Option<&Arc<Image>>) {
        if !self.on {
            return;
        }
        if let Some((for_image, inbox)) = &self.pending
            && let Ok(report) = inbox.try_recv()
        {
            self.overlay = Some(ctx.load_texture(
                "tiling-check",
                overlay_image(&report),
                TextureOptions::LINEAR,
            ));
            self.analysed = Some(Arc::clone(for_image));
            self.report = Some(report);
            self.pending = None;
        }
        let Some(image) = image else { return };
        let known = |held: Option<&Arc<Image>>| held.is_some_and(|held| Arc::ptr_eq(held, image));
        if known(self.analysed.as_ref()) || known(self.pending.as_ref().map(|(i, _)| i)) {
            return;
        }
        let (sender, inbox) = channel();
        let (job, repaint) = (Arc::clone(image), ctx.clone());
        let started =
            std::thread::Builder::new().name("seams-tiling".to_owned()).spawn(move || {
                let report = analyse(&job);
                // The app may have moved on; a closed channel is fine.
                let _ = sender.send(report);
                repaint.request_repaint();
            });
        match started {
            Ok(_) => self.pending = Some((Arc::clone(image), inbox)),
            Err(error) => log::warn!("could not start the tiling check: {error}"),
        }
    }

    /// The report for `image`, when it is up to date.
    pub fn report_for(&self, image: Option<&Arc<Image>>) -> Option<&TilingReport> {
        let current = match (&self.analysed, image) {
            (Some(analysed), Some(image)) => Arc::ptr_eq(analysed, image),
            _ => false,
        };
        self.report.as_ref().filter(|_| self.on && current)
    }

    pub fn is_running(&self) -> bool {
        self.on && self.pending.is_some()
    }

    /// The overlay to draw over `image`, when it is up to date.
    pub fn overlay_for(&self, image: Option<&Arc<Image>>) -> Option<TextureId> {
        self.report_for(image)?;
        self.overlay.as_ref().map(TextureHandle::id)
    }
}

/// Red lines on the seams and orange where content repeats, transparent
/// elsewhere, on the report's grid.
fn overlay_image(report: &TilingReport) -> ColorImage {
    let (width, height) = (report.width.max(1), report.height.max(1));
    let mut pixels = vec![Color32::TRANSPARENT; width * height];
    for y in 0..height {
        for x in 0..width {
            let seam = report
                .columns
                .get(x)
                .copied()
                .unwrap_or(0.0)
                .max(report.rows.get(y).copied().unwrap_or(0.0));
            let repeat = report.heat.get(y * width + x).copied().unwrap_or(0.0);
            let (color, alpha) = if seam * SEAM_ALPHA >= repeat * REPEAT_ALPHA {
                (SEAM_COLOR, seam * SEAM_ALPHA)
            } else {
                (REPEAT_COLOR, repeat * REPEAT_ALPHA)
            };
            if alpha >= 1.0 {
                let [r, g, b] = color;
                pixels[y * width + x] = Color32::from_rgba_unmultiplied(r, g, b, alpha as u8);
            }
        }
    }
    ColorImage::new([width, height], pixels)
}

/// One line describing the seams, for the inspector.
pub fn seam_summary(report: &TilingReport) -> String {
    let lines = report.columns.iter().chain(&report.rows).filter(|s| **s > 0.3).count();
    let edge = report.edge_seam();
    match (lines, edge > 0.3) {
        (0, _) => "No visible seams.".to_owned(),
        (_, true) => format!("Visible seam where the texture wraps ({:.0}%).", edge * 100.0),
        (n, false) => format!("{n} seam line{} inside the texture.", if n == 1 { "" } else { "s" }),
    }
}

/// One line describing repetition, for the inspector.
pub fn repeat_summary(report: &TilingReport) -> String {
    match report.repeat {
        None => "No obvious repetition.".to_owned(),
        Some(repeat) => format!(
            "Repeats {:.0}% alike at a shift of {:+.0}% across, {:+.0}% down.",
            repeat.similarity * 100.0,
            repeat.dx * 100.0,
            repeat.dy * 100.0
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seams_core::tiling::Repeat;

    fn report() -> TilingReport {
        TilingReport {
            width: 4,
            height: 2,
            columns: vec![1.0, 0.0, 0.0, 0.0],
            rows: vec![0.0, 0.0],
            seam: 1.0,
            repeat: Some(Repeat { dx: 0.5, dy: 0.0, similarity: 0.9 }),
            heat: vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    }

    #[test]
    fn the_overlay_marks_seams_red_and_repeats_orange() {
        let image = overlay_image(&report());

        // Colours are premultiplied: compare channels, not absolute values.
        let (seam, repeat) = (image.pixels[0], image.pixels[2]);
        assert!(seam.r() > 2 * seam.g(), "seam column is red: {seam:?}");
        assert!(
            repeat.g() > repeat.b() && repeat.r() > repeat.g(),
            "repeated cell is orange: {repeat:?}"
        );
        assert_eq!(image.pixels[1], Color32::TRANSPARENT);
    }

    #[test]
    fn summaries_name_what_was_found() {
        let found = report();
        assert!(seam_summary(&found).contains("wraps"));
        assert!(repeat_summary(&found).contains("90%"));

        let clean =
            TilingReport { columns: vec![0.0; 4], rows: vec![0.0; 2], repeat: None, ..found };
        assert_eq!(seam_summary(&clean), "No visible seams.");
        assert_eq!(repeat_summary(&clean), "No obvious repetition.");
    }

    #[test]
    fn a_report_belongs_only_to_the_image_it_was_made_for() {
        let (a, b) = (
            Arc::new(Image::filled(64, 64, 3, 0.5).unwrap()),
            Arc::new(Image::filled(64, 64, 3, 0.5).unwrap()),
        );
        let check = TilingCheck {
            on: true,
            analysed: Some(Arc::clone(&a)),
            report: Some(report()),
            ..TilingCheck::default()
        };

        assert!(check.report_for(Some(&a)).is_some());
        assert!(check.report_for(Some(&b)).is_none());
    }
}
