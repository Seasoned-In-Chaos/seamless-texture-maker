//! The channel strip: one thumbnail per image the document can show.

use egui::{Align2, Color32, FontId, Rect, Sense, Stroke, StrokeKind, Ui, pos2, vec2};
use seams_engine::Channel;

use crate::app::SeamsApp;
use crate::textures::Texture;
use crate::theme::{MOTION, SMALL_SIZE, TOKENS, mix};

const THUMBNAIL_SIDE: f32 = 54.0;
const CELL_WIDTH: f32 = 84.0;
const CELL_HEIGHT: f32 = 80.0;
/// Gap between a selected thumbnail and its accent ring.
const RING_GAP: f32 = 3.0;
const FULL_UV: Rect = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));

/// The largest rectangle with the image's aspect ratio that fits `frame`.
fn fit_inside(frame: Rect, image_size: egui::Vec2) -> Rect {
    let scale = (frame.width() / image_size.x.max(1.0)).min(frame.height() / image_size.y.max(1.0));
    Rect::from_center_size(frame.center(), image_size * scale)
}

/// A thumbnail and its name. Selection is a thin accent ring set off from
/// the image; hover brightens the label and shows the ring in a neutral tone.
fn thumbnail(app: &mut SeamsApp, ui: &mut Ui, channel: Channel, texture: Option<Texture>) {
    let (cell, response) = ui.allocate_exact_size(vec2(CELL_WIDTH, CELL_HEIGHT), Sense::click());
    let selected = app.channel == channel;
    let ctx = ui.ctx();
    let hover = ctx.animate_bool_with_time(response.id.with("hover"), response.hovered(), MOTION);
    let chosen = ctx.animate_bool_with_time(response.id.with("chosen"), selected, MOTION);
    let painter = ui.painter();
    let frame = Rect::from_center_size(
        pos2(cell.center().x, cell.top() + RING_GAP + 2.0 + THUMBNAIL_SIDE / 2.0),
        vec2(THUMBNAIL_SIDE, THUMBNAIL_SIDE),
    );
    painter.rect_filled(frame, TOKENS.radius_sm, TOKENS.raised);
    if let Some(texture) = texture {
        let image = egui::epaint::RectShape::filled(
            fit_inside(frame, texture.full_size),
            TOKENS.radius_sm,
            Color32::WHITE,
        )
        .with_texture(texture.id, FULL_UV);
        painter.add(image);
    }
    // A light rim keeps dark maps from dissolving into the strip.
    let rim = Stroke::new(1.0, TOKENS.sheen);
    painter.rect_stroke(frame, TOKENS.radius_sm, rim, StrokeKind::Inside);
    // The ring is neutral while hovered and turns accent once selected.
    let ring_color = mix(TOKENS.raised_active.gamma_multiply(hover), TOKENS.accent, chosen);
    let radius = f32::from(TOKENS.radius_sm) + RING_GAP;
    let ring = Stroke::new(1.5, ring_color);
    painter.rect_stroke(frame.expand(RING_GAP), radius, ring, StrokeKind::Inside);
    let color = mix(TOKENS.text_dim, TOKENS.text, hover.max(chosen));
    let anchor = pos2(cell.center().x, cell.bottom() - 2.0);
    let font = FontId::proportional(SMALL_SIZE);
    painter.text(anchor, Align2::CENTER_BOTTOM, channel.label(), font, color);
    if response.on_hover_text(format!("Show {}", channel.label().to_lowercase())).clicked() {
        app.channel = channel;
    }
}

pub fn show(app: &mut SeamsApp, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    ui.horizontal_centered(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        // Centred under the canvas, like the image and the view bar above.
        let row = CELL_WIDTH * Channel::ALL.len() as f32;
        ui.add_space(((ui.available_width() - row) / 2.0).max(0.0));
        for channel in Channel::ALL {
            let texture = app.channel_texture(&ctx, channel);
            thumbnail(app, ui, channel, texture);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thumbnails_keep_the_image_aspect_ratio() {
        let frame = Rect::from_min_size(pos2(0.0, 0.0), vec2(60.0, 60.0));

        let wide = fit_inside(frame, vec2(400.0, 100.0));
        let tall = fit_inside(frame, vec2(100.0, 400.0));

        assert_eq!(wide.size(), vec2(60.0, 15.0));
        assert_eq!(tall.size(), vec2(15.0, 60.0));
        assert_eq!(wide.center(), frame.center());
    }
}
