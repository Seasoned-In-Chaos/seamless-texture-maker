//! The Retouch tool: Photoshop's Clone Stamp and Healing Brush, for
//! textures.
//!
//! - Alt-click sets where to copy from. The first stroke after that fixes
//!   the distance between the brush and the source, and later strokes keep
//!   it (Photoshop's Aligned option).
//! - A drag paints. Clone copies exactly; Heal copies the source's detail
//!   but matches the colour and brightness where it lands.
//! - With Wrap on, strokes continue across the edges, so a tileable texture
//!   stays tileable; in the Tiled view a stroke can cross a tile border.
//! - `[` and `]` change the size. Each stroke is one undo step.
//!
//! A stroke paints a working copy of the base color and shows it live in a
//! texture of its own; releasing the button hands the copy back to commit.

use std::sync::Arc;

use egui::{
    Color32, ColorImage, Context, CursorIcon, Key, Painter, PointerButton, Pos2, Response, Stroke,
    TextureHandle, TextureOptions, Ui, Vec2, vec2,
};
use seams_core::Image;
use seams_core::retouch::{Brush, Region, RetouchMode, dab, stroke_dabs};
use seams_engine::{DISPLAY_MAX_SIDE, DisplayImage};

use crate::crop::Mapping;
use crate::icons;
use crate::textures::Texture;
use crate::theme::TOKENS;

/// Brush sizes the tool allows, in image pixels.
pub const MIN_RADIUS: f32 = 1.0;
pub const MAX_RADIUS: f32 = 500.0;
/// `[` and `]` change the radius by this factor.
const RADIUS_STEP: f32 = 1.25;
const LIVE_SAMPLING: TextureOptions =
    TextureOptions { magnification: egui::TextureFilter::Nearest, ..TextureOptions::LINEAR };

/// A stroke in progress.
struct StrokeState {
    /// The base color when the stroke began: what every dab reads.
    snapshot: Arc<Image>,
    /// The base color being painted.
    working: Image,
    /// The last dab's centre, in image pixels (unwrapped).
    last: (f32, f32),
}

/// The live view of the working copy while it differs from the session's.
struct Live {
    handle: TextureHandle,
    /// Display pixels per image pixel.
    ratio: f32,
    size: [usize; 2],
}

pub struct RetouchTool {
    pub mode: RetouchMode,
    pub brush: Brush,
    /// Paint and sample across the edges, as the texture tiles.
    pub wrap: bool,
    /// The base color the tool paints on.
    base: Option<Arc<Image>>,
    /// Where Alt-click asked to copy from, until a stroke fixes the offset.
    source: Option<Pos2>,
    /// Source minus brush, fixed by the first stroke after Alt-click.
    offset: Option<Vec2>,
    stroke: Option<StrokeState>,
    live: Option<Live>,
    /// A finished stroke, waiting for the app to commit it.
    finished: Option<Image>,
    /// Something to tell the user, waiting for the app to show it.
    message: Option<&'static str>,
    /// The brush outline and source mark this frame (screen points).
    cursor: Option<(Pos2, f32, Option<Pos2>)>,
}

impl Default for RetouchTool {
    fn default() -> Self {
        Self {
            mode: RetouchMode::Heal,
            brush: Brush::default(),
            wrap: true,
            base: None,
            source: None,
            offset: None,
            stroke: None,
            live: None,
            finished: None,
            message: None,
            cursor: None,
        }
    }
}

impl RetouchTool {
    /// Points the tool at the current base color. A new size forgets the
    /// source; a committed stroke's result drops the live view.
    pub fn sync_base(&mut self, base: &Arc<Image>) {
        if self.base.as_ref().is_some_and(|b| Arc::ptr_eq(b, base)) {
            return;
        }
        let resized = self
            .base
            .as_ref()
            .is_none_or(|b| (b.width(), b.height()) != (base.width(), base.height()));
        if resized {
            self.source = None;
            self.offset = None;
        }
        // A stroke still painting keeps its own copy; it commits when released.
        if self.stroke.is_none() {
            self.live = None;
        }
        self.base = Some(Arc::clone(base));
    }

    /// The live texture while a stroke is shown from the working copy.
    pub fn texture(&self) -> Option<Texture> {
        let live = self.live.as_ref()?;
        let base = self.base.as_ref()?;
        Some(Texture {
            id: live.handle.id(),
            full_size: vec2(base.width() as f32, base.height() as f32),
        })
    }

    pub fn take_finished(&mut self) -> Option<Image> {
        self.finished.take()
    }

    pub fn take_message(&mut self) -> Option<&'static str> {
        self.message.take()
    }

    pub fn has_source(&self) -> bool {
        self.source.is_some() || self.offset.is_some()
    }

    /// `[` and `]` change the size, unless a text field has the keyboard.
    pub fn keys(&mut self, ctx: &Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let (smaller, larger) =
            ctx.input(|i| (i.key_pressed(Key::OpenBracket), i.key_pressed(Key::CloseBracket)));
        let factor = match (smaller, larger) {
            (true, false) => 1.0 / RADIUS_STEP,
            (false, true) => RADIUS_STEP,
            _ => return,
        };
        self.brush.radius = (self.brush.radius * factor).clamp(MIN_RADIUS, MAX_RADIUS);
    }

    /// Handles Alt-click and painting. `tiled` maps the pointer over a tile
    /// grid back into the image.
    pub fn interact(
        &mut self,
        ctx: &Context,
        ui: &Ui,
        response: &Response,
        map: Mapping,
        tiled: bool,
    ) {
        let Some(base) = self.base.clone() else { return };
        let size = vec2(base.width() as f32, base.height() as f32);
        let wrap = self.wrap;
        let to_image = move |p: Pos2| {
            let q = map.to_image(p);
            if tiled && !wrap {
                Pos2::new(q.x.rem_euclid(size.x), q.y.rem_euclid(size.y))
            } else {
                q
            }
        };
        let alt = ui.input(|i| i.modifiers.alt);
        if alt
            && response.clicked_by(PointerButton::Primary)
            && let Some(point) = response.interact_pointer_pos().map(to_image)
        {
            self.source = Some(point);
            self.offset = None;
        }
        let pressed = !alt && response.drag_started_by(PointerButton::Primary);
        let clicked = !alt && response.clicked_by(PointerButton::Primary);
        if pressed || clicked {
            let start = ui.input(|i| i.pointer.press_origin()).or(response.interact_pointer_pos());
            if let Some(point) = start.map(to_image) {
                self.begin(ctx, &base, point);
            }
        }
        if (response.dragged_by(PointerButton::Primary) || clicked)
            && let Some(point) = response.interact_pointer_pos().map(to_image)
        {
            self.paint_to(point);
        }
        if (response.drag_stopped() || clicked)
            && let Some(stroke) = self.stroke.take()
        {
            self.finished = Some(stroke.working);
        }
        self.update_cursor(ui, response, map);
    }

    fn begin(&mut self, ctx: &Context, base: &Arc<Image>, point: Pos2) {
        let offset = match (self.offset, self.source) {
            (Some(offset), _) => offset,
            (None, Some(source)) => source - point,
            (None, None) => {
                self.message = Some("Alt-click where to copy from first.");
                return;
            }
        };
        self.offset = Some(offset);
        self.source = None;
        let mut stroke = StrokeState {
            snapshot: Arc::clone(base),
            working: Image::clone(base),
            last: (point.x, point.y),
        };
        if self.live.is_none() {
            self.live = Live::new(ctx, &stroke.working);
        }
        let regions = self.paint_dab(&mut stroke, (point.x, point.y));
        self.stroke = Some(stroke);
        self.refresh_live(&regions);
    }

    fn paint_to(&mut self, point: Pos2) {
        let Some(mut stroke) = self.stroke.take() else { return };
        let mut regions = Vec::new();
        for centre in stroke_dabs(stroke.last, (point.x, point.y), self.brush.radius) {
            regions.extend(self.paint_dab(&mut stroke, centre));
            stroke.last = centre;
        }
        self.stroke = Some(stroke);
        self.refresh_live(&regions);
    }

    fn paint_dab(&self, stroke: &mut StrokeState, centre: (f32, f32)) -> Vec<Region> {
        let offset = self.offset.unwrap_or(Vec2::ZERO);
        dab(
            &mut stroke.working,
            &stroke.snapshot,
            centre,
            (offset.x, offset.y),
            &self.brush,
            self.mode,
            self.wrap,
        )
    }

    /// Copies the painted regions of the working copy into the live texture.
    fn refresh_live(&mut self, regions: &[Region]) {
        let (Some(live), Some(stroke)) = (&mut self.live, &self.stroke) else { return };
        for region in regions {
            live.update(&stroke.working, *region);
        }
    }

    fn update_cursor(&mut self, ui: &Ui, response: &Response, map: Mapping) {
        let pointer = response
            .hover_pos()
            .or_else(|| self.stroke.as_ref().and_then(|_| ui.input(|i| i.pointer.latest_pos())));
        self.cursor = pointer.map(|at| {
            let scale = map.scale();
            let source = match (self.offset, self.source) {
                (Some(offset), _) => Some(at + offset * scale),
                (None, Some(source)) => Some(map.to_screen(source)),
                (None, None) => None,
            };
            (at, self.brush.radius * scale, source)
        });
        if self.cursor.is_some() {
            ui.ctx().set_cursor_icon(CursorIcon::None);
        }
    }

    /// The brush outline at the pointer and a mark where it copies from.
    pub fn paint(&self, painter: &Painter) {
        let Some((at, radius, source)) = self.cursor else { return };
        let dark = Stroke::new(3.0, TOKENS.canvas.gamma_multiply(0.8));
        let light = Stroke::new(1.0, Color32::WHITE);
        painter.circle_stroke(at, radius.max(2.0), dark);
        painter.circle_stroke(at, radius.max(2.0), light);
        painter.line_segment([at - vec2(4.0, 0.0), at + vec2(4.0, 0.0)], light);
        painter.line_segment([at - vec2(0.0, 4.0), at + vec2(0.0, 4.0)], light);
        if let Some(source) = source {
            icons::paint_cursor(painter, source, icons::SOURCE);
        }
    }
}

impl Live {
    fn new(ctx: &Context, image: &Image) -> Option<Self> {
        let display = DisplayImage::from_image(image, DISPLAY_MAX_SIDE).ok()?;
        let pixels =
            ColorImage::from_rgba_unmultiplied([display.width, display.height], &display.rgba);
        let handle = ctx.load_texture("retouch-live", pixels, LIVE_SAMPLING);
        let ratio = display.width as f32 / image.width().max(1) as f32;
        Some(Self { handle, ratio, size: [display.width, display.height] })
    }

    /// Redraws `region` (image pixels) from `image`, sampling the nearest
    /// pixel for each display pixel.
    fn update(&mut self, image: &Image, [left, top, right, bottom]: Region) {
        let [width, height] = self.size;
        let to_display = |v: usize, round_up: bool| {
            let scaled = v as f32 * self.ratio;
            (if round_up { scaled.ceil() } else { scaled.floor() }) as usize
        };
        let (x0, y0) = (to_display(left, false).min(width), to_display(top, false).min(height));
        let (x1, y1) = (to_display(right, true).min(width), to_display(bottom, true).min(height));
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        let mut rgba = Vec::with_capacity((x1 - x0) * (y1 - y0) * 4);
        for y in y0..y1 {
            let source_y = (((y as f32 + 0.5) / self.ratio) as usize).min(image.height() - 1);
            for x in x0..x1 {
                let source_x = (((x as f32 + 0.5) / self.ratio) as usize).min(image.width() - 1);
                rgba.extend(to_rgba(image.pixel(source_x, source_y)));
            }
        }
        let patch = ColorImage::from_rgba_unmultiplied([x1 - x0, y1 - y0], &rgba);
        self.handle.set_partial([x0, y0], patch, LIVE_SAMPLING);
    }
}

fn to_rgba(pixel: &[f32]) -> [u8; 4] {
    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    match pixel {
        [r, g, b, a] => [byte(*r), byte(*g), byte(*b), byte(*a)],
        [r, g, b] => [byte(*r), byte(*g), byte(*b), 255],
        [v] => [byte(*v), byte(*v), byte(*v), 255],
        _ => [0, 0, 0, 255],
    }
}

/// The texture id of a live view, for tests.
#[cfg(test)]
fn live_id(tool: &RetouchTool) -> Option<egui::TextureId> {
    tool.live.as_ref().map(|live| live.handle.id())
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::pos2;

    fn base() -> Arc<Image> {
        Arc::new(Image::from_fn(128, 128, 3, |x, _, _| if x < 64 { 0.2 } else { 0.8 }).unwrap())
    }

    #[test]
    fn painting_without_a_source_asks_for_one_and_changes_nothing() {
        let ctx = Context::default();
        let mut tool = RetouchTool::default();
        let base = base();
        tool.sync_base(&base);

        tool.begin(&ctx, &base, pos2(20.0, 20.0));

        assert_eq!(tool.take_message(), Some("Alt-click where to copy from first."));
        assert!(tool.stroke.is_none() && live_id(&tool).is_none());
    }

    #[test]
    fn a_stroke_copies_from_the_source_and_keeps_its_offset_for_the_next() {
        let ctx = Context::default();
        let mut tool = RetouchTool { mode: RetouchMode::Clone, ..RetouchTool::default() };
        tool.brush = Brush { radius: 4.0, hardness: 1.0, opacity: 1.0 };
        let base = base();
        tool.sync_base(&base);
        tool.source = Some(pos2(100.0, 20.0));

        tool.begin(&ctx, &base, pos2(20.0, 20.0));
        tool.paint_to(pos2(30.0, 20.0));
        let painted = tool.stroke.take().map(|s| s.working).unwrap();

        assert!((painted.pixel(25, 20)[0] - 0.8).abs() < 1e-5, "copied from the right half");
        assert_eq!(painted.pixel(25, 40)[0], 0.2, "only along the stroke");
        assert_eq!(tool.offset, Some(vec2(80.0, 0.0)), "aligned: kept for the next stroke");
        assert!(tool.texture().is_some(), "shown live");
    }

    #[test]
    fn a_new_base_of_another_size_forgets_the_source() {
        let mut tool = RetouchTool { offset: Some(vec2(5.0, 5.0)), ..RetouchTool::default() };
        tool.sync_base(&base());
        tool.offset = Some(vec2(5.0, 5.0));

        tool.sync_base(&Arc::new(Image::filled(64, 64, 3, 0.5).unwrap()));

        assert!(!tool.has_source());
    }

    #[test]
    fn brackets_change_the_size_within_limits() {
        let mut tool = RetouchTool::default();
        tool.brush.radius = MAX_RADIUS;
        let ctx = Context::default();
        let press = |key| egui::RawInput {
            events: vec![egui::Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: Default::default(),
            }],
            ..Default::default()
        };

        // No renderer here: discard the font-atlas uploads egui queued.
        ctx.run_ui(press(Key::CloseBracket), |ui| tool.keys(ui.ctx())).textures_delta.clear();
        assert_eq!(tool.brush.radius, MAX_RADIUS);
        ctx.run_ui(press(Key::OpenBracket), |ui| tool.keys(ui.ctx())).textures_delta.clear();
        assert!((tool.brush.radius - MAX_RADIUS / RADIUS_STEP).abs() < 1e-3);
    }
}
