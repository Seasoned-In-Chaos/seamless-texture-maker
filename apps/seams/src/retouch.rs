//! The Retouch tool: Photoshop's Clone Stamp, Healing Brush and Spot Healing
//! Brush, for textures.
//!
//! - Alt-click sets where to copy from (Clone and Heal). With Aligned on,
//!   the first stroke fixes the distance between brush and source and later
//!   strokes keep it; off, every stroke starts again from the source point.
//! - A drag paints. Clone copies exactly. Heal shows the copy while drawing
//!   and, on release, matches it to the colour and brightness where it
//!   landed. Spot heal needs no source: it fills the stroke from around it.
//! - Size, hardness, opacity, flow, spacing and smoothing work as in
//!   Photoshop. `[` and `]` change the size, Shift with them the hardness;
//!   digits set the opacity (Shift: flow); Alt+right-drag sizes the brush
//!   on the canvas (sideways) and sets its hardness (up and down).
//! - With Wrap on, strokes continue across the edges, so a tileable texture
//!   stays tileable; in the Tiled view a stroke can cross a tile border.
//!
//! A stroke paints a working copy of the base color and shows it live in a
//! texture of its own. Healing runs on a worker thread after release; the
//! finished image goes back to the app to commit as one undo step.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, TryRecvError, channel};

use egui::epaint::{Mesh, Vertex};
use egui::{
    Align2, Color32, ColorImage, Context, CursorIcon, FontId, Key, Modifiers, Painter,
    PointerButton, Pos2, Response, Shape, Stroke, TextureHandle, TextureId, TextureOptions, Ui,
    Vec2, pos2, vec2,
};
use seams_core::Image;
use seams_core::retouch::{
    Brush, Region, RetouchMode, Smoother, SpotMethod, Stroke as Coverage, dabs_towards,
};
use seams_engine::{DISPLAY_MAX_SIDE, DisplayImage, RetouchSettings};

use crate::crop::Mapping;
use crate::icons;
use crate::textures::Texture;
use crate::theme::TOKENS;

/// Brush sizes the tool allows, in image pixels.
pub const MIN_RADIUS: f32 = 1.0;
pub const MAX_RADIUS: f32 = 1000.0;
/// `[` and `]` change the radius by this factor; Shift with them, the hardness by this much.
const RADIUS_STEP: f32 = 1.25;
const HARDNESS_STEP: f32 = 0.25;
/// Screen points of vertical drag that take the hardness from 0 % to 100 %.
const HARDNESS_TRAVEL: f32 = 200.0;
/// How much of the source shows under the brush before painting.
const OVERLAY_OPACITY: f32 = 0.75;
/// How a Spot heal stroke shows while it is drawn: dark, half covering.
const SPOT_TINT: f32 = 0.12;
const SPOT_TINT_OPACITY: f32 = 0.55;
const LIVE_SAMPLING: TextureOptions =
    TextureOptions { magnification: egui::TextureFilter::Nearest, ..TextureOptions::LINEAR };

/// A stroke in progress.
struct StrokeState {
    /// The base color when the stroke began: what every dab reads.
    snapshot: Arc<Image>,
    /// The base color being painted.
    working: Image,
    /// How far the stroke has covered each pixel.
    coverage: Coverage,
    /// The last dab's centre, in image pixels (unwrapped).
    last: (f32, f32),
    /// The brush trailing the pointer.
    smoother: Smoother,
    /// Source minus brush for this stroke.
    offset: Vec2,
    mode: RetouchMode,
}

/// A heal solving on a worker thread.
struct Healing {
    result: Receiver<Image>,
}

/// An Alt+right-drag changing the brush.
#[derive(Clone, Copy)]
struct Resize {
    /// Where the drag began (screen): the circle stays there.
    anchor: Pos2,
    /// The radius and hardness when it began.
    start: (f32, f32),
}

/// The live view of the working copy while it differs from the session's.
struct Live {
    handle: TextureHandle,
    /// Display pixels per image pixel.
    ratio: f32,
    size: [usize; 2],
}

/// The brush outline, source mark and overlay this frame.
struct Cursor {
    at: Pos2,
    radius: f32,
    source: Option<Pos2>,
    /// Source minus brush, in image pixels, for the overlay.
    overlay: Option<Vec2>,
}

pub struct RetouchTool {
    pub mode: RetouchMode,
    pub spot: SpotMethod,
    pub brush: Brush,
    /// Photoshop's Smoothing, `0..=1`.
    pub smoothing: f32,
    /// Keep the brush-to-source distance from stroke to stroke.
    pub aligned: bool,
    /// Paint and sample across the edges, as the texture tiles.
    pub wrap: bool,
    /// Show the source inside the brush before painting.
    pub overlay: bool,
    /// The base color the tool paints on.
    base: Option<Arc<Image>>,
    /// Where Alt-click set the source.
    anchor: Option<Pos2>,
    /// Source minus brush, fixed by the first stroke after Alt-click (Aligned).
    aligned_offset: Option<Vec2>,
    stroke: Option<StrokeState>,
    healing: Option<Healing>,
    resize: Option<Resize>,
    live: Option<Live>,
    /// A finished stroke, waiting for the app to commit it.
    finished: Option<Image>,
    /// Something to tell the user, waiting for the app to show it.
    message: Option<&'static str>,
    cursor: Option<Cursor>,
}

impl Default for RetouchTool {
    fn default() -> Self {
        Self::new(&RetouchSettings::default())
    }
}

impl RetouchTool {
    /// A tool with the remembered options.
    pub fn new(settings: &RetouchSettings) -> Self {
        let settings = settings.sanitized();
        Self {
            mode: settings.mode,
            spot: settings.spot,
            brush: settings.brush,
            smoothing: settings.smoothing,
            aligned: settings.aligned,
            wrap: settings.wrap,
            overlay: settings.overlay,
            base: None,
            anchor: None,
            aligned_offset: None,
            stroke: None,
            healing: None,
            resize: None,
            live: None,
            finished: None,
            message: None,
            cursor: None,
        }
    }

    /// The options to remember.
    pub fn settings(&self) -> RetouchSettings {
        RetouchSettings {
            mode: self.mode,
            spot: self.spot,
            brush: self.brush,
            smoothing: self.smoothing,
            aligned: self.aligned,
            wrap: self.wrap,
            overlay: self.overlay,
        }
    }

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
            self.anchor = None;
            self.aligned_offset = None;
        }
        // A stroke still painting or healing keeps its own copy; it commits when done.
        if self.stroke.is_none() && self.healing.is_none() {
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
        self.anchor.is_some() || self.aligned_offset.is_some()
    }

    pub fn is_healing(&self) -> bool {
        self.healing.is_some()
    }

    /// Forgets the source, so the next Alt-click starts afresh.
    pub fn clear_source(&mut self) {
        self.anchor = None;
        self.aligned_offset = None;
    }

    /// Photoshop's keys, unless a text field has the keyboard: `[` `]` size,
    /// Shift+`[` `]` hardness, digits opacity, Shift+digits flow. Consumed,
    /// so the app's own shortcuts on those keys wait while the tool is open.
    pub fn keys(&mut self, ctx: &Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let pressed =
            |key: Key, modifiers: Modifiers| ctx.input_mut(|i| i.consume_key(modifiers, key));
        let brush = &mut self.brush;
        if pressed(Key::OpenBracket, Modifiers::SHIFT) {
            brush.hardness = (brush.hardness - HARDNESS_STEP).clamp(0.0, 1.0);
        }
        if pressed(Key::CloseBracket, Modifiers::SHIFT) {
            brush.hardness = (brush.hardness + HARDNESS_STEP).clamp(0.0, 1.0);
        }
        if pressed(Key::OpenBracket, Modifiers::NONE) {
            brush.radius = (brush.radius / RADIUS_STEP).clamp(MIN_RADIUS, MAX_RADIUS);
        }
        if pressed(Key::CloseBracket, Modifiers::NONE) {
            brush.radius = (brush.radius * RADIUS_STEP).clamp(MIN_RADIUS, MAX_RADIUS);
        }
        const DIGITS: [Key; 10] = [
            Key::Num1,
            Key::Num2,
            Key::Num3,
            Key::Num4,
            Key::Num5,
            Key::Num6,
            Key::Num7,
            Key::Num8,
            Key::Num9,
            Key::Num0,
        ];
        for (i, key) in DIGITS.into_iter().enumerate() {
            let value = (i + 1) as f32 / 10.0;
            if pressed(key, Modifiers::SHIFT) {
                brush.flow = value;
            }
            if pressed(key, Modifiers::NONE) {
                brush.opacity = value;
            }
        }
    }

    /// Handles Alt-click, painting and the brush gesture, and collects a
    /// finished heal. `tiled` maps the pointer over a tile grid back into
    /// the image.
    pub fn interact(
        &mut self,
        ctx: &Context,
        ui: &Ui,
        response: &Response,
        map: Mapping,
        tiled: bool,
    ) {
        self.poll_healing(ctx);
        let Some(base) = self.base.clone() else { return };
        let size = vec2(base.width() as f32, base.height() as f32);
        let wrap = self.wrap;
        let to_image = move |p: Pos2| {
            let q = map.to_image(p);
            if tiled && !wrap { pos2(q.x.rem_euclid(size.x), q.y.rem_euclid(size.y)) } else { q }
        };
        let alt = ui.input(|i| i.modifiers.alt);
        if self.resize_gesture(ui, response, map, alt) {
            self.update_cursor(ui, response, map, to_image);
            return;
        }
        if alt
            && self.mode != RetouchMode::SpotHeal
            && response.clicked_by(PointerButton::Primary)
            && let Some(point) = response.interact_pointer_pos().map(to_image)
        {
            self.anchor = Some(point);
            self.aligned_offset = None;
        }
        let pressed = !alt && response.drag_started_by(PointerButton::Primary);
        let clicked = !alt && response.clicked_by(PointerButton::Primary);
        if pressed || clicked {
            let start = ui.input(|i| i.pointer.press_origin()).or(response.interact_pointer_pos());
            if let Some(point) = start.map(to_image) {
                self.begin(ctx, &base, point, ui.input(|i| i.time));
            }
        }
        if (response.dragged_by(PointerButton::Primary) || clicked)
            && let Some(point) = response.interact_pointer_pos().map(to_image)
        {
            self.paint_to(point, ui.input(|i| i.time));
            if self.smoothing > 0.0 {
                // The brush keeps catching up while the pointer rests.
                ctx.request_repaint();
            }
        }
        if response.drag_stopped() || clicked {
            self.end(ctx);
        }
        self.update_cursor(ui, response, map, to_image);
    }

    /// Alt+right-drag: sideways sizes the brush, up and down sets the
    /// hardness (down is harder). Returns true while it is in charge.
    fn resize_gesture(&mut self, ui: &Ui, response: &Response, map: Mapping, alt: bool) -> bool {
        if alt
            && response.drag_started_by(PointerButton::Secondary)
            && let Some(anchor) = ui.input(|i| i.pointer.press_origin())
        {
            self.resize = Some(Resize { anchor, start: (self.brush.radius, self.brush.hardness) });
        }
        let Some(resize) = self.resize else { return false };
        if !ui.input(|i| i.pointer.secondary_down()) {
            self.resize = None;
            return false;
        }
        if let Some(pointer) = ui.input(|i| i.pointer.latest_pos()) {
            let delta = pointer - resize.anchor;
            let radius = resize.start.0 + delta.x / map.scale().max(1e-3);
            self.brush.radius = radius.clamp(MIN_RADIUS, MAX_RADIUS);
            self.brush.hardness = (resize.start.1 + delta.y / HARDNESS_TRAVEL).clamp(0.0, 1.0);
        }
        true
    }

    fn begin(&mut self, ctx: &Context, base: &Arc<Image>, point: Pos2, time: f64) {
        if self.healing.is_some() {
            self.message = Some("Still healing the last stroke…");
            return;
        }
        let offset = match self.mode {
            RetouchMode::SpotHeal => Vec2::ZERO,
            _ => match (self.aligned_offset.filter(|_| self.aligned), self.anchor) {
                (Some(offset), _) => offset,
                (None, Some(anchor)) => anchor - point,
                (None, None) => {
                    self.message = Some("Alt-click where to copy from first.");
                    return;
                }
            },
        };
        if self.mode != RetouchMode::SpotHeal && self.aligned {
            self.aligned_offset = Some(offset);
        }
        let mut smoother = Smoother::new(self.smoothing);
        smoother.push((point.x, point.y), time);
        let mut stroke = StrokeState {
            snapshot: Arc::clone(base),
            working: Image::clone(base),
            coverage: Coverage::new(base.width(), base.height(), self.wrap),
            last: (point.x, point.y),
            smoother,
            offset,
            mode: self.mode,
        };
        if self.live.is_none() {
            self.live = Live::new(ctx, &stroke.working);
        }
        let regions = self.paint_dab(&mut stroke, (point.x, point.y));
        self.stroke = Some(stroke);
        self.refresh_live(&regions);
    }

    fn paint_to(&mut self, point: Pos2, time: f64) {
        let Some(mut stroke) = self.stroke.take() else { return };
        let mut regions = Vec::new();
        for target in stroke.smoother.push((point.x, point.y), time) {
            regions.extend(self.dabs_to(&mut stroke, target));
        }
        self.stroke = Some(stroke);
        self.refresh_live(&regions);
    }

    fn dabs_to(&self, stroke: &mut StrokeState, target: (f32, f32)) -> Vec<Region> {
        let mut regions = Vec::new();
        for centre in dabs_towards(stroke.last, target, self.brush.spacing_px()) {
            regions.extend(self.paint_dab(stroke, centre));
            stroke.last = centre;
        }
        regions
    }

    /// Adds a dab and redraws what it covers: the clone source, or for Spot
    /// heal a dark tint marking what will be filled.
    fn paint_dab(&self, stroke: &mut StrokeState, centre: (f32, f32)) -> Vec<Region> {
        let regions = stroke.coverage.dab(centre, &self.brush);
        let (snapshot, opacity) = (&stroke.snapshot, self.brush.opacity);
        match stroke.mode {
            RetouchMode::SpotHeal => {
                let colors = snapshot.color_channels();
                stroke.coverage.composite(
                    &mut stroke.working,
                    snapshot,
                    &regions,
                    SPOT_TINT_OPACITY,
                    |x, y, out| {
                        out.copy_from_slice(snapshot.pixel(x, y));
                        out[..colors].fill(SPOT_TINT);
                    },
                );
            }
            _ => {
                let offset = (stroke.offset.x, stroke.offset.y);
                stroke.coverage.composite_clone(
                    &mut stroke.working,
                    snapshot,
                    &regions,
                    offset,
                    opacity,
                );
            }
        }
        regions
    }

    /// Ends the stroke: a clone commits now; a heal solves on a worker thread.
    fn end(&mut self, ctx: &Context) {
        let Some(mut stroke) = self.stroke.take() else { return };
        if let Some(target) = stroke.smoother.finish() {
            let regions = self.dabs_to(&mut stroke, target);
            refresh(&mut self.live, &stroke.working, &regions);
        }
        if stroke.coverage.is_empty() {
            return;
        }
        if stroke.mode == RetouchMode::Clone {
            self.finished = Some(stroke.working);
            return;
        }
        let (sender, result) = channel();
        let (brush, spot, repaint) = (self.brush, self.spot, ctx.clone());
        std::thread::spawn(move || {
            let StrokeState { snapshot, coverage, offset, mode, .. } = stroke;
            let mut target = Image::clone(&snapshot);
            match mode {
                RetouchMode::SpotHeal => {
                    coverage.spot_heal(
                        &mut target,
                        &snapshot,
                        spot,
                        brush.radius * 2.0,
                        brush.opacity,
                    );
                }
                _ => {
                    coverage.heal(&mut target, &snapshot, (offset.x, offset.y), brush.opacity);
                }
            }
            // The app may have closed the tool meanwhile; then nobody is listening.
            let _ = sender.send(target);
            repaint.request_repaint();
        });
        self.healing = Some(Healing { result });
    }

    /// Takes a finished heal, if one is ready.
    fn poll_healing(&mut self, ctx: &Context) {
        let Some(healing) = &self.healing else { return };
        match healing.result.try_recv() {
            Ok(image) => {
                self.healing = None;
                self.finished = Some(image);
            }
            Err(TryRecvError::Empty) => {
                ctx.request_repaint_after(std::time::Duration::from_millis(30))
            }
            Err(TryRecvError::Disconnected) => {
                self.healing = None;
                self.live = None;
                self.message = Some("Healing failed; the stroke was not applied.");
            }
        }
    }

    /// Copies the painted regions of the working copy into the live texture.
    fn refresh_live(&mut self, regions: &[Region]) {
        if let Some(stroke) = &self.stroke {
            refresh(&mut self.live, &stroke.working, regions);
        }
    }

    fn update_cursor(
        &mut self,
        ui: &Ui,
        response: &Response,
        map: Mapping,
        to_image: impl Fn(Pos2) -> Pos2,
    ) {
        if let Some(resize) = self.resize {
            let radius = self.brush.radius * map.scale();
            self.cursor = Some(Cursor { at: resize.anchor, radius, source: None, overlay: None });
            ui.ctx().set_cursor_icon(CursorIcon::None);
            return;
        }
        let pointer = response
            .hover_pos()
            .or_else(|| self.stroke.as_ref().and_then(|_| ui.input(|i| i.pointer.latest_pos())));
        self.cursor = pointer.map(|at| {
            let scale = map.scale();
            let here = to_image(at);
            // Where the next (or current) stroke copies from, as source minus brush.
            let offset = match (&self.stroke, self.mode) {
                (_, RetouchMode::SpotHeal) => None,
                (Some(stroke), _) => Some(stroke.offset),
                (None, _) => {
                    self.aligned_offset.filter(|_| self.aligned).or(self.anchor.map(|a| a - here))
                }
            };
            let source = match (offset, self.anchor) {
                (Some(offset), _) if self.stroke.is_some() || self.aligned_offset.is_some() => {
                    Some(at + offset * scale)
                }
                (_, Some(anchor)) => Some(map.to_screen(anchor)),
                _ => None,
            };
            let overlay =
                offset.filter(|_| self.overlay && self.stroke.is_none() && self.healing.is_none());
            Cursor { at, radius: self.brush.radius * scale, source, overlay }
        });
        if self.cursor.is_some() {
            ui.ctx().set_cursor_icon(CursorIcon::None);
        }
    }

    /// The brush outline at the pointer, the source under it, a mark where
    /// it copies from, and the size readout while Alt+right-dragging.
    /// `view` is the texture shown and how it maps to the screen.
    pub fn paint(&self, painter: &Painter, view: Option<(TextureId, Mapping, Vec2)>) {
        let Some(cursor) = &self.cursor else { return };
        let radius = cursor.radius.max(2.0);
        if let (Some(offset), Some((texture, map, size))) = (cursor.overlay, view) {
            paint_overlay(painter, texture, map, size, cursor.at, radius, offset);
        }
        let dark = Stroke::new(3.0, TOKENS.canvas.gamma_multiply(0.8));
        let light = Stroke::new(1.0, Color32::WHITE);
        painter.circle_stroke(cursor.at, radius, dark);
        painter.circle_stroke(cursor.at, radius, light);
        // The hard core, inside the outline, when the brush is soft.
        let core = radius * self.brush.hardness;
        if self.brush.hardness < 0.98 && core > 3.0 {
            painter.circle_stroke(
                cursor.at,
                core,
                Stroke::new(1.0, Color32::WHITE.gamma_multiply(0.45)),
            );
        }
        painter.line_segment([cursor.at - vec2(4.0, 0.0), cursor.at + vec2(4.0, 0.0)], light);
        painter.line_segment([cursor.at - vec2(0.0, 4.0), cursor.at + vec2(0.0, 4.0)], light);
        if let Some(source) = cursor.source {
            icons::paint_cursor(painter, source, icons::SOURCE);
        }
        if self.resize.is_some() {
            let text = format!(
                "Size  {:.0} px\nHardness  {:.0} %",
                self.brush.radius * 2.0,
                self.brush.hardness * 100.0
            );
            let at = cursor.at + vec2(radius + 12.0, 0.0);
            let galley = painter.layout_no_wrap(text, FontId::proportional(13.0), TOKENS.text);
            let rect = Align2::LEFT_CENTER.anchor_size(at, galley.size()).expand(6.0);
            painter.rect_filled(rect, 4.0, TOKENS.surface.gamma_multiply(0.92));
            painter.galley(rect.min + vec2(6.0, 6.0), galley, TOKENS.text);
        }
        if self.healing.is_some() {
            let at = cursor.at + vec2(0.0, radius + 14.0);
            painter.text(
                at,
                Align2::CENTER_TOP,
                "Healing…",
                FontId::proportional(12.0),
                Color32::WHITE,
            );
        }
    }
}

/// Copies `regions` of `working` into the live texture.
fn refresh(live: &mut Option<Live>, working: &Image, regions: &[Region]) {
    if let Some(live) = live {
        for region in regions {
            live.update(working, *region);
        }
    }
}

/// The source's pixels inside the brush circle, as Photoshop's clipped
/// clone-source overlay: a disc of the shown texture, offset to the source.
fn paint_overlay(
    painter: &Painter,
    texture: TextureId,
    map: Mapping,
    size: Vec2,
    at: Pos2,
    radius: f32,
    offset: Vec2,
) {
    const SEGMENTS: usize = 48;
    if size.x <= 0.0 || size.y <= 0.0 {
        return;
    }
    let tint = Color32::WHITE.gamma_multiply(OVERLAY_OPACITY);
    let uv = |screen: Pos2| {
        let image = map.to_image(screen) + offset;
        pos2(image.x / size.x, image.y / size.y)
    };
    let mut mesh = Mesh::with_texture(texture);
    mesh.vertices.push(Vertex { pos: at, uv: uv(at), color: tint });
    for i in 0..=SEGMENTS {
        let angle = i as f32 / SEGMENTS as f32 * std::f32::consts::TAU;
        let pos = at + vec2(angle.cos(), angle.sin()) * radius;
        mesh.vertices.push(Vertex { pos, uv: uv(pos), color: tint });
    }
    for i in 1..=SEGMENTS as u32 {
        mesh.indices.extend([0, i, i + 1]);
    }
    painter.add(Shape::mesh(mesh));
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

    fn base() -> Arc<Image> {
        Arc::new(Image::from_fn(128, 128, 3, |x, _, _| if x < 64 { 0.2 } else { 0.8 }).unwrap())
    }

    fn hard(radius: f32) -> Brush {
        Brush { radius, hardness: 1.0, ..Brush::default() }
    }

    /// Waits for a heal on the worker thread.
    fn wait_for_heal(tool: &mut RetouchTool, ctx: &Context) -> Image {
        for _ in 0..500 {
            tool.poll_healing(ctx);
            if tool.finished.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        tool.take_finished().unwrap()
    }

    #[test]
    fn painting_without_a_source_asks_for_one_and_changes_nothing() {
        let ctx = Context::default();
        let mut tool = RetouchTool::default();
        let base = base();
        tool.sync_base(&base);

        tool.begin(&ctx, &base, pos2(20.0, 20.0), 0.0);

        assert_eq!(tool.take_message(), Some("Alt-click where to copy from first."));
        assert!(tool.stroke.is_none() && live_id(&tool).is_none());
    }

    #[test]
    fn a_clone_stroke_copies_from_the_source_and_aligned_keeps_its_offset() {
        let ctx = Context::default();
        let mut tool =
            RetouchTool { mode: RetouchMode::Clone, smoothing: 0.0, ..RetouchTool::default() };
        tool.brush = hard(4.0);
        let base = base();
        tool.sync_base(&base);
        tool.anchor = Some(pos2(100.0, 20.0));

        tool.begin(&ctx, &base, pos2(20.0, 20.0), 0.0);
        tool.paint_to(pos2(30.0, 20.0), 0.1);
        tool.end(&ctx);
        let painted = tool.take_finished().unwrap();

        assert!((painted.pixel(25, 20)[0] - 0.8).abs() < 1e-5, "copied from the right half");
        assert_eq!(painted.pixel(25, 40)[0], 0.2, "only along the stroke");
        assert_eq!(tool.aligned_offset, Some(vec2(80.0, 0.0)), "kept for the next stroke");
        assert!(tool.texture().is_some(), "shown live");
    }

    #[test]
    fn without_aligned_every_stroke_starts_from_the_source_again() {
        let ctx = Context::default();
        let mut tool =
            RetouchTool { mode: RetouchMode::Clone, aligned: false, ..RetouchTool::default() };
        let base = base();
        tool.sync_base(&base);
        tool.anchor = Some(pos2(100.0, 20.0));

        tool.begin(&ctx, &base, pos2(20.0, 20.0), 0.0);
        tool.end(&ctx);
        tool.begin(&ctx, &base, pos2(30.0, 60.0), 1.0);

        assert_eq!(tool.stroke.as_ref().map(|s| s.offset), Some(vec2(70.0, -40.0)));
    }

    #[test]
    fn a_heal_stroke_finishes_on_a_worker_and_blends_in() {
        let ctx = Context::default();
        let mut tool =
            RetouchTool { mode: RetouchMode::Heal, smoothing: 0.0, ..RetouchTool::default() };
        tool.brush = hard(6.0);
        let base = base();
        tool.sync_base(&base);
        tool.anchor = Some(pos2(100.0, 64.0));

        tool.begin(&ctx, &base, pos2(20.0, 64.0), 0.0);
        tool.paint_to(pos2(30.0, 64.0), 0.1);
        tool.end(&ctx);
        assert!(tool.is_healing());
        let healed = wait_for_heal(&mut tool, &ctx);

        // Flat bright source healed onto flat dark: it takes the dark level.
        assert!((healed.pixel(25, 64)[0] - 0.2).abs() < 0.02, "{}", healed.pixel(25, 64)[0]);
        assert!(!tool.is_healing());
    }

    #[test]
    fn spot_heal_needs_no_source() {
        let ctx = Context::default();
        let mut tool =
            RetouchTool { mode: RetouchMode::SpotHeal, smoothing: 0.0, ..RetouchTool::default() };
        tool.brush = hard(5.0);
        let mut image = Image::filled(96, 96, 3, 0.5).unwrap();
        for y in 46..50 {
            for x in 46..50 {
                image.data_mut()[(y * 96 + x) * 3..(y * 96 + x) * 3 + 3].fill(0.0);
            }
        }
        let base = Arc::new(image);
        tool.sync_base(&base);

        tool.begin(&ctx, &base, pos2(48.0, 48.0), 0.0);
        tool.end(&ctx);
        let healed = wait_for_heal(&mut tool, &ctx);

        assert!((healed.pixel(48, 48)[0] - 0.5).abs() < 0.05, "{}", healed.pixel(48, 48)[0]);
    }

    #[test]
    fn a_new_base_of_another_size_forgets_the_source() {
        let mut tool = RetouchTool::default();
        tool.sync_base(&base());
        tool.aligned_offset = Some(vec2(5.0, 5.0));

        tool.sync_base(&Arc::new(Image::filled(64, 64, 3, 0.5).unwrap()));

        assert!(!tool.has_source());
    }

    fn press(key: Key, modifiers: Modifiers) -> egui::RawInput {
        egui::RawInput {
            events: vec![egui::Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn photoshop_keys_set_size_hardness_opacity_and_flow() {
        let mut tool = RetouchTool {
            brush: Brush { radius: MAX_RADIUS, hardness: 0.5, ..Brush::default() },
            ..RetouchTool::default()
        };
        let ctx = Context::default();
        let mut run = |key, modifiers| {
            // No renderer here: discard the font-atlas uploads egui queued.
            ctx.run_ui(press(key, modifiers), |ui| tool.keys(ui.ctx())).textures_delta.clear();
        };

        run(Key::CloseBracket, Modifiers::NONE);
        run(Key::CloseBracket, Modifiers::SHIFT);
        run(Key::Num5, Modifiers::NONE);
        run(Key::Num3, Modifiers::SHIFT);
        run(Key::OpenBracket, Modifiers::NONE);

        assert!((tool.brush.radius - MAX_RADIUS / RADIUS_STEP).abs() < 1e-3, "size within limits");
        assert_eq!(tool.brush.hardness, 0.75);
        assert_eq!(tool.brush.opacity, 0.5);
        assert!((tool.brush.flow - 0.3).abs() < 1e-6);
    }
}
