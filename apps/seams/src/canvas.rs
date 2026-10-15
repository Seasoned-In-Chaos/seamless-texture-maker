//! The image viewer: pan, zoom, before/after comparison and tiled preview.

use egui::{
    Align, Align2, Color32, CursorIcon, Event, FontId, Key, Layout, MouseWheelUnit, PointerButton,
    Pos2, Rect, Response, RichText, Sense, Shape, Stroke, StrokeKind, TextureId, Ui, UiBuilder,
    Vec2, pos2, vec2,
};

use crate::crop::{CropTool, Mapping};
use crate::retouch::RetouchTool;
use crate::textures::Texture;
use crate::theme::{SMALL_SIZE, TITLE_SIZE, TOKENS, icon_font, semibold};
use crate::{icons, widgets};

const MARGIN: f32 = 24.0;
const SIDE_BY_SIDE_GAP: f32 = 12.0;
const MIN_ZOOM: f32 = 0.01;
const MAX_ZOOM: f32 = 64.0;
/// Fit never enlarges beyond this, so small textures are not shown blurry.
const MAX_FIT_ZOOM: f32 = 2.0;
/// A Zoom-tool drag of this many points doubles, or halves, the zoom.
const SCRUB_DOUBLING: f32 = 100.0;
/// Zoom factor of one wheel notch, as in Photoshop.
const WHEEL_NOTCH_ZOOM: f32 = 1.1;
/// Zoom In and Zoom Out step through these, in percent of actual size, as
/// Photoshop's do. They span `MIN_ZOOM` to `MAX_ZOOM`.
const ZOOM_LEVELS: [f32; 25] = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.25, 8.33, 12.5, 16.67, 25.0, 33.33, 50.0, 66.67, 100.0, 200.0,
    300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 1200.0, 1600.0, 3200.0, 6400.0,
];
const SPLIT_GRAB_DISTANCE: f32 = 8.0;
const CHECKER_SIZE: f32 = 16.0;
/// Dash and gap lengths, in points, of the tile guide lines.
const GUIDE_DASH: f32 = 6.0;
const GUIDE_GAP: f32 = 4.0;
const EMPTY_BLOCK_SIZE: Vec2 = vec2(380.0, 200.0);
const EMPTY_ICON_SIZE: f32 = 32.0;
const CHOOSE_BUTTON_SIZE: Vec2 = vec2(168.0, 34.0);
/// Space kept clear at the bottom of the canvas for the floating view bar.
const VIEW_BAR_RESERVE: f32 = 52.0;
const FULL_UV: Rect = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));

/// How the canvas presents the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ViewMode {
    /// The processed image alone.
    #[default]
    Result,
    /// Before and after in one frame, with a draggable divider.
    Split,
    /// Before and after next to each other.
    SideBySide,
    /// The processed image repeated, to check that it tiles.
    Tiled,
}

impl ViewMode {
    pub const ALL: [(ViewMode, &'static str); 4] = [
        (ViewMode::Result, "Result"),
        (ViewMode::Split, "Split"),
        (ViewMode::SideBySide, "Side by side"),
        (ViewMode::Tiled, "Tiled"),
    ];
}

/// What the canvas should draw this frame.
pub struct Scene {
    pub before: Option<Texture>,
    pub after: Option<Texture>,
    /// Whether a before/after comparison makes sense for the shown image.
    pub can_compare: bool,
    pub tiles: usize,
    pub show_guides: bool,
    pub checker: TextureId,
    /// The tiling check's marks, drawn over the result.
    pub overlay: Option<TextureId>,
    pub loading: bool,
    pub files_hovering: bool,
}

/// The view tool the primary button drives, as in Photoshop's toolbar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NavTool {
    /// Drags do nothing to the view (a tool such as Crop may take them).
    #[default]
    None,
    /// The Hand tool: every drag pans, even one on the split divider.
    Pan,
    /// The Zoom tool: a click steps in (Alt: out), a drag zooms
    /// continuously, right to zoom in and left to zoom out.
    Zoom,
}

/// A tool that works on the base color, shown on the canvas in its place.
pub enum ImageTool<'a> {
    Crop(&'a mut CropTool),
    Retouch(&'a mut RetouchTool),
}

impl ImageTool<'_> {
    fn reborrow(&mut self) -> ImageTool<'_> {
        match self {
            ImageTool::Crop(crop) => ImageTool::Crop(crop),
            ImageTool::Retouch(retouch) => ImageTool::Retouch(retouch),
        }
    }
}

/// A Zoom-tool drag: where it began (relative to the view's middle) and
/// the view then.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Scrub {
    anchor: Vec2,
    zoom: f32,
    pan: Vec2,
}

/// Pan, zoom and comparison state kept between frames.
pub struct CanvasState {
    pub mode: ViewMode,
    pub tool: NavTool,
    /// The icon drawn as the pointer this frame, and where.
    tool_cursor: Option<(Pos2, char)>,
    scrub: Option<Scrub>,
    /// Screen points per full-resolution image pixel.
    zoom: f32,
    pan: Vec2,
    /// While set, the zoom follows the window so the image always fits.
    fit: bool,
    /// Divider position in split mode, as a fraction of the image width.
    split: f32,
    dragging_split: bool,
}

impl Default for CanvasState {
    fn default() -> Self {
        Self {
            mode: ViewMode::default(),
            tool: NavTool::None,
            tool_cursor: None,
            scrub: None,
            zoom: 1.0,
            pan: Vec2::ZERO,
            fit: true,
            split: 0.5,
            dragging_split: false,
        }
    }
}

/// Where each part of the scene lands on screen.
struct Placement {
    /// The frame holding the processed image (or the tile grid).
    after: Rect,
    /// The frame holding the unprocessed image in side-by-side mode.
    before: Option<Rect>,
}

impl CanvasState {
    pub fn fit_to_window(&mut self) {
        self.fit = true;
        self.pan = Vec2::ZERO;
    }

    /// One image pixel per screen pixel.
    pub fn actual_size(&mut self, pixels_per_point: f32) {
        self.fit = false;
        self.pan = Vec2::ZERO;
        self.zoom = 1.0 / pixels_per_point.max(0.1);
    }

    /// Turns `tool` on, or off when it is already on.
    pub fn toggle_tool(&mut self, tool: NavTool) {
        self.tool = if self.tool == tool { NavTool::None } else { tool };
        self.scrub = None;
    }

    /// Steps to the next of Photoshop's zoom levels, keeping the middle of
    /// the view still. Does nothing past the first or last level.
    pub fn zoom_step(&mut self, zoom_in: bool, pixels_per_point: f32) {
        self.zoom_step_about(zoom_in, pixels_per_point, Vec2::ZERO);
    }

    /// [`Self::zoom_step`], keeping the point `from_centre` (relative to the
    /// middle of the view) still.
    fn zoom_step_about(&mut self, zoom_in: bool, pixels_per_point: f32, from_centre: Vec2) {
        let pixels_per_point = pixels_per_point.max(0.1);
        let current = self.zoom_percent(pixels_per_point);
        // A little slack, so a level reached exactly is not stepped to again.
        let next = if zoom_in {
            ZOOM_LEVELS.iter().find(|level| **level > current * 1.001)
        } else {
            ZOOM_LEVELS.iter().rev().find(|level| **level < current * 0.999)
        };
        let Some(percent) = next else { return };
        self.zoom_about(percent / 100.0 / pixels_per_point / self.zoom, from_centre);
    }

    /// A Zoom-tool drag `dx` points to the side of where it began: right
    /// zooms in and left out, doubling every [`SCRUB_DOUBLING`] points,
    /// with the point under the press kept still.
    fn scrub_to(&mut self, scrub: Scrub, dx: f32) {
        let doublings = if dx.is_finite() { (dx / SCRUB_DOUBLING).clamp(-40.0, 40.0) } else { 0.0 };
        let zoom = (scrub.zoom * doublings.exp2()).clamp(MIN_ZOOM, MAX_ZOOM);
        self.pan = scrub.anchor - (scrub.anchor - scrub.pan) * (zoom / scrub.zoom);
        self.zoom = zoom;
        self.fit = false;
    }

    pub fn is_fitted(&self) -> bool {
        self.fit
    }

    /// Zoom as a percentage of actual size.
    pub fn zoom_percent(&self, pixels_per_point: f32) -> f32 {
        self.zoom * pixels_per_point * 100.0
    }

    /// The mode actually used: comparisons need both images.
    fn effective_mode(&self, scene: &Scene) -> ViewMode {
        let comparing = matches!(self.mode, ViewMode::Split | ViewMode::SideBySide);
        if comparing && !(scene.can_compare && scene.before.is_some()) {
            ViewMode::Result
        } else {
            self.mode
        }
    }

    /// Size of everything drawn, in image pixels, plus fixed screen gaps.
    fn content_size(mode: ViewMode, image: Vec2, tiles: usize) -> (Vec2, f32) {
        match mode {
            ViewMode::Result | ViewMode::Split => (image, 0.0),
            ViewMode::SideBySide => (vec2(image.x * 2.0, image.y), SIDE_BY_SIDE_GAP),
            ViewMode::Tiled => (image * tiles as f32, 0.0),
        }
    }

    fn fit_zoom(area: Rect, content: Vec2, gap: f32) -> f32 {
        let available =
            (area.size() - Vec2::splat(MARGIN * 2.0) - vec2(gap, 0.0)).max(Vec2::splat(1.0));
        (available.x / content.x.max(1.0)).min(available.y / content.y.max(1.0)).min(MAX_FIT_ZOOM)
    }

    fn place(&self, area: Rect, mode: ViewMode, image: Vec2, tiles: usize) -> Placement {
        let (content, gap) = Self::content_size(mode, image, tiles);
        let size = content * self.zoom + vec2(gap, 0.0);
        let whole = Rect::from_center_size(area.center() + self.pan, size);
        if mode != ViewMode::SideBySide {
            return Placement { after: whole, before: None };
        }
        let half = vec2((size.x - gap) / 2.0, size.y);
        let before = Rect::from_min_size(whole.min, half);
        let after = Rect::from_min_size(pos2(whole.max.x - half.x, whole.min.y), half);
        Placement { after, before: Some(before) }
    }

    /// Draws the canvas and handles its input. With an image tool open, the
    /// base color is shown (tiled, for the Retouch tool in the Tiled view)
    /// and the tool takes primary drags. Returns true when the empty state's
    /// open button was clicked.
    pub fn show(&mut self, ui: &mut Ui, scene: &Scene, mut tool: Option<ImageTool<'_>>) -> bool {
        let (whole, response) =
            ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let painter = ui.painter_at(whole);
        painter.rect_filled(whole, 0, TOKENS.canvas);
        self.tool_cursor = None;
        // The image is centred in the part the floating view bar leaves free.
        let area = if scene.after.is_some() {
            whole.with_max_y(whole.max.y - VIEW_BAR_RESERVE)
        } else {
            whole
        };

        let shown = match &tool {
            Some(ImageTool::Retouch(retouch)) => retouch.texture().or(scene.before),
            Some(ImageTool::Crop(_)) => scene.before,
            None => scene.after,
        };
        let Some(after) = shown.or(scene.after) else {
            let button_clicked = show_placeholder(ui, area, scene);
            return button_clicked || (!scene.loading && response.double_clicked());
        };
        let mode = match &tool {
            Some(ImageTool::Retouch(_)) if self.mode == ViewMode::Tiled => ViewMode::Tiled,
            Some(_) => ViewMode::Result,
            None => self.effective_mode(scene),
        };
        let (content, gap) = Self::content_size(mode, after.full_size, scene.tiles);
        if self.fit {
            self.zoom = Self::fit_zoom(area, content, gap);
        }
        let placement = self.place(area, mode, after.full_size, scene.tiles);
        let map = Mapping::new(placement.after, after.full_size);
        self.handle_input(
            ui,
            &response,
            area,
            mode,
            &placement,
            tool.as_mut().map(|t| (t.reborrow(), map)),
        );
        // As in Photoshop with Overscroll off: the image cannot be lost off
        // screen, whatever moved it this frame.
        self.pan = clamp_pan(self.pan, area.size(), content * self.zoom + vec2(gap, 0.0));

        let placement = self.place(area, mode, after.full_size, scene.tiles);
        self.paint(&painter, scene, after, mode, &placement);
        match tool {
            Some(ImageTool::Crop(crop)) => {
                crop.paint(&painter, Mapping::new(placement.after, after.full_size));
            }
            Some(ImageTool::Retouch(retouch)) => retouch.paint(&painter),
            None => {}
        }
        if let Some((at, icon)) = self.tool_cursor {
            icons::paint_cursor(&painter, at, icon);
        }
        false
    }

    fn handle_input(
        &mut self,
        ui: &Ui,
        response: &Response,
        area: Rect,
        mode: ViewMode,
        placement: &Placement,
        image_tool: Option<(ImageTool<'_>, Mapping)>,
    ) {
        let pointer = response.hover_pos();
        // Holding Space borrows the pan tool, unless the keys are for a text field.
        let space_held =
            ui.input(|i| i.key_down(Key::Space)) && !ui.ctx().egui_wants_keyboard_input();
        let panning = self.tool == NavTool::Pan || space_held;
        let zooming = self.tool == NavTool::Zoom && !space_held;
        // An image tool takes primary drags unless a view tool or Space is
        // in charge; a middle drag pans either way.
        let cropping = image_tool.is_some() && !panning && !zooming;
        match image_tool.filter(|_| cropping) {
            Some((ImageTool::Crop(crop), map)) => crop.interact(ui, response, map),
            Some((ImageTool::Retouch(retouch), map)) => {
                retouch.interact(ui.ctx(), ui, response, map, mode == ViewMode::Tiled);
            }
            None => {}
        }
        if zooming {
            self.zoom_tool(ui, response, area);
        }
        let divider_x = placement.after.left() + placement.after.width() * self.split;
        let near_divider = !panning
            && !zooming
            && mode == ViewMode::Split
            && pointer.is_some_and(|p| (p.x - divider_x).abs() <= SPLIT_GRAB_DISTANCE);
        let middle_drag = response.dragged_by(PointerButton::Middle);
        // As in Photoshop, a plain drag does not move the view: it takes the
        // pan tool, Space, or the middle button.
        let pan_drag = !self.dragging_split
            && (middle_drag || (panning && response.dragged_by(PointerButton::Primary)));
        if cropping && !middle_drag {
            // The Crop tool set its own cursor.
        } else if near_divider || self.dragging_split {
            ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);
        } else if pan_drag {
            self.show_tool_cursor(ui, icons::HAND_GRAB);
        } else if panning && response.hovered() {
            self.show_tool_cursor(ui, icons::HAND);
        } else if zooming && response.hovered() {
            let out = ui.input(|i| i.modifiers.alt);
            self.show_tool_cursor(ui, if out { icons::ZOOM_OUT } else { icons::ZOOM_IN });
        }
        if response.drag_started() {
            // Only the primary button moves the divider; a middle drag always pans.
            self.dragging_split = near_divider && ui.input(|i| i.pointer.primary_down());
        }
        if self.dragging_split || pan_drag {
            self.drag(response, placement);
        }
        if response.drag_stopped() {
            self.dragging_split = false;
            self.scrub = None;
        }
        // With the Zoom tool or an image tool, clicks act; a double-click is two of them.
        if response.double_clicked() && !zooming && !cropping {
            self.fit_to_window();
        }
        if let Some(pointer) = pointer {
            let (factor, scroll) = read_wheel(ui);
            self.zoom_about(factor, pointer - area.center());
            if scroll != Vec2::ZERO {
                self.pan += scroll;
                self.fit = false;
            }
        }
    }

    /// Shows a view tool's Lucide icon as the pointer, as in the view bar.
    fn show_tool_cursor(&mut self, ui: &Ui, icon: char) {
        ui.ctx().set_cursor_icon(CursorIcon::None);
        self.tool_cursor = ui.input(|i| i.pointer.latest_pos()).map(|at| (at, icon));
    }

    /// The Zoom tool, as in Photoshop with Scrubby Zoom on: a click steps
    /// in at the pointer (Alt: out); a drag zooms continuously about the
    /// point where it began.
    fn zoom_tool(&mut self, ui: &Ui, response: &Response, area: Rect) {
        let centre = area.center();
        if response.drag_started_by(PointerButton::Primary)
            && let Some(origin) = ui.input(|i| i.pointer.press_origin())
        {
            self.scrub = Some(Scrub { anchor: origin - centre, zoom: self.zoom, pan: self.pan });
        }
        if response.dragged_by(PointerButton::Primary)
            && let (Some(scrub), Some(pointer)) = (self.scrub, response.interact_pointer_pos())
        {
            self.scrub_to(scrub, pointer.x - (centre + scrub.anchor).x);
        }
        if response.clicked_by(PointerButton::Primary)
            && let Some(pointer) = response.interact_pointer_pos()
        {
            let out = ui.input(|i| i.modifiers.alt);
            self.zoom_step_about(!out, ui.ctx().pixels_per_point(), pointer - centre);
        }
    }

    fn drag(&mut self, response: &Response, placement: &Placement) {
        if self.dragging_split {
            if let Some(pointer) = response.interact_pointer_pos() {
                let fraction =
                    (pointer.x - placement.after.left()) / placement.after.width().max(1.0);
                self.split = fraction.clamp(0.0, 1.0);
            }
        } else {
            self.pan += response.drag_delta();
            self.fit = false;
        }
    }

    /// Multiplies the zoom by `factor`, keeping the point `from_centre` (the
    /// pointer, relative to the middle of the view) still.
    fn zoom_about(&mut self, factor: f32, from_centre: Vec2) {
        if !factor.is_finite() || factor <= 0.0 || (factor - 1.0).abs() < 1e-4 {
            return;
        }
        let zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        let applied = zoom / self.zoom;
        self.pan = from_centre - (from_centre - self.pan) * applied;
        self.zoom = zoom;
        self.fit = false;
    }

    fn paint(
        &self,
        painter: &egui::Painter,
        scene: &Scene,
        after: Texture,
        mode: ViewMode,
        placement: &Placement,
    ) {
        let frame = placement.after;
        match mode {
            ViewMode::Result => draw_image(painter, scene.checker, after.id, frame, FULL_UV),
            ViewMode::Tiled => draw_tiles(painter, scene, after.id, frame),
            ViewMode::Split => {
                if let Some(before) = scene.before {
                    self.draw_split(painter, scene.checker, before.id, after.id, frame);
                }
            }
            ViewMode::SideBySide => {
                if let (Some(before), Some(before_frame)) = (scene.before, placement.before) {
                    draw_image(painter, scene.checker, before.id, before_frame, FULL_UV);
                    draw_tag(painter, before_frame.left_top(), Align2::LEFT_TOP, "Before");
                }
                draw_image(painter, scene.checker, after.id, frame, FULL_UV);
                draw_tag(painter, frame.left_top(), Align2::LEFT_TOP, "After");
            }
        }
        if let Some(overlay) = scene.overlay {
            // Over every copy of the result: each tile in the tiled view.
            let tiles = if mode == ViewMode::Tiled { scene.tiles.max(1) } else { 1 };
            let tile = frame.size() / tiles as f32;
            for row in 0..tiles {
                for column in 0..tiles {
                    let min = frame.min + vec2(column as f32 * tile.x, row as f32 * tile.y);
                    painter.image(overlay, Rect::from_min_size(min, tile), FULL_UV, Color32::WHITE);
                }
            }
        }
        draw_rim(painter, frame);
        if let Some(before_frame) = placement.before.filter(|_| mode == ViewMode::SideBySide) {
            draw_rim(painter, before_frame);
        }
    }

    fn draw_split(
        &self,
        painter: &egui::Painter,
        checker: TextureId,
        before: TextureId,
        after: TextureId,
        frame: Rect,
    ) {
        let divider_x = frame.left() + frame.width() * self.split;
        let left = Rect::from_min_max(frame.min, pos2(divider_x, frame.max.y));
        let right = Rect::from_min_max(pos2(divider_x, frame.min.y), frame.max);
        let left_uv = Rect::from_min_max(pos2(0.0, 0.0), pos2(self.split, 1.0));
        let right_uv = Rect::from_min_max(pos2(self.split, 0.0), pos2(1.0, 1.0));
        draw_image(painter, checker, before, left, left_uv);
        draw_image(painter, checker, after, right, right_uv);
        painter.vline(divider_x, frame.y_range(), Stroke::new(1.5, Color32::WHITE));
        let handle = Rect::from_center_size(pos2(divider_x, frame.center().y), vec2(6.0, 36.0));
        painter.rect_filled(handle, TOKENS.radius_sm, Color32::WHITE);
        draw_tag(painter, frame.left_top(), Align2::LEFT_TOP, "Before");
        draw_tag(painter, frame.right_top(), Align2::RIGHT_TOP, "After");
    }
}

/// What the mouse wheel and trackpad do to the view this frame: a zoom
/// factor about the pointer, and a scroll in screen points.
///
/// - The wheel zooms by exactly ×1.1 per notch, on the frame the notch
///   arrives. egui's smoothed tail of the notch is ignored, so the zoom does
///   not drift on after the wheel stops.
/// - A pinch (or Ctrl + wheel, which egui reports as one) zooms.
/// - Shift + wheel scrolls sideways and Alt + wheel scrolls up and down.
fn read_wheel(ui: &Ui) -> (f32, Vec2) {
    let notch = ui.ctx().options(|o| o.input_options.line_scroll_speed).max(1.0);
    ui.input(|i| {
        let pinch = i.zoom_delta();
        if pinch.is_finite() && pinch > 0.0 && (pinch - 1.0).abs() > f32::EPSILON {
            return (pinch, Vec2::ZERO);
        }
        let notches: f32 = i
            .events
            .iter()
            .filter_map(|event| match event {
                Event::MouseWheel { unit, delta, modifiers, .. } if modifiers.is_none() => {
                    Some(match unit {
                        MouseWheelUnit::Line | MouseWheelUnit::Page => delta.y,
                        MouseWheelUnit::Point => delta.y / notch,
                    })
                }
                _ => None,
            })
            .sum();
        if notches != 0.0 {
            return (WHEEL_NOTCH_ZOOM.powf(notches), Vec2::ZERO);
        }
        let scrolling = i.modifiers.shift || i.modifiers.alt;
        let scroll = i.smooth_scroll_delta;
        if scrolling && scroll.is_finite() { (1.0, scroll) } else { (1.0, Vec2::ZERO) }
    })
}

/// Keeps the image findable, per axis, as Photoshop does with Overscroll
/// off: content larger than the view cannot pull its edges inside the view,
/// and content smaller than the view stays wholly inside it. `pan` is the
/// content centre's offset from the view centre.
fn clamp_pan(pan: Vec2, view: Vec2, content: Vec2) -> Vec2 {
    let axis = |pan: f32, view: f32, content: f32| {
        let limit = ((content - view) / 2.0).abs();
        if pan.is_finite() && limit.is_finite() { pan.clamp(-limit, limit) } else { 0.0 }
    };
    vec2(axis(pan.x, view.x, content.x), axis(pan.y, view.y, content.y))
}

fn draw_image(
    painter: &egui::Painter,
    checker: TextureId,
    texture: TextureId,
    frame: Rect,
    uv: Rect,
) {
    if frame.width() <= 0.0 || frame.height() <= 0.0 {
        return;
    }
    let checker_uv = Rect::from_min_size(Pos2::ZERO, frame.size() / CHECKER_SIZE);
    painter.image(checker, frame, checker_uv, Color32::WHITE);
    painter.image(texture, frame, uv, Color32::WHITE);
}

/// A light rim just outside the image, so dark textures keep a visible edge
/// against the canvas without any of their pixels being covered.
fn draw_rim(painter: &egui::Painter, frame: Rect) {
    painter.rect_stroke(frame, 0, Stroke::new(1.0, TOKENS.sheen), StrokeKind::Outside);
}

fn draw_tiles(painter: &egui::Painter, scene: &Scene, texture: TextureId, frame: Rect) {
    let tiles = scene.tiles.max(1);
    let tile = frame.size() / tiles as f32;
    for row in 0..tiles {
        for column in 0..tiles {
            let min = frame.min + vec2(column as f32 * tile.x, row as f32 * tile.y);
            draw_image(painter, scene.checker, texture, Rect::from_min_size(min, tile), FULL_UV);
        }
    }
    if !scene.show_guides {
        return;
    }
    let stroke = Stroke::new(1.0, TOKENS.accent.gamma_multiply(0.7));
    for index in 1..tiles {
        let x = frame.left() + index as f32 * tile.x;
        let y = frame.top() + index as f32 * tile.y;
        painter.extend(Shape::dashed_line(
            &[pos2(x, frame.top()), pos2(x, frame.bottom())],
            stroke,
            GUIDE_DASH,
            GUIDE_GAP,
        ));
        painter.extend(Shape::dashed_line(
            &[pos2(frame.left(), y), pos2(frame.right(), y)],
            stroke,
            GUIDE_DASH,
            GUIDE_GAP,
        ));
    }
}

/// A small caption on a dark backing, placed inside the frame corner.
fn draw_tag(painter: &egui::Painter, corner: Pos2, anchor: Align2, text: &str) {
    let inset = vec2(if anchor == Align2::RIGHT_TOP { -8.0 } else { 8.0 }, 8.0);
    let font = FontId::proportional(SMALL_SIZE);
    let galley = painter.layout_no_wrap(text.to_owned(), font, TOKENS.text);
    let text_rect = anchor.anchor_size(corner + inset, galley.size());
    painter.rect_filled(text_rect.expand2(vec2(6.0, 3.0)), TOKENS.radius_sm, TOKENS.scrim);
    painter.galley(text_rect.min, galley, TOKENS.text);
}

/// What the canvas shows when there is no image: a loading note, or an
/// invitation to open a texture. Returns true when open was clicked.
fn show_placeholder(ui: &mut Ui, area: Rect, scene: &Scene) -> bool {
    // No card: the invitation sits directly on the canvas. An outline only
    // appears while a file is being dragged over the window.
    if scene.files_hovering {
        let outline = Stroke::new(1.5, TOKENS.accent);
        ui.painter().rect_stroke(area.shrink(12.0), TOKENS.radius_lg, outline, StrokeKind::Inside);
    }
    let block = Rect::from_center_size(area.center(), EMPTY_BLOCK_SIZE);
    let builder = UiBuilder::new().max_rect(block).layout(Layout::top_down(Align::Center));
    let mut content = ui.new_child(builder);
    if scene.loading {
        content.add_space(40.0);
        content.spinner();
        widgets::hint(&mut content, "Opening texture");
        return false;
    }
    let title = if scene.files_hovering { "Drop to open" } else { "Open a texture" };
    let icon = RichText::new(icons::IMAGE_PLUS).font(icon_font(EMPTY_ICON_SIZE));
    content.label(icon.color(TOKENS.text_dim));
    content.add_space(8.0);
    content.label(RichText::new(title).font(semibold(TITLE_SIZE)));
    widgets::hint(&mut content, "Drop an image here, or choose a file.");
    content.add_space(14.0);
    let button = content
        .allocate_ui(CHOOSE_BUTTON_SIZE, |ui| widgets::primary_button(ui, "Choose file", true));
    let clicked = button.inner.clicked();
    content.add_space(10.0);
    widgets::hint(&mut content, "PNG, JPG, TIFF, TGA or EXR. 64 to 8192 px per side.");
    clicked
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area() -> Rect {
        Rect::from_min_size(Pos2::ZERO, vec2(1000.0, 800.0))
    }

    #[test]
    fn fit_zoom_keeps_the_whole_image_inside_the_margins() {
        let zoom = CanvasState::fit_zoom(area(), vec2(4000.0, 2000.0), 0.0);

        assert!(4000.0 * zoom <= 1000.0 - 2.0 * MARGIN + 0.01);
        assert!(2000.0 * zoom <= 800.0 - 2.0 * MARGIN + 0.01);
    }

    #[test]
    fn a_large_image_cannot_pull_its_edges_inside_the_view() {
        let (view, content) = (vec2(1000.0, 800.0), vec2(3000.0, 2000.0));

        assert_eq!(clamp_pan(vec2(5000.0, -5000.0), view, content), vec2(1000.0, -600.0));
        assert_eq!(clamp_pan(vec2(300.0, 100.0), view, content), vec2(300.0, 100.0));
    }

    #[test]
    fn a_small_image_stays_wholly_inside_the_view() {
        let (view, content) = (vec2(1000.0, 800.0), vec2(400.0, 200.0));

        assert_eq!(clamp_pan(vec2(-900.0, 900.0), view, content), vec2(-300.0, 300.0));
    }

    #[test]
    fn hostile_numbers_recentre_instead_of_losing_the_image() {
        let view = vec2(1000.0, 800.0);

        assert_eq!(clamp_pan(vec2(f32::NAN, f32::INFINITY), view, view * 2.0), Vec2::ZERO);
        assert_eq!(clamp_pan(vec2(10.0, 10.0), view, vec2(f32::NAN, 100.0)), vec2(0.0, 10.0));
    }

    #[test]
    fn a_zoom_tool_drag_doubles_every_hundred_points_about_the_press() {
        let mut canvas = CanvasState { zoom: 1.0, pan: Vec2::ZERO, ..CanvasState::default() };
        let scrub = Scrub { anchor: vec2(200.0, 0.0), zoom: 1.0, pan: Vec2::ZERO };

        canvas.scrub_to(scrub, 100.0);
        assert!((canvas.zoom - 2.0).abs() < 1e-5);
        // The pressed point, 200 points right of the middle, stays put.
        assert_eq!(canvas.pan, vec2(-200.0, 0.0));

        canvas.scrub_to(scrub, -200.0);
        assert!((canvas.zoom - 0.25).abs() < 1e-5);
        canvas.scrub_to(scrub, f32::NAN);
        assert_eq!(canvas.zoom, 1.0);
    }

    #[test]
    fn a_zoom_tool_click_steps_about_the_pointer() {
        let mut canvas = CanvasState { zoom: 1.0, fit: true, ..CanvasState::default() };

        canvas.zoom_step_about(true, 1.0, vec2(100.0, 50.0));

        assert!((canvas.zoom_percent(1.0) - 200.0).abs() < 0.01);
        assert_eq!(canvas.pan, vec2(-100.0, -50.0));
    }

    #[test]
    fn view_tools_are_exclusive_and_toggle_off() {
        let mut canvas = CanvasState::default();

        canvas.toggle_tool(NavTool::Pan);
        canvas.toggle_tool(NavTool::Zoom);
        assert_eq!(canvas.tool, NavTool::Zoom);
        canvas.toggle_tool(NavTool::Zoom);
        assert_eq!(canvas.tool, NavTool::None);
    }

    #[test]
    fn zoom_steps_move_to_the_neighbouring_photoshop_levels() {
        let mut canvas = CanvasState { zoom: 0.62, fit: true, ..CanvasState::default() };

        canvas.zoom_step(true, 1.0);
        assert!((canvas.zoom_percent(1.0) - 66.67).abs() < 0.01);
        assert!(!canvas.is_fitted());

        canvas.zoom_step(false, 1.0);
        assert!((canvas.zoom_percent(1.0) - 50.0).abs() < 0.01);
    }

    #[test]
    fn zoom_steps_keep_the_middle_of_the_view_still_and_stop_at_the_ends() {
        let mut canvas =
            CanvasState { zoom: 1.0, pan: vec2(40.0, -20.0), ..CanvasState::default() };

        canvas.zoom_step(true, 1.0);
        assert_eq!(canvas.pan, vec2(80.0, -40.0), "the pan scales with the zoom");

        let mut largest = CanvasState { zoom: MAX_ZOOM, ..CanvasState::default() };
        largest.zoom_step(true, 1.0);
        assert_eq!(largest.zoom, MAX_ZOOM);
    }

    #[test]
    fn fit_zoom_does_not_enlarge_small_images_without_limit() {
        assert_eq!(CanvasState::fit_zoom(area(), vec2(64.0, 64.0), 0.0), MAX_FIT_ZOOM);
    }

    #[test]
    fn tiled_content_grows_with_the_tile_count() {
        let (two, _) = CanvasState::content_size(ViewMode::Tiled, vec2(100.0, 50.0), 2);
        let (three, _) = CanvasState::content_size(ViewMode::Tiled, vec2(100.0, 50.0), 3);

        assert_eq!(two, vec2(200.0, 100.0));
        assert_eq!(three, vec2(300.0, 150.0));
    }

    #[test]
    fn side_by_side_places_two_equal_frames_with_a_gap() {
        let state = CanvasState { zoom: 1.0, ..CanvasState::default() };

        let placement = state.place(area(), ViewMode::SideBySide, vec2(200.0, 100.0), 2);
        let before = placement.before.unwrap();

        assert_eq!(before.size(), placement.after.size());
        assert!((placement.after.left() - before.right() - SIDE_BY_SIDE_GAP).abs() < 0.01);
    }

    #[test]
    fn actual_size_maps_one_image_pixel_to_one_screen_pixel() {
        let mut state = CanvasState::default();

        state.actual_size(2.0);

        assert!((state.zoom_percent(2.0) - 100.0).abs() < 1e-3);
        assert!(!state.is_fitted());
    }
}
