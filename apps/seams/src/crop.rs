//! The Crop tool, after Photoshop's Classic Mode: the frame moves and turns
//! over the base color, which stays put.
//!
//! - Inside the untouched frame (the whole image), a drag draws a new frame:
//!   Shift makes it square, Alt draws it from the centre.
//! - Inside the frame, a drag moves it.
//! - On an edge or corner, a drag resizes it along its own axes: Shift keeps
//!   its proportions, Alt resizes about the centre, and a ratio preset
//!   always holds.
//! - Outside the frame, a drag turns it about its centre, Shift in 15° steps.
//! - Enter applies, Esc cancels.
//!
//! An upright frame stays inside the image and on whole pixels, so applying
//! it copies pixels exactly. Frame geometry is in base color pixels.
//!
//! Perspective mode, after Photoshop's Perspective Crop tool: each corner
//! moves on its own, and applying straightens the four-cornered region into
//! a rectangle.
//!
//! - A corner drag moves that corner; an edge drag moves both its corners.
//! - Inside the region, a drag moves it; outside it (or inside the untouched
//!   whole image), a drag draws a new rectangle to start from.
//! - A move that would fold the region inward is refused.
//! - The output size is measured from the region's edges unless set.

use egui::{
    Align2, Color32, CursorIcon, FontId, Key, Mesh, Painter, PointerButton, Pos2, Rect, Response,
    Shape, Stroke, Ui, Vec2, pos2, vec2,
};
use seams_core::crop::CropFrame;
use seams_core::image::{MAX_DIMENSION, MIN_DIMENSION};
use seams_core::perspective::Quad;

use crate::icons;
use crate::theme::{SMALL_SIZE, TOKENS};

/// How far (screen points) from an edge the pointer still grabs it.
const HANDLE_GRAB: f32 = 10.0;
/// A corner grabs from this many times as far as an edge does.
const CORNER_GRAB_SCALE: f32 = 1.6;
/// Side of a drawn handle, in screen points.
const HANDLE_SIZE: f32 = 7.0;
/// Shift turns the frame in steps of this many degrees.
const TURN_STEP: f32 = 15.0;
/// The smallest frame a gesture may leave: the smallest texture SEAMS takes.
const MIN_SIDE: f32 = MIN_DIMENSION as f32;
/// Far enough past the frame to cover any canvas with the shield.
const FAR: f32 = 100_000.0;

/// Fixed proportions the frame can keep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ratio {
    Free,
    /// The proportions of the image being cropped.
    Original,
    Square,
    FourThree,
    ThreeFour,
    SixteenNine,
    NineSixteen,
    TwoOne,
}

impl Ratio {
    pub const ALL: [Ratio; 8] = [
        Ratio::Free,
        Ratio::Original,
        Ratio::Square,
        Ratio::FourThree,
        Ratio::ThreeFour,
        Ratio::SixteenNine,
        Ratio::NineSixteen,
        Ratio::TwoOne,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Ratio::Free => "Free",
            Ratio::Original => "Original",
            Ratio::Square => "1 : 1 (square)",
            Ratio::FourThree => "4 : 3",
            Ratio::ThreeFour => "3 : 4",
            Ratio::SixteenNine => "16 : 9",
            Ratio::NineSixteen => "9 : 16",
            Ratio::TwoOne => "2 : 1",
        }
    }

    /// Width over height, or `None` when free.
    fn value(self, image: Vec2) -> Option<f32> {
        match self {
            Ratio::Free => None,
            Ratio::Original => Some(image.x / image.y.max(1.0)),
            Ratio::Square => Some(1.0),
            Ratio::FourThree => Some(4.0 / 3.0),
            Ratio::ThreeFour => Some(3.0 / 4.0),
            Ratio::SixteenNine => Some(16.0 / 9.0),
            Ratio::NineSixteen => Some(9.0 / 16.0),
            Ratio::TwoOne => Some(2.0),
        }
    }
}

/// How the frame is shaped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CropMode {
    /// A rectangle that may be turned.
    Classic,
    /// Four free corners, straightened into a rectangle.
    Perspective,
}

impl CropMode {
    pub const ALL: [(CropMode, &'static str); 2] =
        [(CropMode::Classic, "Classic"), (CropMode::Perspective, "Perspective")];
}

/// What applying the tool should do.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Commit {
    Frame(CropFrame),
    Perspective { quad: Quad, width: usize, height: usize },
}

/// A crop gesture in progress.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Drag {
    /// Perspective: moving corner `index` of `start`; `offset` is the
    /// pointer's distance from the corner when grabbed.
    Corner { index: usize, offset: Vec2, start: Quad },
    /// Perspective: moving edge `index` (from corner `index` to the next) of
    /// `start`, grabbed at `grab`.
    Edge { index: usize, grab: Pos2, start: Quad },
    /// Perspective: moving the whole region `start`, grabbed at `grab`.
    Shift { grab: Pos2, start: Quad },
    /// Drawing a new frame from `anchor` (image pixels).
    Draw { anchor: Pos2 },
    /// Moving the frame `start`, grabbed at `grab`.
    Move { grab: Pos2, start: CropFrame },
    /// Dragging an edge or corner: `hx`/`hy` are -1 (left/top), 1
    /// (right/bottom) or 0. `offset` is the pointer's distance from the
    /// handle when grabbed, in the frame's own axes.
    Resize { hx: f32, hy: f32, offset: Vec2, start: CropFrame },
    /// Turning the frame `start`, grabbed at angle `grab` (degrees).
    Turn { grab: f32, start: CropFrame },
}

/// What lies under a point.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Hit {
    Handle(f32, f32),
    Inside,
    Outside,
}

/// Where the image is on screen: its top-left corner and screen points per
/// image pixel.
#[derive(Debug, Clone, Copy)]
pub struct Mapping {
    origin: Pos2,
    scale: f32,
}

impl Mapping {
    /// For an image of `image_size` pixels drawn in `rect`.
    pub fn new(rect: Rect, image_size: Vec2) -> Self {
        Self { origin: rect.min, scale: (rect.width() / image_size.x.max(1.0)).max(1e-6) }
    }

    /// Screen points per image pixel.
    pub fn scale(self) -> f32 {
        self.scale
    }

    pub fn to_image(self, screen: Pos2) -> Pos2 {
        ((screen - self.origin) / self.scale).to_pos2()
    }

    pub fn to_screen(self, image: Pos2) -> Pos2 {
        self.origin + image.to_vec2() * self.scale
    }
}

/// What lies under a point in perspective mode.
#[derive(Debug, Clone, Copy, PartialEq)]
enum QuadHit {
    Corner(usize),
    Edge(usize),
    Inside,
    Outside,
}

/// The Crop tool's state while it is open.
pub struct CropTool {
    mode: CropMode,
    frame: CropFrame,
    ratio: Ratio,
    /// The region in perspective mode.
    quad: Quad,
    /// A size set for the straightened region; `None` measures it.
    output: Option<(f32, f32)>,
    /// Size of the image being cropped, in pixels.
    image: Vec2,
    /// The frame is still the default whole image: a drag inside it draws a
    /// new frame instead of moving it.
    untouched: bool,
    drag: Option<Drag>,
    /// Where to draw the turn cursor (screen points), while the pointer
    /// would turn the frame. The system has no turn cursor of its own.
    turn_cursor: Option<Pos2>,
}

impl CropTool {
    pub fn new(width: usize, height: usize) -> Self {
        let frame = CropFrame::whole(width, height);
        Self {
            mode: CropMode::Classic,
            frame,
            ratio: Ratio::Free,
            quad: Quad::from_frame(&frame),
            output: None,
            image: vec2(width as f32, height as f32),
            untouched: true,
            drag: None,
            turn_cursor: None,
        }
    }

    pub fn mode(&self) -> CropMode {
        self.mode
    }

    /// Switches mode. Perspective starts from the classic frame's corners.
    pub fn set_mode(&mut self, mode: CropMode) {
        if mode == CropMode::Perspective && self.mode != mode {
            self.quad = Quad::from_frame(&self.frame);
            self.output = None;
        }
        self.mode = mode;
        self.drag = None;
    }

    /// What applying should do.
    pub fn commit(&self) -> Commit {
        match self.mode {
            CropMode::Classic => Commit::Frame(self.frame),
            CropMode::Perspective => {
                let (width, height) = self.size();
                Commit::Perspective {
                    quad: self.quad,
                    width: width as usize,
                    height: height as usize,
                }
            }
        }
    }

    /// True when the straightened size was set by hand.
    pub fn has_set_size(&self) -> bool {
        self.output.is_some()
    }

    /// Goes back to measuring the straightened size from the region.
    pub fn measure_size(&mut self) {
        self.output = None;
    }

    /// True when the tool was made for an image of this size.
    pub fn fits(&self, width: usize, height: usize) -> bool {
        self.image == vec2(width as f32, height as f32)
    }

    pub fn ratio(&self) -> Ratio {
        self.ratio
    }

    /// The result's size in whole pixels.
    pub fn size(&self) -> (f32, f32) {
        let (width, height) = match self.mode {
            CropMode::Classic => (self.frame.width, self.frame.height),
            CropMode::Perspective => self.output.unwrap_or_else(|| self.quad.measured_size()),
        };
        let side = |v: f32| v.round().clamp(MIN_SIDE, MAX_DIMENSION as f32);
        (side(width), side(height))
    }

    pub fn degrees(&self) -> f32 {
        self.frame.degrees
    }

    /// Chooses fixed proportions and reshapes the frame to them about its
    /// centre, keeping its width where the image allows.
    pub fn set_ratio(&mut self, ratio: Ratio) {
        self.ratio = ratio;
        if let Some(value) = ratio.value(self.image) {
            self.frame.height = self.frame.width / value;
            self.fit_size_to_image(value);
            self.untouched = false;
        }
        self.frame = constrain(self.frame, self.image);
    }

    pub fn set_width(&mut self, width: f32) {
        if self.mode == CropMode::Perspective {
            self.output = Some((width, self.size().1));
            return;
        }
        self.frame.width = width.max(MIN_SIDE);
        if let Some(value) = self.ratio.value(self.image) {
            self.frame.height = self.frame.width / value;
            self.fit_size_to_image(value);
        }
        self.edited();
    }

    pub fn set_height(&mut self, height: f32) {
        if self.mode == CropMode::Perspective {
            self.output = Some((self.size().0, height));
            return;
        }
        self.frame.height = height.max(MIN_SIDE);
        if let Some(value) = self.ratio.value(self.image) {
            self.frame.width = self.frame.height * value;
            self.fit_size_to_image(value);
        }
        self.edited();
    }

    pub fn set_degrees(&mut self, degrees: f32) {
        self.frame.degrees = normalize_degrees(degrees);
        self.edited();
    }

    fn edited(&mut self) {
        self.untouched = false;
        self.frame = constrain(self.frame, self.image);
    }

    /// Shrinks an upright frame with proportions `value` until it fits.
    fn fit_size_to_image(&mut self, value: f32) {
        if !self.frame.is_upright() {
            return;
        }
        let f = &mut self.frame;
        if f.width > self.image.x {
            f.width = self.image.x;
            f.height = f.width / value;
        }
        if f.height > self.image.y {
            f.height = self.image.y;
            f.width = f.height * value;
        }
    }

    /// Enter applies (`Some(true)`) and Esc cancels (`Some(false)`), unless
    /// a text field has the keyboard.
    pub fn key_command(ui: &egui::Context) -> Option<bool> {
        // An open dropdown takes Esc and Enter for itself.
        if ui.egui_wants_keyboard_input() || egui::Popup::is_any_open(ui) {
            return None;
        }
        ui.input(|i| {
            if i.key_pressed(Key::Enter) {
                Some(true)
            } else if i.key_pressed(Key::Escape) {
                Some(false)
            } else {
                None
            }
        })
    }

    /// Handles the crop gestures of a primary-button drag on the canvas.
    pub fn interact(&mut self, ui: &Ui, response: &Response, map: Mapping) {
        if self.mode == CropMode::Perspective {
            return self.interact_perspective(ui, response, map);
        }
        let tolerance = HANDLE_GRAB / map.scale;
        let hover = response.hover_pos().map(|p| map.to_image(p));
        let modifiers = ui.input(|i| i.modifiers);
        if response.drag_started_by(PointerButton::Primary)
            && let Some(point) = press_point(ui, response).map(|p| map.to_image(p))
        {
            self.drag = Some(self.begin(point, tolerance));
        }
        if response.dragged_by(PointerButton::Primary)
            && let (Some(drag), Some(point)) =
                (self.drag, response.interact_pointer_pos().map(|p| map.to_image(p)))
        {
            let ratio = self.ratio.value(self.image);
            let (keep, centred) = (modifiers.shift, modifiers.alt);
            self.frame = match drag {
                Drag::Draw { anchor } => {
                    let point = clamp_to(point, self.image);
                    draw(anchor, point, ratio.or(keep.then_some(1.0)), centred)
                }
                Drag::Move { grab, start } => moved(start, point - grab),
                Drag::Resize { hx, hy, offset, start } => {
                    let local = to_local(&start, point) - offset;
                    let ratio = ratio.or(keep.then(|| start.width / start.height.max(1.0)));
                    resize(start, hx, hy, local, ratio, centred)
                }
                Drag::Turn { grab, start } => {
                    let degrees = start.degrees + angle_of(point, &start) - grab;
                    let degrees =
                        if keep { (degrees / TURN_STEP).round() * TURN_STEP } else { degrees };
                    CropFrame { degrees: normalize_degrees(degrees), ..start }
                }
                Drag::Corner { .. } | Drag::Edge { .. } | Drag::Shift { .. } => self.frame,
            };
            self.frame = constrain(self.frame, self.image);
            self.untouched = false;
        }
        if response.drag_stopped() {
            self.drag = None;
        }
        let hit = hover.map(|point| hit(&self.frame, point, tolerance));
        let turning = matches!(self.drag, Some(Drag::Turn { .. }))
            || (self.drag.is_none() && hit == Some(Hit::Outside));
        // The pointer itself while turning; the hover position may be gone
        // once the drag leaves the canvas.
        let pointer = ui.input(|i| i.pointer.latest_pos());
        self.turn_cursor = pointer.filter(|_| turning && (hover.is_some() || self.drag.is_some()));
        let icon = if self.turn_cursor.is_some() { CursorIcon::None } else { self.cursor(hit) };
        ui.ctx().set_cursor_icon(icon);
    }

    fn begin(&self, point: Pos2, tolerance: f32) -> Drag {
        let start = self.frame;
        match hit(&start, point, tolerance) {
            Hit::Handle(hx, hy) => {
                let handle = vec2(hx * start.width / 2.0, hy * start.height / 2.0);
                let local = to_local(&start, point);
                // Only the grabbed axes keep an offset; the other follows the frame.
                let offset = vec2(
                    if hx != 0.0 { local.x - handle.x } else { 0.0 },
                    if hy != 0.0 { local.y - handle.y } else { 0.0 },
                );
                Drag::Resize { hx, hy, offset, start }
            }
            Hit::Inside if self.untouched => Drag::Draw { anchor: clamp_to(point, self.image) },
            Hit::Inside => Drag::Move { grab: point, start },
            Hit::Outside => Drag::Turn { grab: angle_of(point, &start), start },
        }
    }

    fn cursor(&self, hit: Option<Hit>) -> CursorIcon {
        match (self.drag, hit) {
            (Some(Drag::Move { .. } | Drag::Shift { .. } | Drag::Edge { .. }), _) => {
                CursorIcon::Grabbing
            }
            (Some(Drag::Corner { .. }), _) => CursorIcon::Crosshair,
            (Some(Drag::Draw { .. }), _) => CursorIcon::Crosshair,
            (Some(Drag::Turn { .. }), _) => CursorIcon::AllScroll,
            (Some(Drag::Resize { hx, hy, start, .. }), _) => resize_cursor(hx, hy, start.degrees),
            (None, Some(Hit::Handle(hx, hy))) => resize_cursor(hx, hy, self.frame.degrees),
            (None, Some(Hit::Inside)) if self.untouched => CursorIcon::Crosshair,
            (None, Some(Hit::Inside)) => CursorIcon::Move,
            (None, Some(Hit::Outside)) => CursorIcon::AllScroll,
            (None, None) => CursorIcon::Default,
        }
    }

    /// Perspective mode's gestures.
    fn interact_perspective(&mut self, ui: &Ui, response: &Response, map: Mapping) {
        self.turn_cursor = None;
        let tolerance = HANDLE_GRAB / map.scale;
        let to_image = |p: Pos2| map.to_image(p);
        if response.drag_started_by(PointerButton::Primary)
            && let Some(point) = press_point(ui, response).map(to_image)
        {
            let start = self.quad;
            self.drag = Some(match quad_hit(&start, point, tolerance) {
                QuadHit::Corner(index) => {
                    Drag::Corner { index, offset: point - corner_pos(&start, index), start }
                }
                QuadHit::Edge(index) => Drag::Edge { index, grab: point, start },
                // A new rectangle only replaces the untouched whole image:
                // a near miss must never throw away corners placed by hand.
                QuadHit::Inside | QuadHit::Outside if self.untouched => {
                    Drag::Draw { anchor: clamp_to(point, self.image) }
                }
                QuadHit::Inside => Drag::Shift { grab: point, start },
                QuadHit::Outside => return,
            });
        }
        if response.dragged_by(PointerButton::Primary)
            && let (Some(drag), Some(point)) =
                (self.drag, response.interact_pointer_pos().map(to_image))
        {
            let modifiers = ui.input(|i| i.modifiers);
            let moved = match drag {
                Drag::Corner { index, offset, start } => {
                    let mut quad = start;
                    quad.corners[index] = to_array(clamp_to(point - offset, self.image));
                    quad
                }
                Drag::Edge { index, grab, start } => {
                    let mut quad = start;
                    let by = point - grab;
                    for corner in [index, (index + 1) % 4] {
                        let at = corner_pos(&start, corner) + by;
                        quad.corners[corner] = to_array(clamp_to(at, self.image));
                    }
                    quad
                }
                Drag::Shift { grab, start } => shift_inside(start, point - grab, self.image),
                Drag::Draw { anchor } => {
                    let point = clamp_to(point, self.image);
                    let shape = modifiers.shift.then_some(1.0);
                    Quad::from_frame(&draw(anchor, point, shape, modifiers.alt))
                }
                _ => self.quad,
            };
            // A region folded inward cannot be straightened: keep the last good one.
            if moved.is_convex() {
                self.quad = moved;
            }
            self.untouched = false;
        }
        if response.drag_stopped() {
            self.drag = None;
        }
        let hover = response.hover_pos().map(to_image);
        let cursor = match (self.drag, hover.map(|point| quad_hit(&self.quad, point, tolerance))) {
            (Some(Drag::Corner { .. }), _) | (None, Some(QuadHit::Corner(_))) => {
                CursorIcon::Crosshair
            }
            (Some(Drag::Edge { .. } | Drag::Shift { .. }), _) => CursorIcon::Grabbing,
            (None, Some(QuadHit::Edge(_))) => CursorIcon::Grab,
            (None, Some(QuadHit::Inside)) if !self.untouched => CursorIcon::Move,
            (_, Some(_)) | (Some(_), None) => CursorIcon::Crosshair,
            (None, None) => CursorIcon::Default,
        };
        ui.ctx().set_cursor_icon(cursor);
    }

    /// Perspective mode's overlay: the shaded outside, the region with a
    /// grid that follows its perspective, and its corner and edge handles.
    fn paint_perspective(&self, painter: &Painter, map: Mapping) {
        let screen = |index: usize| map.to_screen(corner_pos(&self.quad, index));
        let corners = [0, 1, 2, 3].map(screen);
        painter.add(Shape::mesh(shield(corners, TOKENS.scrim)));
        if let Some(homography) = self.quad.homography() {
            let at = |u: f32, v: f32| map.to_screen(Pos2::from(homography.map(u, v)));
            let grid = Stroke::new(1.0, Color32::WHITE.gamma_multiply(0.35));
            for t in [0.25, 0.5, 0.75] {
                painter.line_segment([at(t, 0.0), at(t, 1.0)], grid);
                painter.line_segment([at(0.0, t), at(1.0, t)], grid);
            }
        }
        let mut outline = corners.to_vec();
        outline.push(corners[0]);
        painter.add(Shape::line(outline, Stroke::new(1.0, Color32::WHITE)));
        for index in 0..4 {
            let middle = lerp(corners[index], corners[(index + 1) % 4], 0.5);
            painter.add(handle(middle, 0.0));
            painter.add(handle(corners[index], 0.0));
        }
        if self.drag.is_some() {
            let (width, height) = self.size();
            let top =
                corners.iter().copied().fold(corners[0], |a, b| if b.y < a.y { b } else { a });
            tag(painter, top - vec2(0.0, 10.0), &format!("{width:.0} × {height:.0} px"));
        }
    }

    /// Shades everything outside the frame and draws the frame, its thirds,
    /// its handles and, during a gesture, its size or angle.
    pub fn paint(&self, painter: &Painter, map: Mapping) {
        if self.mode == CropMode::Perspective {
            return self.paint_perspective(painter, map);
        }
        let corners = corners(&self.frame).map(|corner| map.to_screen(corner));
        painter.add(Shape::mesh(shield(corners, TOKENS.scrim)));
        let thirds = Stroke::new(1.0, Color32::WHITE.gamma_multiply(0.35));
        for t in [1.0 / 3.0, 2.0 / 3.0] {
            painter.line_segment(
                [lerp(corners[0], corners[1], t), lerp(corners[3], corners[2], t)],
                thirds,
            );
            painter.line_segment(
                [lerp(corners[0], corners[3], t), lerp(corners[1], corners[2], t)],
                thirds,
            );
        }
        let mut outline = corners.to_vec();
        outline.push(corners[0]);
        painter.add(Shape::line(outline, Stroke::new(1.0, Color32::WHITE)));
        for (hx, hy) in HANDLES {
            let centre = map.to_screen(to_world(
                &self.frame,
                vec2(hx * self.frame.width / 2.0, hy * self.frame.height / 2.0),
            ));
            painter.add(handle(centre, self.frame.degrees));
        }
        if self.drag.is_some_and(|drag| !matches!(drag, Drag::Turn { .. })) {
            let (width, height) = self.size();
            let top =
                corners.iter().copied().fold(corners[0], |a, b| if b.y < a.y { b } else { a });
            tag(painter, top - vec2(0.0, 10.0), &format!("{width:.0} × {height:.0} px"));
        }
        if let Some(pointer) = self.turn_cursor {
            // The system has no turn cursor: Lucide's turning arrow stands in.
            icons::paint_cursor(painter, pointer, icons::ROTATE);
            if matches!(self.drag, Some(Drag::Turn { .. })) {
                let label = format!("{:.1}°", self.frame.degrees);
                tag(painter, pointer + vec2(0.0, icons::CURSOR_SIZE + 12.0), &label);
            }
        }
    }
}

/// The eight handles, as (x, y) sides of the frame.
const HANDLES: [(f32, f32); 8] = [
    (-1.0, -1.0),
    (0.0, -1.0),
    (1.0, -1.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 0.0),
];

fn corner_pos(quad: &Quad, index: usize) -> Pos2 {
    let [x, y] = quad.corners[index % 4];
    pos2(x, y)
}

fn to_array(point: Pos2) -> [f32; 2] {
    [point.x, point.y]
}

fn quad_hit(quad: &Quad, point: Pos2, tolerance: f32) -> QuadHit {
    let corners = [0, 1, 2, 3].map(|index| corner_pos(quad, index));
    // Corners reach further than edges, as in classic mode; the nearest wins.
    let reach = tolerance * CORNER_GRAB_SCALE;
    let nearest =
        (0..4).min_by(|a, b| corners[*a].distance(point).total_cmp(&corners[*b].distance(point)));
    if let Some(index) = nearest.filter(|index| corners[*index].distance(point) <= reach) {
        return QuadHit::Corner(index);
    }
    for index in 0..4 {
        let (a, b) = (corners[index], corners[(index + 1) % 4]);
        if distance_to_segment(point, a, b) <= tolerance {
            return QuadHit::Edge(index);
        }
    }
    // Inside a clockwise convex region, every edge turns the same way to the point.
    let inside = (0..4).all(|index| {
        let (a, b) = (corners[index], corners[(index + 1) % 4]);
        (b - a).x * (point - a).y - (b - a).y * (point - a).x >= 0.0
    });
    if inside { QuadHit::Inside } else { QuadHit::Outside }
}

fn distance_to_segment(point: Pos2, a: Pos2, b: Pos2) -> f32 {
    let edge = b - a;
    let t = ((point - a).dot(edge) / edge.length_sq().max(1e-6)).clamp(0.0, 1.0);
    point.distance(a + edge * t)
}

/// `start` moved by `by`, held so its corners stay inside the image.
fn shift_inside(start: Quad, by: Vec2, image: Vec2) -> Quad {
    let xs = start.corners.map(|[x, _]| x);
    let ys = start.corners.map(|[_, y]| y);
    let min = pos2(
        xs.iter().copied().fold(f32::MAX, f32::min),
        ys.iter().copied().fold(f32::MAX, f32::min),
    );
    let max = pos2(
        xs.iter().copied().fold(f32::MIN, f32::max),
        ys.iter().copied().fold(f32::MIN, f32::max),
    );
    let by = vec2(
        by.x.clamp(-min.x, (image.x - max.x).max(-min.x)),
        by.y.clamp(-min.y, (image.y - max.y).max(-min.y)),
    );
    Quad { corners: start.corners.map(|[x, y]| [x + by.x, y + by.y]) }
}

fn centre(frame: &CropFrame) -> Pos2 {
    pos2(frame.centre_x, frame.centre_y)
}

/// A point in the frame's own axes, relative to its centre.
fn to_local(frame: &CropFrame, point: Pos2) -> Vec2 {
    let (sin, cos) = frame.degrees.to_radians().sin_cos();
    let d = point - centre(frame);
    vec2(cos * d.x + sin * d.y, -sin * d.x + cos * d.y)
}

/// A point given in the frame's own axes, back in image pixels.
fn to_world(frame: &CropFrame, local: Vec2) -> Pos2 {
    let (sin, cos) = frame.degrees.to_radians().sin_cos();
    centre(frame) + vec2(cos * local.x - sin * local.y, sin * local.x + cos * local.y)
}

/// Top-left, top-right, bottom-right and bottom-left, in image pixels.
fn corners(frame: &CropFrame) -> [Pos2; 4] {
    let (w, h) = (frame.width / 2.0, frame.height / 2.0);
    [vec2(-w, -h), vec2(w, -h), vec2(w, h), vec2(-w, h)].map(|corner| to_world(frame, corner))
}

/// Where the drag began. egui only reports a drag once the pointer has
/// moved a few points, so the pointer's position then is already past a
/// handle that was pressed on its outer side; the press itself is not.
fn press_point(ui: &Ui, response: &Response) -> Option<Pos2> {
    ui.input(|i| i.pointer.press_origin()).or_else(|| response.interact_pointer_pos())
}

fn hit(frame: &CropFrame, point: Pos2, tolerance: f32) -> Hit {
    let local = to_local(frame, point);
    let (w, h) = (frame.width / 2.0, frame.height / 2.0);
    // Corners get a larger square of their own, so a press just off a corner
    // resizes instead of turning the frame.
    let reach = tolerance * CORNER_GRAB_SCALE;
    let near = |value: f32, half: f32| {
        if (value + half).abs() <= reach {
            -1.0
        } else if (value - half).abs() <= reach {
            1.0
        } else {
            0.0
        }
    };
    let (cx, cy) = (near(local.x, w), near(local.y, h));
    if cx != 0.0 && cy != 0.0 {
        return Hit::Handle(cx, cy);
    }
    let along_y = local.y.abs() <= h + tolerance;
    let along_x = local.x.abs() <= w + tolerance;
    let side = |value: f32, half: f32, along: bool| {
        if !along {
            0.0
        } else if (value + half).abs() <= tolerance {
            -1.0
        } else if (value - half).abs() <= tolerance {
            1.0
        } else {
            0.0
        }
    };
    let (hx, hy) = (side(local.x, w, along_y), side(local.y, h, along_x));
    if hx != 0.0 || hy != 0.0 {
        Hit::Handle(hx, hy)
    } else if local.x.abs() <= w && local.y.abs() <= h {
        Hit::Inside
    } else {
        Hit::Outside
    }
}

/// The angle (degrees, clockwise on screen) of `point` seen from the centre.
fn angle_of(point: Pos2, frame: &CropFrame) -> f32 {
    let d = point - centre(frame);
    d.y.atan2(d.x).to_degrees()
}

/// Into `(-180, 180]`.
fn normalize_degrees(degrees: f32) -> f32 {
    if !degrees.is_finite() {
        return 0.0;
    }
    let wrapped = degrees.rem_euclid(360.0);
    if wrapped > 180.0 { wrapped - 360.0 } else { wrapped }
}

fn clamp_to(point: Pos2, image: Vec2) -> Pos2 {
    pos2(point.x.clamp(0.0, image.x), point.y.clamp(0.0, image.y))
}

fn moved(start: CropFrame, by: Vec2) -> CropFrame {
    CropFrame { centre_x: start.centre_x + by.x, centre_y: start.centre_y + by.y, ..start }
}

/// An upright frame from `anchor` to `point`, or centred on `anchor`.
fn draw(anchor: Pos2, point: Pos2, ratio: Option<f32>, centred: bool) -> CropFrame {
    let d = point - anchor;
    let scale = if centred { 2.0 } else { 1.0 };
    let (mut width, mut height) =
        ((d.x.abs() * scale).max(MIN_SIDE), (d.y.abs() * scale).max(MIN_SIDE));
    if let Some(ratio) = ratio {
        if width / height > ratio {
            height = width / ratio;
        } else {
            width = height * ratio;
        }
    }
    let direction = vec2(if d.x < 0.0 { -1.0 } else { 1.0 }, if d.y < 0.0 { -1.0 } else { 1.0 });
    let centre = if centred { anchor } else { anchor + direction * vec2(width, height) / 2.0 };
    CropFrame { centre_x: centre.x, centre_y: centre.y, width, height, degrees: 0.0 }
}

/// `start` with the grabbed sides moved to `local` (in its own axes).
fn resize(
    start: CropFrame,
    hx: f32,
    hy: f32,
    local: Vec2,
    ratio: Option<f32>,
    centred: bool,
) -> CropFrame {
    let half = vec2(start.width / 2.0, start.height / 2.0);
    let (mut lo, mut hi) = (-half, half);
    let axis = |side: f32, value: f32, lo: &mut f32, hi: &mut f32| {
        if side == 0.0 {
            return;
        }
        if centred {
            let reach = (value * side).max(MIN_SIDE / 2.0);
            (*lo, *hi) = (-reach, reach);
        } else if side > 0.0 {
            *hi = value.max(*lo + MIN_SIDE);
        } else {
            *lo = value.min(*hi - MIN_SIDE);
        }
    };
    axis(hx, local.x, &mut lo.x, &mut hi.x);
    axis(hy, local.y, &mut lo.y, &mut hi.y);
    if let Some(ratio) = ratio {
        let (mut width, mut height) = (hi.x - lo.x, hi.y - lo.y);
        if hx != 0.0 && hy != 0.0 {
            if width / height > ratio {
                height = width / ratio;
            } else {
                width = height * ratio;
            }
        } else if hx != 0.0 {
            height = width / ratio;
        } else {
            width = height * ratio;
        }
        let place = |side: f32, size: f32, lo: &mut f32, hi: &mut f32| {
            if centred || side == 0.0 {
                let middle = (*lo + *hi) / 2.0;
                (*lo, *hi) = (middle - size / 2.0, middle + size / 2.0);
            } else if side > 0.0 {
                *hi = *lo + size;
            } else {
                *lo = *hi - size;
            }
        };
        place(hx, width, &mut lo.x, &mut hi.x);
        place(hy, height, &mut lo.y, &mut hi.y);
    }
    let middle = to_world(&start, (lo + hi) / 2.0);
    CropFrame {
        centre_x: middle.x,
        centre_y: middle.y,
        width: hi.x - lo.x,
        height: hi.y - lo.y,
        degrees: start.degrees,
    }
}

/// Keeps a frame usable: at least the minimum size, and an upright one
/// inside the image and on whole pixels. A turned frame may reach past the
/// image; the crop repeats edge pixels there.
fn constrain(frame: CropFrame, image: Vec2) -> CropFrame {
    let mut f = frame;
    f.width = f.width.max(MIN_SIDE.min(image.x));
    f.height = f.height.max(MIN_SIDE.min(image.y));
    if !f.is_upright() {
        return f;
    }
    f.degrees = 0.0;
    f.width = f.width.min(image.x).round();
    f.height = f.height.min(image.y).round();
    let left = (f.centre_x - f.width / 2.0).clamp(0.0, image.x - f.width).round();
    let top = (f.centre_y - f.height / 2.0).clamp(0.0, image.y - f.height).round();
    f.centre_x = left + f.width / 2.0;
    f.centre_y = top + f.height / 2.0;
    f
}

fn resize_cursor(hx: f32, hy: f32, degrees: f32) -> CursorIcon {
    // The direction the handle pulls, on screen, folded into 0..180°.
    let angle = (hy.atan2(hx).to_degrees() + degrees).rem_euclid(180.0);
    match ((angle + 22.5) / 45.0) as u32 % 4 {
        0 => CursorIcon::ResizeHorizontal,
        1 => CursorIcon::ResizeNwSe,
        2 => CursorIcon::ResizeVertical,
        _ => CursorIcon::ResizeNeSw,
    }
}

fn lerp(a: Pos2, b: Pos2, t: f32) -> Pos2 {
    a + (b - a) * t
}

/// Shade covering everything outside the convex quad `corners` (clockwise
/// on screen): a band beyond each edge and a block beyond each corner, so
/// nothing is shaded twice.
fn shield(corners: [Pos2; 4], color: Color32) -> Mesh {
    let mut mesh = Mesh::default();
    let mut quad = |points: [Pos2; 4]| {
        let first = mesh.vertices.len() as u32;
        for point in points {
            mesh.colored_vertex(point, color);
        }
        mesh.add_triangle(first, first + 1, first + 2);
        mesh.add_triangle(first, first + 2, first + 3);
    };
    let normal = |a: Pos2, b: Pos2| {
        let edge = b - a;
        vec2(edge.y, -edge.x).normalized()
    };
    for i in 0..4 {
        let (a, b, c) = (corners[i], corners[(i + 1) % 4], corners[(i + 2) % 4]);
        let (n, next) = (normal(a, b) * FAR, normal(b, c) * FAR);
        quad([a, b, b + n, a + n]);
        quad([b, b + n, b + n + next, b + next]);
    }
    mesh
}

/// A small square handle at `centre`, turned with the frame.
fn handle(centre: Pos2, degrees: f32) -> Shape {
    let (sin, cos) = degrees.to_radians().sin_cos();
    let h = HANDLE_SIZE / 2.0;
    let points = [vec2(-h, -h), vec2(h, -h), vec2(h, h), vec2(-h, h)]
        .map(|p| centre + vec2(cos * p.x - sin * p.y, sin * p.x + cos * p.y))
        .to_vec();
    Shape::convex_polygon(points, Color32::WHITE, Stroke::new(1.0, TOKENS.canvas))
}

/// A short label on a dark backing, centred above `at`.
fn tag(painter: &Painter, at: Pos2, text: &str) {
    let font = FontId::monospace(SMALL_SIZE);
    let galley = painter.layout_no_wrap(text.to_owned(), font, TOKENS.text);
    let rect = Align2::CENTER_BOTTOM.anchor_size(at, galley.size());
    painter.rect_filled(rect.expand2(vec2(6.0, 3.0)), TOKENS.radius_sm, TOKENS.scrim);
    painter.galley(rect.min, galley, TOKENS.text);
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMAGE: Vec2 = vec2(1000.0, 800.0);

    fn upright(left: f32, top: f32, width: f32, height: f32) -> CropFrame {
        CropFrame {
            centre_x: left + width / 2.0,
            centre_y: top + height / 2.0,
            width,
            height,
            degrees: 0.0,
        }
    }

    #[test]
    fn points_are_found_on_handles_inside_and_outside() {
        let frame = upright(100.0, 100.0, 400.0, 300.0);

        assert_eq!(hit(&frame, pos2(102.0, 98.0), 8.0), Hit::Handle(-1.0, -1.0));
        assert_eq!(hit(&frame, pos2(300.0, 403.0), 8.0), Hit::Handle(0.0, 1.0));
        assert_eq!(hit(&frame, pos2(300.0, 250.0), 8.0), Hit::Inside);
        assert_eq!(hit(&frame, pos2(700.0, 250.0), 8.0), Hit::Outside);
    }

    #[test]
    fn a_press_just_off_a_corner_grabs_the_corner_instead_of_turning() {
        let frame = upright(100.0, 100.0, 400.0, 300.0);

        // 12 px out on both axes: past an edge's reach, inside a corner's.
        assert_eq!(hit(&frame, pos2(88.0, 88.0), 10.0), Hit::Handle(-1.0, -1.0));
        assert_eq!(hit(&frame, pos2(512.0, 412.0), 10.0), Hit::Handle(1.0, 1.0));
        // Further out it turns, as in Photoshop.
        assert_eq!(hit(&frame, pos2(70.0, 70.0), 10.0), Hit::Outside);
    }

    #[test]
    fn hit_testing_follows_a_turned_frame() {
        let frame = CropFrame { degrees: 90.0, ..upright(100.0, 100.0, 400.0, 200.0) };

        // Turned a quarter, the frame's right edge faces down the image.
        assert_eq!(hit(&frame, pos2(300.0, 400.0), 8.0), Hit::Handle(1.0, 0.0));
        assert_eq!(hit(&frame, pos2(300.0, 395.0), 8.0), Hit::Handle(1.0, 0.0));
        assert_eq!(hit(&frame, pos2(500.0, 200.0), 8.0), Hit::Outside);
    }

    #[test]
    fn dragging_the_right_edge_keeps_the_left_edge() {
        let start = upright(100.0, 100.0, 400.0, 300.0);

        let frame = resize(start, 1.0, 0.0, vec2(300.0, 0.0), None, false);

        assert_eq!(frame.width, 500.0);
        assert_eq!(frame.centre_x - frame.width / 2.0, 100.0);
        assert_eq!(frame.height, 300.0);
    }

    #[test]
    fn a_corner_keeps_the_ratio_and_the_opposite_corner() {
        let start = upright(0.0, 0.0, 400.0, 400.0);

        let frame = resize(start, 1.0, 1.0, vec2(400.0, 250.0), Some(1.0), false);

        assert_eq!((frame.width, frame.height), (600.0, 600.0));
        assert_eq!((frame.centre_x - 300.0, frame.centre_y - 300.0), (0.0, 0.0));
    }

    #[test]
    fn alt_resizes_about_the_centre() {
        let start = upright(100.0, 100.0, 400.0, 300.0);

        let frame = resize(start, 1.0, 0.0, vec2(300.0, 0.0), None, true);

        assert_eq!(frame.width, 600.0);
        assert_eq!(frame.centre_x, start.centre_x);
    }

    #[test]
    fn a_resize_never_goes_below_the_minimum() {
        let start = upright(100.0, 100.0, 400.0, 300.0);

        let frame = resize(start, -1.0, 0.0, vec2(500.0, 0.0), None, false);

        assert_eq!(frame.width, MIN_SIDE);
    }

    #[test]
    fn shift_draws_a_square_and_alt_draws_from_the_centre() {
        let square = draw(pos2(100.0, 100.0), pos2(400.0, 200.0), Some(1.0), false);
        assert_eq!((square.width, square.height), (300.0, 300.0));
        assert_eq!((square.centre_x, square.centre_y), (250.0, 250.0));

        let centred = draw(pos2(500.0, 400.0), pos2(600.0, 450.0), None, true);
        assert_eq!((centred.centre_x, centred.centre_y), (500.0, 400.0));
        assert_eq!((centred.width, centred.height), (200.0, 100.0));
    }

    #[test]
    fn an_upright_frame_stays_inside_the_image_on_whole_pixels() {
        let outside = upright(-50.3, 700.6, 300.2, 250.0);

        let frame = constrain(outside, IMAGE);

        assert_eq!(frame.centre_x - frame.width / 2.0, 0.0);
        assert_eq!(frame.centre_y + frame.height / 2.0, 800.0);
        assert_eq!(frame.width, 300.0);
    }

    #[test]
    fn a_turned_frame_may_reach_past_the_image() {
        let turned = CropFrame { degrees: 10.0, ..upright(-50.0, -50.0, 300.0, 300.0) };

        assert_eq!(constrain(turned, IMAGE), turned);
    }

    #[test]
    fn angles_wrap_into_a_half_turn_either_way() {
        assert_eq!(normalize_degrees(190.0), -170.0);
        assert_eq!(normalize_degrees(-190.0), 170.0);
        assert_eq!(normalize_degrees(180.0), 180.0);
        assert_eq!(normalize_degrees(f32::NAN), 0.0);
    }

    #[test]
    fn a_ratio_reshapes_the_frame_to_fit_the_image() {
        let mut tool = CropTool::new(1000, 800);

        tool.set_ratio(Ratio::SixteenNine);
        let (width, height) = tool.size();
        assert!((width / height - 16.0 / 9.0).abs() < 0.01, "{width} x {height}");
        assert!(width <= 1000.0 && height <= 800.0);

        tool.set_ratio(Ratio::Square);
        assert_eq!(tool.size(), (800.0, 800.0));
    }

    fn keystone() -> Quad {
        Quad { corners: [[300.0, 100.0], [700.0, 100.0], [900.0, 700.0], [100.0, 700.0]] }
    }

    #[test]
    fn perspective_points_are_found_on_corners_edges_and_inside() {
        let quad = keystone();

        assert_eq!(quad_hit(&quad, pos2(903.0, 698.0), 8.0), QuadHit::Corner(2));
        assert_eq!(quad_hit(&quad, pos2(500.0, 104.0), 8.0), QuadHit::Edge(0));
        // On the slanted left edge, halfway down.
        assert_eq!(quad_hit(&quad, pos2(200.0, 400.0), 8.0), QuadHit::Edge(3));
        assert_eq!(quad_hit(&quad, pos2(500.0, 400.0), 8.0), QuadHit::Inside);
        assert_eq!(quad_hit(&quad, pos2(150.0, 150.0), 8.0), QuadHit::Outside);
    }

    #[test]
    fn a_perspective_corner_grabs_from_further_than_an_edge() {
        let quad = keystone();

        // 14 px from the top-right corner: past an edge's 10, inside a corner's 16.
        assert_eq!(quad_hit(&quad, pos2(710.0, 90.0), 10.0), QuadHit::Corner(1));
    }

    #[test]
    fn perspective_mode_starts_from_the_classic_frame() {
        let mut tool = CropTool::new(1000, 800);
        tool.set_ratio(Ratio::Square);

        tool.set_mode(CropMode::Perspective);

        let Commit::Perspective { quad, width, height } = tool.commit() else {
            panic!("expected a perspective commit");
        };
        assert_eq!(quad, Quad::from_frame(&tool.frame));
        assert_eq!((width, height), (800, 800));
    }

    #[test]
    fn a_set_size_overrides_the_measured_one_until_measured_again() {
        let mut tool = CropTool::new(1000, 800);
        tool.set_mode(CropMode::Perspective);
        tool.quad = keystone();
        assert_eq!(tool.size(), (600.0, 632.0));

        tool.set_width(512.0);
        assert_eq!(tool.size(), (512.0, 632.0));
        assert!(tool.has_set_size());

        tool.measure_size();
        assert_eq!(tool.size(), (600.0, 632.0));
    }

    #[test]
    fn moving_the_region_stops_at_the_image_edge() {
        let moved = shift_inside(keystone(), vec2(-500.0, 500.0), IMAGE);

        assert_eq!(moved.corners[3], [0.0, 800.0]);
        assert_eq!(moved.corners[0], [200.0, 200.0]);
    }

    #[test]
    fn the_shield_covers_the_outside_and_leaves_the_frame_clear() {
        let corners =
            [pos2(100.0, 100.0), pos2(300.0, 100.0), pos2(300.0, 200.0), pos2(100.0, 200.0)];

        let mesh = shield(corners, Color32::BLACK);

        let covered = |p: Pos2| {
            mesh.indices.chunks(3).any(|t| {
                let [a, b, c] = [0, 1, 2].map(|k| mesh.vertices[t[k] as usize].pos);
                let sign = |p1: Pos2, p2: Pos2, p3: Pos2| {
                    (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)
                };
                let (d1, d2, d3) = (sign(p, a, b), sign(p, b, c), sign(p, c, a));
                let negative = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
                let positive = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
                !(negative && positive)
            })
        };
        assert!(!covered(pos2(200.0, 150.0)), "inside the frame");
        for outside in [pos2(50.0, 50.0), pos2(200.0, 50.0), pos2(350.0, 150.0), pos2(350.0, 250.0)]
        {
            assert!(covered(outside), "{outside:?}");
        }
    }
}
