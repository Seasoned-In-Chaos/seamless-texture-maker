//! Reusable controls built from the theme tokens.
//!
//! The custom-painted ones (slider, segmented control, primary button) exist
//! because the look is carried by tone and shape, which the stock widgets
//! outline instead.

use std::ops::RangeInclusive;

use egui::style::WidgetVisuals;
use egui::{
    Align, Align2, Color32, DragValue, FontId, Frame, Id, Key, Layout, Margin, Rect, Response,
    RichText, Sense, Shadow, Shape, Stroke, StrokeKind, Ui, WidgetInfo, WidgetType, pos2, vec2,
};

use crate::icons;
use crate::theme::{BODY_SIZE, CONTROL_HEIGHT, MOTION, SMALL_SIZE, TOKENS, medium, mix};

const VALUE_FIELD_WIDTH: f32 = 52.0;
const SLIDER_LABEL_WIDTH: f32 = 124.0;
const PRIMARY_HEIGHT: f32 = 34.0;
const SECTION_GAP: f32 = 24.0;
const SEGMENT_PADDING: f32 = 12.0;
/// The least side padding an option keeps when a row is short of space.
const MIN_SEGMENT_PADDING: f32 = 4.0;
const SEGMENT_GAP: f32 = 2.0;
const SEGMENT_INSET: i8 = 2;
/// Width of the line around a segmented control.
const SEGMENT_OUTLINE: f32 = 1.0;
const RAIL_THICKNESS: f32 = 2.0;
const KNOB_RADIUS: f32 = 5.5;
/// How far the accent halo reaches past the knob of an active slider.
const KNOB_HALO: f32 = 5.0;
/// Space between an icon toggle's edge and its icon.
const ICON_INSET: f32 = 4.0;
const CHEVRON_SIZE: f32 = 14.0;
/// Width of the keyboard-focus ring around a control.
const FOCUS_RING: f32 = 1.5;
const CHECK_SIDE: f32 = 16.0;
const CHECK_GAP: f32 = 8.0;
/// Arrow keys move a focused slider by this fraction of its range.
const KEY_STEP: f32 = 0.01;

/// How a slider's value is presented.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Unit {
    /// A fraction shown as a whole-number percentage.
    Percent,
    /// A plain number with this many decimals.
    Number(usize),
    Degrees,
}

impl Unit {
    fn scale(self) -> f32 {
        if self == Unit::Percent { 100.0 } else { 1.0 }
    }

    fn decimals(self) -> usize {
        match self {
            Unit::Number(decimals) => decimals,
            Unit::Percent | Unit::Degrees => 0,
        }
    }

    fn suffix(self) -> &'static str {
        match self {
            Unit::Percent => "%",
            Unit::Degrees => "°",
            Unit::Number(_) => "",
        }
    }
}

/// Eases between 0 and 1 as `on` changes, so states fade instead of snapping.
fn settle(ui: &Ui, id: Id, on: bool) -> f32 {
    ui.ctx().animate_bool_with_time(id, on, MOTION)
}

/// The face of a selected option: a lighter tone with a soft shadow under it
/// and a faint light edge along the top. `strength` fades it in.
fn lifted(rect: Rect, strength: f32) -> Shape {
    let radius = TOKENS.radius_sm;
    // A hard one-pixel shadow: a blurred one would smear the face's edge.
    let shadow = Shadow {
        offset: [0, 1],
        blur: 0,
        spread: 0,
        color: TOKENS.shadow.gamma_multiply(strength * 0.6),
    };
    let inset = f32::from(radius);
    let top = rect.top() + 0.5;
    let edge = [pos2(rect.left() + inset, top), pos2(rect.right() - inset, top)];
    Shape::Vec(vec![
        Shape::from(shadow.as_shape(rect, radius)),
        Shape::rect_filled(rect, radius, TOKENS.raised_active.gamma_multiply(strength)),
        Shape::line_segment(edge, Stroke::new(1.0, TOKENS.sheen.gamma_multiply(strength))),
    ])
}

/// An accent outline just outside `rect` while the control has keyboard focus.
fn focus_ring(ui: &Ui, response: &Response, rect: Rect, radius: u8) {
    if response.has_focus() {
        let stroke = Stroke::new(FOCUS_RING, TOKENS.accent);
        ui.painter().rect_stroke(rect, radius, stroke, StrokeKind::Outside);
    }
}

/// A quiet group label followed by its controls. Groups are separated by
/// space alone.
pub fn section(ui: &mut Ui, title: &str, add_contents: impl FnOnce(&mut Ui)) {
    ui.label(RichText::new(title).font(medium(SMALL_SIZE)).color(TOKENS.text_dim));
    // Scoped, so spacing a section changes for its own controls does not leak.
    ui.scope(add_contents);
    ui.add_space(SECTION_GAP);
}

/// Secondary explanatory text.
pub fn hint(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text).size(SMALL_SIZE).color(TOKENS.text_dim));
}

/// Paints text left-aligned and vertically centred in `cell`, clipped to it.
fn paint_label(ui: &Ui, cell: Rect, text: &str, color: Color32) {
    let galley =
        ui.painter().layout_no_wrap(text.to_owned(), FontId::proportional(BODY_SIZE), color);
    let origin = pos2(cell.left(), cell.center().y - galley.size().y / 2.0);
    ui.painter().with_clip_rect(cell).galley(origin, galley, color);
}

/// The draggable part of a slider: a thin rail, an accent fill and a knob.
fn track(ui: &mut Ui, label: &str, value: &mut f32, range: &RangeInclusive<f32>) {
    let size = vec2(ui.available_width().max(KNOB_RADIUS * 4.0), CONTROL_HEIGHT);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click_and_drag());
    let (left, right) = (rect.left() + KNOB_RADIUS, rect.right() - KNOB_RADIUS);
    let span = range.end() - range.start();

    if let Some(pointer) = response.interact_pointer_pos() {
        let fraction = ((pointer.x - left) / (right - left).max(1.0)).clamp(0.0, 1.0);
        *value = range.start() + fraction * span;
        response.request_focus();
    }
    if response.has_focus() {
        let steps = ui.input(|i| {
            i32::from(i.key_pressed(Key::ArrowRight)) - i32::from(i.key_pressed(Key::ArrowLeft))
        });
        *value = (*value + steps as f32 * span * KEY_STEP).clamp(*range.start(), *range.end());
    }

    let fraction = if span > 0.0 { ((*value - range.start()) / span).clamp(0.0, 1.0) } else { 0.0 };
    let knob = pos2(left + (right - left) * fraction, rect.center().y);
    let rail = Rect::from_min_max(
        pos2(left, knob.y - RAIL_THICKNESS / 2.0),
        pos2(right, knob.y + RAIL_THICKNESS / 2.0),
    );
    let active = response.hovered() || response.dragged() || response.has_focus();
    let lift = settle(ui, response.id.with("active"), active);
    let painter = ui.painter();
    painter.rect_filled(rail, RAIL_THICKNESS / 2.0, TOKENS.raised_active);
    painter.rect_filled(rail.with_max_x(knob.x), RAIL_THICKNESS / 2.0, TOKENS.accent);
    // An active knob grows a little inside an accent halo; at rest it only
    // casts a small shadow.
    let halo = TOKENS.accent.gamma_multiply(0.22 * lift);
    painter.circle_filled(knob, KNOB_RADIUS + KNOB_HALO * lift, halo);
    let shadow = TOKENS.shadow.gamma_multiply(0.5);
    painter.circle_filled(knob + vec2(0.0, 1.0), KNOB_RADIUS + 0.5, shadow);
    painter.circle_filled(knob, KNOB_RADIUS + lift, TOKENS.text);
    response.widget_info(|| WidgetInfo::slider(true, f64::from(*value), label));
}

/// The editable number beside a slider: plain text until hovered or edited.
fn value_field(ui: &mut Ui, shown: &mut f32, range: RangeInclusive<f32>, unit: Unit) {
    ui.scope(|ui| {
        // The theme's text override would paint the value at full strength;
        // clearing it lets each state's own text colour through.
        ui.visuals_mut().override_text_color = None;
        let widgets = &mut ui.visuals_mut().widgets;
        widgets.inactive.weak_bg_fill = Color32::TRANSPARENT;
        widgets.inactive.bg_fill = Color32::TRANSPARENT;
        widgets.inactive.bg_stroke = Stroke::NONE;
        widgets.inactive.fg_stroke = Stroke::new(1.0, TOKENS.text_dim);
        let field = DragValue::new(shown)
            .speed(f64::from(range.end() - range.start()) / 200.0)
            .range(range)
            .fixed_decimals(unit.decimals())
            .suffix(unit.suffix());
        ui.add_sized(vec2(VALUE_FIELD_WIDTH, CONTROL_HEIGHT - 4.0), field);
    });
}

/// A labelled slider on one row: name, track, then an editable value.
/// Returns true when the value changed.
pub fn slider(
    ui: &mut Ui,
    label: &str,
    value: &mut f32,
    range: RangeInclusive<f32>,
    unit: Unit,
) -> bool {
    let scale = unit.scale();
    let shown_range = range.start() * scale..=range.end() * scale;
    let mut shown = *value * scale;
    let before = shown;
    ui.horizontal(|ui| {
        // Painted, not added as a widget: a fixed cell keeps every track
        // starting at the same x.
        let label_size = vec2(SLIDER_LABEL_WIDTH, CONTROL_HEIGHT);
        let (cell, _) = ui.allocate_exact_size(label_size, Sense::hover());
        paint_label(ui, cell, label, TOKENS.text);
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            value_field(ui, &mut shown, shown_range.clone(), unit);
            track(ui, label, &mut shown, &shown_range);
        });
    });
    if (shown - before).abs() <= f32::EPSILON {
        return false;
    }
    *value = shown / scale;
    true
}

/// A label on the left and a control on the right of one row.
pub fn row<R>(ui: &mut Ui, label: &str, add_control: impl FnOnce(&mut Ui) -> R) -> R {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.with_layout(Layout::right_to_left(Align::Center), add_control).inner
    })
    .inner
}

/// Lucide's chevron, drawn in place of the stock filled triangle.
fn chevron(ui: &Ui, rect: Rect, _visuals: &WidgetVisuals, _is_open: bool) {
    let icon = Rect::from_center_size(rect.center(), vec2(CHEVRON_SIZE, CHEVRON_SIZE));
    icons::paint(ui.painter(), icon, icons::CHEVRON_DOWN, TOKENS.text_dim);
}

/// A dropdown over a fixed list of values. Returns true when the choice changed.
pub fn dropdown<T: Copy + PartialEq>(
    ui: &mut Ui,
    id: &str,
    current: &mut T,
    options: &[T],
    label: impl Fn(T) -> &'static str,
) -> bool {
    let before = *current;
    egui::ComboBox::from_id_salt(id).selected_text(label(*current)).icon(chevron).show_ui(
        ui,
        |ui| {
            for option in options {
                ui.selectable_value(current, *option, label(*option));
            }
        },
    );
    *current != before
}

fn label_width(ui: &Ui, label: &str) -> f32 {
    let font = FontId::proportional(BODY_SIZE);
    ui.painter().layout_no_wrap(label.to_owned(), font, TOKENS.text).size().x
}

/// A standalone toggle: text on a tone that lifts when hovered and stays
/// lifted when selected.
pub fn toggle_button(ui: &mut Ui, label: &str, selected: bool) -> Response {
    segment(ui, label, selected, true, SEGMENT_PADDING)
}

/// One option of a segmented control. With `paints_selection` off, the
/// caller draws the selected face itself (see [`segmented`]).
fn segment(
    ui: &mut Ui,
    label: &str,
    selected: bool,
    paints_selection: bool,
    padding: f32,
) -> Response {
    let size = vec2(label_width(ui, label) + padding * 2.0, CONTROL_HEIGHT - 4.0);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    let color = toggle_face(ui, &response, rect, selected, paints_selection);
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        label,
        FontId::proportional(BODY_SIZE),
        color,
    );
    let enabled = ui.is_enabled();
    response.widget_info(|| WidgetInfo::selected(WidgetType::Button, enabled, selected, label));
    response
}

/// The face of a toggle: a tone on hover, lifted while selected, and the
/// focus ring. Returns the colour its content should be drawn in.
fn toggle_face(
    ui: &Ui,
    response: &Response,
    rect: Rect,
    selected: bool,
    paints_selection: bool,
) -> Color32 {
    let hover = settle(ui, response.id.with("hover"), response.hovered());
    let chosen = settle(ui, response.id.with("chosen"), selected);
    let painter = ui.painter();
    let hover_fill = TOKENS.raised_hover.gamma_multiply(hover * (1.0 - chosen));
    painter.rect_filled(rect, TOKENS.radius_sm, hover_fill);
    if paints_selection && chosen > 0.0 {
        painter.add(lifted(rect, chosen));
    }
    focus_ring(ui, response, rect, TOKENS.radius_sm);
    mix(TOKENS.text_dim, TOKENS.text, hover.max(chosen))
}

/// A square toggle that shows an icon instead of text. `paint_icon` draws
/// the icon in the given colour inside the rect it is given; `label` names
/// the toggle for screen readers.
pub fn icon_toggle(
    ui: &mut Ui,
    label: &str,
    selected: bool,
    paint_icon: impl FnOnce(&egui::Painter, Rect, Color32),
) -> Response {
    let side = CONTROL_HEIGHT - 4.0;
    let (rect, response) = ui.allocate_exact_size(vec2(side, side), Sense::click());
    let color = toggle_face(ui, &response, rect, selected, true);
    paint_icon(ui.painter(), rect.shrink(ICON_INSET), color);
    let enabled = ui.is_enabled();
    response.widget_info(|| WidgetInfo::selected(WidgetType::Button, enabled, selected, label));
    response
}

/// The width [`segmented`] will take for these options.
pub fn segmented_width<T>(ui: &Ui, options: &[(T, &'static str)]) -> f32 {
    let labels: f32 = options.iter().map(|(_, label)| label_width(ui, label)).sum();
    let count = options.len() as f32;
    let gaps = SEGMENT_GAP * (count - 1.0).max(0.0);
    let edges = (f32::from(SEGMENT_INSET) + SEGMENT_OUTLINE) * 2.0;
    labels + count * SEGMENT_PADDING * 2.0 + gaps + edges
}

/// A row of mutually exclusive options in one recessed track. The selected
/// face slides to the option that was picked. Returns the option that was
/// clicked.
pub fn segmented<T: Copy + PartialEq>(
    ui: &mut Ui,
    current: Option<T>,
    options: &[(T, &'static str)],
) -> Option<T> {
    let mut clicked = None;
    let track = Frame::NONE
        .fill(TOKENS.raised)
        .stroke(Stroke::new(SEGMENT_OUTLINE, TOKENS.outline))
        .corner_radius(TOKENS.radius)
        .inner_margin(Margin::same(SEGMENT_INSET));
    track.show(ui, |ui| {
        // Options give up side padding before the row outgrows its space.
        let labels: f32 = options.iter().map(|(_, label)| label_width(ui, label)).sum();
        let count = options.len().max(1) as f32;
        let free = ui.available_width() - labels - SEGMENT_GAP * (count - 1.0);
        let padding = (free / (2.0 * count)).clamp(MIN_SEGMENT_PADDING, SEGMENT_PADDING);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = SEGMENT_GAP;
            // Reserved now so the face is painted under the labels.
            let face = ui.painter().add(Shape::Noop);
            let origin = ui.cursor().left();
            let mut chosen = None;
            for (option, label) in options {
                let selected = current == Some(*option);
                let response = segment(ui, label, selected, false, padding);
                if selected {
                    chosen = Some(response.rect);
                }
                if response.clicked() && !selected {
                    clicked = Some(*option);
                }
            }
            let Some(target) = chosen else { return };
            // Animated relative to the track, so moving the track (a resize)
            // does not leave the face trailing behind.
            let (ctx, id) = (ui.ctx(), ui.id().with("face"));
            let offset =
                ctx.animate_value_with_time(id.with("offset"), target.left() - origin, MOTION);
            let width = ctx.animate_value_with_time(id.with("width"), target.width(), MOTION);
            let rect = Rect::from_min_size(
                pos2(origin + offset, target.top()),
                vec2(width, target.height()),
            );
            ui.painter().set(face, lifted(rect, 1.0));
        });
    });
    clicked
}

/// The one prominent action in a panel: light on dark, as wide as the space
/// it is given. Disabled, it stays readable but does not respond.
pub fn primary_button(ui: &mut Ui, label: &str, enabled: bool) -> Response {
    let size = vec2(ui.available_width(), PRIMARY_HEIGHT);
    let sense = if enabled { Sense::click() } else { Sense::hover() };
    let (rect, response) = ui.allocate_exact_size(size, sense);
    let hover = settle(ui, response.id.with("hover"), enabled && response.hovered());
    let (fill, color) = if enabled {
        (mix(TOKENS.primary, TOKENS.primary_hover, hover), TOKENS.primary_text)
    } else {
        (TOKENS.raised, TOKENS.text_dim)
    };
    // A pressed button sits a pixel lower, like a physical one.
    let pressed = enabled && response.is_pointer_button_down_on();
    let face = if pressed { rect.translate(vec2(0.0, 1.0)) } else { rect };
    ui.painter().rect_filled(face, TOKENS.radius, fill);
    ui.painter().text(face.center(), Align2::CENTER_CENTER, label, medium(BODY_SIZE), color);
    focus_ring(ui, &response, face, TOKENS.radius);
    response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, enabled, label));
    response
}

/// A checkbox whose state reads from its fill: recessed when off, accent
/// with a dark tick when on. Returns true when it was toggled.
pub fn checkbox(ui: &mut Ui, checked: &mut bool, label: &str) -> bool {
    let text_width = label_width(ui, label);
    let size = vec2(CHECK_SIDE + CHECK_GAP + text_width, CONTROL_HEIGHT - 4.0);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    if response.clicked() {
        *checked = !*checked;
    }
    let side = vec2(CHECK_SIDE, CHECK_SIDE);
    let mark = Rect::from_center_size(pos2(rect.left() + CHECK_SIDE / 2.0, rect.center().y), side);
    let on = settle(ui, response.id.with("on"), *checked);
    let hover = settle(ui, response.id.with("hover"), response.hovered());
    let painter = ui.painter();
    let off_fill = mix(TOKENS.raised_hover, TOKENS.raised_active, hover);
    painter.rect_filled(mark, TOKENS.radius_sm, mix(off_fill, TOKENS.accent, on));
    // The outline defines the box while it is off and fades as it fills.
    let outline = Stroke::new(1.0, TOKENS.outline.gamma_multiply(1.0 - on));
    painter.rect_stroke(mark, TOKENS.radius_sm, outline, StrokeKind::Inside);
    focus_ring(ui, &response, mark, TOKENS.radius_sm);
    if on > 0.0 {
        let tick = Stroke::new(1.8, TOKENS.primary_text.gamma_multiply(on));
        let (c, s) = (mark.center(), CHECK_SIDE);
        let points =
            [c + vec2(-0.25 * s, 0.0), c + vec2(-0.06 * s, 0.2 * s), c + vec2(0.27 * s, -0.2 * s)];
        painter.line_segment([points[0], points[1]], tick);
        painter.line_segment([points[1], points[2]], tick);
    }
    let text_at = pos2(mark.right() + CHECK_GAP, rect.center().y);
    painter.text(text_at, Align2::LEFT_CENTER, label, FontId::proportional(BODY_SIZE), TOKENS.text);
    let (enabled, state) = (ui.is_enabled(), *checked);
    response.widget_info(|| WidgetInfo::selected(WidgetType::Checkbox, enabled, state, label));
    response.clicked()
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Context, Event, PointerButton, Pos2, RawInput};
    use std::cell::Cell;

    /// A headless egui context with the app theme, ready to lay out widgets.
    fn context() -> Context {
        let ctx = Context::default();
        crate::theme::apply(&ctx);
        frame(&ctx, Vec::new(), &mut |_| {});
        ctx
    }

    fn frame(ctx: &Context, events: Vec<Event>, add: &mut dyn FnMut(&mut Ui)) {
        let screen = Rect::from_min_size(Pos2::ZERO, vec2(400.0, 200.0));
        let input = RawInput { screen_rect: Some(screen), events, ..RawInput::default() };
        let mut output = ctx.run_ui(input, |ui| add(ui));
        // No renderer here: discard the font-atlas uploads egui queued.
        output.textures_delta.clear();
    }

    /// Draws a 0-to-1 slider track bound to `value`.
    fn draw_track(ui: &mut Ui, value: &Cell<f32>) {
        let mut current = value.get();
        track(ui, "Strength", &mut current, &(0.0..=1.0));
        value.set(current);
    }

    fn button_event(pos: Pos2, pressed: bool) -> Event {
        let (button, modifiers) = (PointerButton::Primary, Default::default());
        Event::PointerButton { pos, button, pressed, modifiers }
    }

    /// Moves to `pos`, presses and releases, drawing the widgets each frame.
    fn click(ctx: &Context, pos: Pos2, add: &mut dyn FnMut(&mut Ui)) {
        frame(ctx, vec![Event::PointerMoved(pos)], add);
        frame(ctx, vec![button_event(pos, true)], add);
        frame(ctx, vec![button_event(pos, false)], add);
        frame(ctx, Vec::new(), add);
    }

    #[test]
    fn clicking_a_checkbox_toggles_it_each_time() {
        let ctx = context();
        let mut checked = false;
        let mut toggles = 0;
        let mut draw = |ui: &mut Ui| toggles += i32::from(checkbox(ui, &mut checked, "Invert"));

        click(&ctx, pos2(8.0, 12.0), &mut draw);
        click(&ctx, pos2(8.0, 12.0), &mut draw);
        click(&ctx, pos2(8.0, 12.0), &mut draw);

        assert_eq!(toggles, 3);
        assert!(checked);
    }

    #[test]
    fn clicking_the_slider_track_sets_the_value_under_the_pointer() {
        let ctx = context();
        let value = Cell::new(0.0f32);
        let mut draw = |ui: &mut Ui| draw_track(ui, &value);

        click(&ctx, pos2(200.0, 14.0), &mut draw);
        assert!((value.get() - 0.5).abs() < 0.02, "middle of the track gave {}", value.get());

        click(&ctx, pos2(399.0, 14.0), &mut draw);
        assert_eq!(value.get(), 1.0);
    }

    #[test]
    fn arrow_keys_step_a_focused_slider_within_its_range() {
        let ctx = context();
        let value = Cell::new(0.0f32);
        let mut draw = |ui: &mut Ui| draw_track(ui, &value);
        click(&ctx, pos2(5.5, 14.0), &mut draw);
        let key = |key, pressed| Event::Key {
            key,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: Default::default(),
        };

        frame(&ctx, vec![key(Key::ArrowRight, true)], &mut draw);
        frame(&ctx, vec![key(Key::ArrowRight, false)], &mut draw);
        assert!((value.get() - KEY_STEP).abs() < 1e-4, "one step right gave {}", value.get());

        frame(&ctx, vec![key(Key::ArrowLeft, true)], &mut draw);
        frame(&ctx, vec![key(Key::ArrowLeft, true)], &mut draw);
        assert_eq!(value.get(), 0.0, "the value must not go below the range");
    }

    #[test]
    fn slider_reports_a_change_only_when_the_value_moved() {
        let ctx = context();
        let mut value = 0.25f32;
        let mut changes = 0;
        let mut draw = |ui: &mut Ui| {
            changes += i32::from(slider(ui, "Fade", &mut value, 0.0..=1.0, Unit::Percent));
        };

        frame(&ctx, Vec::new(), &mut draw);
        frame(&ctx, Vec::new(), &mut draw);

        assert_eq!(changes, 0);
        assert_eq!(value, 0.25);
    }

    #[test]
    fn segmented_control_returns_the_clicked_option_and_predicts_its_width() {
        let ctx = context();
        let options = [(1, "One"), (2, "Two"), (3, "Three")];
        let (clicked, predicted, actual) = (Cell::new(None), Cell::new(0.0), Cell::new(0.0));
        let mut draw = |ui: &mut Ui| {
            predicted.set(segmented_width(ui, &options));
            let before = ui.cursor().left();
            ui.horizontal(|ui| {
                clicked.set(clicked.get().or(segmented(ui, Some(1), &options)));
                actual.set(ui.min_rect().right() - before);
            });
        };

        frame(&ctx, Vec::new(), &mut draw);
        let second = pos2(predicted.get() * 0.5, 14.0);
        click(&ctx, second, &mut draw);

        assert_eq!(clicked.get(), Some(2));
        let (predicted, actual) = (predicted.get(), actual.get());
        assert!((predicted - actual).abs() < 1.0, "predicted {predicted}, laid out {actual}");
    }

    #[test]
    fn a_click_does_not_focus_a_toggle_but_tab_does() {
        let ctx = context();
        let focused = Cell::new(false);
        let mut draw = |ui: &mut Ui| focused.set(toggle_button(ui, "Guides", false).has_focus());

        click(&ctx, pos2(10.0, 12.0), &mut draw);
        assert!(!focused.get(), "a mouse click must not show the focus ring");

        let tab = |pressed| Event::Key {
            key: Key::Tab,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: Default::default(),
        };
        frame(&ctx, vec![tab(true)], &mut draw);
        frame(&ctx, vec![tab(false)], &mut draw);
        frame(&ctx, Vec::new(), &mut draw);
        assert!(focused.get(), "tab must focus the toggle");
    }

    #[test]
    fn a_disabled_primary_button_ignores_clicks() {
        let ctx = context();
        let (mut enabled_clicks, mut disabled_clicks) = (0, 0);
        let mut draw = |ui: &mut Ui| {
            enabled_clicks += i32::from(primary_button(ui, "Export", true).clicked());
            disabled_clicks += i32::from(primary_button(ui, "Apply", false).clicked());
        };

        click(&ctx, pos2(200.0, 17.0), &mut draw);
        click(&ctx, pos2(200.0, 17.0 + PRIMARY_HEIGHT + 8.0), &mut draw);

        assert_eq!(enabled_clicks, 1);
        assert_eq!(disabled_clicks, 0);
    }
}
