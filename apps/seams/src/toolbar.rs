//! The top bar (menus and the workflow stage) and the floating view bar
//! that sits over the bottom of the canvas.

use egui::{
    Align, Align2, Area, Context, Frame, Id, Layout, Margin, Rect, Response, RichText, Stroke, Ui,
    UiBuilder, pos2, vec2,
};
use seams_engine::Stage;

use crate::actions::Action;
use crate::app::SeamsApp;
use crate::canvas::{NavTool, ViewMode};
use crate::theme::{CONTROL_HEIGHT, INSPECTOR_WIDTH, SMALL_SIZE, TOKENS};
use crate::{icons, menu, widgets};

const STAGES: [(Stage, &str); 3] =
    [(Stage::Delight, "Delight"), (Stage::Seamless, "Seamless"), (Stage::Material, "Material")];
const TILE_COUNTS: [u8; 3] = [2, 3, 4];
const TILE_DROPDOWN_WIDTH: f32 = 68.0;
const MENU_WIDTH: f32 = 220.0;
/// Least space between the menus and the stage switcher.
const STAGE_MENU_GAP: f32 = 16.0;
/// Gap between the floating view bar and the bottom of the canvas.
const VIEW_BAR_OFFSET: f32 = 16.0;

fn tile_label(count: u8) -> &'static str {
    match count {
        3 => "3 × 3",
        4 => "4 × 4",
        _ => "2 × 2",
    }
}

fn centred_row(rect: Rect) -> UiBuilder {
    UiBuilder::new().max_rect(rect).layout(Layout::left_to_right(Align::Center))
}

/// Menus on the left, the workflow stage in the middle.
pub fn show_top_bar(app: &mut SeamsApp, ui: &mut Ui) {
    let bar = ui.max_rect();
    let menus = Rect::from_min_size(bar.min, vec2(MENU_WIDTH.min(bar.width()), bar.height()));
    ui.scope_builder(centred_row(menus), |ui| menu::show(app, ui));

    let width = widgets::segmented_width(ui, &STAGES);
    let has_document = app.has_document();
    // Centred over the canvas, not the window, so the stage lines up with
    // the image and the view bar below it. The inspector only shows with a
    // document. The bar's side padding is symmetric, so it cancels out.
    let inspector = if has_document { INSPECTOR_WIDTH } else { 0.0 };
    let centre_x = (bar.left() + bar.right() - inspector) / 2.0;
    // Never under the menus on a narrow window.
    let centre_x = centre_x.max(menus.right() + STAGE_MENU_GAP + width / 2.0);
    let middle = Rect::from_center_size(pos2(centre_x, bar.center().y), vec2(width, bar.height()));
    ui.scope_builder(centred_row(middle), |ui| {
        ui.add_enabled_ui(has_document, |ui| {
            if let Some(stage) = widgets::segmented(ui, Some(app.session.stage()), &STAGES) {
                app.queue(Action::SetStage(stage));
            }
        });
    });
}

fn view_modes(app: &mut SeamsApp, ui: &mut Ui) {
    for (mode, label) in ViewMode::ALL {
        let selected = app.canvas.mode == mode;
        if widgets::toggle_button(ui, label, selected).clicked() && !selected {
            app.queue(Action::SetViewMode(mode));
        }
    }
    if app.canvas.mode != ViewMode::Tiled {
        return;
    }
    ui.spacing_mut().combo_width = TILE_DROPDOWN_WIDTH;
    let tiles = &mut app.settings.view.tiles;
    widgets::dropdown(ui, "tile-count", tiles, &TILE_COUNTS, tile_label);
    let guides = app.settings.view.show_guides;
    if icon_toggle(ui, "Tile guides", guides, icons::GRID)
        .on_hover_text("Tile guides (Ctrl+G): dashed lines where the tiles meet")
        .clicked()
    {
        app.queue(Action::ToggleGuides);
    }
}

/// A toggle showing one of the Lucide icons.
fn icon_toggle(ui: &mut Ui, label: &str, selected: bool, icon: char) -> Response {
    widgets::icon_toggle(ui, label, selected, |painter, rect, color| {
        icons::paint(painter, rect, icon, color);
    })
}

fn zoom_controls(app: &mut SeamsApp, ui: &mut Ui) {
    if icon_toggle(ui, "Crop", app.crop.is_some(), icons::CROP)
        .on_hover_text("Crop (C). Enter applies, Esc cancels.")
        .clicked()
    {
        app.queue(Action::ToggleCrop);
    }
    if icon_toggle(ui, "Retouch", app.retouch.is_some(), icons::RETOUCH)
        .on_hover_text("Retouch (J): clone and heal. Alt-click a source, then paint.")
        .clicked()
    {
        app.queue(Action::ToggleRetouch);
    }
    if icon_toggle(ui, "Check tiling", app.tiling.on, icons::TILING_CHECK)
        .on_hover_text("Check tiling (Ctrl+Shift+T): marks seams and repeated areas.")
        .clicked()
    {
        app.queue(Action::ToggleTilingCheck);
    }
    let tool = app.canvas.tool;
    if icon_toggle(ui, "Pan tool", tool == NavTool::Pan, icons::HAND)
        .on_hover_text("Pan tool (H). Hold Space to pan with any tool.")
        .clicked()
    {
        app.queue(Action::TogglePanTool);
    }
    if icon_toggle(ui, "Zoom tool", tool == NavTool::Zoom, icons::ZOOM_IN)
        .on_hover_text(
            "Zoom tool (Z): click to zoom in, Alt-click to zoom out, drag right or left.",
        )
        .clicked()
    {
        app.queue(Action::ToggleZoomTool);
    }
    if icon_toggle(ui, "Fit to window", app.canvas.is_fitted(), icons::SCAN)
        .on_hover_text("Fit to window (Ctrl+0). Double-click the canvas to do the same.")
        .clicked()
    {
        app.queue(Action::FitView);
    }
    ui.add_space(4.0);
    if icon_toggle(ui, "Zoom out", false, icons::MINUS).on_hover_text("Zoom out (Ctrl+-)").clicked()
    {
        app.queue(Action::ZoomOut);
    }
    let percent = app.canvas.zoom_percent(ui.ctx().pixels_per_point());
    let label = RichText::new(format!("{percent:.0}%")).monospace().size(SMALL_SIZE);
    let zoom =
        egui::Button::new(label.color(TOKENS.text_dim)).frame(false).min_size(vec2(44.0, 0.0));
    if ui.add(zoom).on_hover_text("Click for actual size (Ctrl+1)").clicked() {
        app.queue(Action::ActualSize);
    }
    if icon_toggle(ui, "Zoom in", false, icons::PLUS).on_hover_text("Zoom in (Ctrl+=)").clicked() {
        app.queue(Action::ZoomIn);
    }
}

/// View mode and zoom, floating over the bottom of the canvas.
pub fn show_view_bar(app: &mut SeamsApp, ctx: &Context, canvas: Rect) {
    let anchor = canvas.center_bottom() - vec2(0.0, VIEW_BAR_OFFSET);
    let frame = Frame::NONE
        .fill(TOKENS.popup)
        .stroke(Stroke::new(1.0, TOKENS.outline))
        .corner_radius(TOKENS.radius_lg)
        .inner_margin(Margin::same(4))
        .shadow(egui::Shadow { offset: [0, 4], blur: 12, spread: 0, color: TOKENS.shadow });
    Area::new(Id::new("view-bar")).pivot(Align2::CENTER_BOTTOM).fixed_pos(anchor).show(ctx, |ui| {
        let bar = frame.show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                ui.set_height(CONTROL_HEIGHT - 4.0);
                // Cropping frames the base color alone, so view modes step aside.
                if app.crop.is_none() {
                    view_modes(app, ui);
                    ui.add_space(6.0);
                    ui.separator();
                    ui.add_space(6.0);
                }
                zoom_controls(app, ui);
            });
        });
        // A light edge along the top, just inside the border, lifts the bar
        // off the image behind it.
        let rect = bar.response.rect;
        let span = rect.x_range().shrink(f32::from(TOKENS.radius_lg));
        ui.painter().hline(span, rect.top() + 1.5, Stroke::new(1.0, TOKENS.sheen));
    });
}
