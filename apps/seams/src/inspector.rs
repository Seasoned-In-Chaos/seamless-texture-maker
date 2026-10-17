//! The inspector: settings for the current stage, with export pinned below.

use egui::{DragValue, Frame, Margin, Panel, ScrollArea, Ui};
use seams_core::delight::DelightParams;
use seams_core::pbr::{HeightSource, MapKind, MaterialParams, NormalFilter, NormalFormat};
use seams_core::retouch::{RetouchMode, SpotMethod};
use seams_core::seamless::{Method, OverlapParams, SeamlessParams, SplatParams};
use seams_engine::{Channel, SaveMode, Stage};
use seams_io::FileFormat;

use crate::actions::Action;
use crate::app::SeamsApp;
use crate::crop::{CropMode, Ratio};
use crate::retouch::{MAX_RADIUS, MIN_RADIUS};
use crate::theme::{PANEL_PADDING, TOKENS};
use crate::tiling;
use crate::widgets::{self, Unit};

const DELIGHT_PRESETS: [(DelightParams, &str); 3] = [
    (DelightParams::SOFT, "Soft"),
    (DelightParams::BALANCED, "Balanced"),
    (DelightParams::AGGRESSIVE, "Aggressive"),
];

/// A coherent look for a kind of material: patch size, rotation, outline
/// irregularity and edge softness all move together.
#[derive(Debug, Clone, Copy, PartialEq)]
struct MaterialPreset {
    overlap: OverlapParams,
    splat: SplatParams,
}

const fn preset(overlap: f32, overlap_falloff: f32, splat: [f32; 4]) -> MaterialPreset {
    let [falloff, scale, random_rotation, wobble] = splat;
    MaterialPreset {
        overlap: OverlapParams { x: overlap, y: overlap, falloff: overlap_falloff },
        splat: SplatParams { scale, rotation_deg: 0.0, random_rotation, wobble, falloff, seed: 0 },
    }
}

const MATERIAL_PRESETS: [(MaterialPreset, &str); 4] = [
    (preset(0.20, 0.40, [0.55, 0.45, 1.00, 0.45]), "Stone"),
    // Weave direction has to survive: small patches that barely rotate.
    (preset(0.12, 0.20, [0.25, 0.28, 0.08, 0.12]), "Fabric"),
    // Large, heavily deformed, very soft patches with no straight edges.
    (preset(0.28, 0.75, [0.85, 0.65, 1.00, 0.85]), "Organic"),
    // Aligned patches with tight edges keep machined detail sharp.
    (preset(0.18, 0.06, [0.08, 0.55, 0.00, 0.04]), "Hard surface"),
];

impl MaterialPreset {
    fn matches(&self, params: &SeamlessParams) -> bool {
        let seedless = SplatParams { seed: 0, ..params.splat };
        self.overlap == params.overlap && self.splat == seedless
    }

    fn apply_to(&self, params: &mut SeamlessParams) {
        params.overlap = self.overlap;
        params.splat = SplatParams { seed: params.splat.seed, ..self.splat };
    }
}

fn delight_sliders(ui: &mut Ui, params: &mut DelightParams) -> bool {
    let controls: [(&str, &mut f32); 8] = [
        ("Shadow removal", &mut params.shadow_removal),
        ("Highlight reduction", &mut params.highlight_reduction),
        ("AO removal", &mut params.ao_removal),
        ("Flatness", &mut params.flatness),
        ("Contrast recovery", &mut params.contrast_recovery),
        ("Detail", &mut params.detail_preservation),
        ("Color preservation", &mut params.color_preservation),
        ("Edge consistency", &mut params.edge_consistency),
    ];
    let mut changed = false;
    for (label, value) in controls {
        changed |= widgets::slider(ui, label, value, 0.0..=1.0, Unit::Percent);
    }
    changed
}

fn delight(app: &mut SeamsApp, ui: &mut Ui) {
    let mut changed = false;
    let params = &mut app.session.params.delight;
    widgets::section(ui, "Preset", |ui| {
        let current = DELIGHT_PRESETS.iter().map(|(preset, _)| *preset).find(|p| p == params);
        if let Some(preset) = widgets::segmented(ui, current, &DELIGHT_PRESETS) {
            *params = preset;
            changed = true;
        }
    });
    widgets::section(ui, "Adjustments", |ui| {
        changed |= delight_sliders(ui, params);
        if ui.button("Reset").clicked() {
            *params = DelightParams::default();
            changed = true;
        }
    });
    if changed {
        app.session.edited();
    }
    widgets::hint(ui, "Applying makes the delighted image the new base color. Undo with Ctrl+Z.");
    ui.add_space(4.0);
    if widgets::primary_button(ui, "Apply delight", app.session.can_apply_delight()).clicked() {
        app.queue(Action::ApplyDelight);
    }
}

fn overlap_controls(ui: &mut Ui, overlap: &mut OverlapParams) -> bool {
    let mut changed = widgets::slider(ui, "Overlap X", &mut overlap.x, 0.0..=0.5, Unit::Percent);
    changed |= widgets::slider(ui, "Overlap Y", &mut overlap.y, 0.0..=0.5, Unit::Percent);
    changed | widgets::slider(ui, "Edge falloff", &mut overlap.falloff, 0.0..=1.0, Unit::Percent)
}

fn splat_controls(ui: &mut Ui, splat: &mut SplatParams) -> bool {
    let mut changed = widgets::slider(ui, "Patch size", &mut splat.scale, 0.1..=1.5, Unit::Percent);
    changed |= widgets::slider(ui, "Rotation", &mut splat.rotation_deg, 0.0..=360.0, Unit::Degrees);
    changed |= widgets::slider(
        ui,
        "Random rotation",
        &mut splat.random_rotation,
        0.0..=1.0,
        Unit::Percent,
    );
    changed |= widgets::slider(ui, "Wobble", &mut splat.wobble, 0.0..=1.0, Unit::Percent);
    changed |= widgets::slider(ui, "Edge falloff", &mut splat.falloff, 0.0..=1.0, Unit::Percent);
    widgets::row(ui, "Variation", |ui| {
        if ui.button("Shuffle").on_hover_text("Try a different patch layout").clicked() {
            splat.seed = splat.seed.wrapping_add(1);
            changed = true;
        }
        changed |= ui.add(egui::DragValue::new(&mut splat.seed)).changed();
    });
    changed
}

fn method_settings(ui: &mut Ui, params: &mut SeamlessParams) -> bool {
    match params.method {
        Method::OffsetCrossfade => {
            widgets::slider(ui, "Fade width", &mut params.fade_amount.0, 0.0..=1.0, Unit::Percent)
        }
        Method::Mirror => {
            widgets::hint(ui, "Mirror tiling has no settings.");
            false
        }
        Method::None => {
            widgets::hint(ui, "The base color is used as it is.");
            false
        }
        Method::Overlap => overlap_controls(ui, &mut params.overlap),
        Method::Splat => splat_controls(ui, &mut params.splat),
    }
}

fn material_presets(ui: &mut Ui, params: &mut SeamlessParams) -> bool {
    let current =
        MATERIAL_PRESETS.iter().map(|(preset, _)| *preset).find(|preset| preset.matches(params));
    let Some(preset) = widgets::segmented(ui, current, &MATERIAL_PRESETS) else {
        return false;
    };
    preset.apply_to(params);
    true
}

fn seamless(app: &mut SeamsApp, ui: &mut Ui) {
    let params = &mut app.session.params.seamless;
    let mut changed = false;
    widgets::section(ui, "Method", |ui| {
        ui.spacing_mut().combo_width = ui.available_width();
        changed |= widgets::dropdown(ui, "method", &mut params.method, &Method::ALL, Method::label);
        widgets::hint(ui, params.method.description());
    });
    if matches!(params.method, Method::Overlap | Method::Splat) {
        widgets::section(ui, "Material preset", |ui| changed |= material_presets(ui, params));
    }
    widgets::section(ui, "Settings", |ui| changed |= method_settings(ui, params));
    let can_bake = params.method != Method::None && app.session.can_bake_seamless();
    if changed {
        app.session.edited();
    }
    widgets::section(ui, "Retouch", |ui| {
        widgets::hint(
            ui,
            "To paint out seams this method left, make the result the base color, then use \
             the Retouch tool (J).",
        );
        ui.add_space(4.0);
        if ui.add_enabled(can_bake, egui::Button::new("Use result as base")).clicked() {
            app.queue(Action::BakeSeamless);
        }
    });
}

fn normal_controls(ui: &mut Ui, params: &mut MaterialParams) -> bool {
    let normal = &mut params.normal;
    let mut changed = widgets::row(ui, "Filter", |ui| {
        widgets::dropdown(
            ui,
            "normal-filter",
            &mut normal.filter,
            &NormalFilter::ALL,
            NormalFilter::label,
        )
    });
    changed |= widgets::row(ui, "Format", |ui| {
        widgets::dropdown(
            ui,
            "normal-format",
            &mut normal.format,
            &NormalFormat::ALL,
            NormalFormat::label,
        )
    });
    changed |= widgets::slider(ui, "Strength", &mut normal.scale, 0.1..=8.0, Unit::Number(2));
    changed |= widgets::slider(ui, "Minimum Z", &mut normal.min_z, 0.0..=1.0, Unit::Number(2));
    changed |= widgets::checkbox(ui, &mut normal.wrap, "Sample across edges (tiling)");
    ui.horizontal(|ui| {
        changed |= widgets::checkbox(ui, &mut normal.invert_x, "Invert X");
        changed |= widgets::checkbox(ui, &mut normal.invert_y, "Invert Y");
    });
    changed | widgets::checkbox(ui, &mut normal.invert_height, "Invert height")
}

fn roughness_controls(ui: &mut Ui, params: &mut MaterialParams) -> bool {
    let roughness = &mut params.roughness;
    let mut changed =
        widgets::slider(ui, "Intensity", &mut roughness.intensity, 0.0..=1.0, Unit::Percent);
    changed |= widgets::slider(ui, "Contrast", &mut roughness.contrast, 0.0..=1.0, Unit::Percent);
    changed | widgets::checkbox(ui, &mut roughness.invert, "Invert (glossiness)")
}

fn ao_controls(ui: &mut Ui, params: &mut MaterialParams) -> bool {
    let ao = &mut params.ao;
    let mut changed = widgets::slider(ui, "Strength", &mut ao.strength, 0.0..=1.0, Unit::Number(2));
    changed |= widgets::slider(ui, "Mean", &mut ao.mean, 0.0..=1.0, Unit::Number(2));
    changed |= widgets::slider(ui, "Range", &mut ao.range, 0.0..=1.0, Unit::Number(2));
    changed |=
        widgets::slider(ui, "Blur / sharpen", &mut ao.blur_sharp, -32.0..=32.0, Unit::Number(0));
    changed |= widgets::checkbox(ui, &mut ao.invert, "Invert");
    if ui.button("Reset").clicked() {
        *ao = Default::default();
        changed = true;
    }
    changed
}

fn displacement_controls(ui: &mut Ui, params: &mut MaterialParams) -> bool {
    let displacement = &mut params.displacement;
    let mut changed =
        widgets::slider(ui, "Contrast", &mut displacement.contrast, -1.0..=1.0, Unit::Number(2));
    changed |= widgets::slider(
        ui,
        "Blur / sharpen",
        &mut displacement.blur_sharp,
        -32.0..=32.0,
        Unit::Number(0),
    );
    changed |= widgets::checkbox(ui, &mut displacement.invert, "Invert");
    if ui.button("Reset").clicked() {
        *displacement = Default::default();
        changed = true;
    }
    changed
}

fn opacity_controls(ui: &mut Ui, params: &mut MaterialParams) -> bool {
    let opacity = &mut params.opacity;
    let changed =
        widgets::slider(ui, "Threshold", &mut opacity.threshold, 0.0..=1.0, Unit::Percent);
    changed | widgets::slider(ui, "Edge softness", &mut opacity.softness, 0.0..=1.0, Unit::Percent)
}

type MapControls = fn(&mut Ui, &mut MaterialParams) -> bool;

const MAP_SECTIONS: [(MapKind, &str, MapControls); 5] = [
    (MapKind::Normal, "Normal", normal_controls),
    (MapKind::Roughness, "Roughness", roughness_controls),
    (MapKind::AmbientOcclusion, "Ambient occlusion", ao_controls),
    (MapKind::Displacement, "Displacement", displacement_controls),
    (MapKind::Opacity, "Opacity", opacity_controls),
];

fn material(app: &mut SeamsApp, ui: &mut Ui) {
    let params = &mut app.session.params.material;
    let mut changed = false;
    widgets::section(ui, "Height source", |ui| {
        ui.spacing_mut().combo_width = ui.available_width();
        let source = &mut params.height_source;
        changed |= widgets::dropdown(ui, "height", source, &HeightSource::ALL, HeightSource::label);
        widgets::hint(ui, "The channel every map is derived from.");
    });
    // With a map selected only its settings show; base color shows them all.
    for (kind, title, controls) in MAP_SECTIONS {
        if app.channel == Channel::BaseColor || app.channel == Channel::Map(kind) {
            widgets::section(ui, title, |ui| changed |= controls(ui, params));
        }
    }
    if changed {
        app.session.material_edited();
    }
}

const RETOUCH_MODES: [(RetouchMode, &str); 3] = [
    (RetouchMode::Heal, "Heal"),
    (RetouchMode::Clone, "Clone"),
    (RetouchMode::SpotHeal, "Spot heal"),
];
const SPOT_METHODS: [(SpotMethod, &str); 2] =
    [(SpotMethod::ContentAware, "Content-aware"), (SpotMethod::ProximityMatch, "Proximity")];

/// The Retouch tool's options, shown in place of the stage's while it is open.
fn retouch_panel(app: &mut SeamsApp, ui: &mut Ui) {
    let Some(tool) = app.retouch.as_mut() else { return };
    widgets::section(ui, "Tool", |ui| {
        if let Some(mode) = widgets::segmented(ui, Some(tool.mode), &RETOUCH_MODES) {
            tool.mode = mode;
        }
        let what = match tool.mode {
            RetouchMode::Heal => {
                "Copies detail from the source and matches it to the colour and light where it lands."
            }
            RetouchMode::Clone => "Copies the source's pixels exactly.",
            RetouchMode::SpotHeal => {
                "Paint over a blemish: it is filled from what surrounds it. No source needed."
            }
        };
        widgets::hint(ui, what);
        if tool.mode == RetouchMode::SpotHeal {
            if let Some(method) = widgets::segmented(ui, Some(tool.spot), &SPOT_METHODS) {
                tool.spot = method;
            }
            let how = match tool.spot {
                SpotMethod::ContentAware => {
                    "Rebuilds the area patch by patch from its surroundings."
                }
                SpotMethod::ProximityMatch => "Copies the nearby area that matches best. Faster.",
            };
            widgets::hint(ui, how);
        }
    });
    widgets::section(ui, "Brush", |ui| {
        let mut size = tool.brush.radius * 2.0;
        let diameter = DragValue::new(&mut size)
            .range(MIN_RADIUS * 2.0..=MAX_RADIUS * 2.0)
            .speed(1.0)
            .fixed_decimals(0)
            .suffix(" px");
        if widgets::row(ui, "Size", |ui| ui.add(diameter).changed()) {
            tool.brush.radius = (size / 2.0).clamp(MIN_RADIUS, MAX_RADIUS);
        }
        widgets::slider(ui, "Hardness", &mut tool.brush.hardness, 0.0..=1.0, Unit::Percent);
        widgets::slider(ui, "Opacity", &mut tool.brush.opacity, 0.01..=1.0, Unit::Percent);
        widgets::slider(ui, "Flow", &mut tool.brush.flow, 0.01..=1.0, Unit::Percent);
        widgets::slider(ui, "Spacing", &mut tool.brush.spacing, 0.01..=2.0, Unit::Percent);
        widgets::slider(ui, "Smoothing", &mut tool.smoothing, 0.0..=0.95, Unit::Percent);
        widgets::hint(
            ui,
            "Lower hardness gives a softer edge. Opacity caps a stroke; flow sets how fast overlapping dabs build up to it. Smoothing steadies a shaky line.",
        );
    });
    widgets::section(ui, "Options", |ui| {
        if tool.mode != RetouchMode::SpotHeal {
            widgets::checkbox(ui, &mut tool.aligned, "Aligned");
            widgets::checkbox(ui, &mut tool.overlay, "Show source under the brush");
        }
        widgets::checkbox(ui, &mut tool.wrap, "Wrap around edges");
    });
    if tool.mode != RetouchMode::SpotHeal {
        widgets::section(ui, "Source", |ui| {
            let status = if tool.has_source() {
                "Source set."
            } else {
                "No source yet: Alt-click where to copy from."
            };
            widgets::hint(ui, status);
            if tool.has_source() && ui.button("Clear source").clicked() {
                tool.clear_source();
            }
        });
    }
    if tool.is_healing() {
        widgets::hint(ui, "Healing the last stroke…");
    }
    widgets::hint(
        ui,
        "[ and ] change the size, Shift+[ and Shift+] the hardness. 1 to 0 set the opacity, Shift with them the flow. Alt+right-drag on the canvas: sideways for size, up and down for hardness. With Wrap on, strokes continue across the edges, so the texture keeps tiling; paint over tile borders in the Tiled view.",
    );
    ui.add_space(12.0);
    if ui.button("Done").clicked() {
        app.queue(Action::ToggleRetouch);
    }
}

/// What the tiling check found, above the stage's settings while it is on.
fn tiling_section(app: &mut SeamsApp, ui: &mut Ui) {
    let source = app.session.map_source();
    widgets::section(ui, "Tiling check", |ui| match app.tiling.report_for(source) {
        Some(report) => {
            ui.label(tiling::seam_summary(report));
            ui.label(tiling::repeat_summary(report));
            widgets::hint(
                ui,
                "Red marks seams, orange marks repeated areas. Tiled view shows them where tiles meet.",
            );
        }
        None if app.tiling.is_running() => widgets::hint(ui, "Checking…"),
        None => widgets::hint(ui, "Waiting for the result…"),
    });
}

/// An editable size in whole pixels, within the sizes SEAMS takes.
fn pixels(value: &mut f32) -> DragValue<'_> {
    DragValue::new(value).range(64.0..=8192.0).speed(1.0).fixed_decimals(0).suffix(" px")
}

/// The Crop tool's options, shown in place of the stage's while it is open.
fn crop_panel(app: &mut SeamsApp, ui: &mut Ui) {
    let Some(tool) = app.crop.as_mut() else { return };
    widgets::section(ui, "Mode", |ui| {
        if let Some(mode) = widgets::segmented(ui, Some(tool.mode()), &CropMode::ALL) {
            tool.set_mode(mode);
        }
    });
    let perspective = tool.mode() == CropMode::Perspective;
    if !perspective {
        widgets::section(ui, "Aspect ratio", |ui| {
            let mut ratio = tool.ratio();
            if widgets::dropdown(ui, "crop-ratio", &mut ratio, &Ratio::ALL, Ratio::label) {
                tool.set_ratio(ratio);
            }
        });
    }
    let title = if perspective { "Result size" } else { "Frame" };
    widgets::section(ui, title, |ui| {
        let (mut width, mut height) = tool.size();
        if widgets::row(ui, "Width", |ui| ui.add(pixels(&mut width)).changed()) {
            tool.set_width(width);
        }
        if widgets::row(ui, "Height", |ui| ui.add(pixels(&mut height)).changed()) {
            tool.set_height(height);
        }
        if perspective {
            let measured = !tool.has_set_size();
            let label =
                if measured { "Measured from the corners" } else { "Measure from the corners" };
            if ui.add_enabled(!measured, egui::Button::new(label)).clicked() {
                tool.measure_size();
            }
            return;
        }
        let mut degrees = tool.degrees();
        let angle = DragValue::new(&mut degrees)
            .range(-180.0..=180.0)
            .speed(0.1)
            .fixed_decimals(1)
            .suffix("°");
        if widgets::row(ui, "Angle", |ui| ui.add(angle).changed()) {
            tool.set_degrees(degrees);
        }
    });
    let hint = if perspective {
        "Drag each corner onto a corner of the surface, such as a wall or floor shot at an \
         angle. Drag an edge to move both its corners, or inside to move the whole shape. \
         Applying straightens it into a rectangle."
    } else {
        "Drag inside the frame to move it, an edge or corner to resize it, and outside it to \
         turn it. Shift keeps proportions or snaps the angle; Alt works from the centre."
    };
    widgets::hint(ui, hint);
    ui.add_space(12.0);
    if widgets::primary_button(ui, "Apply crop", true).clicked() {
        app.queue(Action::ApplyCrop);
    }
    ui.add_space(4.0);
    if ui.button("Cancel").clicked() {
        app.queue(Action::CancelCrop);
    }
}

/// The file format and the export button, pinned to the bottom. For the
/// base color, the export can instead replace the file that was opened; this
/// is the same setting as the File menu's quick save option.
fn export_footer(app: &mut SeamsApp, ui: &mut Ui) {
    let base_color = app.channel == Channel::BaseColor;
    let mut replace = base_color && app.settings.save_mode == SaveMode::Overwrite;
    if base_color && widgets::checkbox(ui, &mut replace, "Replace original file") {
        app.settings.save_mode = if replace { SaveMode::Overwrite } else { SaveMode::NewFile };
    }
    let source = app.session.document().map(|document| document.file_name());
    match source.filter(|_| replace) {
        Some(name) => widgets::hint(ui, &format!("Saves over {name} in its own format.")),
        None => {
            widgets::row(ui, "Format", |ui| {
                let format = &mut app.settings.export_format;
                widgets::dropdown(ui, "export-format", format, &FileFormat::ALL, FileFormat::label);
            });
        }
    }
    ui.add_space(2.0);
    let (label, action) = if replace {
        ("Replace original".to_owned(), Action::Save)
    } else {
        (format!("Export {}", app.channel.label().to_lowercase()), Action::ExportChannel)
    };
    if widgets::primary_button(ui, &label, app.session.can_export(app.channel)).clicked() {
        app.queue(action);
    }
}

pub fn show(app: &mut SeamsApp, ui: &mut Ui) {
    let padded = Frame::NONE.fill(TOKENS.surface).inner_margin(Margin::same(PANEL_PADDING));
    Panel::bottom("inspector-export").frame(padded).show(ui, |ui| export_footer(app, ui));
    padded.show(ui, |ui| {
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            if app.tiling.on {
                tiling_section(app, ui);
            }
            if app.crop.is_some() {
                return crop_panel(app, ui);
            }
            if app.retouch.is_some() {
                return retouch_panel(app, ui);
            }
            match app.session.stage() {
                Stage::Delight => delight(app, ui),
                Stage::Seamless => seamless(app, ui),
                Stage::Material => material(app, ui),
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applying_a_preset_makes_it_the_matching_one_and_keeps_the_seed() {
        let mut params = SeamlessParams::default();
        params.splat.seed = 42;
        let (fabric, _) = MATERIAL_PRESETS[1];

        fabric.apply_to(&mut params);

        assert!(fabric.matches(&params));
        assert_eq!(params.splat.seed, 42);
        assert!(!MATERIAL_PRESETS[0].0.matches(&params));
    }

    #[test]
    fn presets_are_distinct_and_within_valid_ranges() {
        for (i, (preset, label)) in MATERIAL_PRESETS.iter().enumerate() {
            let params = SeamlessParams {
                overlap: preset.overlap,
                splat: preset.splat,
                ..Default::default()
            };
            assert_eq!(params.sanitized(), params, "{label} is out of range");
            assert!(MATERIAL_PRESETS.iter().skip(i + 1).all(|(other, _)| other != preset));
        }
    }
}
