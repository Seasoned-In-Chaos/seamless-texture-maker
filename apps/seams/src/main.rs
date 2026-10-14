//! SEAMS: seamless texture and PBR map studio.
//!
//! This binary is a thin egui shell. Everything it does goes through
//! `seams-engine`; nothing here touches pixels directly.

// Release builds are a GUI application: no console window on Windows.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod actions;
mod app;
mod canvas;
mod channels;
mod crop;
mod dialogs;
mod gpu_maps;
mod gpu_preview;
mod gpu_seamless;
mod icons;
mod inspector;
mod logging;
mod menu;
mod status;
mod textures;
mod theme;
mod toolbar;
mod widgets;

use egui::{IconData, ViewportBuilder};
use seams_engine::Settings;

use crate::app::SeamsApp;

const DEFAULT_WINDOW_SIZE: [f32; 2] = [1440.0, 900.0];
const MIN_WINDOW_SIZE: [f32; 2] = [980.0, 640.0];
const ICON_PNG: &[u8] = include_bytes!("../../../resources/icon.png");

/// The window icon, or none if the embedded image cannot be decoded.
fn window_icon() -> Option<IconData> {
    let decoded = image::load_from_memory_with_format(ICON_PNG, image::ImageFormat::Png).ok()?;
    let small = decoded.thumbnail(256, 256).into_rgba8();
    let (width, height) = small.dimensions();
    Some(IconData { rgba: small.into_raw(), width, height })
}

fn viewport(settings: &Settings) -> ViewportBuilder {
    let size = settings.window.map_or(DEFAULT_WINDOW_SIZE, |window| {
        [window.width.max(MIN_WINDOW_SIZE[0]), window.height.max(MIN_WINDOW_SIZE[1])]
    });
    let mut builder = ViewportBuilder::default()
        .with_title("SEAMS")
        .with_app_id("SeasonedInChaos.SeamlessTextureMaker")
        .with_inner_size(size)
        .with_min_inner_size(MIN_WINDOW_SIZE)
        .with_maximized(settings.window.is_some_and(|window| window.maximized))
        .with_drag_and_drop(true);
    if let Some(icon) = window_icon() {
        builder = builder.with_icon(icon);
    }
    builder
}

fn main() -> eframe::Result {
    logging::init();
    log::info!("starting SEAMS {}", env!("CARGO_PKG_VERSION"));
    let settings_path = Settings::default_path();
    let settings = settings_path.as_deref().map(Settings::load).unwrap_or_default();
    let options = eframe::NativeOptions { viewport: viewport(&settings), ..Default::default() };
    eframe::run_native(
        "SEAMS",
        options,
        Box::new(move |creation| {
            let gpu = creation.wgpu_render_state.as_ref().and_then(gpu_preview::GpuPreview::new);
            let mut app = SeamsApp::new(&creation.egui_ctx, gpu, settings, settings_path)?;
            // A path on the command line ("Open with", or a file dropped on the exe).
            if let Some(path) = std::env::args_os().nth(1) {
                app.queue(actions::Action::OpenPath(path.into()));
            }
            Ok(Box::new(app))
        }),
    )
}
