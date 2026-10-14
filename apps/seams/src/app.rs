//! Application state and the frame loop.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use egui::{CentralPanel, Context, Frame, Margin, Panel, Ui, ViewportCommand};
use seams_engine::{
    Channel, EngineError, Event, SaveMode, Session, Settings, Stage, WindowSettings,
};
use seams_io::FileFormat;

use crate::actions::{ALIASES, Action, WITH_SHORTCUTS};
use crate::canvas::{CanvasState, NavTool, Scene, ViewMode};
use crate::crop::{Commit, CropMode, CropTool};
use crate::gpu_preview::GpuPreview;
use crate::textures::{Slot, Texture, Textures};
use crate::theme::{
    CHANNEL_STRIP_HEIGHT, INSPECTOR_WIDTH, PANEL_PADDING, STATUS_HEIGHT, TOKENS, TOP_BAR_HEIGHT,
};
use crate::{channels, dialogs, inspector, status, toolbar};

const NOTICE_DURATION: Duration = Duration::from_secs(6);
const ERROR_DURATION: Duration = Duration::from_secs(12);
/// Repaint interval while background work is running, to animate the spinner.
const BUSY_REPAINT: Duration = Duration::from_millis(100);
const APP_NAME: &str = "SEAMS";

/// A short message shown in the status bar.
pub struct Notice {
    pub text: String,
    pub is_error: bool,
    shown_at: Instant,
}

/// The modal dialog on screen, if any.
#[derive(Debug, Clone, PartialEq)]
pub enum Dialog {
    None,
    Shortcuts,
    About,
    /// Asks before replacing the source file.
    ConfirmOverwrite(PathBuf),
}

pub struct SeamsApp {
    pub session: Session,
    pub settings: Settings,
    settings_path: Option<PathBuf>,
    pub canvas: CanvasState,
    pub textures: Textures,
    /// Draws the material maps live on the GPU; `None` keeps them on the CPU.
    gpu: Option<GpuPreview>,
    /// The image shown on the canvas and targeted by export.
    pub channel: Channel,
    pub notice: Option<Notice>,
    pub dialog: Dialog,
    /// The Crop tool, while it is open.
    pub crop: Option<CropTool>,
    actions: Vec<Action>,
    files_hovering: bool,
}

impl SeamsApp {
    pub fn new(
        ctx: &Context,
        gpu: Option<GpuPreview>,
        settings: Settings,
        settings_path: Option<PathBuf>,
    ) -> Result<Self, EngineError> {
        crate::theme::apply(ctx);
        // Ctrl+= and Ctrl+- zoom the canvas, as in Photoshop, not the interface.
        ctx.options_mut(|options| options.zoom_with_keyboard = false);
        // The app is dark whatever the system setting, so the title bar is too.
        ctx.send_viewport_cmd(ViewportCommand::SetTheme(egui::SystemTheme::Dark));
        let repaint = ctx.clone();
        let notify = Arc::new(move || repaint.request_repaint());
        let session = Session::new(settings.params, settings.stage, notify)?;
        Ok(Self {
            session,
            settings,
            settings_path,
            canvas: CanvasState::default(),
            textures: Textures::default(),
            gpu,
            channel: Channel::BaseColor,
            notice: None,
            dialog: Dialog::None,
            crop: None,
            actions: Vec::new(),
            files_hovering: false,
        })
    }

    /// Schedules an action to run before the next frame is drawn.
    pub fn queue(&mut self, action: Action) {
        self.actions.push(action);
    }

    pub fn has_document(&self) -> bool {
        self.session.document().is_some()
    }

    pub fn inform(&mut self, text: impl Into<String>) {
        self.notice = Some(Notice { text: text.into(), is_error: false, shown_at: Instant::now() });
    }

    pub fn report_error(&mut self, text: impl Into<String>) {
        self.notice = Some(Notice { text: text.into(), is_error: true, shown_at: Instant::now() });
    }

    /// How previews are drawn, for the About dialog.
    pub fn preview_description(&self) -> String {
        match &self.gpu {
            Some(gpu) => format!("Live previews on the GPU: {}", gpu.adapter()),
            None => "Live previews on the CPU".to_owned(),
        }
    }

    /// The texture showing `channel`, uploading or drawing it as needed.
    pub fn channel_texture(&mut self, ctx: &Context, channel: Channel) -> Option<Texture> {
        let kind = match channel {
            Channel::BaseColor => {
                // The Delight stage shows the delighted image, not the seamless one.
                let gpu = self.gpu.as_ref().filter(|_| self.session.stage() != Stage::Delight);
                let live = gpu.and_then(GpuPreview::seamless_texture);
                return live
                    .or_else(|| self.textures.sync(ctx, Slot::After, self.session.after_view()));
            }
            Channel::Map(kind) => kind,
        };
        if let Some(live) = self.gpu.as_ref().and_then(|gpu| gpu.map_texture(kind)) {
            return Some(live);
        }
        self.textures.sync(ctx, Slot::Map(kind), self.session.channel_view(channel))
    }

    /// Redraws the live maps for this frame's settings, and has the session
    /// make display copies only of what the GPU is not showing.
    fn sync_gpu(&mut self, ctx: &Context) {
        let Some(gpu) = &mut self.gpu else { return };
        if let Some(fault) = gpu.fault() {
            log::error!("{fault}; previews use the CPU for the rest of the session");
            gpu.release();
            self.gpu = None;
            self.session.set_map_views(true);
            self.report_error("The GPU stopped responding. Previews now use the CPU.");
            return;
        }
        let session = &self.session;
        let live_maps = gpu.update_maps(ctx, session.map_source(), &session.params.material);
        let seamless_source =
            session.seamless_source().filter(|_| session.stage() != Stage::Delight);
        gpu.update_seamless(ctx, seamless_source, &session.params.seamless);
        self.session.set_map_views(!live_maps);
    }

    fn expire_notice(&mut self) {
        let expired = self.notice.as_ref().is_some_and(|notice| {
            let lifetime = if notice.is_error { ERROR_DURATION } else { NOTICE_DURATION };
            notice.shown_at.elapsed() > lifetime
        });
        if expired {
            self.notice = None;
        }
    }

    fn on_event(&mut self, ctx: &Context, event: Event) {
        match event {
            Event::Opened(path) => {
                // A frame drawn for the previous texture means nothing on this one.
                self.crop = None;
                self.settings.remember_file(&path);
                self.channel = Channel::BaseColor;
                self.canvas.fit_to_window();
                let name = file_name(&path);
                ctx.send_viewport_cmd(ViewportCommand::Title(format!("{name} - {APP_NAME}")));
                self.inform(format!("Opened {name}"));
            }
            Event::Saved(path) => self.inform(format!("Saved {}", file_name(&path))),
            Event::Failed(message) => self.report_error(message),
        }
    }

    fn collect_dropped_files(&mut self, ctx: &Context) {
        self.files_hovering = ctx.input(|i| !i.raw.hovered_files.is_empty());
        let dropped = ctx.input(|i| i.raw.dropped_files.first().map(|file| file.path().to_owned()));
        if let Some(path) = dropped {
            self.queue(Action::OpenPath(path));
        }
    }

    fn collect_shortcuts(&mut self, ctx: &Context) {
        let typing = ctx.egui_wants_keyboard_input();
        for command in WITH_SHORTCUTS {
            let Some(shortcut) = &command.shortcut else { continue };
            // Shift + a letter is typing too.
            let modifiers = shortcut.modifiers;
            let is_bare_key = !(modifiers.command || modifiers.ctrl || modifiers.alt);
            if typing && is_bare_key {
                continue;
            }
            if ctx.input_mut(|i| i.consume_shortcut(shortcut)) {
                self.queue(command.action.clone());
            }
        }
        for (shortcut, action) in &ALIASES {
            if ctx.input_mut(|i| i.consume_shortcut(shortcut)) {
                self.queue(action.clone());
            }
        }
    }

    fn remember_window(&mut self, ctx: &Context) {
        let (size, maximized) = ctx.input(|i| {
            let viewport = i.viewport();
            (viewport.inner_rect.map(|rect| rect.size()), viewport.maximized.unwrap_or(false))
        });
        let previous = self.settings.window;
        // Keep the restored size while maximized, so un-maximizing later works.
        let (width, height) = match (maximized, previous, size) {
            (true, Some(window), _) => (window.width, window.height),
            (_, _, Some(size)) => (size.x, size.y),
            _ => return,
        };
        self.settings.window = Some(WindowSettings { width, height, maximized });
    }

    fn run(&mut self, ctx: &Context, action: Action) {
        match action {
            Action::Open => self.open_with_dialog(),
            Action::OpenPath(path) => self.session.open(path),
            Action::Save => self.quick_save(),
            Action::SaveAs => self.export_with_dialog(Channel::BaseColor),
            Action::ExportChannel => self.export_with_dialog(self.channel),
            Action::OverwriteSource => self.overwrite_source(),
            Action::ToggleSaveMode => {
                self.settings.save_mode = match self.settings.save_mode {
                    SaveMode::NewFile => SaveMode::Overwrite,
                    SaveMode::Overwrite => SaveMode::NewFile,
                };
            }
            Action::Exit => ctx.send_viewport_cmd(ViewportCommand::Close),
            Action::Undo => self.step_history(Session::undo, "Undo", "Nothing to undo"),
            Action::Redo => self.step_history(Session::redo, "Redo", "Nothing to redo"),
            Action::ApplyDelight => self.apply_delight(),
            Action::ToggleCrop => self.toggle_crop(),
            Action::PerspectiveCrop => self.perspective_crop(),
            Action::ApplyCrop => self.apply_crop(),
            Action::CancelCrop => self.crop = None,
            Action::SetStage(stage) => self.set_stage(stage),
            Action::SetViewMode(mode) => self.canvas.mode = mode,
            Action::ToggleGuides => {
                self.settings.view.show_guides = !self.settings.view.show_guides
            }
            Action::TogglePanTool => self.canvas.toggle_tool(NavTool::Pan),
            Action::ToggleZoomTool => self.canvas.toggle_tool(NavTool::Zoom),
            Action::ZoomIn => self.canvas.zoom_step(true, ctx.pixels_per_point()),
            Action::ZoomOut => self.canvas.zoom_step(false, ctx.pixels_per_point()),
            Action::FitView => self.canvas.fit_to_window(),
            Action::ActualSize => self.canvas.actual_size(ctx.pixels_per_point()),
            Action::ShowShortcuts => self.dialog = Dialog::Shortcuts,
            Action::ShowAbout => self.dialog = Dialog::About,
        }
    }

    /// Runs an action behind a last-resort guard: a panic becomes an error in
    /// the status bar instead of closing the window with the texture open.
    fn run_guarded(&mut self, ctx: &Context, action: Action) {
        let name = format!("{action:?}");
        let outcome = catch_unwind(AssertUnwindSafe(|| self.run(ctx, action)));
        if outcome.is_err() {
            log::error!("action {name} panicked");
            self.report_error("That action failed unexpectedly. Your texture is still open.");
        }
    }

    fn open_with_dialog(&mut self) {
        let mut dialog = rfd::FileDialog::new()
            .set_title("Open texture")
            .add_filter("Images", &FileFormat::EXTENSIONS);
        if let Some(folder) = self.settings.last_directory.as_ref().filter(|dir| dir.is_dir()) {
            dialog = dialog.set_directory(folder);
        }
        if let Some(path) = dialog.pick_file() {
            self.session.open(path);
        }
    }

    fn set_stage(&mut self, stage: Stage) {
        self.session.set_stage(stage);
        self.settings.stage = stage;
        // Delight is judged against the original, so show both.
        self.canvas.mode =
            if stage == Stage::Delight { ViewMode::SideBySide } else { ViewMode::Result };
        if stage != Stage::Material {
            self.channel = Channel::BaseColor;
        }
    }

    fn step_history(
        &mut self,
        step: fn(&mut Session) -> seams_engine::Result<bool>,
        done: &str,
        nothing: &str,
    ) {
        match step(&mut self.session) {
            Ok(true) => self.inform(done),
            Ok(false) => self.inform(nothing),
            Err(error) => self.report_error(error.to_string()),
        }
    }

    fn apply_delight(&mut self) {
        match self.session.apply_delight() {
            Ok(()) => self.inform("Delight applied to the base color"),
            Err(error) => self.report_error(error.to_string()),
        }
    }

    fn toggle_crop(&mut self) {
        if self.crop.take().is_some() {
            return;
        }
        let Some(base) = self.session.document().map(|document| document.base()) else { return };
        self.crop = Some(CropTool::new(base.width(), base.height()));
    }

    fn perspective_crop(&mut self) {
        if self.crop.is_none() {
            self.toggle_crop();
        }
        if let Some(tool) = &mut self.crop {
            tool.set_mode(CropMode::Perspective);
        }
    }

    fn apply_crop(&mut self) {
        let Some(tool) = self.crop.take() else { return };
        let applied = match tool.commit() {
            Commit::Frame(frame) => self.session.apply_crop(&frame),
            Commit::Perspective { quad, width, height } => {
                self.session.apply_perspective_crop(&quad, width, height)
            }
        };
        match applied {
            Ok(()) => {
                let (width, height) = tool.size();
                self.inform(format!("Cropped to {width:.0} × {height:.0} px"));
            }
            Err(error) => {
                self.report_error(format!("Could not crop: {error}"));
                self.crop = Some(tool);
            }
        }
    }

    /// Closes the Crop tool with the document, and starts it over when undo
    /// or redo changed the size of the image it was framing.
    fn sync_crop(&mut self) {
        let size = self
            .session
            .document()
            .map(|document| (document.base().width(), document.base().height()));
        match (&self.crop, size) {
            (Some(_), None) => self.crop = None,
            (Some(tool), Some((width, height))) if !tool.fits(width, height) => {
                self.crop = Some(CropTool::new(width, height));
            }
            _ => {}
        }
    }

    fn quick_save(&mut self) {
        let Some(source) = self.session.document().map(|document| document.path().to_owned())
        else {
            return self.report_error(EngineError::NoDocument.to_string());
        };
        match self.settings.save_mode {
            SaveMode::NewFile => self.export_with_dialog(Channel::BaseColor),
            SaveMode::Overwrite => self.dialog = Dialog::ConfirmOverwrite(source),
        }
    }

    fn overwrite_source(&mut self) {
        let Some(source) = self.session.document().map(|document| document.path().to_owned())
        else {
            return;
        };
        let format = FileFormat::from_path(&source).unwrap_or(self.settings.export_format);
        self.export(Channel::BaseColor, source, format);
    }

    fn export_with_dialog(&mut self, channel: Channel) {
        if !self.session.can_export(channel) {
            let reason = if self.has_document() {
                EngineError::RenderPending
            } else {
                EngineError::NoDocument
            };
            return self.report_error(reason.to_string());
        }
        let format = self.settings.export_format;
        let source = self.session.document().map(|document| document.path().to_owned());
        let suggested = seams_io::output_path(source.as_deref(), channel.file_suffix(), format);
        let mut dialog = rfd::FileDialog::new()
            .set_title(format!("Export {}", channel.label().to_lowercase()))
            .add_filter(format.label(), &[format.extension()]);
        for other in FileFormat::ALL.into_iter().filter(|other| *other != format) {
            dialog = dialog.add_filter(other.label(), &[other.extension()]);
        }
        if let Some(name) = suggested.file_name() {
            dialog = dialog.set_file_name(name.to_string_lossy());
        }
        if let Some(folder) = suggested.parent().filter(|dir| dir.is_dir()) {
            dialog = dialog.set_directory(folder);
        }
        if let Some(path) = dialog.save_file() {
            let format = FileFormat::from_path(&path).unwrap_or(format);
            self.export(channel, path.with_extension(extension_or(&path, format)), format);
        }
    }

    fn export(&mut self, channel: Channel, path: PathBuf, format: FileFormat) {
        if let Err(error) = self.session.export(channel, path, format) {
            self.report_error(error.to_string());
        }
    }

    fn show_canvas(&mut self, ui: &mut Ui) {
        let ctx = ui.ctx().clone();
        let before = self.textures.sync(&ctx, Slot::Before, self.session.before_view());
        let after = self.channel_texture(&ctx, self.channel);
        let scene = Scene {
            before,
            after,
            can_compare: self.channel == Channel::BaseColor,
            tiles: usize::from(self.settings.view.tiles),
            show_guides: self.settings.view.show_guides,
            checker: self.textures.checker(&ctx),
            loading: self.session.activity().loading,
            files_hovering: self.files_hovering,
        };
        let area = ui.max_rect();
        if self.canvas.show(ui, &scene, self.crop.as_mut()) {
            self.queue(Action::Open);
        }
        if scene.after.is_some() {
            toolbar::show_view_bar(self, &ctx, area);
        }
    }

    fn save_settings(&mut self) {
        self.settings.params = self.session.params;
        self.settings.stage = self.session.stage();
        let Some(path) = &self.settings_path else { return };
        if let Err(error) = self.settings.save(path) {
            log::warn!("could not save settings to {}: {error}", path.display());
        }
    }
}

fn file_name(path: &Path) -> String {
    path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default()
}

/// The extension to save with: the one typed, or the format's when missing.
fn extension_or(path: &Path, format: FileFormat) -> String {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some(extension) if FileFormat::from_extension(extension).is_some() => extension.to_owned(),
        _ => format.extension().to_owned(),
    }
}

fn bar_frame() -> Frame {
    Frame::NONE.fill(TOKENS.surface).inner_margin(Margin::symmetric(PANEL_PADDING, 0))
}

impl eframe::App for SeamsApp {
    fn logic(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        for event in self.session.tick() {
            self.on_event(ctx, event);
        }
        self.collect_dropped_files(ctx);
        self.sync_crop();
        if self.crop.is_some() && self.dialog == Dialog::None {
            match CropTool::key_command(ctx) {
                Some(true) => self.queue(Action::ApplyCrop),
                Some(false) => self.queue(Action::CancelCrop),
                None => {}
            }
        }
        self.collect_shortcuts(ctx);
        self.remember_window(ctx);
        self.expire_notice();
        for action in std::mem::take(&mut self.actions) {
            self.run_guarded(ctx, action);
        }
        if let Some(wait) = self.session.next_wakeup() {
            ctx.request_repaint_after(wait);
        }
        if self.session.activity().is_busy() {
            ctx.request_repaint_after(BUSY_REPAINT);
        }
    }

    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        Panel::top("top-bar")
            .exact_size(TOP_BAR_HEIGHT)
            .frame(bar_frame())
            .show(ui, |ui| toolbar::show_top_bar(self, ui));
        Panel::bottom("status")
            .exact_size(STATUS_HEIGHT)
            .frame(bar_frame())
            .show(ui, |ui| status::show(self, ui));
        if self.has_document() {
            Panel::right("inspector")
                .exact_size(INSPECTOR_WIDTH)
                .frame(Frame::NONE.fill(TOKENS.surface))
                .show(ui, |ui| inspector::show(self, ui));
            Panel::bottom("channels")
                .exact_size(CHANNEL_STRIP_HEIGHT)
                .frame(bar_frame())
                .show(ui, |ui| channels::show(self, ui));
        }
        CentralPanel::default().frame(Frame::NONE).show(ui, |ui| self.show_canvas(ui));
        dialogs::show(self, &ctx);
        // After the panels, so a slider moved this frame is drawn this frame:
        // the maps are rendered before the interface that shows them.
        self.sync_gpu(&ctx);
        if !self.actions.is_empty() {
            ctx.request_repaint();
        }
    }

    fn on_exit(&mut self) {
        self.save_settings();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_or_unknown_extensions_take_the_format_default() {
        assert_eq!(extension_or(Path::new("out"), FileFormat::Tiff), "tiff");
        assert_eq!(extension_or(Path::new("out.backup"), FileFormat::Png), "png");
    }

    #[test]
    fn a_supported_extension_the_user_typed_is_kept() {
        assert_eq!(extension_or(Path::new("out.JPEG"), FileFormat::Png), "JPEG");
        assert_eq!(extension_or(Path::new("out.tif"), FileFormat::Tiff), "tif");
    }
}
