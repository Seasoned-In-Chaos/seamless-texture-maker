//! The editing session: one document, its settings and its background renders.
//!
//! The front end calls methods here in response to input and calls
//! [`Session::tick`] every frame. All pixel work runs on worker threads; the
//! session only ever swaps finished results in.
//!
//! Renders are tagged with a generation number. Each edit starts a new
//! generation, and results from older ones are discarded when they arrive, so
//! a slow render can never overwrite a newer one.

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};

use seams_core::Image;
use seams_core::crop::{CropFrame, crop};
use seams_core::delight::DelightParams;
use seams_core::filter::resize_area;
use seams_core::image::MIN_DIMENSION;
use seams_core::pbr::{MapKind, MaterialParams};
use seams_core::perspective::{Quad, perspective_crop};
use seams_io::{BitDepth, FileFormat, Loaded, SaveOptions};

use crate::display::DisplayImage;
use crate::document::Document;
use crate::error::{EngineError, Result};
use crate::model::{Channel, Params, Stage};
use crate::pipeline::{FULL_CACHE, Keyed, MAP_CACHE, MapCache, PREVIEW_CACHE, RenderCache};
use crate::worker::{Mailbox, Policy, Worker};

/// Longest side of the image used for live previews.
const PREVIEW_MAX_SIDE: usize = 768;
/// Longest side of any image handed to the front end for display.
const DISPLAY_MAX_SIDE: usize = 4096;
/// How long input must pause before the full-resolution render starts.
const SETTLE_DELAY: Duration = Duration::from_millis(250);
const JPEG_QUALITY: u8 = 95;

/// Called from worker threads when a result is ready, to wake the front end.
pub type Notify = Arc<dyn Fn() + Send + Sync>;

/// A display image with a revision the front end can use to detect changes.
#[derive(Debug, Clone)]
pub struct View {
    pub image: Arc<DisplayImage>,
    /// Unique per view; changes whenever the pixels do.
    pub revision: u64,
}

/// Something the front end should tell the user about.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    Opened(PathBuf),
    Saved(PathBuf),
    Failed(String),
}

/// Which background work is in flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Activity {
    pub loading: bool,
    pub rendering: bool,
    pub generating_maps: bool,
    pub saving: bool,
}

impl Activity {
    pub fn is_busy(self) -> bool {
        self.loading || self.rendering || self.generating_maps || self.saving
    }
}

#[derive(Clone)]
struct RenderRequest {
    generation: u64,
    source: Arc<Image>,
    params: Params,
    /// Preview only the delight step, as the Delight stage shows it.
    delight_only: bool,
}

struct MapsRequest {
    generation: u64,
    source: Arc<Image>,
    params: MaterialParams,
    /// Also build a display copy of each map.
    views: bool,
}

enum IoRequest {
    Load(PathBuf),
    Save { image: Arc<Image>, path: PathBuf, options: SaveOptions },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Preview,
    Delighted,
    Seamless,
}

struct MapResult {
    kind: MapKind,
    image: Arc<Image>,
    /// Absent when the request asked for no display copies.
    display: Option<DisplayImage>,
}

enum Outcome {
    Loaded { path: PathBuf, loaded: Box<Loaded>, display: DisplayImage },
    LoadFailed(String),
    Rendered { generation: u64, phase: Phase, image: Arc<Image>, display: DisplayImage },
    RenderFailed { generation: u64, message: String },
    Maps { generation: u64, maps: Vec<MapResult> },
    MapsFailed { generation: u64, message: String },
    Saved(PathBuf),
    SaveFailed(String),
}

/// Where workers send what they produce.
#[derive(Clone)]
struct Outbox {
    sender: Sender<Outcome>,
    notify: Notify,
}

impl Outbox {
    fn send(&self, outcome: Outcome) {
        // The receiver only disappears when the session is being dropped.
        if self.sender.send(outcome).is_ok() {
            (self.notify)();
        }
    }
}

/// Runs a job, turning both errors and panics into a message. A panic in a
/// worker must surface as a failed job, never take the application down.
fn guarded<T>(job: impl FnOnce() -> Result<T>) -> std::result::Result<T, String> {
    match catch_unwind(AssertUnwindSafe(job)) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error)) => Err(error.to_string()),
        Err(_) => {
            log::error!("a background job panicked");
            Err("an internal error interrupted processing".to_owned())
        }
    }
}

/// Shrinks the source for previews, keeping the short side usable.
fn preview_source(source: &Image) -> Result<Image> {
    let (width, height) = (source.width(), source.height());
    let (longest, shortest) = (width.max(height) as f32, width.min(height) as f32);
    let fit = PREVIEW_MAX_SIDE as f32 / longest;
    let keep_usable = MIN_DIMENSION as f32 / shortest;
    let scale = fit.max(keep_usable).min(1.0);
    if scale >= 1.0 {
        return Ok(source.clone());
    }
    let scaled = |side: usize| ((side as f32 * scale).round() as usize).max(MIN_DIMENSION);
    Ok(resize_area(source, scaled(width), scaled(height))?)
}

/// Holds the preview worker's state between requests.
struct Previewer {
    cache: RenderCache,
    /// The shrunken copy of the most recent source, keyed by its buffer.
    shrunk: Option<(Arc<Image>, Keyed)>,
}

impl Previewer {
    fn new() -> Self {
        Self { cache: RenderCache::new(PREVIEW_CACHE), shrunk: None }
    }

    fn shrunk_source(&mut self, source: &Arc<Image>) -> Result<Keyed> {
        if let Some((original, small)) = &self.shrunk
            && Arc::ptr_eq(original, source)
        {
            return Ok(small.clone());
        }
        let small = Keyed::source(Arc::new(preview_source(source)?));
        self.shrunk = Some((Arc::clone(source), small.clone()));
        Ok(small)
    }

    fn render(&mut self, request: &RenderRequest) -> Result<Outcome> {
        let small = self.shrunk_source(&request.source)?;
        let delighted = self.cache.delighted(&small, &request.params.delight)?;
        let result = if request.delight_only {
            delighted
        } else {
            self.cache.seamless(&delighted, &request.params.seamless)?
        };
        rendered(request.generation, Phase::Preview, result.image)
    }
}

fn rendered(generation: u64, phase: Phase, image: Arc<Image>) -> Result<Outcome> {
    let display = DisplayImage::from_image(&image, DISPLAY_MAX_SIDE)?;
    Ok(Outcome::Rendered { generation, phase, image, display })
}

fn send_render(outbox: &Outbox, generation: u64, result: std::result::Result<Outcome, String>) {
    outbox.send(result.unwrap_or_else(|message| Outcome::RenderFailed { generation, message }));
}

/// Renders at full resolution: the delighted image first, then the seamless
/// one, stopping in between when a newer request is already waiting.
fn render_full(
    cache: &mut RenderCache,
    request: &RenderRequest,
    mailbox: &Mailbox<RenderRequest>,
    outbox: &Outbox,
) {
    let generation = request.generation;
    let started = Instant::now();
    let source = Keyed::source(Arc::clone(&request.source));
    let delighted = guarded(|| cache.delighted(&source, &request.params.delight));
    let delighted = match delighted {
        Ok(delighted) => delighted,
        Err(message) => return outbox.send(Outcome::RenderFailed { generation, message }),
    };
    let shown = guarded(|| rendered(generation, Phase::Delighted, Arc::clone(&delighted.image)));
    send_render(outbox, generation, shown);
    if mailbox.has_waiting() {
        return;
    }
    let seamless = guarded(|| {
        let result = cache.seamless(&delighted, &request.params.seamless)?;
        rendered(generation, Phase::Seamless, result.image)
    });
    let (width, height) = (request.source.width(), request.source.height());
    log::info!(
        "rendered {width}x{height} with {:?} in {} ms",
        request.params.seamless.method,
        started.elapsed().as_millis()
    );
    send_render(outbox, generation, seamless);
}

fn generate_maps(cache: &mut MapCache, request: &MapsRequest) -> Result<Outcome> {
    let started = Instant::now();
    let mut maps = Vec::new();
    for (kind, image) in cache.generate(&request.source, &request.params)? {
        let display = match request.views {
            true => Some(DisplayImage::from_image(&image, DISPLAY_MAX_SIDE)?),
            false => None,
        };
        maps.push(MapResult { kind, image, display });
    }
    log::info!("generated {} maps in {} ms", maps.len(), started.elapsed().as_millis());
    Ok(Outcome::Maps { generation: request.generation, maps })
}

fn load_file(path: &Path) -> Result<Outcome> {
    let loaded = seams_io::load(path)?;
    loaded.image.validate_dimensions()?;
    let display = DisplayImage::from_image(&loaded.image, DISPLAY_MAX_SIDE)?;
    Ok(Outcome::Loaded { path: path.to_owned(), loaded: Box::new(loaded), display })
}

fn handle_io(request: IoRequest) -> Outcome {
    match request {
        IoRequest::Load(path) => guarded(|| load_file(&path)).unwrap_or_else(Outcome::LoadFailed),
        IoRequest::Save { image, path, options } => {
            match guarded(|| Ok(seams_io::save(&image, &path, &options)?)) {
                Ok(()) => Outcome::Saved(path),
                Err(message) => Outcome::SaveFailed(message),
            }
        }
    }
}

/// The four background threads.
struct Workers {
    preview: Worker<RenderRequest>,
    full: Worker<RenderRequest>,
    maps: Worker<MapsRequest>,
    io: Worker<IoRequest>,
}

impl Workers {
    fn start(outbox: &Outbox) -> std::io::Result<Self> {
        let (to_ui, mut previewer) = (outbox.clone(), Previewer::new());
        let preview = Worker::spawn(
            "seams-preview",
            Policy::LatestOnly,
            move |request: RenderRequest, _| {
                let generation = request.generation;
                send_render(&to_ui, generation, guarded(|| previewer.render(&request)));
            },
        )?;

        let (to_ui, mut cache) = (outbox.clone(), RenderCache::new(FULL_CACHE));
        let full = Worker::spawn(
            "seams-render",
            Policy::LatestOnly,
            move |request: RenderRequest, mailbox| {
                render_full(&mut cache, &request, mailbox, &to_ui);
            },
        )?;

        let (to_ui, mut cache) = (outbox.clone(), MapCache::new(MAP_CACHE));
        let maps =
            Worker::spawn("seams-maps", Policy::LatestOnly, move |request: MapsRequest, _| {
                let generation = request.generation;
                let outcome = guarded(|| generate_maps(&mut cache, &request));
                to_ui.send(
                    outcome.unwrap_or_else(|message| Outcome::MapsFailed { generation, message }),
                );
            })?;

        let to_ui = outbox.clone();
        let io = Worker::spawn("seams-io", Policy::Queue, move |request: IoRequest, _| {
            to_ui.send(handle_io(request));
        })?;
        Ok(Self { preview, full, maps, io })
    }
}

/// A finished render and the generation it belongs to.
struct Rendered {
    generation: u64,
    image: Arc<Image>,
    view: View,
}

struct MapEntry {
    image: Arc<Image>,
    /// Absent while the front end draws the maps itself.
    view: Option<View>,
}

/// One document, its settings and its background work.
pub struct Session {
    /// The settings the next render will use. Change them freely, then call
    /// [`Session::edited`] or [`Session::material_edited`].
    pub params: Params,
    stage: Stage,
    document: Option<Document>,
    before: Option<View>,
    preview: Option<Rendered>,
    delighted: Option<Rendered>,
    /// The base colour and delight settings `delighted` was rendered from.
    delighted_from: Option<(Arc<Image>, DelightParams)>,
    seamless: Option<Rendered>,
    maps: BTreeMap<MapKind, MapEntry>,
    edit_generation: u64,
    rendered_generation: u64,
    maps_generation: u64,
    maps_done_generation: u64,
    next_revision: u64,
    settle: Option<(Instant, RenderRequest)>,
    /// Whether finished maps come with display copies.
    map_views: bool,
    /// When the maps are due to be generated, while that waits for a pause.
    maps_settle: Option<Instant>,
    loading: bool,
    saves_in_flight: usize,
    events: Vec<Event>,
    outcomes: Receiver<Outcome>,
    workers: Workers,
}

impl Session {
    /// Starts the background workers. `notify` is called from those threads
    /// whenever a result is waiting for [`Session::tick`].
    pub fn new(params: Params, stage: Stage, notify: Notify) -> Result<Self> {
        let (sender, outcomes) = channel();
        let workers =
            Workers::start(&Outbox { sender, notify }).map_err(EngineError::WorkerStart)?;
        Ok(Self {
            params,
            stage,
            document: None,
            before: None,
            preview: None,
            delighted: None,
            delighted_from: None,
            seamless: None,
            maps: BTreeMap::new(),
            edit_generation: 0,
            rendered_generation: 0,
            maps_generation: 0,
            maps_done_generation: 0,
            next_revision: 1,
            settle: None,
            map_views: true,
            maps_settle: None,
            loading: false,
            saves_in_flight: 0,
            events: Vec::new(),
            outcomes,
            workers,
        })
    }

    pub fn document(&self) -> Option<&Document> {
        self.document.as_ref()
    }

    pub fn stage(&self) -> Stage {
        self.stage
    }

    pub fn activity(&self) -> Activity {
        let has_document = self.document.is_some();
        Activity {
            loading: self.loading,
            rendering: has_document && self.rendered_generation != self.edit_generation,
            generating_maps: has_document && self.maps_done_generation != self.maps_generation,
            saving: self.saves_in_flight > 0,
        }
    }

    /// Starts loading a file. An [`Event::Opened`] or [`Event::Failed`]
    /// follows from [`Session::tick`].
    pub fn open(&mut self, path: PathBuf) {
        self.loading = true;
        self.workers.io.post(IoRequest::Load(path));
    }

    /// Switches workflow stage and refreshes the preview for it if needed.
    pub fn set_stage(&mut self, stage: Stage) {
        if self.stage == stage {
            return;
        }
        self.stage = stage;
        if self.current(self.wanted()).is_none() {
            self.preview = None;
            if let Some(request) = self.render_request() {
                self.workers.preview.post(request);
            }
        }
    }

    /// Call after changing delight or seamless settings: refreshes the live
    /// preview now and the full-resolution render once input pauses.
    pub fn edited(&mut self) {
        self.edit_generation += 1;
        let Some(request) = self.render_request() else { return };
        self.workers.preview.post(request.clone());
        self.settle = Some((Instant::now() + SETTLE_DELAY, request));
    }

    /// Call after changing material settings: regenerates the maps.
    pub fn material_edited(&mut self) {
        if self.document.is_none() {
            return;
        }
        self.maps_generation += 1;
        if self.map_views {
            self.post_maps();
        } else {
            // Nothing on screen waits for these maps, so they are generated
            // once input pauses instead of on every change.
            self.maps_settle = Some(Instant::now() + SETTLE_DELAY);
        }
    }

    fn post_maps(&mut self) {
        self.maps_settle = None;
        let Some(source) = self.map_source().cloned() else { return };
        self.workers.maps.post(MapsRequest {
            generation: self.maps_generation,
            source,
            params: self.params.material,
            views: self.map_views,
        });
    }

    /// The full-resolution image the seamless step starts from, when it is
    /// known for the current delight settings: the base colour while delight
    /// is off, else the delighted render once it has arrived.
    pub fn seamless_source(&self) -> Option<&Arc<Image>> {
        let document = self.document.as_ref()?;
        if self.params.delight.is_identity() {
            return Some(document.base());
        }
        let (base, settings) = self.delighted_from.as_ref()?;
        let current = Arc::ptr_eq(base, document.base()) && *settings == self.params.delight;
        self.delighted.as_ref().filter(|_| current).map(|delighted| &delighted.image)
    }

    /// The image the material maps are derived from: the latest seamless
    /// result, or the base colour until there is one.
    pub fn map_source(&self) -> Option<&Arc<Image>> {
        let document = self.document.as_ref()?;
        Some(self.seamless.as_ref().map_or(document.base(), |seamless| &seamless.image))
    }

    /// Chooses whether finished maps come with display copies.
    ///
    /// A front end that draws the maps itself (on the GPU) turns them off:
    /// [`Session::channel_view`] then has nothing for a map, and the maps are
    /// generated only for export, once input pauses. Turning them back on
    /// rebuilds the copies.
    pub fn set_map_views(&mut self, enabled: bool) {
        if self.map_views == enabled {
            return;
        }
        self.map_views = enabled;
        if enabled {
            self.material_edited();
        } else {
            self.maps.values_mut().for_each(|entry| entry.view = None);
        }
    }

    /// True when the Delight stage has a result that can be committed.
    pub fn can_apply_delight(&self) -> bool {
        !self.params.delight.is_identity() && self.current(&self.delighted).is_some()
    }

    /// Makes the delighted image the new base colour and resets the delight
    /// controls. Undoable.
    pub fn apply_delight(&mut self) -> Result<()> {
        if self.document.is_none() {
            return Err(EngineError::NoDocument);
        }
        let Some(delighted) = self.current(&self.delighted) else {
            return Err(EngineError::RenderPending);
        };
        let (image, view) = (Arc::clone(&delighted.image), delighted.view.clone());
        if let Some(document) = &mut self.document {
            document.commit(image);
        }
        self.before = Some(view);
        self.params.delight = Default::default();
        self.edited();
        Ok(())
    }

    /// Crops the base colour to `frame` (in base colour pixels) and makes
    /// the result the new base colour. Undoable.
    pub fn apply_crop(&mut self, frame: &CropFrame) -> Result<()> {
        self.replace_base(|base| crop(base, frame))
    }

    /// Straightens the region `quad` of the base colour into a `width` ×
    /// `height` image and makes it the new base colour. Undoable.
    pub fn apply_perspective_crop(
        &mut self,
        quad: &Quad,
        width: usize,
        height: usize,
    ) -> Result<()> {
        self.replace_base(|base| perspective_crop(base, quad, width, height))
    }

    /// Commits `edit` of the base colour, leaving the document untouched
    /// when it fails.
    fn replace_base(
        &mut self,
        edit: impl FnOnce(&Image) -> seams_core::Result<Image>,
    ) -> Result<()> {
        let document = self.document.as_mut().ok_or(EngineError::NoDocument)?;
        let edited = Arc::new(edit(document.base())?);
        let display = DisplayImage::from_image(&edited, DISPLAY_MAX_SIDE)?;
        document.commit(edited);
        self.before = Some(self.new_view(display));
        self.edited();
        Ok(())
    }

    pub fn can_undo(&self) -> bool {
        self.document.as_ref().is_some_and(Document::can_undo)
    }

    pub fn can_redo(&self) -> bool {
        self.document.as_ref().is_some_and(Document::can_redo)
    }

    /// Steps the base colour back; returns false when there is nothing to undo.
    pub fn undo(&mut self) -> Result<bool> {
        self.step_history(Document::undo)
    }

    /// Steps the base colour forward; returns false when there is nothing to redo.
    pub fn redo(&mut self) -> Result<bool> {
        self.step_history(Document::redo)
    }

    fn step_history(&mut self, step: fn(&mut Document) -> bool) -> Result<bool> {
        let Some(document) = &mut self.document else { return Ok(false) };
        if !step(document) {
            return Ok(false);
        }
        let display = DisplayImage::from_image(document.base(), DISPLAY_MAX_SIDE)?;
        self.before = Some(self.new_view(display));
        self.edited();
        Ok(true)
    }

    /// The base colour as loaded or last committed.
    pub fn before_view(&self) -> Option<&View> {
        self.before.as_ref()
    }

    /// The processed base colour for the current stage: the up-to-date
    /// full-resolution result when there is one, else the live preview, else
    /// the previous result so the canvas never blanks while rendering.
    pub fn after_view(&self) -> Option<&View> {
        let wanted = self.wanted();
        self.current(wanted)
            .or_else(|| self.current(&self.preview))
            .or(wanted.as_ref())
            .map(|rendered| &rendered.view)
    }

    /// What to show for `channel`.
    pub fn channel_view(&self, channel: Channel) -> Option<&View> {
        match channel {
            Channel::BaseColor => self.after_view(),
            Channel::Map(kind) => self.maps.get(&kind).and_then(|entry| entry.view.as_ref()),
        }
    }

    /// True when `channel` is up to date and can be exported.
    pub fn can_export(&self, channel: Channel) -> bool {
        self.export_source(channel).is_ok()
    }

    /// Queues a save of `channel`. An [`Event::Saved`] or [`Event::Failed`]
    /// follows from [`Session::tick`].
    pub fn export(&mut self, channel: Channel, path: PathBuf, format: FileFormat) -> Result<()> {
        let (image, bit_depth) = self.export_source(channel)?;
        let dpi = self.document.as_ref().and_then(|document| document.meta().dpi);
        let options =
            SaveOptions { format: Some(format), bit_depth, jpeg_quality: JPEG_QUALITY, dpi };
        self.saves_in_flight += 1;
        self.workers.io.post(IoRequest::Save { image, path, options });
        Ok(())
    }

    fn export_source(&self, channel: Channel) -> Result<(Arc<Image>, BitDepth)> {
        let document = self.document.as_ref().ok_or(EngineError::NoDocument)?;
        let seamless = self.current(&self.seamless).ok_or(EngineError::RenderPending)?;
        match channel {
            Channel::BaseColor => Ok((Arc::clone(&seamless.image), document.meta().bit_depth)),
            Channel::Map(kind) => {
                if self.maps_done_generation != self.maps_generation {
                    return Err(EngineError::RenderPending);
                }
                let entry = self.maps.get(&kind).ok_or(EngineError::MapNotReady(kind.label()))?;
                // Normals need the precision; scalar maps do not benefit.
                let depth =
                    if kind == MapKind::Normal { BitDepth::Sixteen } else { BitDepth::Eight };
                Ok((Arc::clone(&entry.image), depth))
            }
        }
    }

    /// Absorbs finished background work and starts any render that is due.
    /// Call once per frame; returns what the user should be told.
    pub fn tick(&mut self) -> Vec<Event> {
        self.start_due_render(Instant::now());
        while let Ok(outcome) = self.outcomes.try_recv() {
            self.absorb(outcome);
        }
        std::mem::take(&mut self.events)
    }

    /// How long until [`Session::tick`] next has timed work to do.
    pub fn next_wakeup(&self) -> Option<Duration> {
        let render = self.settle.as_ref().map(|(due, _)| *due);
        let due = render.into_iter().chain(self.maps_settle).min()?;
        Some(due.saturating_duration_since(Instant::now()))
    }

    fn start_due_render(&mut self, now: Instant) {
        if self.settle.as_ref().is_some_and(|(due, _)| *due <= now)
            && let Some((_, request)) = self.settle.take()
        {
            self.workers.full.post(request);
        }
        if self.maps_settle.is_some_and(|due| due <= now) {
            self.post_maps();
        }
    }

    fn render_request(&self) -> Option<RenderRequest> {
        let document = self.document.as_ref()?;
        Some(RenderRequest {
            generation: self.edit_generation,
            source: Arc::clone(document.base()),
            params: self.params,
            delight_only: self.stage == Stage::Delight,
        })
    }

    /// The full-resolution result the current stage displays.
    fn wanted(&self) -> &Option<Rendered> {
        if self.stage == Stage::Delight { &self.delighted } else { &self.seamless }
    }

    /// `rendered`, only if it belongs to the current generation.
    fn current<'a>(&self, rendered: &'a Option<Rendered>) -> Option<&'a Rendered> {
        rendered.as_ref().filter(|r| r.generation == self.edit_generation)
    }

    fn new_view(&mut self, display: DisplayImage) -> View {
        let revision = self.next_revision;
        self.next_revision += 1;
        View { image: Arc::new(display), revision }
    }

    fn absorb(&mut self, outcome: Outcome) {
        match outcome {
            Outcome::Loaded { path, loaded, display } => self.on_loaded(path, *loaded, display),
            Outcome::LoadFailed(message) => {
                self.loading = false;
                self.fail(format!("Could not open the texture: {message}"));
            }
            Outcome::Rendered { generation, phase, image, display } => {
                self.on_rendered(generation, phase, image, display);
            }
            Outcome::RenderFailed { generation, message } => {
                if generation == self.edit_generation {
                    self.rendered_generation = generation;
                    self.fail(format!("Processing failed: {message}"));
                }
            }
            Outcome::Maps { generation, maps } => self.on_maps(generation, maps),
            Outcome::MapsFailed { generation, message } => {
                if generation == self.maps_generation {
                    self.maps_done_generation = generation;
                    self.fail(format!("Map generation failed: {message}"));
                }
            }
            Outcome::Saved(path) => self.on_save_finished(Event::Saved(path)),
            Outcome::SaveFailed(message) => {
                self.on_save_finished(Event::Failed(format!("Could not save: {message}")));
            }
        }
    }

    fn fail(&mut self, message: String) {
        log::warn!("{message}");
        self.events.push(Event::Failed(message));
    }

    fn on_save_finished(&mut self, event: Event) {
        self.saves_in_flight = self.saves_in_flight.saturating_sub(1);
        self.events.push(event);
    }

    fn on_loaded(&mut self, path: PathBuf, loaded: Loaded, display: DisplayImage) {
        self.loading = false;
        self.before = Some(self.new_view(display));
        self.preview = None;
        self.delighted = None;
        self.delighted_from = None;
        self.seamless = None;
        self.maps.clear();
        self.document = Some(Document::new(path.clone(), loaded.meta, loaded.image));
        self.events.push(Event::Opened(path));
        self.edited();
    }

    fn on_rendered(
        &mut self,
        generation: u64,
        phase: Phase,
        image: Arc<Image>,
        display: DisplayImage,
    ) {
        if generation != self.edit_generation {
            return;
        }
        let view = self.new_view(display);
        let rendered = Some(Rendered { generation, image, view });
        match phase {
            Phase::Preview => self.preview = rendered,
            Phase::Delighted => {
                // The generation matched, so these are the settings it used.
                let base = self.document.as_ref().map(|document| Arc::clone(document.base()));
                self.delighted_from = base.map(|base| (base, self.params.delight));
                self.delighted = rendered;
            }
            Phase::Seamless => {
                self.seamless = rendered;
                self.rendered_generation = generation;
                self.material_edited();
            }
        }
    }

    fn on_maps(&mut self, generation: u64, maps: Vec<MapResult>) {
        if generation != self.maps_generation {
            return;
        }
        self.maps_done_generation = generation;
        for map in maps {
            let view = map.display.map(|display| self.new_view(display));
            self.maps.insert(map.kind, MapEntry { image: map.image, view });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seams_core::seamless::Method;

    const TIMEOUT: Duration = Duration::from_secs(30);

    fn session() -> Session {
        Session::new(Params::default(), Stage::Seamless, Arc::new(|| {})).unwrap()
    }

    fn write_texture(dir: &Path, name: &str, size: usize) -> PathBuf {
        let image = Image::from_fn(size, size, 3, |x, y, c| {
            (0.2 + 0.6 * x as f32 / size as f32 + 0.1 * ((y / 7 + c) % 2) as f32).min(1.0)
        })
        .unwrap();
        let path = dir.join(name);
        seams_io::save(&image, &path, &SaveOptions::default()).unwrap();
        path
    }

    /// Ticks until `done` holds, collecting events; panics on timeout.
    fn pump(session: &mut Session, mut done: impl FnMut(&Session, &[Event]) -> bool) -> Vec<Event> {
        let deadline = Instant::now() + TIMEOUT;
        let mut events = Vec::new();
        loop {
            events.extend(session.tick());
            if done(session, &events) {
                return events;
            }
            assert!(Instant::now() < deadline, "timed out; events so far: {events:?}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn settled(session: &Session, _: &[Event]) -> bool {
        session.document().is_some() && !session.activity().is_busy()
    }

    fn open(session: &mut Session, path: PathBuf) {
        session.open(path);
        pump(session, settled);
    }

    #[test]
    fn opening_a_texture_renders_everything() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_texture(dir.path(), "rock.png", 128);
        let mut session = session();

        session.open(path.clone());
        let events = pump(&mut session, settled);

        assert_eq!(events, vec![Event::Opened(path)]);
        assert!(session.before_view().is_some());
        assert!(Channel::ALL.iter().all(|c| session.channel_view(*c).is_some()));
        assert!(Channel::ALL.iter().all(|c| session.can_export(*c)));
    }

    #[test]
    fn opening_a_missing_file_reports_failure_and_stays_usable() {
        let mut session = session();

        session.open(PathBuf::from("no/such/texture.png"));
        let events = pump(&mut session, |s, events| !s.activity().loading && !events.is_empty());

        assert!(matches!(events.as_slice(), [Event::Failed(message)] if message.contains("open")));
        assert!(session.document().is_none());
    }

    #[test]
    fn undersized_texture_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_texture(dir.path(), "tiny.png", 16);
        let mut session = session();

        session.open(path);
        let events = pump(&mut session, |s, events| !s.activity().loading && !events.is_empty());

        assert!(matches!(events.as_slice(), [Event::Failed(message)] if message.contains("small")));
    }

    #[test]
    fn editing_settings_replaces_the_result() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = session();
        open(&mut session, write_texture(dir.path(), "a.png", 128));
        let before = session.after_view().unwrap().revision;

        session.params.seamless.method = Method::Mirror;
        session.edited();
        assert!(session.activity().rendering);
        pump(&mut session, settled);

        let after = session.after_view().unwrap();
        assert_ne!(after.revision, before);
        assert_eq!((after.image.width, after.image.height), (256, 256));
    }

    #[test]
    fn delight_stage_shows_the_delighted_image_and_apply_commits_it() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = session();
        open(&mut session, write_texture(dir.path(), "lit.png", 128));
        session.set_stage(Stage::Delight);
        assert!(!session.can_apply_delight(), "nothing to apply while controls are off");

        session.params.delight = DelightParams::AGGRESSIVE;
        session.edited();
        pump(&mut session, settled);
        assert!(session.can_apply_delight());
        session.apply_delight().unwrap();
        pump(&mut session, settled);

        assert!(session.can_undo());
        assert_eq!(session.params.delight, DelightParams::default());
    }

    #[test]
    fn cropping_commits_a_smaller_base_colour_that_undo_restores() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = session();
        open(&mut session, write_texture(dir.path(), "c.png", 128));
        let frame = CropFrame { width: 96.0, height: 64.0, ..CropFrame::whole(128, 128) };

        session.apply_crop(&frame).unwrap();
        pump(&mut session, settled);

        let base = session.document().unwrap().base();
        assert_eq!((base.width(), base.height()), (96, 64));
        let before = session.before_view().unwrap();
        assert_eq!((before.image.width, before.image.height), (96, 64));
        assert!(session.undo().unwrap());
        assert_eq!(session.document().unwrap().base().width(), 128);
    }

    #[test]
    fn a_perspective_crop_commits_the_straightened_size() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = session();
        open(&mut session, write_texture(dir.path(), "p.png", 128));
        let quad = Quad { corners: [[20.0, 10.0], [110.0, 10.0], [125.0, 120.0], [5.0, 120.0]] };

        session.apply_perspective_crop(&quad, 100, 90).unwrap();
        pump(&mut session, settled);

        let base = session.document().unwrap().base();
        assert_eq!((base.width(), base.height()), (100, 90));
        assert!(session.can_undo());
    }

    #[test]
    fn a_crop_below_the_minimum_size_leaves_the_document_alone() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = session();
        open(&mut session, write_texture(dir.path(), "s.png", 128));
        let frame = CropFrame { width: 10.0, ..CropFrame::whole(128, 128) };

        assert!(session.apply_crop(&frame).is_err());
        assert!(!session.can_undo());
    }

    #[test]
    fn undo_and_redo_restore_the_base_colour() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = session();
        open(&mut session, write_texture(dir.path(), "u.png", 128));
        let original = Arc::clone(session.document().unwrap().base());
        session.params.delight = DelightParams::BALANCED;
        session.edited();
        pump(&mut session, settled);
        session.apply_delight().unwrap();
        pump(&mut session, settled);
        let committed = Arc::clone(session.document().unwrap().base());

        assert!(session.undo().unwrap());
        assert!(Arc::ptr_eq(session.document().unwrap().base(), &original));
        assert!(session.redo().unwrap());
        assert!(Arc::ptr_eq(session.document().unwrap().base(), &committed));
        assert!(!session.redo().unwrap());
    }

    #[test]
    fn export_writes_each_channel_at_a_sensible_depth() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = session();
        open(&mut session, write_texture(dir.path(), "e.png", 128));
        let base = dir.path().join("out_seamless.png");
        let normal = dir.path().join("out_normal.png");

        session.export(Channel::BaseColor, base.clone(), FileFormat::Png).unwrap();
        session.export(Channel::Map(MapKind::Normal), normal.clone(), FileFormat::Png).unwrap();
        let events = pump(&mut session, |_, events| events.len() == 2);

        assert_eq!(events, vec![Event::Saved(base.clone()), Event::Saved(normal.clone())]);
        assert_eq!(seams_io::load(&base).unwrap().meta.bit_depth, BitDepth::Eight);
        assert_eq!(seams_io::load(&normal).unwrap().meta.bit_depth, BitDepth::Sixteen);
    }

    #[test]
    fn export_is_refused_without_a_document_or_while_rendering() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = session();
        let target = dir.path().join("x.png");

        let no_document = session.export(Channel::BaseColor, target.clone(), FileFormat::Png);
        open(&mut session, write_texture(dir.path(), "p.png", 128));
        session.params.seamless.overlap.x = 0.4;
        session.edited();
        let while_rendering = session.export(Channel::BaseColor, target, FileFormat::Png);

        assert!(matches!(no_document, Err(EngineError::NoDocument)));
        assert!(matches!(while_rendering, Err(EngineError::RenderPending)));
    }

    #[test]
    fn changing_material_settings_regenerates_maps_only() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = session();
        open(&mut session, write_texture(dir.path(), "m.png", 128));
        let base_revision = session.after_view().unwrap().revision;
        let roughness = Channel::Map(MapKind::Roughness);
        let roughness_revision = session.channel_view(roughness).unwrap().revision;

        session.params.material.roughness.invert = true;
        session.material_edited();
        pump(&mut session, settled);

        assert_eq!(session.after_view().unwrap().revision, base_revision);
        assert_ne!(session.channel_view(roughness).unwrap().revision, roughness_revision);
    }

    #[test]
    fn maps_without_views_are_generated_for_export_only() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = session();
        session.set_map_views(false);
        open(&mut session, write_texture(dir.path(), "g.png", 128));
        let normal = Channel::Map(MapKind::Normal);

        assert!(session.channel_view(normal).is_none());
        assert!(session.channel_view(Channel::BaseColor).is_some());
        assert!(session.can_export(normal));
    }

    #[test]
    fn map_changes_without_views_wait_for_input_to_pause() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = session();
        session.set_map_views(false);
        open(&mut session, write_texture(dir.path(), "w.png", 128));

        session.params.material.roughness.invert = true;
        session.material_edited();
        session.tick();

        assert!(session.activity().generating_maps);
        assert!(session.next_wakeup().is_some_and(|wait| wait <= SETTLE_DELAY));
        assert!(!session.can_export(Channel::Map(MapKind::Roughness)));
        pump(&mut session, settled);
        assert!(session.can_export(Channel::Map(MapKind::Roughness)));
    }

    #[test]
    fn turning_map_views_back_on_rebuilds_them() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = session();
        open(&mut session, write_texture(dir.path(), "v.png", 128));
        let maps = || Channel::ALL.into_iter().filter(|c| *c != Channel::BaseColor);

        session.set_map_views(false);
        assert!(maps().all(|map| session.channel_view(map).is_none()));
        session.set_map_views(true);
        pump(&mut session, settled);

        assert!(maps().all(|map| session.channel_view(map).is_some()));
    }

    #[test]
    fn the_seamless_step_starts_from_the_base_colour_while_delight_is_off() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = session();
        assert!(session.seamless_source().is_none());
        open(&mut session, write_texture(dir.path(), "b.png", 128));

        let base = Arc::clone(session.document().unwrap().base());

        assert!(Arc::ptr_eq(session.seamless_source().unwrap(), &base));
    }

    #[test]
    fn the_seamless_step_starts_from_the_delighted_render_once_it_arrives() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = session();
        open(&mut session, write_texture(dir.path(), "d.png", 128));
        let base = Arc::clone(session.document().unwrap().base());

        session.params.delight = DelightParams::BALANCED;
        session.edited();
        let while_rendering = session.seamless_source().is_some();
        pump(&mut session, settled);
        let delighted = Arc::clone(session.seamless_source().unwrap());
        session.params.delight = DelightParams::AGGRESSIVE;

        assert!(!while_rendering, "the previous render is for other settings");
        assert!(!Arc::ptr_eq(&delighted, &base));
        assert!(session.seamless_source().is_none(), "settings changed again");
    }

    #[test]
    fn maps_are_derived_from_the_seamless_result() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = session();
        assert!(session.map_source().is_none());
        session.params.seamless.method = Method::Mirror;
        open(&mut session, write_texture(dir.path(), "s.png", 128));

        // Mirror tiling doubles the size, which the base colour never has.
        assert_eq!(session.map_source().unwrap().width(), 256);
    }

    #[test]
    fn preview_source_keeps_narrow_images_usable() {
        let wide = Image::filled(4096, 128, 3, 0.5).unwrap();
        let small = Image::filled(200, 100, 3, 0.5).unwrap();

        let shrunk = preview_source(&wide).unwrap();

        assert!(shrunk.height() >= MIN_DIMENSION && shrunk.width() < 4096);
        assert_eq!(preview_source(&small).unwrap(), small);
    }

    #[test]
    fn guarded_turns_a_panic_into_an_error_message() {
        let result: std::result::Result<(), String> = guarded(|| panic!("boom"));

        assert!(result.unwrap_err().contains("internal error"));
    }
}
