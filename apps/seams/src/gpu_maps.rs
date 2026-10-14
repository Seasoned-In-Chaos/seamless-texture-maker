//! The material maps, drawn live on the GPU. See [`crate::gpu_preview`].

use std::sync::Arc;
use std::time::Instant;

use eframe::egui_wgpu::RenderState;
use egui::{Context, TextureId, vec2};
use seams_core::Image;
use seams_core::pbr::{HeightField, HeightSource, MapKind, MaterialParams};
use seams_gpu::{DeviceHealth, GpuField, MapRenderer, Unsupported};

use crate::gpu_preview::Background;
use crate::textures::Texture;

const MAP_COUNT: usize = MapKind::ALL.len();

fn slot(kind: MapKind) -> Option<usize> {
    MapKind::ALL.iter().position(|candidate| *candidate == kind)
}

/// What a height field is made from: one image, read one way.
#[derive(Clone)]
struct FieldSource {
    image: Arc<Image>,
    height_source: HeightSource,
}

impl FieldSource {
    fn is(&self, other: &FieldSource) -> bool {
        Arc::ptr_eq(&self.image, &other.image) && self.height_source == other.height_source
    }
}

pub struct GpuMaps {
    state: RenderState,
    renderer: MapRenderer,
    /// What the field on the GPU was made from.
    resident: Option<FieldSource>,
    /// A height field being extracted and uploaded.
    upload: Option<(FieldSource, Background<GpuField>)>,
    /// The source the GPU could not take; it is shown from the CPU.
    refused: Option<FieldSource>,
    /// The settings each map texture was last drawn with.
    drawn: [Option<u64>; MAP_COUNT],
    ids: [Option<TextureId>; MAP_COUNT],
}

impl GpuMaps {
    pub fn new(state: &RenderState) -> Result<Self, Unsupported> {
        Ok(Self {
            state: state.clone(),
            renderer: MapRenderer::new(&state.device, &state.queue)?,
            resident: None,
            upload: None,
            refused: None,
            drawn: [None; MAP_COUNT],
            ids: [None; MAP_COUNT],
        })
    }

    pub fn set_health(&mut self, health: DeviceHealth) {
        self.renderer.set_health(health);
    }

    /// Brings the map textures up to date with `source` and `params`.
    ///
    /// Returns true while the GPU is the one showing the maps of `source`,
    /// and false when the caller should show the CPU maps instead.
    pub fn update(
        &mut self,
        ctx: &Context,
        source: Option<&Arc<Image>>,
        params: &MaterialParams,
    ) -> bool {
        let Some(image) = source else {
            self.release();
            return false;
        };
        let wanted = FieldSource { image: Arc::clone(image), height_source: params.height_source };
        self.finish_upload(ctx);
        let is_wanted = |held: &Option<FieldSource>| held.as_ref().is_some_and(|s| s.is(&wanted));
        if !is_wanted(&self.resident) && !is_wanted(&self.refused) && self.upload.is_none() {
            self.start_upload(ctx, wanted.clone());
        }
        if is_wanted(&self.refused) {
            self.release();
            return false;
        }
        self.draw(params);
        // `draw` refuses the source when the GPU turns out unable to draw it.
        !is_wanted(&self.refused)
    }

    fn refuse(&mut self, source: FieldSource, reason: &dyn std::fmt::Display) {
        log::info!("maps of this image are previewed on the CPU: {reason}");
        self.refused = Some(source);
    }

    /// Extracts and uploads the height field of `source` in the background.
    fn start_upload(&mut self, ctx: &Context, source: FieldSource) {
        let image = &source.image;
        let has_alpha = image.channels() == 4;
        let fits = seams_gpu::fits(&self.state.device, image.width(), image.height(), has_alpha);
        if let Err(reason) = fits {
            return self.refuse(source, &reason);
        }
        let (device, queue) = (self.state.device.clone(), self.state.queue.clone());
        let job_source = source.clone();
        let job = move || {
            let started = Instant::now();
            let field = HeightField::from_image(&job_source.image, job_source.height_source);
            let uploaded = GpuField::upload(&device, &queue, &field);
            let (width, height, elapsed) = (field.width, field.height, started.elapsed());
            log::info!(
                "height field {width}x{height} sent to the GPU in {} ms",
                elapsed.as_millis()
            );
            uploaded
        };
        match Background::spawn(ctx, job) {
            Ok(upload) => self.upload = Some((source, upload)),
            Err(error) => self.refuse(source, &error),
        }
    }

    /// Adopts a finished upload: its field becomes the one maps are drawn from.
    fn finish_upload(&mut self, ctx: &Context) {
        let Some(outcome) = self.upload.as_ref().and_then(|(_, upload)| upload.poll()) else {
            return;
        };
        let Some((source, _)) = self.upload.take() else { return };
        match outcome {
            Ok(field) => {
                if self.renderer.set_field(field) {
                    self.register_targets();
                }
                self.resident = Some(source);
                self.drawn = [None; MAP_COUNT];
                // The textures were not there when this frame was laid out.
                ctx.request_repaint();
            }
            Err(reason) => self.refuse(source, &reason),
        }
    }

    /// Points the interface's texture ids at the renderer's current targets.
    fn register_targets(&mut self) {
        let device = &self.state.device;
        let sampler = crate::gpu_preview::sampler;
        let mut egui_renderer = self.state.renderer.write();
        for (kind, id) in MapKind::ALL.iter().zip(&mut self.ids) {
            let Some(view) = self.renderer.target(*kind) else { continue };
            match id {
                Some(id) => {
                    egui_renderer.update_egui_texture_from_wgpu_texture_with_sampler_options(
                        device,
                        view,
                        sampler(),
                        *id,
                    );
                }
                None => {
                    *id = Some(egui_renderer.register_native_texture_with_sampler_options(
                        device,
                        view,
                        sampler(),
                    ))
                }
            }
        }
    }

    /// Redraws every map whose settings changed since it was last drawn.
    fn draw(&mut self, params: &MaterialParams) {
        if self.resident.is_none() {
            return;
        }
        for (kind, drawn) in MapKind::ALL.iter().zip(&mut self.drawn) {
            let key = params.cache_key(*kind);
            if *drawn == Some(key) {
                continue;
            }
            if let Err(reason) = self.renderer.render(*kind, params) {
                log::warn!("maps of this image are previewed on the CPU: {reason}");
                self.refused = self.resident.take();
                return;
            }
            *drawn = Some(key);
        }
    }

    /// The texture showing `kind`, once the GPU has drawn it.
    pub fn texture(&self, kind: MapKind) -> Option<Texture> {
        let index = slot(kind)?;
        self.drawn.get(index).copied().flatten()?;
        let id = self.ids.get(index).copied().flatten()?;
        let (width, height) = self.renderer.size()?;
        Some(Texture { id, full_size: vec2(width as f32, height as f32) })
    }

    /// Frees the field, the map textures and their ids. The next
    /// [`GpuMaps::update`] with a source the GPU can take sets them up again.
    pub fn release(&mut self) {
        let mut egui_renderer = self.state.renderer.write();
        for id in self.ids.iter_mut().filter_map(Option::take) {
            egui_renderer.free_texture(&id);
        }
        self.renderer.clear();
        self.resident = None;
        self.upload = None;
        self.drawn = [None; MAP_COUNT];
    }
}
