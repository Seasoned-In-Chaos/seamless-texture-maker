//! The seamless base colour, drawn live on the GPU. See [`crate::gpu_preview`].

use std::sync::Arc;
use std::time::Instant;

use eframe::egui_wgpu::RenderState;
use egui::{Context, TextureId, vec2};
use seams_core::Image;
use seams_core::seamless::SeamlessParams;
use seams_gpu::{DeviceHealth, GpuImage, SeamlessRenderer, Unsupported, seamless};

use crate::gpu_preview::Background;
use crate::textures::Texture;

fn is(held: &Option<Arc<Image>>, wanted: &Arc<Image>) -> bool {
    held.as_ref().is_some_and(|image| Arc::ptr_eq(image, wanted))
}

pub struct GpuSeamless {
    state: RenderState,
    renderer: SeamlessRenderer,
    /// The image the source texture on the GPU was made from.
    resident: Option<Arc<Image>>,
    /// An image being converted and uploaded.
    upload: Option<(Arc<Image>, Background<GpuImage>)>,
    /// The image the GPU could not take; it is shown from the CPU.
    refused: Option<Arc<Image>>,
    /// The settings the result texture was last drawn with.
    drawn: Option<u64>,
    id: Option<TextureId>,
}

impl GpuSeamless {
    pub fn new(state: &RenderState) -> Result<Self, Unsupported> {
        Ok(Self {
            state: state.clone(),
            renderer: SeamlessRenderer::new(&state.device, &state.queue)?,
            resident: None,
            upload: None,
            refused: None,
            drawn: None,
            id: None,
        })
    }

    pub fn set_health(&mut self, health: DeviceHealth) {
        self.renderer.set_health(health);
    }

    /// Brings the result texture up to date with `source` and `params`.
    ///
    /// Returns true while the GPU is the one showing the seamless result of
    /// `source`, and false when the caller should show the CPU result: there
    /// is no source, the method is not one the GPU draws, or the image does
    /// not fit.
    pub fn update(
        &mut self,
        ctx: &Context,
        source: Option<&Arc<Image>>,
        params: &SeamlessParams,
    ) -> bool {
        let device = &self.state.device;
        let fits = |image: &&Arc<Image>| {
            seamless::fits(device, image.width(), image.height(), params.method).is_ok()
        };
        let Some(image) = source.filter(fits) else {
            self.release();
            return false;
        };
        self.finish_upload(ctx);
        if !is(&self.resident, image) && !is(&self.refused, image) && self.upload.is_none() {
            self.start_upload(ctx, Arc::clone(image));
        }
        if is(&self.refused, image) {
            self.release();
            return false;
        }
        self.draw(params);
        // `draw` refuses the image when the GPU turns out unable to draw it.
        !is(&self.refused, image)
    }

    fn refuse(&mut self, image: Arc<Image>, reason: &dyn std::fmt::Display) {
        log::info!("this image is made seamless on the CPU only: {reason}");
        self.refused = Some(image);
    }

    /// Converts and uploads `image` in the background.
    fn start_upload(&mut self, ctx: &Context, image: Arc<Image>) {
        let (device, queue) = (self.state.device.clone(), self.state.queue.clone());
        let job_image = Arc::clone(&image);
        let job = move || {
            let started = Instant::now();
            let uploaded = GpuImage::upload(&device, &queue, &job_image);
            let (width, height, elapsed) =
                (job_image.width(), job_image.height(), started.elapsed());
            log::info!("image {width}x{height} sent to the GPU in {} ms", elapsed.as_millis());
            uploaded
        };
        match Background::spawn(ctx, job) {
            Ok(upload) => self.upload = Some((image, upload)),
            Err(error) => self.refuse(image, &error),
        }
    }

    /// Adopts a finished upload: it becomes the image the result is drawn from.
    fn finish_upload(&mut self, ctx: &Context) {
        let Some(outcome) = self.upload.as_ref().and_then(|(_, upload)| upload.poll()) else {
            return;
        };
        let Some((image, _)) = self.upload.take() else { return };
        match outcome {
            Ok(uploaded) => {
                self.renderer.set_source(uploaded);
                self.resident = Some(image);
                self.drawn = None;
                // The texture was not there when this frame was laid out.
                ctx.request_repaint();
            }
            Err(reason) => self.refuse(image, &reason),
        }
    }

    /// Points the interface's texture id at the renderer's current target.
    fn register_target(&mut self) {
        let Some(view) = self.renderer.target() else { return };
        let device = &self.state.device;
        let sampler = crate::gpu_preview::sampler;
        let mut egui_renderer = self.state.renderer.write();
        match self.id {
            Some(id) => {
                egui_renderer.update_egui_texture_from_wgpu_texture_with_sampler_options(
                    device,
                    view,
                    sampler(),
                    id,
                );
            }
            None => {
                self.id = Some(egui_renderer.register_native_texture_with_sampler_options(
                    device,
                    view,
                    sampler(),
                ))
            }
        }
    }

    /// Redraws the result when the settings changed since it was last drawn.
    fn draw(&mut self, params: &SeamlessParams) {
        let key = params.cache_key();
        if self.resident.is_none() || self.drawn == Some(key) {
            return;
        }
        match self.renderer.render(params) {
            Ok(resized) => {
                if resized {
                    self.register_target();
                }
                self.drawn = Some(key);
            }
            Err(reason) => {
                log::warn!("this image is made seamless on the CPU only: {reason}");
                self.refused = self.resident.take();
            }
        }
    }

    /// The texture showing the seamless result, once the GPU has drawn it.
    pub fn texture(&self) -> Option<Texture> {
        self.drawn?;
        let (width, height) = self.renderer.size()?;
        Some(Texture { id: self.id?, full_size: vec2(width as f32, height as f32) })
    }

    /// Frees the source, the result texture and its id. The next
    /// [`GpuSeamless::update`] with an image the GPU can take sets them up
    /// again.
    pub fn release(&mut self) {
        if let Some(id) = self.id.take() {
            self.state.renderer.write().free_texture(&id);
        }
        self.renderer.clear();
        self.resident = None;
        self.upload = None;
        self.drawn = None;
    }
}
