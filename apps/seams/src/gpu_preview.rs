//! Live previews drawn on the GPU.
//!
//! The session computes every result on the CPU, and that is what export
//! writes. This module draws the same results with `seams-gpu` into textures
//! the interface shows directly, so a slider is followed at full resolution
//! without waiting for the CPU: the material maps ([`GpuMaps`]) and the
//! seamless base colour ([`GpuSeamless`]).
//!
//! Whatever the GPU cannot take (an image too large for a texture or the
//! memory budget, splat synthesis) is reported by the `update_*` methods, and
//! the caller shows the CPU result instead. A lost device is reported by
//! [`GpuPreview::fault`]; the caller then drops the whole preview.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, TryRecvError, channel};

use eframe::egui_wgpu::RenderState;
use eframe::wgpu;
use egui::Context;
use seams_core::Image;
use seams_core::pbr::{MapKind, MaterialParams};
use seams_core::seamless::SeamlessParams;
use seams_gpu::{DeviceHealth, Fault, Unsupported};

use crate::gpu_maps::GpuMaps;
use crate::gpu_seamless::GpuSeamless;
use crate::textures::Texture;

/// Set to any value to keep every preview on the CPU.
const DISABLE_VARIABLE: &str = "SEAMS_CPU_PREVIEW";

/// Something being prepared for the GPU on a background thread.
///
/// Converting and uploading a 4K image is tens of megabytes of work, too
/// much for the interface thread.
pub struct Background<T> {
    result: Receiver<Result<T, Unsupported>>,
}

impl<T: Send + 'static> Background<T> {
    /// Runs `job` on its own thread and asks for a repaint once it is done.
    pub fn spawn(
        ctx: &Context,
        job: impl FnOnce() -> Result<T, Unsupported> + Send + 'static,
    ) -> std::io::Result<Self> {
        let (sender, result) = channel();
        let repaint = ctx.clone();
        let run = move || {
            // The receiver is gone only when the preview was shut down.
            if sender.send(job()).is_ok() {
                repaint.request_repaint();
            }
        };
        std::thread::Builder::new().name("seams-gpu-upload".to_owned()).spawn(run)?;
        Ok(Self { result })
    }

    /// The job's outcome once it has finished; `None` while it is running.
    pub fn poll(&self) -> Option<Result<T, Unsupported>> {
        match self.result.try_recv() {
            Ok(outcome) => Some(outcome),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                Some(Err(Unsupported("the upload was interrupted".to_owned())))
            }
        }
    }
}

pub struct GpuPreview {
    health: DeviceHealth,
    adapter: String,
    maps: GpuMaps,
    seamless: GpuSeamless,
}

impl GpuPreview {
    /// Sets up the previews on the interface's own device. Returns `None`,
    /// leaving every preview on the CPU, when they are disabled or the
    /// device cannot run them.
    pub fn new(state: &RenderState) -> Option<Self> {
        if std::env::var_os(DISABLE_VARIABLE).is_some() {
            log::info!("GPU preview disabled by {DISABLE_VARIABLE}");
            return None;
        }
        let renderers = GpuMaps::new(state).and_then(|maps| Ok((maps, GpuSeamless::new(state)?)));
        let (mut maps, mut seamless) = match renderers {
            Ok(renderers) => renderers,
            Err(reason) => {
                log::warn!("GPU preview unavailable: {reason}");
                return None;
            }
        };
        // From here a lost device marks this flag instead of panicking.
        let health = DeviceHealth::watch(&state.device);
        maps.set_health(health.clone());
        seamless.set_health(health.clone());
        let info = state.adapter.get_info();
        let adapter = format!("{} ({:?})", info.name, info.backend);
        log::info!("GPU preview on {adapter}");
        Some(Self { health, adapter, maps, seamless })
    }

    /// The graphics adapter and API in use, for display.
    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    /// Why the GPU can no longer be used, if it cannot.
    pub fn fault(&self) -> Option<Fault> {
        self.health.fault()
    }

    /// Brings the map textures up to date. Returns true while the GPU is the
    /// one showing the maps of `source`.
    pub fn update_maps(
        &mut self,
        ctx: &Context,
        source: Option<&Arc<Image>>,
        params: &MaterialParams,
    ) -> bool {
        self.maps.update(ctx, source.filter(|_| self.health.is_ok()), params)
    }

    /// Brings the seamless texture up to date. Returns true while the GPU is
    /// the one showing the seamless result of `source`.
    pub fn update_seamless(
        &mut self,
        ctx: &Context,
        source: Option<&Arc<Image>>,
        params: &SeamlessParams,
    ) -> bool {
        self.seamless.update(ctx, source.filter(|_| self.health.is_ok()), params)
    }

    /// The texture showing the map `kind`, once the GPU has drawn it.
    pub fn map_texture(&self, kind: MapKind) -> Option<Texture> {
        self.maps.texture(kind)
    }

    /// The texture showing the seamless base colour, once the GPU has drawn it.
    pub fn seamless_texture(&self) -> Option<Texture> {
        self.seamless.texture()
    }

    /// Frees everything held on the GPU.
    pub fn release(&mut self) {
        self.maps.release();
        self.seamless.release();
    }
}

/// How the interface samples a preview texture: smoothed when shrunk, exact
/// pixels when enlarged.
pub fn sampler() -> wgpu::SamplerDescriptor<'static> {
    wgpu::SamplerDescriptor {
        label: Some("seams_preview"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    }
}
