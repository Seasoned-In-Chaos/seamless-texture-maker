//! Pieces every renderer in this crate shares.

use std::task::{Context, Poll, Waker};

use crate::{DeviceHealth, Unsupported};

/// Row alignment wgpu requires when copying a texture into a buffer.
const COPY_ROW_ALIGNMENT: u32 = 256;
const RGBA8_BYTES: u32 = 4;

/// A texture and the view passes bind.
pub(crate) struct Tex {
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
}

impl Tex {
    pub(crate) fn new(
        device: &wgpu::Device,
        label: &str,
        size: (u32, u32),
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { texture, view }
    }

    /// Fills the whole texture from tightly packed rows.
    pub(crate) fn write(
        &self,
        queue: &wgpu::Queue,
        bytes: &[u8],
        size: (u32, u32),
        bytes_per_pixel: u32,
    ) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size.0 * bytes_per_pixel),
                rows_per_image: Some(size.1),
            },
            wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
        );
    }
}

/// Runs `build` and returns the first validation or internal error it raised.
pub(crate) fn first_error<T>(
    device: &wgpu::Device,
    build: impl FnOnce() -> T,
) -> (T, Option<wgpu::Error>) {
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let built = build();
    let mut context = Context::from_waker(Waker::noop());
    let mut error = None;
    // Scopes pop in reverse order.
    for pop in [internal.pop(), validation.pop()] {
        if let Poll::Ready(Some(raised)) = std::pin::pin!(pop).poll(&mut context) {
            error.get_or_insert(raised);
        }
    }
    (built, error)
}

/// Reads an `Rgba8Unorm` texture back as tightly packed rows. Blocks until
/// the GPU has finished; meant for tests and tools.
pub(crate) fn read_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    health: &DeviceHealth,
    texture: &wgpu::Texture,
    size: (u32, u32),
) -> Result<Vec<u8>, Unsupported> {
    let (width, height) = size;
    let row_bytes = width * RGBA8_BYTES;
    let padded = row_bytes.div_ceil(COPY_ROW_ALIGNMENT) * COPY_ROW_ALIGNMENT;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("seams_readback"),
        size: u64::from(padded) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    queue.submit([encoder.finish()]);
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    if !health.wait(device) {
        let stopped = || Unsupported("the GPU stopped before the read-back".to_owned());
        return Err(health.fault().map_or_else(stopped, Unsupported::from));
    }
    let failed = |error| Unsupported(format!("the GPU read-back failed: {error:?}"));
    let mapped = buffer.slice(..).get_mapped_range().map_err(failed)?;
    let mut pixels = Vec::with_capacity(row_bytes as usize * height as usize);
    for row in mapped.chunks(padded as usize) {
        pixels.extend_from_slice(row.get(..row_bytes as usize).unwrap_or(&[]));
    }
    Ok(pixels)
}

/// The vertex state of a pass that covers its whole target with one triangle,
/// using the module's `vs` entry point.
pub(crate) fn fullscreen_vertex(module: &wgpu::ShaderModule) -> wgpu::VertexState<'_> {
    wgpu::VertexState {
        module,
        entry_point: Some("vs"),
        buffers: &[],
        compilation_options: Default::default(),
    }
}
