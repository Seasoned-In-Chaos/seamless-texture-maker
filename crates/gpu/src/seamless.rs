//! GPU preview of the seamless step.
//!
//! Draws the result of [`seams_core::seamless`] for the methods that are one
//! formula per pixel: offset cross-fade, mirror tiling and overlap blend.
//! Splat synthesis scatters and averages many patches and stays on the CPU;
//! [`supports`] tells the two apart.
//!
//! The source is held as 8-bit colour, which is all a preview shows, and the
//! result is drawn into one `Rgba8Unorm` texture of premultiplied colour.
//! Export never reads it: the CPU code is the reference, and
//! `tests/seamless_parity.rs` holds the two within one 8-bit step.

use seams_core::Image;
use seams_core::seamless::{
    Method, SeamlessParams, fade_half_width, feather_radius, overlap_extent,
};

use crate::common::{Tex, first_error, fullscreen_vertex, read_rgba8};
use crate::{DeviceHealth, Fault, MEMORY_BUDGET, TARGET_FORMAT, Unsupported};

const SHADER: &str = include_str!("seamless.wgsl");
const UNIFORM_SIZE: u64 = 3 * 16;
const RGBA8_BYTES: u64 = 4;

// Method selectors shared with seamless.wgsl.
const OFFSET_CROSSFADE: i32 = 0;
const MIRROR: i32 = 1;
const OVERLAP: i32 = 2;

/// Whether the GPU can preview `method`. None needs no preview work: the
/// CPU result is the source itself.
pub fn supports(method: Method) -> bool {
    !matches!(method, Method::Splat | Method::None)
}

/// Size of the result for a source of `size`: mirror tiling doubles it.
pub fn output_size(method: Method, size: (u32, u32)) -> (u32, u32) {
    match method {
        Method::Mirror => (size.0.saturating_mul(2), size.1.saturating_mul(2)),
        Method::OffsetCrossfade | Method::Overlap | Method::Splat | Method::None => size,
    }
}

/// Checks that `method` can be previewed on `device` for a source this size.
pub fn fits(
    device: &wgpu::Device,
    width: usize,
    height: usize,
    method: Method,
) -> Result<(), Unsupported> {
    if !supports(method) {
        return Err(Unsupported(format!("{} is previewed on the CPU", method.label())));
    }
    if width == 0 || height == 0 {
        return Err(Unsupported("the image is empty".to_owned()));
    }
    let limit = u64::from(device.limits().max_texture_dimension_2d);
    let clamped = |side: usize| u32::try_from(side).unwrap_or(u32::MAX);
    let (out_width, out_height) = output_size(method, (clamped(width), clamped(height)));
    if u64::from(out_width) > limit || u64::from(out_height) > limit {
        return Err(Unsupported(format!(
            "a {out_width} x {out_height} px result is beyond this GPU's {limit} px texture limit"
        )));
    }
    let source = width as u64 * height as u64;
    let result = u64::from(out_width) * u64::from(out_height);
    let needed = (source + result).saturating_mul(RGBA8_BYTES);
    if needed > MEMORY_BUDGET {
        return Err(Unsupported(format!(
            "the preview needs {} MB of GPU memory; the budget is {} MB",
            needed >> 20,
            MEMORY_BUDGET >> 20
        )));
    }
    Ok(())
}

/// The uniform block of seamless.wgsl.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct Uniforms {
    size: [i32; 4],
    mode: [i32; 4],
    a: [f32; 4],
}

impl Uniforms {
    /// The settings of `params` for a source of `size`.
    fn new(params: &SeamlessParams, size: (u32, u32)) -> Self {
        let params = params.sanitized();
        let (width, height) = (size.0 as usize, size.1 as usize);
        let whole = [size.0 as i32, size.1 as i32];
        match params.method {
            Method::OffsetCrossfade | Method::Splat | Method::None => {
                let radius = feather_radius(width, height, params.fade_amount.0);
                Self {
                    size: [whole[0], whole[1], whole[0], whole[1]],
                    mode: [OFFSET_CROSSFADE, radius as i32, 0, 0],
                    a: [0.0; 4],
                }
            }
            Method::Mirror => Self {
                size: [whole[0], whole[1], whole[0], whole[1]],
                mode: [MIRROR, 0, 0, 0],
                a: [0.0; 4],
            },
            Method::Overlap => {
                let columns = overlap_extent(width, params.overlap.x) as i32;
                let rows = overlap_extent(height, params.overlap.y) as i32;
                Self {
                    size: [whole[0], whole[1], whole[0] - columns, whole[1] - rows],
                    mode: [OVERLAP, 0, columns, rows],
                    a: [fade_half_width(params.overlap.falloff), 0.0, 0.0, 0.0],
                }
            }
        }
    }

    fn bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(UNIFORM_SIZE as usize);
        out.extend(self.size.iter().chain(&self.mode).flat_map(|v| v.to_le_bytes()));
        out.extend(self.a.iter().flat_map(|v| v.to_le_bytes()));
        out
    }
}

fn to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// An image resident on the GPU as 8-bit colour with straight alpha.
pub struct GpuImage {
    texture: Tex,
    size: (u32, u32),
}

impl GpuImage {
    /// Converts and uploads `image`. Safe to call from any thread: the
    /// upload is queued and takes effect with the next submission.
    pub fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &Image,
    ) -> Result<Self, Unsupported> {
        let limit = device.limits().max_texture_dimension_2d as usize;
        let (width, height) = (image.width(), image.height());
        if width == 0 || height == 0 || width > limit || height > limit {
            return Err(Unsupported(format!(
                "{width} x {height} px does not fit this GPU's {limit} px texture limit"
            )));
        }
        let bytes: Vec<u8> = image
            .data()
            .chunks(image.channels())
            .flat_map(|pixel| match pixel {
                [r, g, b, a] => [to_byte(*r), to_byte(*g), to_byte(*b), to_byte(*a)],
                [r, g, b] => [to_byte(*r), to_byte(*g), to_byte(*b), 255],
                [gray] => [to_byte(*gray), to_byte(*gray), to_byte(*gray), 255],
                _ => [0, 0, 0, 255],
            })
            .collect();
        let size = (width as u32, height as u32);
        let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let texture = Tex::new(device, "seams_seamless_source", size, TARGET_FORMAT, usage);
        texture.write(queue, &bytes, size, RGBA8_BYTES as u32);
        Ok(Self { texture, size })
    }

    /// Width and height in pixels.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }
}

struct Kit {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
}

impl Kit {
    fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("seams_seamless"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let uniform = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZeroU64::new(UNIFORM_SIZE),
            },
            count: None,
        };
        let source = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                // Read with `textureLoad`, never filtered.
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("seams_seamless"),
            entries: &[uniform, source],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("seams_seamless"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fs_seamless"),
            layout: Some(&pipeline_layout),
            vertex: fullscreen_vertex(&module),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_seamless"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: TARGET_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        Self { layout, pipeline }
    }
}

/// Draws the seamless result of one resident image.
pub struct SeamlessRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    health: DeviceHealth,
    kit: Kit,
    uniforms: wgpu::Buffer,
    source: Option<GpuImage>,
    /// The result texture and its size, which follows the method.
    target: Option<(Tex, (u32, u32))>,
}

impl SeamlessRenderer {
    /// Builds the pipeline on `device`. Fails, without panicking, when the
    /// device cannot run it.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Self, Unsupported> {
        let (kit, error) = first_error(device, || Kit::new(device));
        if let Some(error) = error {
            return Err(Unsupported(format!("the seamless shader could not be set up: {error}")));
        }
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("seams_seamless_uniforms"),
            size: UNIFORM_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            health: DeviceHealth::new(),
            kit,
            uniforms,
            source: None,
            target: None,
        })
    }

    /// Shares the device's health flag; no GPU work is issued once it faults.
    pub fn set_health(&mut self, health: DeviceHealth) {
        self.health = health;
    }

    /// Why the GPU can no longer be used, if it cannot.
    pub fn fault(&self) -> Option<Fault> {
        self.health.fault()
    }

    fn check(&self) -> Result<(), Unsupported> {
        self.health.fault().map_or(Ok(()), |fault| Err(fault.into()))
    }

    /// Makes `image` the one the result is drawn from.
    pub fn set_source(&mut self, image: GpuImage) {
        self.source = Some(image);
    }

    /// Frees the source and the result.
    pub fn clear(&mut self) {
        self.source = None;
        self.target = None;
    }

    /// The result texture, once something has been drawn.
    pub fn target(&self) -> Option<&wgpu::TextureView> {
        self.target.as_ref().map(|(texture, _)| &texture.view)
    }

    /// Size of the result texture.
    pub fn size(&self) -> Option<(u32, u32)> {
        self.target.as_ref().map(|(_, size)| *size)
    }

    /// Draws the seamless result for `params`. The work is queued; it has
    /// run by the time anything submitted later reads the texture.
    ///
    /// Returns true when the result changed size, which replaces the texture:
    /// a view taken from [`SeamlessRenderer::target`] before must be taken
    /// again.
    pub fn render(&mut self, params: &SeamlessParams) -> Result<bool, Unsupported> {
        self.check()?;
        let missing = || Unsupported("no image is loaded".to_owned());
        let source_size = self.source.as_ref().map(GpuImage::size).ok_or_else(missing)?;
        let (width, height) = (source_size.0 as usize, source_size.1 as usize);
        fits(&self.device, width, height, params.method)?;

        let size = output_size(params.method, source_size);
        let resized = self.size() != Some(size);
        if resized {
            let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC;
            let texture = Tex::new(&self.device, "seams_seamless", size, TARGET_FORMAT, usage);
            self.target = Some((texture, size));
        }
        let source = self.source.as_ref().ok_or_else(missing)?;
        let (target, _) = self.target.as_ref().ok_or_else(missing)?;

        self.queue.write_buffer(&self.uniforms, 0, &Uniforms::new(params, source_size).bytes());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("seams_seamless"),
            layout: &self.kit.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniforms.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&source.texture.view),
                },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("seams_seamless"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("seams_seamless"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.kit.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        self.check()?;
        Ok(resized)
    }

    /// Reads the result back as tightly packed premultiplied RGBA8 rows.
    /// Blocks until the GPU has finished; meant for tests and tools.
    pub fn read_back(&self) -> Result<Vec<u8>, Unsupported> {
        self.check()?;
        let missing = || Unsupported("nothing has been drawn".to_owned());
        let (target, size) = self.target.as_ref().ok_or_else(missing)?;
        read_rgba8(&self.device, &self.queue, &self.health, &target.texture, *size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seams_core::seamless::{FadeAmount, OverlapParams};

    #[test]
    fn splat_is_the_only_method_left_to_the_cpu() {
        let supported: Vec<Method> = Method::ALL.into_iter().filter(|m| supports(*m)).collect();

        assert_eq!(supported, [Method::OffsetCrossfade, Method::Mirror, Method::Overlap]);
    }

    #[test]
    fn only_mirror_tiling_changes_the_result_size() {
        assert_eq!(output_size(Method::Mirror, (300, 200)), (600, 400));
        assert_eq!(output_size(Method::Overlap, (300, 200)), (300, 200));
        assert_eq!(output_size(Method::OffsetCrossfade, (300, 200)), (300, 200));
    }

    #[test]
    fn the_uniform_block_is_exactly_the_size_the_shader_declares() {
        assert_eq!(Uniforms::default().bytes().len() as u64, UNIFORM_SIZE);
    }

    #[test]
    fn cross_fade_uniforms_carry_the_reference_radius() {
        let params = SeamlessParams {
            method: Method::OffsetCrossfade,
            fade_amount: FadeAmount(0.75),
            ..SeamlessParams::default()
        };

        let uniforms = Uniforms::new(&params, (400, 300));

        assert_eq!(uniforms.mode, [OFFSET_CROSSFADE, feather_radius(400, 300, 0.75) as i32, 0, 0]);
        assert_eq!(uniforms.size, [400, 300, 400, 300]);
    }

    #[test]
    fn overlap_uniforms_describe_the_tile_left_after_the_overlap() {
        let params = SeamlessParams {
            method: Method::Overlap,
            overlap: OverlapParams { x: 0.25, y: 0.5, falloff: 1.0 },
            ..SeamlessParams::default()
        };

        let uniforms = Uniforms::new(&params, (400, 300));

        assert_eq!(uniforms.mode, [OVERLAP, 0, 100, 150]);
        assert_eq!(uniforms.size, [400, 300, 300, 150]);
        assert_eq!(uniforms.a[0], fade_half_width(1.0));
    }

    #[test]
    fn out_of_range_settings_are_clamped_like_the_reference() {
        let params = SeamlessParams {
            method: Method::Overlap,
            overlap: OverlapParams { x: 9.0, y: -1.0, falloff: f32::NAN },
            ..SeamlessParams::default()
        };

        let uniforms = Uniforms::new(&params, (200, 200));

        assert_eq!(uniforms.mode, [OVERLAP, 0, 100, 0]);
        assert!(uniforms.a[0].is_finite());
    }
}
