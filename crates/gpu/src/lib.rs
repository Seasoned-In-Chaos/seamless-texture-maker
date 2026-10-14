//! GPU preview of the material maps.
//!
//! Renders the maps of [`seams_core::pbr`] with wgpu, at full resolution and
//! fast enough to follow a slider. The CPU code stays the reference: export
//! always uses it, and `tests/parity.rs` holds the two within one 8-bit step
//! of each other.
//!
//! - **Residency.** The height field lives on the GPU as one `R32Float`
//!   texture ([`GpuField`]), with the source alpha beside it when the image
//!   has one. It is uploaded once per source image, never per setting.
//! - **Planning.** A map and its settings become a short list of fragment
//!   passes ([`plan`]): one for most maps, four when a map is blurred or
//!   sharpened (shape, blur across, blur down, finish).
//! - **Targets.** Each map is drawn into its own `Rgba8Unorm` texture holding
//!   premultiplied colour, ready to be shown as it is.
//!
//! Anything the GPU cannot take (an image beyond the texture limit or the
//! memory budget, a lost device) is reported as [`Unsupported`]; callers then
//! show the CPU result instead. The structure follows PhotoCraft's
//! `photocraft-gpu`: a CPU oracle, parity tests and a device health flag.

mod common;
pub mod health;
pub mod seamless;

use std::fmt;

use seams_core::filter::{auto_blur_radius, gaussian_kernel};
use seams_core::pbr::{
    BlurSharp, HeightField, MapKind, MaterialParams, NORMAL_Z_BASIS, NormalFormat, NormalParams,
    SlopeKernel,
};

use common::{Tex, first_error, fullscreen_vertex, read_rgba8};
pub use health::{DeviceHealth, Fault};
pub use seamless::{GpuImage, SeamlessRenderer};

/// Format of the map textures: 8-bit, premultiplied, gamma-encoded.
pub const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const FIELD_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
const ALPHA_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// GPU memory one image's maps may take. A larger image uses the CPU.
pub const MEMORY_BUDGET: u64 = 1536 << 20;
const FIELD_BYTES: u64 = 4;
const ALPHA_BYTES: u64 = 1;
const TARGET_BYTES: u64 = 4;
/// Two scalar textures, used while a map is blurred or sharpened.
const SCRATCH_BYTES: u64 = 8;

const SHADER: &str = include_str!("maps.wgsl");
/// Kernel weights the shader's uniform block holds.
const KERNEL_SLOTS: usize = 84;
const UNIFORM_SIZE: u64 = (4 * 16 + KERNEL_SLOTS * 4) as u64;

// Bit flags and selectors shared with maps.wgsl.
const WRAP: u32 = 1;
const INVERT_HEIGHT: u32 = 2;
const HAS_ALPHA: u32 = 4;
const INVERT_RESULT: u32 = 8;
const USE_BLURRED: u32 = 1;
const USE_SHARPENED: u32 = 2;
const BLUR_ACROSS: u32 = 0;
const BLUR_DOWN: u32 = 1;

/// The GPU cannot do what was asked; the CPU result should be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unsupported(pub String);

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Unsupported {}

impl From<Fault> for Unsupported {
    fn from(fault: Fault) -> Self {
        Self(fault.to_string())
    }
}

/// Position of a map in [`MapKind::ALL`], used to index the targets.
fn slot(kind: MapKind) -> usize {
    match kind {
        MapKind::Normal => 0,
        MapKind::Roughness => 1,
        MapKind::AmbientOcclusion => 2,
        MapKind::Displacement => 3,
        MapKind::Opacity => 4,
    }
}

/// GPU memory, in bytes, the maps of an image of this size take at most.
pub fn working_set(width: usize, height: usize, has_alpha: bool) -> u64 {
    let pixels = width as u64 * height as u64;
    let alpha = if has_alpha { ALPHA_BYTES } else { 0 };
    let targets = TARGET_BYTES * MapKind::ALL.len() as u64;
    pixels.saturating_mul(FIELD_BYTES + alpha + targets + SCRATCH_BYTES)
}

/// Checks that an image of this size can be previewed on `device`.
pub fn fits(
    device: &wgpu::Device,
    width: usize,
    height: usize,
    has_alpha: bool,
) -> Result<(), Unsupported> {
    let limit = device.limits().max_texture_dimension_2d as usize;
    if width == 0 || height == 0 {
        return Err(Unsupported("the image is empty".to_owned()));
    }
    if width > limit || height > limit {
        return Err(Unsupported(format!(
            "{width} x {height} px is beyond this GPU's {limit} px texture limit"
        )));
    }
    let needed = working_set(width, height, has_alpha);
    if needed > MEMORY_BUDGET {
        return Err(Unsupported(format!(
            "{width} x {height} px needs {} MB of GPU memory; the preview budget is {} MB",
            needed >> 20,
            MEMORY_BUDGET >> 20
        )));
    }
    Ok(())
}

/// The fragment program a pass runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Program {
    Normal,
    Shape,
    Blur,
    Scalar,
}

/// Where a pass writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Output {
    /// The map's own texture.
    Target,
    /// One of the two scalar scratch textures.
    Scratch(usize),
}

/// The uniform block of maps.wgsl.
#[derive(Debug, Clone, PartialEq, Default)]
struct Uniforms {
    dims: [i32; 4],
    flags: [u32; 4],
    a: [f32; 4],
    b: [f32; 4],
    kernel: Vec<f32>,
}

impl Uniforms {
    fn bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(UNIFORM_SIZE as usize);
        out.extend(self.dims.iter().flat_map(|v| v.to_le_bytes()));
        out.extend(self.flags.iter().flat_map(|v| v.to_le_bytes()));
        out.extend(self.a.iter().chain(&self.b).flat_map(|v| v.to_le_bytes()));
        let weights = self.kernel.iter().copied().chain(std::iter::repeat(0.0));
        out.extend(weights.take(KERNEL_SLOTS).flat_map(f32::to_le_bytes));
        out
    }
}

/// One fragment pass over the whole field.
#[derive(Debug, Clone, PartialEq)]
struct Pass {
    program: Program,
    output: Output,
    /// The scratch texture read as the scalar field, if the program reads one.
    input: Option<usize>,
    uniforms: Uniforms,
}

fn flag(on: bool, bit: u32) -> u32 {
    if on { bit } else { 0 }
}

fn as_float(on: bool) -> f32 {
    if on { 1.0 } else { 0.0 }
}

fn normal_pass(params: &NormalParams, size: [i32; 2], alpha: u32) -> Pass {
    let kernel = SlopeKernel::for_filter(params.filter);
    let scale = params.scale.clamp(0.1, 8.0) * kernel.gain;
    let flip_y = params.invert_y != (params.format == NormalFormat::DirectX);
    let min_z = if params.min_z.is_finite() { params.min_z.clamp(0.0, 1.0) } else { 0.0 };
    let flags = alpha | flag(params.wrap, WRAP) | flag(params.invert_height, INVERT_HEIGHT);
    let uniforms = Uniforms {
        dims: [size[0], size[1], kernel.size as i32, 0],
        flags: [flags, 0, 0, 0],
        a: [
            if params.invert_x { -scale } else { scale },
            if flip_y { -scale } else { scale },
            NORMAL_Z_BASIS,
            min_z,
        ],
        kernel: kernel.weights,
        ..Uniforms::default()
    };
    Pass { program: Program::Normal, output: Output::Target, input: None, uniforms }
}

/// How one scalar map is shaped and finished.
struct ScalarSettings {
    /// Selects the formula in the shader's `shape`.
    mode: i32,
    a: [f32; 4],
    blur_sharp: f32,
    invert_result: bool,
}

/// The settings of a scalar map, or `None` for the normal map.
fn scalar_settings(kind: MapKind, params: &MaterialParams) -> Option<ScalarSettings> {
    let plain = |mode, a| ScalarSettings { mode, a, blur_sharp: 0.0, invert_result: false };
    match kind {
        MapKind::Normal => None,
        MapKind::Roughness => {
            let r = &params.roughness;
            Some(plain(0, [1.0 + r.contrast * 2.0, r.intensity * 2.0, as_float(r.invert), 0.0]))
        }
        MapKind::AmbientOcclusion => {
            let ao = &params.ao;
            Some(ScalarSettings {
                mode: 1,
                a: [ao.range.max(1e-4), ao.mean, ao.strength, 0.0],
                blur_sharp: ao.blur_sharp,
                invert_result: ao.invert,
            })
        }
        MapKind::Displacement => {
            let d = &params.displacement;
            let contrast = d.contrast.clamp(-1.0, 0.999);
            let factor = (contrast + 1.0) / (1.0 - contrast);
            Some(ScalarSettings {
                mode: 2,
                a: [factor, as_float(d.invert), 0.0, 0.0],
                blur_sharp: d.blur_sharp,
                invert_result: false,
            })
        }
        MapKind::Opacity => {
            let o = &params.opacity;
            let width = (o.softness * 0.45).max(0.01);
            Some(plain(3, [1.0 - o.threshold, width, as_float(o.softness > 0.0), 0.0]))
        }
    }
}

/// The passes that draw `kind` for a field of `size`.
fn plan(kind: MapKind, params: &MaterialParams, size: [i32; 2], has_alpha: bool) -> Vec<Pass> {
    let alpha = flag(has_alpha, HAS_ALPHA);
    let Some(settings) = scalar_settings(kind, params) else {
        return vec![normal_pass(&params.normal, size, alpha)];
    };
    let shaped = Uniforms {
        dims: [size[0], size[1], 0, settings.mode],
        a: settings.a,
        ..Uniforms::default()
    };
    let finish = |use_field: u32, gain: f32, input: Option<usize>| Pass {
        program: Program::Scalar,
        output: Output::Target,
        input,
        uniforms: Uniforms {
            flags: [alpha | flag(settings.invert_result, INVERT_RESULT), use_field, 0, 0],
            b: [gain, 0.0, 0.0, 0.0],
            ..shaped.clone()
        },
    };
    let Some(blur) = BlurSharp::from_setting(settings.blur_sharp) else {
        return vec![finish(0, 0.0, None)];
    };
    let radius = auto_blur_radius(blur.sigma);
    let taps = gaussian_kernel(blur.sigma, radius);
    let blur_pass = |axis: u32, from: usize, to: usize| Pass {
        program: Program::Blur,
        output: Output::Scratch(to),
        input: Some(from),
        uniforms: Uniforms {
            dims: [size[0], size[1], radius as i32, 0],
            flags: [0, 0, axis, 0],
            kernel: taps.clone(),
            ..Uniforms::default()
        },
    };
    let (use_field, gain) = match blur.sharpen {
        Some(gain) => (USE_SHARPENED, gain),
        None => (USE_BLURRED, 0.0),
    };
    let finished = finish(use_field, gain, Some(0));
    vec![
        Pass { program: Program::Shape, output: Output::Scratch(0), input: None, uniforms: shaped },
        blur_pass(BLUR_ACROSS, 0, 1),
        blur_pass(BLUR_DOWN, 1, 0),
        finished,
    ]
}

/// A height field resident on the GPU.
pub struct GpuField {
    values: Tex,
    alpha: Option<Tex>,
    size: (u32, u32),
}

impl GpuField {
    /// Uploads `field`. Safe to call from any thread: the upload is queued
    /// and takes effect with the next submission.
    pub fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        field: &HeightField,
    ) -> Result<Self, Unsupported> {
        fits(device, field.width, field.height, field.alpha.is_some())?;
        let pixels = field.width * field.height;
        let alpha_len = field.alpha.as_ref().map_or(pixels, Vec::len);
        if field.values.len() != pixels || alpha_len != pixels {
            return Err(Unsupported("the height field does not match its size".to_owned()));
        }
        let size = (field.width as u32, field.height as u32);
        let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let values = Tex::new(device, "seams_height", size, FIELD_FORMAT, usage);
        values.write(queue, bytemuck::cast_slice(&field.values), size, FIELD_BYTES as u32);
        let alpha = field.alpha.as_ref().map(|alpha| {
            let bytes: Vec<u8> =
                alpha.iter().map(|a| (a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
            let texture = Tex::new(device, "seams_alpha", size, ALPHA_FORMAT, usage);
            texture.write(queue, &bytes, size, ALPHA_BYTES as u32);
            texture
        });
        Ok(Self { values, alpha, size })
    }

    /// Width and height in pixels.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            multisampled: false,
            // Every texture is read with `textureLoad`, never filtered.
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

/// The shader module's pipelines and the layout their bind groups share.
struct Kit {
    layout: wgpu::BindGroupLayout,
    normal: wgpu::RenderPipeline,
    shape: wgpu::RenderPipeline,
    blur: wgpu::RenderPipeline,
    scalar: wgpu::RenderPipeline,
}

impl Kit {
    fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("seams_maps"),
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
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("seams_maps"),
            entries: &[uniform, texture_entry(1), texture_entry(2), texture_entry(3)],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("seams_maps"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &'static str, format: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                vertex: fullscreen_vertex(&module),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        Self {
            normal: pipeline("fs_normal", TARGET_FORMAT),
            shape: pipeline("fs_shape", FIELD_FORMAT),
            blur: pipeline("fs_blur", FIELD_FORMAT),
            scalar: pipeline("fs_scalar", TARGET_FORMAT),
            layout,
        }
    }

    fn pipeline(&self, program: Program) -> &wgpu::RenderPipeline {
        match program {
            Program::Normal => &self.normal,
            Program::Shape => &self.shape,
            Program::Blur => &self.blur,
            Program::Scalar => &self.scalar,
        }
    }
}

/// Draws material maps for one resident height field.
pub struct MapRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    health: DeviceHealth,
    kit: Kit,
    /// Bound where a pass reads no alpha or scalar field.
    placeholder: Tex,
    /// One uniform buffer per pass of a render, reused between renders.
    uniforms: Vec<wgpu::Buffer>,
    field: Option<GpuField>,
    /// One texture per map, in [`MapKind::ALL`] order, at the field's size.
    targets: Vec<Tex>,
    /// Scalar textures for blurring, created the first time a map needs them.
    scratch: Option<[Tex; 2]>,
}

impl MapRenderer {
    /// Builds the pipelines on `device`. Fails, without panicking, when the
    /// device cannot run them.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Self, Unsupported> {
        let (kit, error) = first_error(device, || Kit::new(device));
        if let Some(error) = error {
            return Err(Unsupported(format!("the map shaders could not be set up: {error}")));
        }
        let usage = wgpu::TextureUsages::TEXTURE_BINDING;
        let placeholder = Tex::new(device, "seams_placeholder", (1, 1), FIELD_FORMAT, usage);
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            health: DeviceHealth::new(),
            kit,
            placeholder,
            uniforms: Vec::new(),
            field: None,
            targets: Vec::new(),
            scratch: None,
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

    /// Size of the resident field, if there is one.
    pub fn size(&self) -> Option<(u32, u32)> {
        self.field.as_ref().map(GpuField::size)
    }

    /// Makes `field` the one maps are drawn from. Returns true when its size
    /// differs from the previous field's, which replaces every map texture:
    /// views taken from [`MapRenderer::target`] before must be taken again.
    pub fn set_field(&mut self, field: GpuField) -> bool {
        let resized = self.size() != Some(field.size());
        if resized {
            let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC;
            self.targets = MapKind::ALL
                .iter()
                .map(|kind| {
                    Tex::new(&self.device, kind.label(), field.size(), TARGET_FORMAT, usage)
                })
                .collect();
            self.scratch = None;
        }
        self.field = Some(field);
        resized
    }

    /// Frees the field and every map texture.
    pub fn clear(&mut self) {
        self.field = None;
        self.targets.clear();
        self.scratch = None;
    }

    /// The texture `kind` is drawn into, once a field is set.
    pub fn target(&self, kind: MapKind) -> Option<&wgpu::TextureView> {
        self.targets.get(slot(kind)).map(|target| &target.view)
    }

    fn scratch_textures(device: &wgpu::Device, size: (u32, u32)) -> [Tex; 2] {
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        ["seams_scratch_0", "seams_scratch_1"]
            .map(|label| Tex::new(device, label, size, FIELD_FORMAT, usage))
    }

    /// Draws `kind` with `params` into its texture. The work is queued; it
    /// has run by the time anything submitted later reads the texture.
    pub fn render(&mut self, kind: MapKind, params: &MaterialParams) -> Result<(), Unsupported> {
        self.check()?;
        let missing = || Unsupported("no height field is loaded".to_owned());
        let size = self.size().ok_or_else(missing)?;
        let has_alpha = self.field.as_ref().is_some_and(|field| field.alpha.is_some());
        let passes = plan(kind, params, [size.0 as i32, size.1 as i32], has_alpha);

        if self.scratch.is_none() && passes.iter().any(|pass| pass.output != Output::Target) {
            self.scratch = Some(Self::scratch_textures(&self.device, size));
        }
        while self.uniforms.len() < passes.len() {
            self.uniforms.push(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("seams_maps_uniforms"),
                size: UNIFORM_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        let field = self.field.as_ref().ok_or_else(missing)?;
        let target = self.targets.get(slot(kind)).ok_or_else(missing)?;
        let scratch = |index: usize| {
            let textures = self.scratch.as_ref().and_then(|textures| textures.get(index));
            textures.map(|texture| &texture.view).ok_or_else(missing)
        };
        let alpha = field.alpha.as_ref().unwrap_or(&self.placeholder);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("seams_maps") });
        for (pass, uniforms) in passes.iter().zip(&self.uniforms) {
            self.queue.write_buffer(uniforms, 0, &pass.uniforms.bytes());
            let output = match pass.output {
                Output::Target => &target.view,
                Output::Scratch(index) => scratch(index)?,
            };
            let input = match pass.input {
                Some(index) => scratch(index)?,
                None => &self.placeholder.view,
            };
            let textures = [&field.values.view, &alpha.view, input];
            self.draw(&mut encoder, pass.program, output, uniforms, textures);
        }
        self.queue.submit([encoder.finish()]);
        self.check()
    }

    fn draw(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        program: Program,
        output: &wgpu::TextureView,
        uniforms: &wgpu::Buffer,
        textures: [&wgpu::TextureView; 3],
    ) {
        let [height, alpha, field] = textures.map(wgpu::BindingResource::TextureView);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("seams_maps"),
            layout: &self.kit.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: height },
                wgpu::BindGroupEntry { binding: 2, resource: alpha },
                wgpu::BindGroupEntry { binding: 3, resource: field },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("seams_maps"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
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
        pass.set_pipeline(self.kit.pipeline(program));
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Reads `kind` back as tightly packed premultiplied RGBA8 rows. Blocks
    /// until the GPU has finished; meant for tests and tools.
    pub fn read_back(&self, kind: MapKind) -> Result<Vec<u8>, Unsupported> {
        self.check()?;
        let missing = || Unsupported("no height field is loaded".to_owned());
        let (width, height) = self.size().ok_or_else(missing)?;
        let target = self.targets.get(slot(kind)).ok_or_else(missing)?;
        read_rgba8(&self.device, &self.queue, &self.health, &target.texture, (width, height))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seams_core::pbr::NormalFilter;

    const SIZE: [i32; 2] = [64, 48];

    #[test]
    fn the_uniform_block_is_exactly_the_size_the_shader_declares() {
        let empty = Uniforms::default();
        let full = Uniforms { kernel: vec![1.0; 500], ..Uniforms::default() };

        assert_eq!(UNIFORM_SIZE, 400);
        assert_eq!(empty.bytes().len(), 400);
        assert_eq!(full.bytes().len(), 400);
    }

    #[test]
    fn uniform_fields_are_laid_out_in_declaration_order() {
        let uniforms = Uniforms {
            dims: [1, 2, 3, 4],
            flags: [5, 6, 7, 8],
            a: [0.5, 0.0, 0.0, 0.0],
            b: [0.25, 0.0, 0.0, 0.0],
            kernel: vec![2.0],
        };

        let bytes = uniforms.bytes();

        assert_eq!(bytes[0..4], 1i32.to_le_bytes());
        assert_eq!(bytes[12..16], 4i32.to_le_bytes());
        assert_eq!(bytes[16..20], 5u32.to_le_bytes());
        assert_eq!(bytes[32..36], 0.5f32.to_le_bytes());
        assert_eq!(bytes[48..52], 0.25f32.to_le_bytes());
        assert_eq!(bytes[64..68], 2.0f32.to_le_bytes());
    }

    #[test]
    fn every_kernel_the_shader_may_receive_fits_its_uniform_block() {
        for filter in NormalFilter::ALL {
            assert!(SlopeKernel::for_filter(filter).weights.len() <= KERNEL_SLOTS, "{filter:?}");
        }
        for setting in [-32.0, 32.0, -1.0e9] {
            let blur = BlurSharp::from_setting(setting).unwrap();
            let taps = 2 * auto_blur_radius(blur.sigma) + 1;
            assert!(taps <= KERNEL_SLOTS, "setting {setting} needs {taps} taps");
        }
    }

    #[test]
    fn maps_without_blur_take_one_pass_into_their_target() {
        let params = MaterialParams::default();

        for kind in MapKind::ALL {
            let passes = plan(kind, &params, SIZE, false);

            assert_eq!(passes.len(), 1, "{kind:?}");
            assert_eq!(passes[0].output, Output::Target);
            assert_eq!(passes[0].input, None);
        }
    }

    #[test]
    fn a_blurred_map_is_shaped_blurred_both_ways_then_finished() {
        let mut params = MaterialParams::default();
        params.ao.blur_sharp = -8.0;

        let passes = plan(MapKind::AmbientOcclusion, &params, SIZE, false);

        let programs: Vec<Program> = passes.iter().map(|pass| pass.program).collect();
        assert_eq!(programs, [Program::Shape, Program::Blur, Program::Blur, Program::Scalar]);
        assert_eq!(passes[1].uniforms.flags[2], BLUR_ACROSS);
        assert_eq!(passes[2].uniforms.flags[2], BLUR_DOWN);
        assert_eq!(passes[3].uniforms.flags[1], USE_BLURRED);
        // No pass reads the texture it writes.
        for pass in &passes {
            assert!(pass.input.is_none_or(|input| pass.output != Output::Scratch(input)));
        }
    }

    #[test]
    fn a_positive_setting_sharpens_with_the_reference_gain() {
        let mut params = MaterialParams::default();
        params.displacement.blur_sharp = 16.0;

        let passes = plan(MapKind::Displacement, &params, SIZE, false);

        let finish = &passes[3].uniforms;
        assert_eq!(finish.flags[1], USE_SHARPENED);
        assert_eq!(finish.b[0], BlurSharp::from_setting(16.0).unwrap().sharpen.unwrap());
    }

    #[test]
    fn flags_follow_the_settings_and_the_alpha_channel() {
        let mut params = MaterialParams::default();
        params.normal.wrap = false;
        params.normal.invert_height = true;
        params.ao.invert = true;

        let normal = plan(MapKind::Normal, &params, SIZE, true);
        let ao = plan(MapKind::AmbientOcclusion, &params, SIZE, false);

        assert_eq!(normal[0].uniforms.flags[0], HAS_ALPHA | INVERT_HEIGHT);
        assert_eq!(ao[0].uniforms.flags[0], INVERT_RESULT);
    }

    #[test]
    fn directx_and_invert_y_cancel_out() {
        let mut params = MaterialParams::default();
        let opengl = plan(MapKind::Normal, &params, SIZE, false);
        params.normal.format = NormalFormat::DirectX;
        let directx = plan(MapKind::Normal, &params, SIZE, false);
        params.normal.invert_y = true;
        let both = plan(MapKind::Normal, &params, SIZE, false);

        assert_eq!(directx[0].uniforms.a[1], -opengl[0].uniforms.a[1]);
        assert_eq!(both[0].uniforms.a, opengl[0].uniforms.a);
    }

    #[test]
    fn the_working_set_counts_every_texture_an_image_needs() {
        let megapixel = 1024 * 1024;

        assert_eq!(working_set(1024, 1024, false), megapixel * 32);
        assert_eq!(working_set(1024, 1024, true), megapixel * 33);
        // A 4K texture fits the budget; an 8K one does not.
        assert!(working_set(4096, 4096, true) <= MEMORY_BUDGET);
        assert!(working_set(8192, 8192, false) > MEMORY_BUDGET);
    }
}
