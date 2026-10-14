//! GPU against CPU for the seamless step: offset cross-fade, mirror tiling
//! and overlap blend must match `seams_core::seamless` within one 8-bit step.
//! The tests pass without checking anything when the machine has no GPU
//! adapter, and say so on stderr.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{Mutex, MutexGuard, PoisonError};

use seams_core::Image;
use seams_core::seamless::{FadeAmount, Method, OverlapParams, SeamlessParams, make_seamless};
use seams_gpu::{GpuImage, SeamlessRenderer, seamless};

/// Largest difference allowed, in 8-bit steps.
const TOLERANCE: i32 = 1;

/// Several wgpu instances at once crash some drivers, so each test holds
/// this lock until its device is dropped.
static GPU_LOCK: Mutex<()> = Mutex::new(());

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: SeamlessRenderer,
    _lock: MutexGuard<'static, ()>,
}

fn gpu() -> Option<Gpu> {
    let lock = GPU_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    let instance = wgpu::Instance::default();
    let options = wgpu::RequestAdapterOptions::default();
    let adapter = match pollster::block_on(instance.request_adapter(&options)) {
        Ok(adapter) => adapter,
        Err(error) => {
            eprintln!("skipping GPU parity tests: no adapter ({error})");
            return None;
        }
    };
    let descriptor = wgpu::DeviceDescriptor::default();
    let (device, queue) = pollster::block_on(adapter.request_device(&descriptor)).ok()?;
    let renderer = match SeamlessRenderer::new(&device, &queue) {
        Ok(renderer) => renderer,
        Err(error) => {
            eprintln!("skipping GPU parity tests: {error}");
            return None;
        }
    };
    Some(Gpu { device, queue, renderer, _lock: lock })
}

/// A clearly non-tiling texture whose values are exact 8-bit steps, as an
/// 8-bit file's are, so the GPU's 8-bit source holds the same numbers.
fn source(width: usize, height: usize, channels: usize) -> Image {
    Image::from_fn(width, height, channels, |x, y, c| {
        let (fx, fy) = (x as f32 / width as f32, y as f32 / height as f32);
        let value = if c == 3 {
            0.3 + 0.7 * fy
        } else {
            let ramp = 0.15 + 0.6 * fx + 0.2 * fy;
            let grain = ((x * 5 + y * 11 + c * 3) % 13) as f32 / 13.0;
            ramp + 0.1 * (grain - 0.5) + 0.05 * c as f32
        };
        (value.clamp(0.0, 1.0) * 255.0).round() / 255.0
    })
    .unwrap()
}

fn to_byte(value: f32) -> i32 {
    (value.clamp(0.0, 1.0) * 255.0 + 0.5) as i32
}

/// The reference result as the premultiplied RGBA8 the GPU target holds.
fn reference(image: &Image, params: &SeamlessParams) -> (Vec<i32>, (u32, u32)) {
    let result = make_seamless(image, params).unwrap();
    let mut out = Vec::with_capacity(result.width() * result.height() * 4);
    for pixel in result.data().chunks(result.channels()) {
        let (rgb, alpha) = match pixel {
            [r, g, b, a] => ([*r, *g, *b], a.clamp(0.0, 1.0)),
            [r, g, b] => ([*r, *g, *b], 1.0),
            [gray] => ([*gray; 3], 1.0),
            _ => panic!("unexpected channel count"),
        };
        out.extend(rgb.map(|v| to_byte(v.clamp(0.0, 1.0) * alpha)));
        out.push(to_byte(alpha));
    }
    (out, (result.width() as u32, result.height() as u32))
}

fn assert_parity(gpu: &mut Gpu, image: &Image, params: &SeamlessParams) {
    let resident = GpuImage::upload(&gpu.device, &gpu.queue, image).unwrap();
    gpu.renderer.set_source(resident);
    gpu.renderer.render(params).unwrap();
    let actual = gpu.renderer.read_back().unwrap();
    let (expected, size) = reference(image, params);

    assert_eq!(gpu.renderer.size(), Some(size), "{params:?}: size");
    assert_eq!(actual.len(), expected.len());
    let worst = actual
        .iter()
        .zip(&expected)
        .enumerate()
        .map(|(i, (a, e))| ((i32::from(*a) - e).abs(), i))
        .max()
        .unwrap();
    let width = size.0 as usize;
    let (x, y) = ((worst.1 / 4) % width, (worst.1 / 4) / width);
    assert!(
        worst.0 <= TOLERANCE,
        "differs by {} at ({x}, {y}) channel {} with {params:?}",
        worst.0,
        worst.1 % 4
    );
}

fn with_method(method: Method) -> SeamlessParams {
    SeamlessParams { method, ..SeamlessParams::default() }
}

#[test]
fn offset_cross_fade_matches_at_every_fade_width() {
    let Some(mut gpu) = gpu() else { return };
    for fade in [0.0, 0.1, 0.5, 1.0] {
        let params = SeamlessParams {
            fade_amount: FadeAmount(fade),
            ..with_method(Method::OffsetCrossfade)
        };

        assert_parity(&mut gpu, &source(120, 96, 3), &params);
    }
}

#[test]
fn offset_cross_fade_matches_for_odd_and_narrow_sizes() {
    let Some(mut gpu) = gpu() else { return };
    let params =
        SeamlessParams { fade_amount: FadeAmount(1.0), ..with_method(Method::OffsetCrossfade) };

    for (width, height) in [(97, 65), (64, 201), (255, 64)] {
        assert_parity(&mut gpu, &source(width, height, 3), &params);
    }
}

#[test]
fn mirror_tiling_matches_and_doubles_the_size() {
    let Some(mut gpu) = gpu() else { return };

    assert_parity(&mut gpu, &source(90, 70, 3), &with_method(Method::Mirror));
    assert_eq!(gpu.renderer.size(), Some((180, 140)));
}

#[test]
fn overlap_blend_matches_across_amounts_and_falloffs() {
    let Some(mut gpu) = gpu() else { return };
    let cases = [
        (0.2, 0.2, 0.1),
        (0.5, 0.5, 1.0),
        (0.35, 0.0, 0.5),
        (0.0, 0.4, 0.0),
        (0.0, 0.0, 0.3),
        (0.13, 0.27, 0.75),
    ];
    for (x, y, falloff) in cases {
        let params = SeamlessParams {
            overlap: OverlapParams { x, y, falloff },
            ..with_method(Method::Overlap)
        };

        assert_parity(&mut gpu, &source(128, 100, 3), &params);
    }
}

#[test]
fn every_gpu_method_carries_alpha_and_handles_grayscale() {
    let Some(mut gpu) = gpu() else { return };
    for method in Method::ALL.into_iter().filter(|method| seamless::supports(*method)) {
        assert_parity(&mut gpu, &source(96, 80, 4), &with_method(method));
        assert_parity(&mut gpu, &source(96, 80, 1), &with_method(method));
    }
}

#[test]
fn switching_method_replaces_the_result_only_when_its_size_changes() {
    let Some(mut gpu) = gpu() else { return };
    let resident = GpuImage::upload(&gpu.device, &gpu.queue, &source(80, 64, 3)).unwrap();
    gpu.renderer.set_source(resident);

    let first = gpu.renderer.render(&with_method(Method::OffsetCrossfade)).unwrap();
    let same_size = gpu.renderer.render(&with_method(Method::Overlap)).unwrap();
    let doubled = gpu.renderer.render(&with_method(Method::Mirror)).unwrap();

    assert_eq!((first, same_size, doubled), (true, false, true));
}

#[test]
fn splat_and_missing_sources_are_refused_not_panicked() {
    let Some(mut gpu) = gpu() else { return };

    let no_source = gpu.renderer.render(&with_method(Method::OffsetCrossfade));
    let resident = GpuImage::upload(&gpu.device, &gpu.queue, &source(64, 64, 3)).unwrap();
    gpu.renderer.set_source(resident);
    let splat = gpu.renderer.render(&with_method(Method::Splat));

    assert!(no_source.is_err());
    assert!(splat.unwrap_err().to_string().contains("CPU"));
}

#[test]
fn a_mirrored_result_beyond_the_texture_limit_is_refused() {
    let Some(gpu) = gpu() else { return };
    let limit = gpu.device.limits().max_texture_dimension_2d as usize;

    let refused = seamless::fits(&gpu.device, limit, limit, Method::Mirror);
    let accepted = seamless::fits(&gpu.device, 2048, 2048, Method::Mirror);

    assert!(refused.is_err());
    assert!(accepted.is_ok());
}
