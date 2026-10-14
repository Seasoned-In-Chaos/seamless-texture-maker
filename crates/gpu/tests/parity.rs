//! GPU against CPU: every map, with every setting that changes its shape,
//! must match the reference in `seams_core::pbr` within one 8-bit step.
//! The tests pass without checking anything when the machine has no GPU
//! adapter, and say so on stderr.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{Mutex, MutexGuard, PoisonError};

use seams_core::Image;
use seams_core::pbr::{
    HeightField, HeightSource, MapKind, MaterialParams, NormalFilter, NormalFormat, generate_map,
};
use seams_gpu::{GpuField, MapRenderer};

/// Largest difference allowed, in 8-bit steps.
const TOLERANCE: i32 = 1;
const SIZE: (usize, usize) = (96, 80);

/// Several wgpu instances at once crash some drivers, so each test holds
/// this lock until its device is dropped.
static GPU_LOCK: Mutex<()> = Mutex::new(());

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: MapRenderer,
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
    let renderer = match MapRenderer::new(&device, &queue) {
        Ok(renderer) => renderer,
        Err(error) => {
            eprintln!("skipping GPU parity tests: {error}");
            return None;
        }
    };
    Some(Gpu { device, queue, renderer, _lock: lock })
}

/// A texture with detail at several scales, values across the whole range
/// and, with four channels, a varying alpha.
fn source(channels: usize) -> Image {
    Image::from_fn(SIZE.0, SIZE.1, channels, |x, y, c| {
        let (fx, fy) = (x as f32 / SIZE.0 as f32, y as f32 / SIZE.1 as f32);
        if c == 3 {
            return 0.25 + 0.75 * fx;
        }
        let waves = (fx * 19.0 + c as f32).sin() * (fy * 13.0).cos();
        let grain = ((x * 7 + y * 13 + c * 5) % 11) as f32 / 11.0;
        (0.5 + 0.35 * waves + 0.15 * (grain - 0.5)).clamp(0.0, 1.0)
    })
    .unwrap()
}

fn to_byte(value: f32) -> i32 {
    (value.clamp(0.0, 1.0) * 255.0 + 0.5) as i32
}

/// The reference map as the premultiplied RGBA8 the GPU targets hold.
fn reference(kind: MapKind, field: &HeightField, params: &MaterialParams) -> Vec<i32> {
    let map = generate_map(kind, field, params).unwrap();
    let mut out = Vec::with_capacity(map.width() * map.height() * 4);
    for pixel in map.data().chunks(map.channels()) {
        let (rgb, alpha) = match pixel {
            [r, g, b, a] => ([*r, *g, *b], *a),
            [r, g, b] => ([*r, *g, *b], 1.0),
            [gray] => ([*gray; 3], 1.0),
            _ => panic!("unexpected channel count"),
        };
        let alpha_byte = to_byte(alpha);
        let coverage = alpha_byte as f32 / 255.0;
        out.extend(rgb.map(|v| to_byte(v.clamp(0.0, 1.0) * coverage)));
        out.push(alpha_byte);
    }
    out
}

/// Renders `kinds` on both sides and asserts they agree.
fn assert_parity(gpu: &mut Gpu, image: &Image, params: &MaterialParams, kinds: &[MapKind]) {
    let field = HeightField::from_image(image, params.height_source);
    let resident = GpuField::upload(&gpu.device, &gpu.queue, &field).unwrap();
    gpu.renderer.set_field(resident);
    for kind in kinds {
        gpu.renderer.render(*kind, params).unwrap();
        let actual = gpu.renderer.read_back(*kind).unwrap();
        let expected = reference(*kind, &field, params);

        assert_eq!(actual.len(), expected.len(), "{kind:?}: size");
        let worst = actual
            .iter()
            .zip(&expected)
            .enumerate()
            .map(|(i, (a, e))| ((i32::from(*a) - e).abs(), i))
            .max()
            .unwrap();
        let (x, y) = ((worst.1 / 4) % SIZE.0, (worst.1 / 4) / SIZE.0);
        assert!(
            worst.0 <= TOLERANCE,
            "{kind:?} differs by {} at ({x}, {y}) channel {} with {params:?}",
            worst.0,
            worst.1 % 4
        );
    }
}

#[test]
fn every_map_matches_with_default_settings() {
    let Some(mut gpu) = gpu() else { return };

    assert_parity(&mut gpu, &source(3), &MaterialParams::default(), &MapKind::ALL);
}

#[test]
fn every_map_carries_the_source_alpha() {
    let Some(mut gpu) = gpu() else { return };

    assert_parity(&mut gpu, &source(4), &MaterialParams::default(), &MapKind::ALL);
}

#[test]
fn a_grayscale_source_matches() {
    let Some(mut gpu) = gpu() else { return };

    assert_parity(&mut gpu, &source(1), &MaterialParams::default(), &MapKind::ALL);
}

#[test]
fn every_height_source_matches() {
    let Some(mut gpu) = gpu() else { return };
    for height_source in HeightSource::ALL {
        let params = MaterialParams { height_source, ..MaterialParams::default() };

        assert_parity(&mut gpu, &source(4), &params, &MapKind::ALL);
    }
}

#[test]
fn every_normal_filter_matches_wrapped_and_clamped() {
    let Some(mut gpu) = gpu() else { return };
    for filter in NormalFilter::ALL {
        for wrap in [true, false] {
            let mut params = MaterialParams::default();
            params.normal.filter = filter;
            params.normal.wrap = wrap;

            assert_parity(&mut gpu, &source(3), &params, &[MapKind::Normal]);
        }
    }
}

#[test]
fn normal_inversions_formats_and_limits_match() {
    let Some(mut gpu) = gpu() else { return };
    let mut variants = vec![MaterialParams::default(); 6];
    variants[0].normal.invert_x = true;
    variants[1].normal.invert_y = true;
    variants[2].normal.invert_height = true;
    variants[3].normal.format = NormalFormat::DirectX;
    variants[4].normal.scale = 8.0;
    variants[4].normal.min_z = 0.8;
    variants[5].normal.scale = 0.1;
    variants[5].normal.filter = NormalFilter::Sobel5;

    for params in &variants {
        assert_parity(&mut gpu, &source(3), params, &[MapKind::Normal]);
    }
}

#[test]
fn roughness_settings_match() {
    let Some(mut gpu) = gpu() else { return };
    for (intensity, contrast, invert) in [(0.2, 0.0, false), (1.0, 1.0, false), (0.6, 0.4, true)] {
        let mut params = MaterialParams::default();
        params.roughness.intensity = intensity;
        params.roughness.contrast = contrast;
        params.roughness.invert = invert;

        assert_parity(&mut gpu, &source(3), &params, &[MapKind::Roughness]);
    }
}

#[test]
fn blurred_and_sharpened_maps_match() {
    let Some(mut gpu) = gpu() else { return };
    let blurred = [MapKind::AmbientOcclusion, MapKind::Displacement];
    for blur_sharp in [-32.0, -6.0, -0.5, 0.5, 9.0, 32.0] {
        let mut params = MaterialParams::default();
        params.ao.blur_sharp = blur_sharp;
        params.ao.mean = 0.6;
        params.ao.range = 0.5;
        params.displacement.blur_sharp = blur_sharp;

        assert_parity(&mut gpu, &source(4), &params, &blurred);
    }
}

#[test]
fn ambient_occlusion_and_displacement_settings_match() {
    let Some(mut gpu) = gpu() else { return };
    let mut variants = vec![MaterialParams::default(); 4];
    variants[0].ao.strength = 1.0;
    variants[0].ao.mean = 0.3;
    variants[0].ao.range = 0.2;
    variants[1].ao.invert = true;
    variants[1].ao.blur_sharp = -4.0;
    variants[2].displacement.contrast = 0.6;
    variants[2].displacement.invert = true;
    variants[3].displacement.contrast = -1.0;

    for params in &variants {
        let kinds = [MapKind::AmbientOcclusion, MapKind::Displacement];
        assert_parity(&mut gpu, &source(3), params, &kinds);
    }
}

#[test]
fn hard_and_soft_opacity_match() {
    let Some(mut gpu) = gpu() else { return };
    for (threshold, softness) in [(1.0, 0.0), (0.5, 0.0), (0.5, 0.3), (0.2, 1.0), (0.0, 0.0)] {
        let mut params = MaterialParams::default();
        params.opacity.threshold = threshold;
        params.opacity.softness = softness;

        assert_parity(&mut gpu, &source(3), &params, &[MapKind::Opacity]);
    }
}

#[test]
fn a_new_field_of_another_size_replaces_the_maps() {
    let Some(mut gpu) = gpu() else { return };
    let params = MaterialParams::default();
    assert_parity(&mut gpu, &source(3), &params, &[MapKind::Normal]);
    let small = Image::from_fn(64, 64, 3, |x, y, _| ((x ^ y) % 16) as f32 / 15.0).unwrap();
    let field = HeightField::from_image(&small, params.height_source);

    let resident = GpuField::upload(&gpu.device, &gpu.queue, &field).unwrap();
    let resized = gpu.renderer.set_field(resident);
    gpu.renderer.render(MapKind::Roughness, &params).unwrap();

    assert!(resized);
    assert_eq!(gpu.renderer.size(), Some((64, 64)));
    assert_eq!(gpu.renderer.read_back(MapKind::Roughness).unwrap().len(), 64 * 64 * 4);
}

#[test]
fn rendering_without_a_field_is_reported_not_panicked() {
    let Some(mut gpu) = gpu() else { return };

    let result = gpu.renderer.render(MapKind::Normal, &MaterialParams::default());

    assert!(result.is_err());
    assert!(gpu.renderer.target(MapKind::Normal).is_none());
}

#[test]
fn an_image_beyond_the_memory_budget_is_refused() {
    let Some(gpu) = gpu() else { return };

    let refused = seams_gpu::fits(&gpu.device, 8192, 8192, false);
    let accepted = seams_gpu::fits(&gpu.device, 4096, 4096, true);

    assert!(refused.is_err());
    assert!(accepted.is_ok());
}
