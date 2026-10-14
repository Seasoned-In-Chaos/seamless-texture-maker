//! Encoding [`Image`] to files, written atomically.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use image::codecs::jpeg::{JpegEncoder, PixelDensity};
use image::{ExtendedColorType, ImageEncoder};
use seams_core::Image;

use crate::error::{IoError, Result};
use crate::format::{BitDepth, FileFormat};

const INCHES_PER_METRE: f32 = 39.370_08;
const TEMP_PREFIX: &str = ".seams-";

/// How an image should be written.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SaveOptions {
    /// The format to write; `None` takes it from the path's extension.
    pub format: Option<FileFormat>,
    /// Requested sample depth. Formats that cannot hold it fall back to 8-bit.
    pub bit_depth: BitDepth,
    /// JPEG quality, `1..=100`.
    pub jpeg_quality: u8,
    /// Resolution to record, for formats that carry it (PNG, JPEG).
    pub dpi: Option<(f32, f32)>,
}

impl Default for SaveOptions {
    fn default() -> Self {
        Self { format: None, bit_depth: BitDepth::Eight, jpeg_quality: 95, dpi: None }
    }
}

/// Writes `image` to `path`, creating parent folders as needed.
///
/// The file is encoded in memory, written to a temporary file beside the
/// target and renamed into place, so a failure never leaves a partial file.
pub fn save(image: &Image, path: &Path, options: &SaveOptions) -> Result<()> {
    let format = match options.format {
        Some(format) => format,
        None => FileFormat::from_path(path)?,
    };
    let bytes = encode(image, format, options)?;
    write_atomically(path, &bytes)?;
    log::info!("saved {} ({}, {} bytes)", path.display(), format.label(), bytes.len());
    Ok(())
}

fn encode_error(format: FileFormat, error: impl std::fmt::Display) -> IoError {
    IoError::Encode { format: format.label(), reason: error.to_string() }
}

fn encode(image: &Image, format: FileFormat, options: &SaveOptions) -> Result<Vec<u8>> {
    let sixteen = options.bit_depth != BitDepth::Eight && format.supports_sixteen_bit();
    match format {
        FileFormat::Png => encode_png(image, sixteen, options.dpi),
        FileFormat::Jpeg => encode_jpeg(image, options),
        FileFormat::Exr => encode_exr(image),
        FileFormat::Tiff | FileFormat::Tga => encode_generic(image, format, sixteen),
    }
}

fn to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn to_u16(value: f32) -> u16 {
    (value.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16
}

/// Interleaved samples as bytes; 16-bit samples in the given byte order.
fn sample_bytes(image: &Image, sixteen: bool, big_endian: bool) -> Vec<u8> {
    if !sixteen {
        return image.data().iter().map(|v| to_u8(*v)).collect();
    }
    let split = if big_endian { u16::to_be_bytes } else { u16::to_ne_bytes };
    image.data().iter().flat_map(|v| split(to_u16(*v))).collect()
}

/// Drops the alpha channel, for formats without one.
fn without_alpha(image: &Image) -> Result<Image> {
    if !image.has_alpha() {
        return Ok(image.clone());
    }
    let data = image.data().chunks(4).flat_map(|p| p.iter().take(3).copied()).collect();
    Ok(Image::new(image.width(), image.height(), 3, data)?)
}

fn dimensions(image: &Image, format: FileFormat) -> Result<(u32, u32)> {
    let convert = |v: usize| u32::try_from(v).map_err(|e| encode_error(format, e));
    Ok((convert(image.width())?, convert(image.height())?))
}

fn encode_png(image: &Image, sixteen: bool, dpi: Option<(f32, f32)>) -> Result<Vec<u8>> {
    let (width, height) = dimensions(image, FileFormat::Png)?;
    let failed = |e: png::EncodingError| encode_error(FileFormat::Png, e);
    let mut bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut bytes, width, height);
    encoder.set_color(match image.channels() {
        1 => png::ColorType::Grayscale,
        3 => png::ColorType::Rgb,
        _ => png::ColorType::Rgba,
    });
    encoder.set_depth(if sixteen { png::BitDepth::Sixteen } else { png::BitDepth::Eight });
    if let Some((x, y)) = dpi {
        encoder.set_pixel_dims(Some(png::PixelDimensions {
            xppu: (x * INCHES_PER_METRE).round() as u32,
            yppu: (y * INCHES_PER_METRE).round() as u32,
            unit: png::Unit::Meter,
        }));
    }
    let mut writer = encoder.write_header().map_err(failed)?;
    writer.write_image_data(&sample_bytes(image, sixteen, true)).map_err(failed)?;
    writer.finish().map_err(failed)?;
    Ok(bytes)
}

fn encode_jpeg(image: &Image, options: &SaveOptions) -> Result<Vec<u8>> {
    let opaque = without_alpha(image)?;
    let (width, height) = dimensions(&opaque, FileFormat::Jpeg)?;
    let color =
        if opaque.channels() == 1 { ExtendedColorType::L8 } else { ExtendedColorType::Rgb8 };
    let mut bytes = Vec::new();
    let mut encoder = JpegEncoder::new_with_quality(&mut bytes, options.jpeg_quality.clamp(1, 100));
    if let Some((x, _)) = options.dpi {
        encoder.set_pixel_density(PixelDensity::dpi(x.round().clamp(1.0, 65535.0) as u16));
    }
    encoder
        .encode(&sample_bytes(&opaque, false, false), width, height, color)
        .map_err(|e| encode_error(FileFormat::Jpeg, e))?;
    Ok(bytes)
}

fn encode_exr(image: &Image) -> Result<Vec<u8>> {
    let (width, height) = dimensions(image, FileFormat::Exr)?;
    let (color, data): (_, Vec<f32>) = match image.channels() {
        1 => (ExtendedColorType::Rgb32F, image.data().iter().flat_map(|v| [*v; 3]).collect()),
        3 => (ExtendedColorType::Rgb32F, image.data().to_vec()),
        _ => (ExtendedColorType::Rgba32F, image.data().to_vec()),
    };
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let mut cursor = Cursor::new(Vec::new());
    image::codecs::openexr::OpenExrEncoder::new(&mut cursor)
        .write_image(&raw, width, height, color)
        .map_err(|e| encode_error(FileFormat::Exr, e))?;
    Ok(cursor.into_inner())
}

fn color_type(channels: usize, sixteen: bool) -> ExtendedColorType {
    match (channels, sixteen) {
        (1, false) => ExtendedColorType::L8,
        (3, false) => ExtendedColorType::Rgb8,
        (_, false) => ExtendedColorType::Rgba8,
        (1, true) => ExtendedColorType::L16,
        (3, true) => ExtendedColorType::Rgb16,
        (_, true) => ExtendedColorType::Rgba16,
    }
}

fn encode_generic(image: &Image, format: FileFormat, sixteen: bool) -> Result<Vec<u8>> {
    let (width, height) = dimensions(image, format)?;
    let raw = sample_bytes(image, sixteen, false);
    let color = color_type(image.channels(), sixteen);
    let mut cursor = Cursor::new(Vec::new());
    let written = match format {
        FileFormat::Tga => {
            image::codecs::tga::TgaEncoder::new(&mut cursor).write_image(&raw, width, height, color)
        }
        _ => image::codecs::tiff::TiffEncoder::new(&mut cursor)
            .write_image(&raw, width, height, color),
    };
    written.map_err(|e| encode_error(format, e))?;
    Ok(cursor.into_inner())
}

fn temp_path_beside(path: &Path) -> PathBuf {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let name = format!("{TEMP_PREFIX}{}-{nanos}.tmp", std::process::id());
    path.with_file_name(name)
}

fn write_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let failed = |source| IoError::Write { path: path.to_owned(), source };
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(failed)?;
    }
    let temp = temp_path_beside(path);
    let result = std::fs::write(&temp, bytes).and_then(|()| std::fs::rename(&temp, path));
    if result.is_err() {
        // Best effort: the original error is what the caller needs to see.
        let _ = std::fs::remove_file(&temp);
    }
    result.map_err(failed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::load;

    fn gradient(channels: usize) -> Image {
        Image::from_fn(70, 66, channels, |x, y, c| ((x * 3 + y * 5 + c * 40) % 256) as f32 / 255.0)
            .unwrap()
    }

    fn worst_difference(a: &Image, b: &Image) -> f32 {
        a.data().iter().zip(b.data()).map(|(p, q)| (p - q).abs()).fold(0.0, f32::max)
    }

    #[test]
    fn lossless_formats_round_trip_eight_bit_pixels() {
        let dir = tempfile::tempdir().unwrap();
        for format in [FileFormat::Png, FileFormat::Tiff, FileFormat::Tga] {
            for channels in [1, 3, 4] {
                let image = gradient(channels);
                let path = dir.path().join(format!("rt{channels}.{}", format.extension()));

                save(&image, &path, &SaveOptions::default()).unwrap();
                let loaded = load(&path).unwrap();

                assert_eq!(loaded.meta.format, format);
                assert_eq!(loaded.image.channels(), channels, "{format:?}");
                assert!(worst_difference(&loaded.image, &image) < 1e-6, "{format:?} {channels}");
            }
        }
    }

    #[test]
    fn sixteen_bit_png_and_tiff_keep_fine_precision() {
        let dir = tempfile::tempdir().unwrap();
        let image = Image::from_fn(64, 64, 3, |x, y, _| (x * 64 + y) as f32 / 4096.0).unwrap();
        let options = SaveOptions { bit_depth: BitDepth::Sixteen, ..SaveOptions::default() };

        for format in [FileFormat::Png, FileFormat::Tiff] {
            let path = dir.path().join(format!("deep.{}", format.extension()));
            save(&image, &path, &options).unwrap();
            let loaded = load(&path).unwrap();

            assert_eq!(loaded.meta.bit_depth, BitDepth::Sixteen);
            assert!(worst_difference(&loaded.image, &image) < 1.0 / 60000.0, "{format:?}");
        }
    }

    #[test]
    fn sixteen_bit_request_falls_back_for_formats_without_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fallback.tga");
        let options = SaveOptions { bit_depth: BitDepth::Sixteen, ..SaveOptions::default() };

        save(&gradient(3), &path, &options).unwrap();

        assert_eq!(load(&path).unwrap().meta.bit_depth, BitDepth::Eight);
    }

    #[test]
    fn jpeg_drops_alpha_and_stays_close() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("photo.jpg");
        let image = Image::filled(64, 64, 4, 0.5).unwrap();

        save(&image, &path, &SaveOptions::default()).unwrap();
        let loaded = load(&path).unwrap();

        assert_eq!(loaded.image.channels(), 3);
        assert!(loaded.image.data().iter().all(|v| (v - 0.5).abs() < 0.02));
    }

    #[test]
    fn exr_round_trips_float_pixels() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("float.exr");
        let image = Image::from_fn(64, 64, 3, |x, y, c| (x + y + c) as f32 / 200.0).unwrap();

        save(&image, &path, &SaveOptions::default()).unwrap();
        let loaded = load(&path).unwrap();

        assert_eq!(loaded.meta.bit_depth, BitDepth::Float);
        assert!(worst_difference(&loaded.image, &image) < 1e-6);
    }

    #[test]
    fn dpi_survives_a_png_and_jpeg_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let options = SaveOptions { dpi: Some((300.0, 300.0)), ..SaveOptions::default() };

        for name in ["print.png", "print.jpg"] {
            let path = dir.path().join(name);
            save(&gradient(3), &path, &options).unwrap();
            let (x, y) = load(&path).unwrap().meta.dpi.unwrap();

            assert!((x - 300.0).abs() < 0.5 && (y - 300.0).abs() < 0.5, "{name}: {x}x{y}");
        }
    }

    #[test]
    fn save_creates_missing_folders_and_leaves_no_temp_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/deeper/out.png");

        save(&gradient(3), &path, &SaveOptions::default()).unwrap();
        let names: Vec<String> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();

        assert_eq!(names, vec!["out.png"]);
    }

    #[test]
    fn save_overwrites_an_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("same.png");
        save(&Image::filled(64, 64, 3, 0.0).unwrap(), &path, &SaveOptions::default()).unwrap();

        save(&Image::filled(64, 64, 3, 1.0).unwrap(), &path, &SaveOptions::default()).unwrap();

        assert!(load(&path).unwrap().image.data().iter().all(|v| *v == 1.0));
    }

    #[test]
    fn unicode_paths_are_supported() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("текстура_石_🪨.png");

        save(&gradient(3), &path, &SaveOptions::default()).unwrap();

        assert!(load(&path).is_ok());
    }

    #[test]
    fn unsupported_extension_is_rejected_before_writing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.webp");

        let error = save(&gradient(3), &path, &SaveOptions::default()).unwrap_err();

        assert!(matches!(error, IoError::UnsupportedFormat(_)));
        assert!(!path.exists());
    }

    #[test]
    fn explicit_format_overrides_the_extension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("actually_png.dat");
        let options = SaveOptions { format: Some(FileFormat::Png), ..SaveOptions::default() };

        save(&gradient(3), &path, &options).unwrap();

        assert_eq!(load(&path).unwrap().meta.format, FileFormat::Png);
    }
}
