//! Decoding image files into [`Image`].

use std::io::Cursor;
use std::path::Path;

use image::{DynamicImage, ImageReader};
use seams_core::Image;
use seams_core::image::MAX_DIMENSION;

use crate::error::{IoError, Result};
use crate::format::{BitDepth, FileFormat, ImageMeta};

const METRES_PER_INCH: f32 = 0.0254;
const CENTIMETRES_PER_INCH: f32 = 2.54;

/// A decoded image with its file metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Loaded {
    pub image: Image,
    pub meta: ImageMeta,
}

/// Reads and decodes an image file.
///
/// The format is detected from the file contents, falling back to the
/// extension. Dimensions are checked before any pixels are decoded, so an
/// oversized file is rejected without allocating for it.
pub fn load(path: &Path) -> Result<Loaded> {
    let bytes = read_file(path)?;
    if bytes.starts_with(crate::psd::SIGNATURE) {
        return load_psd(path, &bytes);
    }
    let reader = guessed_reader(&bytes, path)?;
    let format = detected_format(&reader, path)?;
    check_dimensions(&bytes, path)?;

    let decoded = reader.decode().map_err(|e| decode_error(path, e))?;
    let bit_depth = bit_depth_of(&decoded);
    let mut image = to_image(decoded)?;
    image.sanitize();

    let meta = ImageMeta {
        width: image.width(),
        height: image.height(),
        channels: image.channels(),
        bit_depth,
        format,
        dpi: read_dpi(&bytes, format),
        file_size: bytes.len() as u64,
    };
    log::info!("loaded {} ({}x{}, {})", path.display(), meta.width, meta.height, bit_depth.label());
    Ok(Loaded { image, meta })
}

/// Photoshop files are read by SEAMS itself: the image crate has no decoder.
fn load_psd(path: &Path, bytes: &[u8]) -> Result<Loaded> {
    let (width, height) = crate::psd::dimensions(bytes).map_err(|e| decode_error(path, e))?;
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(seams_core::CoreError::TooLarge { width, height, max: MAX_DIMENSION }.into());
    }
    let decoded = crate::psd::decode(bytes).map_err(|e| decode_error(path, e))?;
    let mut image = decoded.image;
    image.sanitize();
    let meta = ImageMeta {
        width: image.width(),
        height: image.height(),
        channels: image.channels(),
        bit_depth: decoded.bit_depth,
        format: FileFormat::Psd,
        dpi: decoded.dpi,
        file_size: bytes.len() as u64,
    };
    log::info!(
        "loaded {} ({}x{}, {})",
        path.display(),
        meta.width,
        meta.height,
        meta.bit_depth.label()
    );
    Ok(Loaded { image, meta })
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    if !path.exists() {
        return Err(IoError::NotFound(path.to_owned()));
    }
    if !path.is_file() {
        return Err(IoError::NotAFile(path.to_owned()));
    }
    std::fs::read(path).map_err(|source| IoError::Read { path: path.to_owned(), source })
}

fn decode_error(path: &Path, error: impl std::fmt::Display) -> IoError {
    IoError::Decode { path: path.to_owned(), reason: error.to_string() }
}

fn guessed_reader<'a>(bytes: &'a [u8], path: &Path) -> Result<ImageReader<Cursor<&'a [u8]>>> {
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| decode_error(path, e))?;
    if reader.format().is_none() {
        // Formats without a signature (TGA) are only identifiable by name.
        if let Ok(format) = image::ImageFormat::from_path(path) {
            reader.set_format(format);
        }
    }
    Ok(reader)
}

fn detected_format(reader: &ImageReader<Cursor<&[u8]>>, path: &Path) -> Result<FileFormat> {
    let detected = reader.format().and_then(|format| match format {
        image::ImageFormat::Png => Some(FileFormat::Png),
        image::ImageFormat::Jpeg => Some(FileFormat::Jpeg),
        image::ImageFormat::Tiff => Some(FileFormat::Tiff),
        image::ImageFormat::Tga => Some(FileFormat::Tga),
        image::ImageFormat::OpenExr => Some(FileFormat::Exr),
        _ => None,
    });
    match detected {
        Some(format) => Ok(format),
        None => FileFormat::from_path(path),
    }
}

fn check_dimensions(bytes: &[u8], path: &Path) -> Result<()> {
    let (width, height) =
        guessed_reader(bytes, path)?.into_dimensions().map_err(|e| decode_error(path, e))?;
    let (width, height) = (width as usize, height as usize);
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(seams_core::CoreError::TooLarge { width, height, max: MAX_DIMENSION }.into());
    }
    Ok(())
}

fn bit_depth_of(image: &DynamicImage) -> BitDepth {
    match image {
        DynamicImage::ImageLuma16(_)
        | DynamicImage::ImageLumaA16(_)
        | DynamicImage::ImageRgb16(_)
        | DynamicImage::ImageRgba16(_) => BitDepth::Sixteen,
        DynamicImage::ImageRgb32F(_) | DynamicImage::ImageRgba32F(_) => BitDepth::Float,
        _ => BitDepth::Eight,
    }
}

fn normalised<T: Copy + Into<f32>>(samples: &[T], max: f32) -> Vec<f32> {
    samples.iter().map(|s| (*s).into() / max).collect()
}

/// Converts any decoded layout to gray, RGB or RGBA `f32`.
fn to_image(decoded: DynamicImage) -> Result<Image> {
    let (width, height) = (decoded.width() as usize, decoded.height() as usize);
    let (channels, data) = match decoded {
        DynamicImage::ImageLuma8(buffer) => (1, normalised(buffer.as_raw(), 255.0)),
        DynamicImage::ImageRgb8(buffer) => (3, normalised(buffer.as_raw(), 255.0)),
        DynamicImage::ImageRgba8(buffer) => (4, normalised(buffer.as_raw(), 255.0)),
        DynamicImage::ImageLuma16(buffer) => (1, normalised(buffer.as_raw(), 65535.0)),
        DynamicImage::ImageRgb16(buffer) => (3, normalised(buffer.as_raw(), 65535.0)),
        DynamicImage::ImageRgba16(buffer) => (4, normalised(buffer.as_raw(), 65535.0)),
        DynamicImage::ImageRgb32F(buffer) => (3, buffer.into_raw()),
        DynamicImage::ImageRgba32F(buffer) => (4, buffer.into_raw()),
        // Gray with alpha has no direct equivalent: expand it to RGBA.
        DynamicImage::ImageLumaA16(_) => (4, normalised(decoded.to_rgba16().as_raw(), 65535.0)),
        other => (4, normalised(other.to_rgba8().as_raw(), 255.0)),
    };
    Ok(Image::new(width, height, channels, data)?)
}

fn read_dpi(bytes: &[u8], format: FileFormat) -> Option<(f32, f32)> {
    match format {
        FileFormat::Png => png_dpi(bytes),
        FileFormat::Jpeg => jfif_dpi(bytes),
        _ => None,
    }
}

fn png_dpi(bytes: &[u8]) -> Option<(f32, f32)> {
    let reader = png::Decoder::new(Cursor::new(bytes)).read_info().ok()?;
    let dims = reader.info().pixel_dims?;
    if dims.unit != png::Unit::Meter {
        return None;
    }
    Some((dims.xppu as f32 * METRES_PER_INCH, dims.yppu as f32 * METRES_PER_INCH))
}

/// Reads the density from a JFIF APP0 segment at the start of a JPEG.
fn jfif_dpi(bytes: &[u8]) -> Option<(f32, f32)> {
    let header = bytes.get(..18)?;
    if header.get(..4)? != [0xFF, 0xD8, 0xFF, 0xE0] || header.get(6..11)? != b"JFIF\0" {
        return None;
    }
    let units = *header.get(13)?;
    let x = f32::from(u16::from_be_bytes([*header.get(14)?, *header.get(15)?]));
    let y = f32::from(u16::from_be_bytes([*header.get(16)?, *header.get(17)?]));
    match units {
        1 => Some((x, y)),
        2 => Some((x * CENTIMETRES_PER_INCH, y * CENTIMETRES_PER_INCH)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_is_reported_as_not_found() {
        let error = load(Path::new("definitely/not/here.png")).unwrap_err();

        assert!(matches!(error, IoError::NotFound(_)));
    }

    #[test]
    fn directory_is_not_a_file() {
        let dir = tempfile::tempdir().unwrap();

        assert!(matches!(load(dir.path()).unwrap_err(), IoError::NotAFile(_)));
    }

    #[test]
    fn garbage_bytes_fail_to_decode_without_panicking() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.png");
        std::fs::write(&path, b"this is not an image at all").unwrap();

        assert!(matches!(load(&path).unwrap_err(), IoError::Decode { .. }));
    }

    #[test]
    fn truncated_png_fails_to_decode_without_panicking() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cut.png");
        let mut bytes = Vec::new();
        image::RgbImage::from_pixel(64, 64, image::Rgb([10, 20, 30]))
            .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
            .unwrap();
        std::fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();

        assert!(load(&path).is_err());
    }

    #[test]
    fn jfif_density_is_parsed_in_both_units() {
        let mut header = vec![0xFF, 0xD8, 0xFF, 0xE0, 0, 16];
        header.extend_from_slice(b"JFIF\0");
        header.extend_from_slice(&[1, 2, 1, 0x01, 0x2C, 0x01, 0x2C]);

        assert_eq!(jfif_dpi(&header), Some((300.0, 300.0)));
        header[13] = 0;
        assert_eq!(jfif_dpi(&header), None);
        assert_eq!(jfif_dpi(&header[..10]), None);
    }
}
