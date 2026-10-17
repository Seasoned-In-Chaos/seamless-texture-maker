//! Adobe Photoshop PSD and PSB files.
//!
//! Reading takes the merged (flattened) image Photoshop stores at the end of
//! every file. When the file was saved without "Maximize compatibility" that
//! image is a white placeholder, so the visible pixel layers are composited
//! instead (normal blending, layer opacity and group visibility; blend modes,
//! masks and effects are not applied). Writing produces a flat file, or a
//! single-layer file when the image has alpha so the transparency survives.
//!
//! The codecs (PackBits RLE, ZIP, ZIP with prediction) and the section layout
//! are ported from the `photocraft-psd` crate (MIT OR Apache-2.0), which
//! follows Adobe's public Photoshop File Formats Specification.

use std::io::Read;

use seams_core::Image;

use crate::format::BitDepth;

/// The first four bytes of every PSD and PSB file.
pub(crate) const SIGNATURE: &[u8; 4] = b"8BPS";

const RESOURCE_RESOLUTION: u16 = 1005;
const RESOURCE_VERSION_INFO: u16 = 1057;
const CHANNEL_TRANSPARENCY: i16 = -1;
/// Tagged blocks whose length is 8 bytes in a PSB file.
const PSB_LONG_KEYS: [&[u8; 4]; 13] = [
    b"LMsk", b"Lr16", b"Lr32", b"Layr", b"Mt16", b"Mt32", b"Mtrn", b"Alph", b"FMsk", b"lnk2",
    b"FEid", b"FXid", b"PxSD",
];

type Result<T> = std::result::Result<T, String>;

/// A decoded file: the flattened image and what is known about it.
pub(crate) struct Decoded {
    pub image: Image,
    pub bit_depth: BitDepth,
    pub dpi: Option<(f32, f32)>,
}

// ============================================================================ reading

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|end| *end <= self.data.len());
        let Some(end) = end else {
            return Err(format!("file ends early (needed {n} bytes at offset {})", self.pos));
        };
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.bytes(N)?);
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.array::<1>()?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_be_bytes(self.array()?))
    }

    fn i16(&mut self) -> Result<i16> {
        Ok(i16::from_be_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.array()?))
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_be_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_be_bytes(self.array()?))
    }

    /// A section length: 4 bytes in a PSD, 8 in a PSB.
    fn length(&mut self, psb: bool) -> Result<usize> {
        let value = if psb { self.u64()? } else { u64::from(self.u32()?) };
        usize::try_from(value).map_err(|_| "section length exceeds memory".to_owned())
    }

    fn section(&mut self, n: usize) -> Result<Reader<'a>> {
        Ok(Reader::new(self.bytes(n)?))
    }

    fn peek(&self, n: usize) -> Option<&'a [u8]> {
        self.data.get(self.pos..self.pos.checked_add(n)?)
    }
}

#[derive(Debug, Clone, Copy)]
struct Header {
    psb: bool,
    channels: usize,
    width: usize,
    height: usize,
    depth: u16,
    mode: u16,
}

mod mode {
    pub const BITMAP: u16 = 0;
    pub const GRAYSCALE: u16 = 1;
    pub const INDEXED: u16 = 2;
    pub const RGB: u16 = 3;
    pub const CMYK: u16 = 4;
    pub const DUOTONE: u16 = 8;
}

impl Header {
    fn read(r: &mut Reader<'_>) -> Result<Self> {
        if r.bytes(4)? != SIGNATURE {
            return Err("not a Photoshop file".to_owned());
        }
        let psb = match r.u16()? {
            1 => false,
            2 => true,
            v => return Err(format!("unknown Photoshop file version {v}")),
        };
        r.bytes(6)?;
        let channels = usize::from(r.u16()?);
        let height = r.u32()? as usize;
        let width = r.u32()? as usize;
        let depth = r.u16()?;
        let mode = r.u16()?;
        if !matches!(depth, 1 | 8 | 16 | 32) {
            return Err(format!("unsupported bit depth {depth}"));
        }
        if channels == 0 || width == 0 || height == 0 {
            return Err("the image is empty".to_owned());
        }
        Ok(Self { psb, channels, width, height, depth, mode })
    }

    /// Channels that hold colour in this mode.
    fn color_channels(&self) -> Result<usize> {
        match self.mode {
            mode::RGB => Ok(3),
            mode::CMYK => Ok(4),
            mode::GRAYSCALE | mode::INDEXED | mode::BITMAP | mode::DUOTONE => Ok(1),
            7 => {
                Err("multichannel images are not supported; convert to RGB in Photoshop".to_owned())
            }
            9 => Err("Lab images are not supported; convert to RGB in Photoshop".to_owned()),
            m => Err(format!("unsupported colour mode {m}")),
        }
    }
}

/// Bytes per row of one plane.
fn row_bytes(width: usize, depth: u16) -> usize {
    if depth == 1 { width.div_ceil(8) } else { width * usize::from(depth / 8) }
}

/// PackBits run-length coding as used by Photoshop RLE.
mod packbits {
    use super::Result;

    /// Encodes one row, appending to `out`. Never emits the `-128` no-op.
    pub fn encode(src: &[u8], out: &mut Vec<u8>) {
        let n = src.len();
        let mut i = 0;
        while i < n {
            let mut run = 1;
            while i + run < n && run < 128 && src[i + run] == src[i] {
                run += 1;
            }
            if run >= 2 {
                out.push((1i16 - run as i16) as i8 as u8);
                out.push(src[i]);
                i += run;
                continue;
            }
            // Literal: extend until a run of three begins or 128 bytes.
            let start = i;
            let mut len = 0;
            while i < n && len < 128 {
                if i + 2 < n && src[i] == src[i + 1] && src[i] == src[i + 2] {
                    break;
                }
                i += 1;
                len += 1;
            }
            out.push((len - 1) as u8);
            out.extend_from_slice(&src[start..start + len]);
        }
    }

    /// Decodes `src` into exactly `expected` bytes appended to `out`.
    pub fn decode_into(src: &[u8], expected: usize, out: &mut Vec<u8>) -> Result<()> {
        let target = out.len() + expected;
        let mut i = 0;
        while out.len() < target {
            let Some(&header) = src.get(i) else {
                return Err("RLE row ends early".to_owned());
            };
            i += 1;
            let header = header as i8;
            if header >= 0 {
                let len = header as usize + 1;
                let Some(literal) = src.get(i..i + len) else {
                    return Err("RLE literal is truncated".to_owned());
                };
                if out.len() + len > target {
                    return Err("RLE literal overflows its row".to_owned());
                }
                out.extend_from_slice(literal);
                i += len;
            } else if header != -128 {
                let len = (1 - isize::from(header)) as usize;
                let Some(&byte) = src.get(i) else {
                    return Err("RLE run is truncated".to_owned());
                };
                i += 1;
                if out.len() + len > target {
                    return Err("RLE run overflows its row".to_owned());
                }
                out.extend(std::iter::repeat_n(byte, len));
            }
        }
        Ok(())
    }
}

/// Decodes `planes` planes of `width x height` samples (after the 2-byte
/// compression field) into planar big-endian bytes.
fn decode_planes(
    compression: u16,
    data: &[u8],
    planes: usize,
    width: usize,
    height: usize,
    depth: u16,
    psb: bool,
) -> Result<Vec<u8>> {
    let rb = row_bytes(width, depth);
    let rows = planes * height;
    let total = rows * rb;
    match compression {
        0 => data
            .get(..total)
            .map(<[u8]>::to_vec)
            .ok_or_else(|| "raw image data is truncated".to_owned()),
        1 => {
            let mut r = Reader::new(data);
            let mut counts = Vec::with_capacity(rows);
            for _ in 0..rows {
                counts.push(if psb { r.u32()? as usize } else { usize::from(r.u16()?) });
            }
            // PackBits expands at most 64x: reject declared sizes the data cannot hold.
            if total > r.remaining().saturating_mul(64) {
                return Err("RLE data is too short for the image size".to_owned());
            }
            let mut out = Vec::new();
            out.try_reserve_exact(total)
                .map_err(|_| "not enough memory for the image".to_owned())?;
            for count in counts {
                packbits::decode_into(r.bytes(count)?, rb, &mut out)?;
            }
            Ok(out)
        }
        2 => zip_decompress(data, total),
        3 => {
            let mut out = zip_decompress(data, total)?;
            unpredict(&mut out, width, rb, depth)?;
            Ok(out)
        }
        c => Err(format!("unknown compression method {c}")),
    }
}

fn zip_decompress(data: &[u8], expected: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.try_reserve_exact(expected).map_err(|_| "not enough memory for the image".to_owned())?;
    flate2::read::ZlibDecoder::new(data)
        .take(expected as u64)
        .read_to_end(&mut out)
        .map_err(|e| format!("ZIP data is corrupt: {e}"))?;
    if out.len() < expected {
        return Err(format!("ZIP data produced {} of {expected} bytes", out.len()));
    }
    Ok(out)
}

/// Reverses Photoshop's horizontal delta prediction, row by row.
fn unpredict(buf: &mut [u8], width: usize, rb: usize, depth: u16) -> Result<()> {
    if rb == 0 {
        return Ok(());
    }
    let mut scratch = Vec::new();
    for row in buf.chunks_exact_mut(rb) {
        match depth {
            1 | 8 => {
                for i in 1..row.len() {
                    row[i] = row[i].wrapping_add(row[i - 1]);
                }
            }
            16 => {
                for i in 1..width {
                    let a = u16::from_be_bytes([row[2 * i], row[2 * i + 1]]);
                    let b = u16::from_be_bytes([row[2 * i - 2], row[2 * i - 1]]);
                    row[2 * i..2 * i + 2].copy_from_slice(&a.wrapping_add(b).to_be_bytes());
                }
            }
            32 => {
                // Each row is split into four byte planes, most significant first.
                for i in 1..rb {
                    row[i] = row[i].wrapping_add(row[i - 1]);
                }
                scratch.clear();
                scratch.extend_from_slice(row);
                for i in 0..width {
                    for k in 0..4 {
                        row[4 * i + k] = scratch[k * width + i];
                    }
                }
            }
            d => return Err(format!("prediction for depth {d}")),
        }
    }
    Ok(())
}

/// One plane of samples as `0..=1` floats (32-bit floats pass through).
fn plane_to_f32(bytes: &[u8], depth: u16, width: usize, height: usize) -> Vec<f32> {
    match depth {
        1 => {
            let rb = width.div_ceil(8);
            let mut out = Vec::with_capacity(width * height);
            for y in 0..height {
                for x in 0..width {
                    let byte = bytes.get(y * rb + x / 8).copied().unwrap_or(0);
                    // A set bit is black.
                    out.push(if (byte >> (7 - x % 8)) & 1 == 1 { 0.0 } else { 1.0 });
                }
            }
            out
        }
        8 => bytes.iter().map(|b| f32::from(*b) / 255.0).collect(),
        16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from(u16::from_be_bytes(*c)) / 65535.0)
            .collect(),
        _ => bytes.as_chunks::<4>().0.iter().map(|c| f32::from_be_bytes(*c)).collect(),
    }
}

/// Colour planes in the file's mode turned into RGB (or gray) planes.
fn to_rgb(header: &Header, color: Vec<Vec<f32>>, palette: &[u8]) -> Vec<Vec<f32>> {
    match header.mode {
        mode::RGB => color,
        mode::CMYK => {
            // Stored inverted (1 = no ink): a naive conversion, no colour management.
            let k = &color[3];
            (0..3).map(|c| color[c].iter().zip(k).map(|(v, k)| v * k).collect()).collect()
        }
        mode::INDEXED => {
            let index = &color[0];
            (0..3)
                .map(|c| {
                    index
                        .iter()
                        .map(|v| {
                            let i = (v * 255.0).round() as usize;
                            f32::from(palette.get(c * 256 + i).copied().unwrap_or(0)) / 255.0
                        })
                        .collect()
                })
                .collect()
        }
        _ => color,
    }
}

/// Interleaves gray or RGB planes, with optional alpha, into an [`Image`].
fn interleave(
    width: usize,
    height: usize,
    mut color: Vec<Vec<f32>>,
    alpha: Option<Vec<f32>>,
) -> Result<Image> {
    let n = width * height;
    if color.len() == 1 && alpha.is_some() {
        // Gray with alpha has no direct equivalent: expand it to RGBA.
        let gray = color.remove(0);
        color = vec![gray.clone(), gray.clone(), gray];
    }
    let channels = color.len() + usize::from(alpha.is_some());
    let mut data = Vec::with_capacity(n * channels);
    for i in 0..n {
        for plane in &color {
            data.push(plane[i]);
        }
        if let Some(alpha) = &alpha {
            data.push(alpha[i]);
        }
    }
    Image::new(width, height, channels, data).map_err(|e| e.to_string())
}

#[derive(Debug, Default)]
struct Resources {
    dpi: Option<(f32, f32)>,
    has_real_merged_data: Option<bool>,
}

fn read_resources(mut r: Reader<'_>) -> Result<Resources> {
    let mut out = Resources::default();
    while r.remaining() >= 12 {
        r.bytes(4)?; // 8BIM (or another signature; the layout is the same)
        let id = r.u16()?;
        let name_len = usize::from(r.u8()?);
        // The Pascal name, length byte included, is padded to an even length.
        r.bytes(name_len + usize::from(name_len % 2 == 0))?;
        let size = r.u32()? as usize;
        let data = r.bytes(size)?;
        if size % 2 == 1 && r.remaining() > 0 {
            r.bytes(1)?;
        }
        let mut d = Reader::new(data);
        match id {
            RESOURCE_RESOLUTION => {
                let fixed = |v: u32| v as f32 / 65536.0;
                let (h, h_unit) = (d.u32().map(fixed), d.u16());
                d.u16().ok();
                let (v, v_unit) = (d.u32().map(fixed), d.u16());
                if let (Ok(h), Ok(hu), Ok(v), Ok(vu)) = (h, h_unit, v, v_unit) {
                    // Unit 2 is pixels per centimetre.
                    let inch = |value: f32, unit: u16| if unit == 2 { value * 2.54 } else { value };
                    out.dpi =
                        Some((inch(h, hu), inch(v, vu))).filter(|(x, y)| *x > 0.0 && *y > 0.0);
                }
            }
            RESOURCE_VERSION_INFO if d.u32().is_ok() => {
                out.has_real_merged_data = d.u8().ok().map(|v| v != 0);
            }
            _ => {}
        }
    }
    Ok(out)
}

#[derive(Debug)]
struct LayerRecord {
    top: i32,
    left: i32,
    bottom: i32,
    right: i32,
    opacity: f32,
    hidden: bool,
    /// The `lsct` section type: 1 or 2 opens a group, 3 closes one.
    divider: Option<u32>,
    /// Channel id and encoded data (compression field first).
    channels: Vec<(i16, Vec<u8>)>,
}

struct LayerInfo {
    merged_alpha: bool,
    layers: Vec<LayerRecord>,
}

/// Padding after a tagged block: the smallest k in 0..=3 such that the next
/// block starts or the section ends. Photoshop pads to 2 or 4 depending on
/// where the block sits, so it is detected rather than assumed.
fn skip_block_padding(r: &mut Reader<'_>) -> Result<()> {
    for k in 0..4 {
        if r.remaining() == k {
            return r.bytes(k).map(|_| ());
        }
        if let Some(next) = r.peek(k + 4)
            && (next[k..].starts_with(b"8BIM") || next[k..].starts_with(b"8B64"))
        {
            return r.bytes(k).map(|_| ());
        }
    }
    Ok(())
}

fn read_layer_info(mut r: Reader<'_>, psb: bool) -> Result<LayerInfo> {
    if r.remaining() < 2 {
        return Ok(LayerInfo { merged_alpha: false, layers: Vec::new() });
    }
    let count = r.i16()?;
    let merged_alpha = count < 0;
    let count = usize::from(count.unsigned_abs());
    let mut records = Vec::with_capacity(count);
    let mut lengths = Vec::with_capacity(count);
    for _ in 0..count {
        let (top, left, bottom, right) = (r.i32()?, r.i32()?, r.i32()?, r.i32()?);
        let channel_count = usize::from(r.u16()?);
        let mut ids = Vec::with_capacity(channel_count);
        for _ in 0..channel_count {
            let id = r.i16()?;
            ids.push((id, r.length(psb)?));
        }
        r.bytes(4)?; // 8BIM
        r.bytes(4)?; // blend mode key
        let opacity = f32::from(r.u8()?) / 255.0;
        r.u8()?; // clipping
        let flags = r.u8()?;
        r.u8()?; // filler
        let extra_len = r.u32()? as usize;
        let mut extra = r.section(extra_len)?;
        let mask_len = extra.u32()? as usize;
        extra.bytes(mask_len)?;
        let ranges_len = extra.u32()? as usize;
        extra.bytes(ranges_len)?;
        let name_len = usize::from(extra.u8()?);
        // The Pascal name, length byte included, is padded to a multiple of 4.
        extra.bytes((name_len + 1).div_ceil(4) * 4 - 1)?;
        let mut divider = None;
        while extra.remaining() >= 12 {
            extra.bytes(4)?;
            let key = extra.array::<4>()?;
            let len = if psb && PSB_LONG_KEYS.contains(&&key) {
                extra.length(true)?
            } else {
                extra.u32()? as usize
            };
            let data = extra.bytes(len)?;
            if &key == b"lsct" && data.len() >= 4 {
                divider = Some(u32::from_be_bytes([data[0], data[1], data[2], data[3]]));
            }
            skip_block_padding(&mut extra)?;
        }
        records.push(LayerRecord {
            top,
            left,
            bottom,
            right,
            opacity,
            hidden: flags & 2 != 0,
            divider,
            channels: Vec::new(),
        });
        lengths.push(ids);
    }
    for (record, ids) in records.iter_mut().zip(lengths) {
        for (id, len) in ids {
            record.channels.push((id, r.bytes(len)?.to_vec()));
        }
    }
    Ok(LayerInfo { merged_alpha, layers: records })
}

/// The layer and mask section: the layer list, from the main layer info or,
/// for 16- and 32-bit files that store it there, an `Lr16` / `Lr32` block.
fn read_layers(mut r: Reader<'_>, psb: bool) -> Result<Option<LayerInfo>> {
    if r.remaining() == 0 {
        return Ok(None);
    }
    let info_len = r.length(psb)?;
    let mut info = read_layer_info(r.section(info_len)?, psb)?;
    if r.remaining() >= 4 {
        let mask_len = r.u32()? as usize;
        r.bytes(mask_len)?;
    }
    while r.remaining() >= 12 {
        r.bytes(4)?;
        let key = r.array::<4>()?;
        let len =
            if psb && PSB_LONG_KEYS.contains(&&key) { r.length(true)? } else { r.u32()? as usize };
        let data = r.bytes(len)?;
        if (&key == b"Lr16" || &key == b"Lr32") && info.layers.is_empty() {
            let nested = read_layer_info(Reader::new(data), psb)?;
            info.layers = nested.layers;
        }
        skip_block_padding(&mut r)?;
    }
    Ok(Some(info))
}

/// Composites the visible pixel layers over transparency, bottom first.
fn composite_layers(header: &Header, info: &LayerInfo, palette: &[u8]) -> Result<Image> {
    let (w, h) = (header.width, header.height);
    let color_count = header.color_channels()?;
    // Visibility, walking from the top so a hidden group hides its children.
    let mut visible = vec![false; info.layers.len()];
    let mut groups = vec![true];
    for (i, layer) in info.layers.iter().enumerate().rev() {
        let parent = groups.last().copied().unwrap_or(true);
        match layer.divider {
            Some(1 | 2) => groups.push(parent && !layer.hidden),
            Some(3) => {
                if groups.len() > 1 {
                    groups.pop();
                }
            }
            _ => visible[i] = parent && !layer.hidden,
        }
    }
    let gray = color_count == 1 && header.mode != mode::INDEXED;
    let out_color = if gray { 1 } else { 3 };
    let mut rgb = vec![vec![0.0f32; w * h]; out_color];
    let mut alpha = vec![0.0f32; w * h];
    for (layer, _) in info.layers.iter().zip(&visible).filter(|(_, v)| **v) {
        let lw = usize::try_from(layer.right - layer.left).unwrap_or(0);
        let lh = usize::try_from(layer.bottom - layer.top).unwrap_or(0);
        if lw == 0 || lh == 0 || layer.opacity <= 0.0 {
            continue;
        }
        let plane = |id: i16| -> Result<Option<Vec<f32>>> {
            let Some((_, data)) = layer.channels.iter().find(|(cid, _)| *cid == id) else {
                return Ok(None);
            };
            let Some((compression, rest)) = data.split_first_chunk::<2>() else { return Ok(None) };
            let bytes = decode_planes(
                u16::from_be_bytes(*compression),
                rest,
                1,
                lw,
                lh,
                header.depth,
                header.psb,
            )?;
            Ok(Some(plane_to_f32(&bytes, header.depth, lw, lh)))
        };
        let mut color = Vec::with_capacity(color_count);
        for id in 0..color_count as i16 {
            let fill = if header.mode == mode::CMYK { 1.0 } else { 0.0 };
            color.push(plane(id)?.unwrap_or_else(|| vec![fill; lw * lh]));
        }
        let color = to_rgb(header, color, palette);
        let layer_alpha = plane(CHANNEL_TRANSPARENCY)?;
        for ly in 0..lh {
            let y = i64::from(layer.top) + ly as i64;
            if !(0..h as i64).contains(&y) {
                continue;
            }
            for lx in 0..lw {
                let x = i64::from(layer.left) + lx as i64;
                if !(0..w as i64).contains(&x) {
                    continue;
                }
                let src = ly * lw + lx;
                let dst = y as usize * w + x as usize;
                let a =
                    layer_alpha.as_ref().map_or(1.0, |p| p[src].clamp(0.0, 1.0)) * layer.opacity;
                if a <= 0.0 {
                    continue;
                }
                let below = alpha[dst];
                let out_a = a + below * (1.0 - a);
                for (c, channel) in rgb.iter_mut().enumerate() {
                    let value = color.get(c).map_or(0.0, |p| p[src]);
                    channel[dst] = (value * a + channel[dst] * below * (1.0 - a)) / out_a;
                }
                alpha[dst] = out_a;
            }
        }
    }
    let opaque = alpha.iter().all(|a| *a >= 1.0);
    interleave(w, h, rgb, if opaque { None } else { Some(alpha) })
}

/// Removes the white matte Photoshop bakes into the merged colour of a
/// transparent document: stored = colour * alpha + (1 - alpha).
fn unmatte(color: Vec<Vec<f32>>, alpha: &[f32]) -> Vec<Vec<f32>> {
    color
        .into_iter()
        .map(|plane| {
            plane
                .iter()
                .zip(alpha)
                .map(|(m, a)| if *a > 0.0 { ((m - (1.0 - a)) / a).clamp(0.0, 1.0) } else { 0.0 })
                .collect()
        })
        .collect()
}

/// The image's width and height, read from the header only.
pub(crate) fn dimensions(bytes: &[u8]) -> Result<(usize, usize)> {
    let header = Header::read(&mut Reader::new(bytes))?;
    Ok((header.width, header.height))
}

/// Decodes a PSD or PSB file into its flattened image.
pub(crate) fn decode(bytes: &[u8]) -> Result<Decoded> {
    let mut r = Reader::new(bytes);
    let header = Header::read(&mut r)?;
    let color_count = header.color_channels()?;
    if header.channels < color_count {
        return Err("the file has fewer channels than its colour mode needs".to_owned());
    }
    let palette_len = r.u32()? as usize;
    let palette = r.bytes(palette_len)?;
    let resources_len = r.u32()? as usize;
    let resources = read_resources(r.section(resources_len)?)?;
    let layers_len = r.length(header.psb)?;
    // A damaged layer section must not stop the merged image from loading.
    let layers = read_layers(r.section(layers_len)?, header.psb).unwrap_or_else(|e| {
        log::warn!("PSD layer section ignored: {e}");
        None
    });

    let bit_depth = match header.depth {
        16 => BitDepth::Sixteen,
        32 => BitDepth::Float,
        _ => BitDepth::Eight,
    };
    let use_layers = resources.has_real_merged_data == Some(false)
        && layers.as_ref().is_some_and(|info| info.layers.iter().any(|l| l.divider.is_none()));
    let image = match layers.as_ref().filter(|_| use_layers) {
        Some(info) => {
            log::info!("PSD has no merged image; compositing its layers");
            composite_layers(&header, info, palette)?
        }
        None => {
            let compression = r.u16()?;
            let rest = &bytes[r.pos..];
            let (w, h) = (header.width, header.height);
            let planes =
                decode_planes(compression, rest, header.channels, w, h, header.depth, header.psb)?;
            let plane_len = row_bytes(w, header.depth) * h;
            let plane = |i: usize| {
                plane_to_f32(&planes[i * plane_len..(i + 1) * plane_len], header.depth, w, h)
            };
            let color = to_rgb(&header, (0..color_count).map(plane).collect(), palette);
            // Without a layer section an extra channel is taken as transparency.
            let has_alpha = header.channels > color_count
                && layers.as_ref().is_none_or(|info| info.merged_alpha);
            let alpha = has_alpha.then(|| plane(color_count));
            let color = match &alpha {
                Some(alpha) => unmatte(color, alpha),
                None => color,
            };
            interleave(w, h, color, alpha)?
        }
    };
    Ok(Decoded { image, bit_depth, dpi: resources.dpi })
}

// ============================================================================ writing

fn put_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_be_bytes());
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_be_bytes());
}

fn to_u32(n: usize, what: &str) -> Result<u32> {
    u32::try_from(n).map_err(|_| format!("{what} is too large for a PSD"))
}

/// RLE-encodes planes: the byte count of every row, then the rows.
fn encode_rle(planes: &[Vec<u8>], rb: usize) -> Result<Vec<u8>> {
    let rows: usize = planes.iter().map(|p| p.len() / rb.max(1)).sum();
    let mut counts = Vec::with_capacity(rows * 2);
    let mut data = Vec::new();
    for plane in planes {
        for row in plane.chunks_exact(rb) {
            let start = data.len();
            packbits::encode(row, &mut data);
            let n = u16::try_from(data.len() - start)
                .map_err(|_| "a row is too wide for a PSD".to_owned())?;
            put_u16(&mut counts, n);
        }
    }
    counts.extend_from_slice(&data);
    Ok(counts)
}

fn resource(out: &mut Vec<u8>, id: u16, data: &[u8]) -> Result<()> {
    out.extend_from_slice(b"8BIM");
    put_u16(out, id);
    out.extend_from_slice(&[0, 0]); // empty name, padded to even
    put_u32(out, to_u32(data.len(), "a resource")?);
    out.extend_from_slice(data);
    if data.len() % 2 == 1 {
        out.push(0);
    }
    Ok(())
}

fn unicode(out: &mut Vec<u8>, text: &str) {
    let units: Vec<u16> = text.encode_utf16().collect();
    put_u32(out, units.len() as u32);
    for unit in units {
        put_u16(out, unit);
    }
}

fn resources(dpi: Option<(f32, f32)>, real_merged: bool) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let (x, y) = dpi.unwrap_or((72.0, 72.0));
    let mut resolution = Vec::new();
    for value in [x, y] {
        put_u32(&mut resolution, (value.clamp(1.0, 30000.0) * 65536.0).round() as u32);
        put_u16(&mut resolution, 1); // pixels per inch
        put_u16(&mut resolution, 1); // display in inches
    }
    resource(&mut out, RESOURCE_RESOLUTION, &resolution)?;
    let mut version = Vec::new();
    put_u32(&mut version, 1);
    version.push(u8::from(real_merged));
    unicode(&mut version, "SEAMS");
    unicode(&mut version, "SEAMS");
    put_u32(&mut version, 1);
    resource(&mut out, RESOURCE_VERSION_INFO, &version)?;
    Ok(out)
}

/// One layer holding the whole image, so its transparency survives.
fn layer_section(planes: &[Vec<u8>], width: usize, height: usize, rb: usize) -> Result<Vec<u8>> {
    let mut info = Vec::new();
    // Negative: the first extra merged channel is the merged transparency.
    info.extend_from_slice(&(-1i16).to_be_bytes());
    for v in [0, 0, to_u32(height, "the height")?, to_u32(width, "the width")?] {
        put_u32(&mut info, v);
    }
    // Channel order in the record: transparency first, then colour.
    let order: Vec<(i16, &Vec<u8>)> =
        std::iter::once((CHANNEL_TRANSPARENCY, &planes[planes.len() - 1]))
            .chain(planes[..planes.len() - 1].iter().enumerate().map(|(i, p)| (i as i16, p)))
            .collect();
    let encoded = order
        .iter()
        .map(|(_, plane)| encode_rle(std::slice::from_ref(*plane), rb))
        .collect::<Result<Vec<_>>>()?;
    put_u16(&mut info, order.len() as u16);
    for ((id, _), data) in order.iter().zip(&encoded) {
        info.extend_from_slice(&id.to_be_bytes());
        put_u32(&mut info, to_u32(data.len() + 2, "a channel")?);
    }
    info.extend_from_slice(b"8BIMnorm");
    info.extend_from_slice(&[255, 0, 0, 0]); // opacity, clipping, flags, filler
    let name = "Layer 0";
    let mut extra = Vec::new();
    put_u32(&mut extra, 0); // no mask
    put_u32(&mut extra, 0); // no blending ranges
    extra.push(name.len() as u8);
    extra.extend_from_slice(name.as_bytes());
    while extra.len() % 4 != 0 {
        extra.push(0);
    }
    let mut luni = Vec::new();
    unicode(&mut luni, name);
    while luni.len() % 4 != 0 {
        luni.push(0);
    }
    extra.extend_from_slice(b"8BIMluni");
    put_u32(&mut extra, luni.len() as u32);
    extra.extend_from_slice(&luni);
    put_u32(&mut info, to_u32(extra.len(), "the layer record")?);
    info.extend_from_slice(&extra);
    for data in &encoded {
        put_u16(&mut info, 1);
        info.extend_from_slice(data);
    }
    while info.len() % 4 != 0 {
        info.push(0);
    }
    let mut section = Vec::new();
    put_u32(&mut section, to_u32(info.len(), "the layer data")?);
    section.extend_from_slice(&info);
    put_u32(&mut section, 0); // no global layer mask
    Ok(section)
}

fn encode_with(
    image: &Image,
    sixteen: bool,
    dpi: Option<(f32, f32)>,
    real_merged: bool,
) -> Result<Vec<u8>> {
    let (w, h) = (image.width(), image.height());
    if w > 30_000 || h > 30_000 {
        return Err("PSD files are limited to 30000 pixels per side".to_owned());
    }
    let channels = image.channels();
    let depth: u16 = if sixteen { 16 } else { 8 };
    let rb = row_bytes(w, depth);
    let sample = |v: f32| -> [u8; 2] {
        if sixteen {
            ((v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16).to_be_bytes()
        } else {
            [(v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8, 0]
        }
    };
    let width_bytes = usize::from(depth / 8);
    let mut planes: Vec<Vec<u8>> = (0..channels)
        .map(|c| {
            image
                .data()
                .iter()
                .skip(c)
                .step_by(channels)
                .flat_map(|v| sample(*v).into_iter().take(width_bytes))
                .collect()
        })
        .collect();
    let has_alpha = image.has_alpha();

    let mut out = Vec::new();
    out.extend_from_slice(SIGNATURE);
    put_u16(&mut out, 1);
    out.extend_from_slice(&[0; 6]);
    put_u16(&mut out, channels as u16);
    put_u32(&mut out, to_u32(h, "the height")?);
    put_u32(&mut out, to_u32(w, "the width")?);
    put_u16(&mut out, depth);
    put_u16(&mut out, if channels == 1 { mode::GRAYSCALE } else { mode::RGB });
    put_u32(&mut out, 0); // no colour mode data
    let res = resources(dpi, real_merged)?;
    put_u32(&mut out, to_u32(res.len(), "the resources")?);
    out.extend_from_slice(&res);
    if has_alpha {
        let section = layer_section(&planes, w, h, rb)?;
        put_u32(&mut out, to_u32(section.len(), "the layer section")?);
        out.extend_from_slice(&section);
    } else {
        put_u32(&mut out, 0);
    }
    if !real_merged {
        // A white placeholder, as Photoshop writes without "Maximize compatibility".
        for plane in &mut planes {
            plane.fill(255);
        }
    } else if has_alpha {
        // Photoshop's merged colour is matted against white.
        for (c, plane) in planes.iter_mut().take(channels - 1).enumerate() {
            *plane = image
                .data()
                .chunks_exact(channels)
                .flat_map(|p| {
                    let a = p[channels - 1].clamp(0.0, 1.0);
                    sample(p[c] * a + 1.0 - a).into_iter().take(width_bytes)
                })
                .collect();
        }
    }
    put_u16(&mut out, 1);
    out.extend_from_slice(&encode_rle(&planes, rb)?);
    Ok(out)
}

/// Encodes `image` as an 8- or 16-bit PSD: gray or RGB, flat when opaque and
/// a single layer when it has alpha.
pub(crate) fn encode(image: &Image, sixteen: bool, dpi: Option<(f32, f32)>) -> Result<Vec<u8>> {
    encode_with(image, sixteen, dpi, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(channels: usize) -> Image {
        Image::from_fn(37, 21, channels, |x, y, c| ((x * 7 + y * 3 + c * 11) % 64) as f32 / 63.0)
            .unwrap()
    }

    /// Largest difference in premultiplied colour and alpha: colour under
    /// low alpha is only as precise as the matted merged image allows.
    fn worst(a: &Image, b: &Image) -> f32 {
        let premultiplied = |image: &Image| -> Vec<f32> {
            let c = image.channels();
            image
                .data()
                .chunks(c)
                .flat_map(|p| {
                    let alpha = if c == 4 { p[3] } else { 1.0 };
                    p.iter().take(c.min(3)).map(move |v| v * alpha).chain(std::iter::once(alpha))
                })
                .collect()
        };
        premultiplied(a)
            .iter()
            .zip(&premultiplied(b))
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn packbits_round_trips_runs_and_literals() {
        let row: Vec<u8> =
            (0..300).map(|i| if i % 50 < 20 { 7 } else { (i * 13 % 256) as u8 }).collect();
        let mut encoded = Vec::new();
        packbits::encode(&row, &mut encoded);
        let mut decoded = Vec::new();
        packbits::decode_into(&encoded, row.len(), &mut decoded).unwrap();
        assert_eq!(decoded, row);
    }

    #[test]
    fn eight_bit_gray_rgb_and_rgba_round_trip() {
        for channels in [1, 3, 4] {
            let image = gradient(channels);
            let back = decode(&encode(&image, false, None).unwrap()).unwrap();
            assert_eq!(back.image.channels(), channels);
            assert_eq!(back.bit_depth, BitDepth::Eight);
            assert!(worst(&image, &back.image) <= 1.0 / 255.0, "{channels} channels");
        }
    }

    #[test]
    fn sixteen_bit_keeps_fine_precision() {
        let image = gradient(4);
        let back = decode(&encode(&image, true, None).unwrap()).unwrap();
        assert_eq!(back.bit_depth, BitDepth::Sixteen);
        assert!(worst(&image, &back.image) < 1e-4);
    }

    #[test]
    fn resolution_survives() {
        let back = decode(&encode(&gradient(3), false, Some((300.0, 150.0))).unwrap()).unwrap();
        let (x, y) = back.dpi.unwrap();
        assert!((x - 300.0).abs() < 0.01 && (y - 150.0).abs() < 0.01);
    }

    #[test]
    fn layers_are_used_when_the_merged_image_is_a_placeholder() {
        let image = gradient(4);
        let bytes = encode_with(&image, false, None, false).unwrap();
        let back = decode(&bytes).unwrap();
        assert_eq!(back.image.channels(), 4);
        assert!(worst(&image, &back.image) <= 1.0 / 255.0);
    }

    #[test]
    fn zip_with_prediction_decodes() {
        let (w, h) = (9, 4);
        let raw: Vec<u8> = (0..w * h * 2).map(|i| (i * 37 % 251) as u8).collect();
        // Predict (inverse of unpredict) for 16-bit, then zlib.
        let mut predicted = raw.clone();
        for row in predicted.chunks_exact_mut(w * 2) {
            for i in (1..w).rev() {
                let a = u16::from_be_bytes([row[2 * i], row[2 * i + 1]]);
                let b = u16::from_be_bytes([row[2 * i - 2], row[2 * i - 1]]);
                row[2 * i..2 * i + 2].copy_from_slice(&a.wrapping_sub(b).to_be_bytes());
            }
        }
        let mut encoder =
            flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut encoder, &predicted).unwrap();
        let zipped = encoder.finish().unwrap();
        assert_eq!(decode_planes(3, &zipped, 1, w, h, 16, false).unwrap(), raw);
    }

    #[test]
    fn garbage_and_truncated_files_fail_without_panicking() {
        assert!(decode(b"8BPS").is_err());
        assert!(decode(b"not a psd at all, just text").is_err());
        let bytes = encode(&gradient(3), false, None).unwrap();
        for cut in [10, 30, bytes.len() / 2, bytes.len() - 3] {
            assert!(decode(&bytes[..cut]).is_err(), "cut at {cut}");
        }
    }

    #[test]
    fn lab_is_rejected_with_a_clear_message() {
        let mut bytes = encode(&gradient(3), false, None).unwrap();
        bytes[24..26].copy_from_slice(&9u16.to_be_bytes());
        assert!(decode(&bytes).err().unwrap().contains("Lab"));
    }
}
