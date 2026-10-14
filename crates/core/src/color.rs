//! sRGB to CIE Lab conversion.
//!
//! Lab values use the 8-bit convention the delighting constants were tuned
//! against: `L` spans `[0, 255]` and `a`/`b` are centred on 128.

const LAB_OFFSET: f32 = 128.0;
const L_SCALE: f32 = 255.0 / 100.0;
const WHITE_D65: [f32; 3] = [0.950_456, 1.0, 1.088_754];
const LAB_EPSILON: f32 = 0.008_856;
const LAB_KAPPA: f32 = 7.787;

/// A colour in scaled Lab: `l` in `[0, 255]`, `a`/`b` centred on 128.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

pub fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 { value / 12.92 } else { ((value + 0.055) / 1.055).powf(2.4) }
}

pub fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.003_130_8 { value * 12.92 } else { 1.055 * value.powf(1.0 / 2.4) - 0.055 }
}

fn lab_forward(t: f32) -> f32 {
    if t > LAB_EPSILON { t.cbrt() } else { LAB_KAPPA * t + 16.0 / 116.0 }
}

fn lab_inverse(t: f32) -> f32 {
    let cubed = t * t * t;
    if cubed > LAB_EPSILON { cubed } else { (t - 16.0 / 116.0) / LAB_KAPPA }
}

/// Converts an sRGB colour with components in `[0, 1]` to scaled Lab.
pub fn rgb_to_lab(rgb: [f32; 3]) -> Lab {
    let [r, g, b] = rgb.map(|v| srgb_to_linear(v.clamp(0.0, 1.0)));
    let x = (0.412_453 * r + 0.357_58 * g + 0.180_423 * b) / WHITE_D65[0];
    let y = 0.212_671 * r + 0.715_16 * g + 0.072_169 * b;
    let z = (0.019_334 * r + 0.119_193 * g + 0.950_227 * b) / WHITE_D65[2];
    let (fx, fy, fz) = (lab_forward(x), lab_forward(y), lab_forward(z));
    Lab {
        l: (116.0 * fy - 16.0) * L_SCALE,
        a: 500.0 * (fx - fy) + LAB_OFFSET,
        b: 200.0 * (fy - fz) + LAB_OFFSET,
    }
}

/// Converts scaled Lab back to sRGB, clamped to `[0, 1]`.
pub fn lab_to_rgb(lab: Lab) -> [f32; 3] {
    let fy = (lab.l / L_SCALE + 16.0) / 116.0;
    let fx = fy + (lab.a - LAB_OFFSET) / 500.0;
    let fz = fy - (lab.b - LAB_OFFSET) / 200.0;
    let x = lab_inverse(fx) * WHITE_D65[0];
    let y = lab_inverse(fy);
    let z = lab_inverse(fz) * WHITE_D65[2];
    let r = 3.240_479 * x - 1.537_15 * y - 0.498_535 * z;
    let g = -0.969_256 * x + 1.875_991 * y + 0.041_556 * z;
    let b = 0.055_648 * x - 0.204_043 * y + 1.057_311 * z;
    [r, g, b].map(|v| linear_to_srgb(v.max(0.0)).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_and_black_map_to_the_ends_of_l() {
        let white = rgb_to_lab([1.0, 1.0, 1.0]);
        let black = rgb_to_lab([0.0, 0.0, 0.0]);

        assert!((white.l - 255.0).abs() < 0.5);
        assert!(black.l.abs() < 0.5);
        assert!((white.a - 128.0).abs() < 0.5 && (white.b - 128.0).abs() < 0.5);
    }

    #[test]
    fn grey_has_neutral_chroma() {
        let grey = rgb_to_lab([0.5, 0.5, 0.5]);

        assert!((grey.a - 128.0).abs() < 0.2);
        assert!((grey.b - 128.0).abs() < 0.2);
    }

    #[test]
    fn conversion_round_trips() {
        for rgb in [[0.2, 0.4, 0.6], [0.9, 0.1, 0.3], [0.01, 0.02, 0.03], [1.0, 1.0, 0.0]] {
            let back = lab_to_rgb(rgb_to_lab(rgb));

            for (a, e) in back.iter().zip(rgb) {
                assert!((a - e).abs() < 2e-3, "{rgb:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn transfer_functions_are_inverse() {
        for v in [0.0, 0.002, 0.04, 0.2, 0.7, 1.0] {
            assert!((linear_to_srgb(srgb_to_linear(v)) - v).abs() < 1e-5);
        }
    }
}
