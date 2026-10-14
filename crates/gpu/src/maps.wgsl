// Material maps from one height field.
//
// Every formula here mirrors `seams_core::pbr`, which stays the reference:
// export always uses it, and tests/parity.rs compares the two.

struct Params {
    // x, y: field size. z: kernel side (normal) or blur radius. w: scalar map.
    dims: vec4<i32>,
    // x: bit flags. y: how the blurred field is used. z: blur axis (0 = x).
    flags: vec4<u32>,
    a: vec4<f32>,
    b: vec4<f32>,
    // Normal: the slope kernel, row-major. Blur: the Gaussian taps.
    kernel: array<vec4<f32>, 21>,
}

@group(0) @binding(0) var<uniform> params: Params;
// The height field, one value in [0, 1] per pixel.
@group(0) @binding(1) var height_tex: texture_2d<f32>;
// The source alpha, when the image has one.
@group(0) @binding(2) var alpha_tex: texture_2d<f32>;
// A scalar field made by an earlier pass: the shaped or the blurred map.
@group(0) @binding(3) var field_tex: texture_2d<f32>;

const WRAP: u32 = 1u;
const INVERT_HEIGHT: u32 = 2u;
const HAS_ALPHA: u32 = 4u;
const INVERT_RESULT: u32 = 8u;

const USE_BLURRED: u32 = 1u;
const USE_SHARPENED: u32 = 2u;

const ROUGHNESS: i32 = 0;
const AMBIENT_OCCLUSION: i32 = 1;
const DISPLACEMENT: i32 = 2;

// One triangle that covers the whole target.
@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
}

fn has(flag: u32) -> bool {
    return (params.flags.x & flag) != 0u;
}

fn weight(index: i32) -> f32 {
    return params.kernel[index / 4][index % 4];
}

fn wrap_index(index: i32, len: i32) -> i32 {
    return ((index % len) + len) % len;
}

// Mirrors about the edge pixel without repeating it: dcb|abcd|cba.
fn reflect_index(index: i32, len: i32) -> i32 {
    if len <= 1 {
        return 0;
    }
    let period = 2 * (len - 1);
    let m = ((index % period) + period) % period;
    return select(period - m, m, m < len);
}

fn edge_index(index: i32, len: i32) -> i32 {
    if has(WRAP) {
        return wrap_index(index, len);
    }
    return clamp(index, 0, len - 1);
}

fn height_at(p: vec2<i32>) -> f32 {
    let h = textureLoad(height_tex, p, 0).r;
    return select(h, 1.0 - h, has(INVERT_HEIGHT));
}

// The colour with the source alpha applied. The targets hold premultiplied
// colour, which is what the interface blends.
fn premultiplied(rgb: vec3<f32>, p: vec2<i32>) -> vec4<f32> {
    var alpha = 1.0;
    if has(HAS_ALPHA) {
        alpha = textureLoad(alpha_tex, p, 0).r;
    }
    return vec4(rgb * alpha, alpha);
}

// a: slope gain for x and y (sign and strength included), the Z of an
// unscaled normal, the smallest Z allowed.
@fragment
fn fs_normal(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(position.xy));
    let size = params.dims.xy;
    let side = params.dims.z;
    let radius = side / 2;
    var du = 0.0;
    var dv = 0.0;
    for (var ky = 0; ky < side; ky++) {
        for (var kx = 0; kx < side; kx++) {
            // The vertical kernel is the transpose of the horizontal one.
            let wu = weight(ky * side + kx);
            let wv = weight(kx * side + ky);
            if wu == 0.0 && wv == 0.0 {
                continue;
            }
            let q = vec2(
                edge_index(p.x + kx - radius, size.x),
                edge_index(p.y + ky - radius, size.y),
            );
            let h = height_at(q);
            du += h * wu;
            dv += h * wv;
        }
    }
    let slope = vec2(du, dv) * params.a.xy;
    let basis = params.a.z;
    let min_z = params.a.w;
    let len = sqrt(dot(slope, slope) + basis * basis);
    var normal = vec3(-slope / len, basis / len);
    if min_z > 0.0 && normal.z < min_z {
        let planar = max(length(normal.xy), 1e-8);
        let reach = sqrt(max(1.0 - min_z * min_z, 0.0));
        normal = vec3(normal.xy * (reach / planar), min_z);
    }
    return premultiplied(clamp(normal * 0.5 + 0.5, vec3(0.0), vec3(1.0)), p);
}

// The value of a scalar map before any blur or sharpen.
fn shape(h: f32) -> f32 {
    let a = params.a;
    switch params.dims.w {
        // a: contrast gain, level, invert.
        case ROUGHNESS {
            let value = select(h, 1.0 - h, a.z > 0.5);
            let contrasted = clamp(0.5 + (value - 0.5) * a.x, 0.0, 1.0);
            return clamp(contrasted * a.y, 0.0, 1.0);
        }
        // a: range, mean, strength.
        case AMBIENT_OCCLUSION {
            let closeness = sqrt(max((a.x - abs(h - a.y)) / a.x, 0.0));
            return closeness + (1.0 - closeness) * (1.0 - a.z);
        }
        // a: contrast factor, invert.
        case DISPLACEMENT {
            let value = a.x * (h - 0.5) + 0.5;
            return clamp(select(value, 1.0 - value, a.y > 0.5), 0.0, 1.0);
        }
        // Opacity. a: cutoff, edge width, soft edge.
        default {
            if a.z > 0.5 {
                return clamp((h - a.x + a.y * 0.5) / a.y, 0.0, 1.0);
            }
            return select(0.0, 1.0, h > a.x);
        }
    }
}

// Writes the shaped field, so it can be blurred.
@fragment
fn fs_shape(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(position.xy));
    return vec4(shape(textureLoad(height_tex, p, 0).r), 0.0, 0.0, 1.0);
}

// One direction of a separable Gaussian blur with reflected borders.
@fragment
fn fs_blur(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(position.xy));
    let radius = params.dims.z;
    let horizontal = params.flags.z == 0u;
    var sum = 0.0;
    for (var tap = 0; tap <= 2 * radius; tap++) {
        var q = p;
        if horizontal {
            q.x = reflect_index(p.x + tap - radius, params.dims.x);
        } else {
            q.y = reflect_index(p.y + tap - radius, params.dims.y);
        }
        sum += textureLoad(field_tex, q, 0).r * weight(tap);
    }
    return vec4(sum, 0.0, 0.0, 1.0);
}

// A finished scalar map. b.x: sharpen gain.
@fragment
fn fs_scalar(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(position.xy));
    let shaped = shape(textureLoad(height_tex, p, 0).r);
    var value = shaped;
    if params.flags.y == USE_BLURRED {
        value = textureLoad(field_tex, p, 0).r;
    } else if params.flags.y == USE_SHARPENED {
        let blurred = textureLoad(field_tex, p, 0).r;
        value = shaped + (shaped - blurred) * params.b.x;
    }
    if has(INVERT_RESULT) {
        value = 1.0 - value;
    }
    return premultiplied(vec3(clamp(value, 0.0, 1.0)), p);
}
