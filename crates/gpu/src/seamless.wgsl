// The seamless step, for the methods that are one formula per pixel.
//
// Every formula here mirrors `seams_core::seamless`, which stays the
// reference: export always uses it, and tests/seamless_parity.rs compares
// the two. Splat synthesis is not here; it stays on the CPU.

struct Params {
    // x, y: source size. z, w: size of the overlap tile before it is resized.
    size: vec4<i32>,
    // x: method. y: cross-fade radius. z, w: overlapped columns and rows.
    mode: vec4<i32>,
    // x: half the width of the overlap fade, as a fraction of the strip.
    a: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
// The image to make seamless: 8-bit, straight alpha.
@group(0) @binding(1) var source_tex: texture_2d<f32>;

const OFFSET_CROSSFADE: i32 = 0;
const MIRROR: i32 = 1;

// One triangle that covers the whole target.
@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
}

fn wrap_index(index: i32, len: i32) -> i32 {
    return ((index % len) + len) % len;
}

fn source(p: vec2<i32>) -> vec4<f32> {
    return textureLoad(source_tex, p, 0);
}

// ---- Offset cross-fade ----

// The source moved by half its size, which puts its borders on the centre.
fn shifted(q: vec2<i32>) -> vec4<f32> {
    let size = params.size.xy;
    let shift = size / 2;
    return source(vec2(wrap_index(q.x - shift.x, size.x), wrap_index(q.y - shift.y, size.y)));
}

// For an index on one side of the seam at `centre`: its distance from the
// seam, and the index the same distance away on the other side.
fn across(index: i32, centre: i32) -> vec2<i32> {
    if index < centre {
        let distance = centre - 1 - index;
        return vec2(distance, centre + distance);
    }
    let distance = index - centre;
    return vec2(distance, centre - 1 - distance);
}

// How much of the pixel across the seam is mixed in: half at the seam,
// falling to nothing at the radius.
fn feather(distance: i32) -> f32 {
    return 0.5 * (1.0 - f32(distance) / f32(params.mode.y));
}

fn reach(len: i32) -> i32 {
    let centre = len / 2;
    return min(params.mode.y, min(centre, len - centre));
}

// The shifted image after the seam down its centre column is feathered.
fn column_seam(q: vec2<i32>) -> vec4<f32> {
    let width = params.size.x;
    let pair = across(q.x, width / 2);
    let own = shifted(q);
    if pair.x >= reach(width) {
        return own;
    }
    return mix(own, shifted(vec2(pair.y, q.y)), feather(pair.x));
}

fn offset_crossfade(p: vec2<i32>) -> vec4<f32> {
    let size = params.size.xy;
    let shift = size / 2;
    // Undoing the shift at the end puts output pixel p at q in the shifted image.
    let q = vec2(wrap_index(p.x + shift.x, size.x), wrap_index(p.y + shift.y, size.y));
    let pair = across(q.y, size.y / 2);
    let own = column_seam(q);
    if pair.x >= reach(size.y) {
        return own;
    }
    return mix(own, column_seam(vec2(q.x, pair.y)), feather(pair.x));
}

// ---- Mirror tiling ----

fn mirror(p: vec2<i32>) -> vec4<f32> {
    let size = params.size.xy;
    return source(vec2(
        select(2 * size.x - 1 - p.x, p.x, p.x < size.x),
        select(2 * size.y - 1 - p.y, p.y, p.y < size.y),
    ));
}

// ---- Overlap blend ----

fn smooth_step(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

// How much of the far edge shows at `index` of an overlap strip `len` long:
// all of it at the start, none at the end.
fn seam_fade(index: i32, len: i32) -> f32 {
    let position = (f32(index) + 0.5) / f32(len);
    let half_width = params.a.x;
    if half_width <= 1e-6 {
        return select(0.0, 1.0, position < 0.5);
    }
    let linear = clamp((0.5 + half_width - position) / (2.0 * half_width), 0.0, 1.0);
    return smooth_step(smooth_step(linear));
}

// The source with its rightmost columns blended over its leftmost.
fn narrowed(p: vec2<i32>) -> vec4<f32> {
    let blend = params.mode.z;
    let near = source(p);
    if p.x >= blend {
        return near;
    }
    let far = source(vec2(params.size.x - blend + p.x, p.y));
    return mix(near, far, seam_fade(p.x, blend));
}

// The narrowed image with its bottom rows blended over its top rows.
fn tile(p: vec2<i32>) -> vec4<f32> {
    let blend = params.mode.w;
    let near = narrowed(p);
    if p.y >= blend {
        return near;
    }
    let far = narrowed(vec2(p.x, params.size.y - blend + p.y));
    return mix(near, far, seam_fade(p.y, blend));
}

// The tile resized back to the source size, sampled across its own edges.
fn overlap(p: vec2<i32>) -> vec4<f32> {
    let tile_size = params.size.zw;
    let scale = vec2<f32>(tile_size) / vec2<f32>(params.size.xy);
    let position = (vec2<f32>(p) + 0.5) * scale - 0.5;
    let base = floor(position);
    let t = position - base;
    let x0 = wrap_index(i32(base.x), tile_size.x);
    let x1 = wrap_index(i32(base.x) + 1, tile_size.x);
    let y0 = wrap_index(i32(base.y), tile_size.y);
    let y1 = wrap_index(i32(base.y) + 1, tile_size.y);
    let top = tile(vec2(x0, y0)) * (1.0 - t.x) + tile(vec2(x1, y0)) * t.x;
    let bottom = tile(vec2(x0, y1)) * (1.0 - t.x) + tile(vec2(x1, y1)) * t.x;
    return top * (1.0 - t.y) + bottom * t.y;
}

@fragment
fn fs_seamless(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(position.xy));
    var color: vec4<f32>;
    switch params.mode.x {
        case OFFSET_CROSSFADE {
            color = offset_crossfade(p);
        }
        case MIRROR {
            color = mirror(p);
        }
        default {
            color = overlap(p);
        }
    }
    color = clamp(color, vec4(0.0), vec4(1.0));
    // The target holds premultiplied colour, which is what the interface blends.
    return vec4(color.rgb * color.a, color.a);
}
