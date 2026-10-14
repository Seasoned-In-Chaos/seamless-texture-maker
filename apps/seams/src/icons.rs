//! Icons from Lucide (ISC license, see `assets/fonts/Lucide-LICENSE.txt`),
//! drawn as glyphs of the bundled icon font. Codepoints are those of
//! lucide-static 1.57.0; updating the font means checking them again.

use egui::{Align2, Color32, Painter, Pos2, Rect, Vec2, vec2};

use crate::theme::{TOKENS, icon_font};

/// `hand`: the pan tool.
pub const HAND: char = '\u{e1d7}';
/// `hand-grab`: panning in progress.
pub const HAND_GRAB: char = '\u{e1e6}';
/// `zoom-in`: the zoom tool, and its pointer.
pub const ZOOM_IN: char = '\u{e1b6}';
/// `zoom-out`: the zoom tool's pointer with Alt held.
pub const ZOOM_OUT: char = '\u{e1b7}';
/// `minus` and `plus`: step the zoom.
pub const MINUS: char = '\u{e11c}';
pub const PLUS: char = '\u{e13d}';
/// `scan`: fit to window.
pub const SCAN: char = '\u{e257}';
/// `grid-2x2`: tile guides.
pub const GRID: char = '\u{e4ff}';
/// `chevron-down`: opens a dropdown.
pub const CHEVRON_DOWN: char = '\u{e06d}';
/// `crop`: the Crop tool.
pub const CROP: char = '\u{e0ab}';
/// `rotate-cw`: the crop frame will turn.
pub const ROTATE: char = '\u{e149}';
/// `image-plus`: open a texture.
pub const IMAGE_PLUS: char = '\u{e1f7}';

/// Size of an icon drawn as the pointer, in screen points.
pub const CURSOR_SIZE: f32 = 18.0;

/// Draws `icon` centred in `rect`, as large as the rect's shorter side.
pub fn paint(painter: &Painter, rect: Rect, icon: char, color: Color32) {
    let size = rect.width().min(rect.height());
    painter.text(rect.center(), Align2::CENTER_CENTER, icon, icon_font(size), color);
}

/// Draws `icon` as the pointer at `at`: white, outlined dark so it shows on
/// any image. For cursors the system lacks; the caller hides the system one.
pub fn paint_cursor(painter: &Painter, at: Pos2, icon: char) {
    let rect = Rect::from_center_size(at, Vec2::splat(CURSOR_SIZE));
    for offset in [vec2(-1.0, 0.0), vec2(1.0, 0.0), vec2(0.0, -1.0), vec2(0.0, 1.0)] {
        paint(painter, rect.translate(offset), icon, TOKENS.canvas);
    }
    paint(painter, rect, icon, Color32::WHITE);
}
