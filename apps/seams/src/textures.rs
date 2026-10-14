//! GPU textures for the images the session exposes.

use std::collections::HashMap;

use egui::{
    ColorImage, Context, TextureFilter, TextureHandle, TextureId, TextureOptions, Vec2, vec2,
};
use seams_core::pbr::MapKind;
use seams_engine::View;

/// Images are smoothed when shrunk and show exact pixels when enlarged.
const IMAGE_SAMPLING: TextureOptions =
    TextureOptions { magnification: TextureFilter::Nearest, ..TextureOptions::LINEAR };

/// Which image a texture holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Slot {
    Before,
    After,
    Map(MapKind),
}

/// A texture ready to draw, with the size of the full-resolution image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Texture {
    pub id: TextureId,
    /// Full-resolution size in pixels, which may exceed the texture's own.
    pub full_size: Vec2,
}

struct Entry {
    revision: u64,
    handle: TextureHandle,
    full_size: Vec2,
}

/// Uploads each view once per revision and reuses the texture afterwards.
#[derive(Default)]
pub struct Textures {
    entries: HashMap<Slot, Entry>,
    checker: Option<TextureHandle>,
}

impl Textures {
    /// The texture for `view`, uploading it if its revision changed.
    pub fn sync(&mut self, ctx: &Context, slot: Slot, view: Option<&View>) -> Option<Texture> {
        let Some(view) = view else {
            self.entries.remove(&slot);
            return None;
        };
        let stale = self.entries.get(&slot).is_none_or(|entry| entry.revision != view.revision);
        if stale {
            let image = &view.image;
            let pixels =
                ColorImage::from_rgba_unmultiplied([image.width, image.height], &image.rgba);
            let handle = ctx.load_texture(format!("{slot:?}"), pixels, IMAGE_SAMPLING);
            let full_size = vec2(image.full_width as f32, image.full_height as f32);
            self.entries.insert(slot, Entry { revision: view.revision, handle, full_size });
        }
        self.entries
            .get(&slot)
            .map(|entry| Texture { id: entry.handle.id(), full_size: entry.full_size })
    }

    /// A small repeating checkerboard, drawn behind images with transparency.
    pub fn checker(&mut self, ctx: &Context) -> TextureId {
        let handle = self.checker.get_or_insert_with(|| {
            let (dark, light) = (egui::Color32::from_gray(48), egui::Color32::from_gray(64));
            let pixels = ColorImage::new([2, 2], vec![dark, light, light, dark]);
            let options = TextureOptions {
                wrap_mode: egui::TextureWrapMode::Repeat,
                ..TextureOptions::NEAREST
            };
            ctx.load_texture("checker", pixels, options)
        });
        handle.id()
    }
}
