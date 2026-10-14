//! The render pipeline (delight, then seamless, then maps) with result reuse.
//!
//! Each worker thread owns its own cache, so no locking is needed. Results are
//! shared as `Arc<Image>`: a cached entry and the copy in use are one buffer.

use std::sync::Arc;

use seams_core::Image;
use seams_core::cache::{CacheLimits, LruCache};
use seams_core::delight::{DelightParams, delight};
use seams_core::key::KeyHasher;
use seams_core::pbr::{HeightField, MapKind, MaterialParams, generate_map};
use seams_core::seamless::{SeamlessParams, make_seamless};

use crate::error::Result;

const MEBIBYTE: usize = 1024 * 1024;

/// Budget for full-resolution renders. A 4K RGB result is about 200 MB, so
/// this keeps a few recent settings instant without holding gigabytes.
pub const FULL_CACHE: CacheLimits = CacheLimits { max_entries: 8, max_bytes: 768 * MEBIBYTE };
/// Budget for preview-size renders, which are small and revisited often.
pub const PREVIEW_CACHE: CacheLimits = CacheLimits { max_entries: 48, max_bytes: 192 * MEBIBYTE };
/// Budget for generated maps beyond the set currently on screen.
pub const MAP_CACHE: CacheLimits = CacheLimits { max_entries: 20, max_bytes: 512 * MEBIBYTE };

fn combine(first: u64, second: u64) -> u64 {
    KeyHasher::new().u64(first).u64(second).finish()
}

/// An image together with the key identifying how it was produced.
#[derive(Debug, Clone)]
pub struct Keyed {
    pub image: Arc<Image>,
    pub key: u64,
}

impl Keyed {
    /// Wraps a source image; its key is its content fingerprint.
    pub fn source(image: Arc<Image>) -> Self {
        let key = image.fingerprint();
        Self { image, key }
    }
}

/// The fingerprint of the most recent source buffer. Sources are shared,
/// immutable `Arc`s, so the same buffer is hashed once, not per request.
#[derive(Default)]
struct LastKey(Option<(Arc<Image>, u64)>);

impl LastKey {
    fn of(&mut self, image: &Arc<Image>) -> u64 {
        if let Some((last, key)) = &self.0
            && Arc::ptr_eq(last, image)
        {
            return *key;
        }
        let key = image.fingerprint();
        self.0 = Some((Arc::clone(image), key));
        key
    }
}

/// Runs delight and seamless, reusing earlier results.
pub struct RenderCache {
    images: LruCache<Arc<Image>>,
    source_key: LastKey,
}

impl RenderCache {
    pub fn new(limits: CacheLimits) -> Self {
        Self { images: LruCache::new(limits), source_key: LastKey::default() }
    }

    /// `image` as a source, keyed by its content.
    pub fn source(&mut self, image: &Arc<Image>) -> Keyed {
        Keyed { image: Arc::clone(image), key: self.source_key.of(image) }
    }

    fn get_or_render(
        &mut self,
        key: u64,
        render: impl FnOnce() -> Result<Image>,
    ) -> Result<Arc<Image>> {
        if let Some(cached) = self.images.get(key) {
            return Ok(cached);
        }
        let image = Arc::new(render()?);
        self.images.insert(key, Arc::clone(&image));
        Ok(image)
    }

    /// The delighted image. With every control off this is the source itself.
    pub fn delighted(&mut self, source: &Keyed, params: &DelightParams) -> Result<Keyed> {
        if params.is_identity() {
            return Ok(source.clone());
        }
        let key = combine(source.key, params.cache_key());
        let image = self.get_or_render(key, || Ok(delight(&source.image, params)?))?;
        Ok(Keyed { image, key })
    }

    /// The tileable image made from an already delighted one.
    pub fn seamless(&mut self, delighted: &Keyed, params: &SeamlessParams) -> Result<Keyed> {
        let key = combine(delighted.key, params.cache_key());
        let image = self.get_or_render(key, || Ok(make_seamless(&delighted.image, params)?))?;
        Ok(Keyed { image, key })
    }
}

/// Generates material maps, recomputing only those whose settings changed.
pub struct MapCache {
    maps: LruCache<Arc<Image>>,
    source_key: LastKey,
}

impl MapCache {
    pub fn new(limits: CacheLimits) -> Self {
        Self { maps: LruCache::new(limits), source_key: LastKey::default() }
    }

    /// Every map for `source`, each taken from the cache when still valid.
    pub fn generate(
        &mut self,
        source: &Arc<Image>,
        params: &MaterialParams,
    ) -> Result<Vec<(MapKind, Arc<Image>)>> {
        let source_key = self.source_key.of(source);
        let mut field: Option<HeightField> = None;
        let mut result = Vec::with_capacity(MapKind::ALL.len());
        for kind in MapKind::ALL {
            let key = combine(source_key, params.cache_key(kind));
            let map = match self.maps.get(key) {
                Some(cached) => cached,
                None => {
                    let heights = field.get_or_insert_with(|| {
                        HeightField::from_image(source, params.height_source)
                    });
                    let map = Arc::new(generate_map(kind, heights, params)?);
                    self.maps.insert(key, Arc::clone(&map));
                    map
                }
            };
            result.push((kind, map));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seams_core::seamless::Method;

    fn source() -> Keyed {
        let image =
            Image::from_fn(96, 96, 3, |x, y, c| ((x * 5 + y * 3 + c * 11) % 97) as f32 / 97.0);
        Keyed::source(Arc::new(image.unwrap()))
    }

    #[test]
    fn identity_delight_returns_the_source_buffer() {
        let source = source();
        let mut cache = RenderCache::new(FULL_CACHE);

        let delighted = cache.delighted(&source, &DelightParams::default()).unwrap();

        assert!(Arc::ptr_eq(&delighted.image, &source.image));
        assert_eq!(delighted.key, source.key);
    }

    #[test]
    fn repeated_render_reuses_the_cached_buffer() {
        let source = source();
        let params = SeamlessParams::default();
        let mut cache = RenderCache::new(FULL_CACHE);

        let first = cache.seamless(&source, &params).unwrap();
        let second = cache.seamless(&source, &params).unwrap();

        assert!(Arc::ptr_eq(&first.image, &second.image));
    }

    #[test]
    fn changed_settings_render_again() {
        let source = source();
        let mut cache = RenderCache::new(FULL_CACHE);
        let mirror = SeamlessParams { method: Method::Mirror, ..SeamlessParams::default() };

        let fade = cache.seamless(&source, &SeamlessParams::default()).unwrap();
        let mirrored = cache.seamless(&source, &mirror).unwrap();

        assert_ne!(fade.key, mirrored.key);
        assert_eq!(mirrored.image.width(), 192);
    }

    #[test]
    fn delight_changes_the_key_seamless_builds_on() {
        let source = source();
        let mut cache = RenderCache::new(FULL_CACHE);

        let delighted = cache.delighted(&source, &DelightParams::BALANCED).unwrap();

        assert_ne!(delighted.key, source.key);
        assert!(!Arc::ptr_eq(&delighted.image, &source.image));
    }

    #[test]
    fn changing_one_map_setting_recomputes_only_that_map() {
        let source = source();
        let mut cache = MapCache::new(MAP_CACHE);
        let base = MaterialParams::default();
        let mut changed = base;
        changed.roughness.contrast = 0.6;

        let first = cache.generate(&source.image, &base).unwrap();
        let second = cache.generate(&source.image, &changed).unwrap();

        for ((kind, before), (_, after)) in first.iter().zip(&second) {
            let reused = Arc::ptr_eq(before, after);
            assert_eq!(reused, *kind != MapKind::Roughness, "{kind:?}");
        }
    }

    #[test]
    fn changing_the_height_source_recomputes_every_map() {
        let source = source();
        let mut cache = MapCache::new(MAP_CACHE);
        let base = MaterialParams::default();
        let mut changed = base;
        changed.height_source = seams_core::pbr::HeightSource::Red;

        let first = cache.generate(&source.image, &base).unwrap();
        let second = cache.generate(&source.image, &changed).unwrap();

        assert!(first.iter().zip(&second).all(|((_, a), (_, b))| !Arc::ptr_eq(a, b)));
    }
}
