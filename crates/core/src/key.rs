//! Builds `u64` cache keys from parameter values.
//!
//! `f32` is not `Hash`, so parameter structs describe themselves through this
//! small builder instead of deriving it. Keys are only compared within one
//! process, so the hasher does not need to be stable across builds.

use std::hash::{DefaultHasher, Hasher};

/// A chainable hasher for cache keys.
#[derive(Default)]
pub struct KeyHasher(DefaultHasher);

impl KeyHasher {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn u64(&mut self, value: u64) -> &mut Self {
        self.0.write_u64(value);
        self
    }

    pub fn usize(&mut self, value: usize) -> &mut Self {
        self.0.write_usize(value);
        self
    }

    pub fn bool(&mut self, value: bool) -> &mut Self {
        self.0.write_u8(u8::from(value));
        self
    }

    /// Hashes the bit pattern, treating `-0.0` and `0.0` as the same value.
    pub fn f32(&mut self, value: f32) -> &mut Self {
        let normalised = if value == 0.0 { 0.0 } else { value };
        self.0.write_u32(normalised.to_bits());
        self
    }

    pub fn slice(&mut self, values: &[f32]) -> &mut Self {
        for value in values {
            self.f32(*value);
        }
        self
    }

    pub fn finish(&self) -> u64 {
        self.0.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_inputs_give_equal_keys() {
        let key = |v: f32| KeyHasher::new().f32(v).bool(true).u64(7).finish();

        assert_eq!(key(0.25), key(0.25));
        assert_ne!(key(0.25), key(0.26));
    }

    #[test]
    fn signed_zero_hashes_like_zero() {
        assert_eq!(KeyHasher::new().f32(0.0).finish(), KeyHasher::new().f32(-0.0).finish());
    }
}
