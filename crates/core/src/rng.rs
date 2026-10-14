//! A small deterministic random number generator.
//!
//! Splat layouts must be reproducible from a seed across runs and releases, so
//! the generator is defined here (SplitMix64) instead of depending on a crate
//! whose default algorithm may change between versions.

/// SplitMix64: fast, well distributed, and fully determined by its seed.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        // 24 random bits fill an f32 mantissa exactly, so 1.0 is never produced.
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// A value in `[low, high)`.
    pub fn uniform(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }

    /// An index in `[0, bound)`; `0` when `bound` is zero.
    pub fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() % bound as u64) as usize
    }

    /// Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_same_sequence() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);

        let first: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let second: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();

        assert_eq!(first, second);
    }

    #[test]
    fn next_f32_stays_in_unit_interval() {
        let mut rng = Rng::new(1);

        assert!((0..10_000).map(|_| rng.next_f32()).all(|v| (0.0..1.0).contains(&v)));
    }

    #[test]
    fn below_respects_bound_and_handles_zero() {
        let mut rng = Rng::new(9);

        assert!((0..1000).map(|_| rng.below(7)).all(|v| v < 7));
        assert_eq!(rng.below(0), 0);
    }

    #[test]
    fn shuffle_keeps_every_item() {
        let mut items: Vec<usize> = (0..50).collect();

        Rng::new(3).shuffle(&mut items);
        let mut sorted = items.clone();
        sorted.sort_unstable();

        assert_ne!(items, (0..50).collect::<Vec<_>>());
        assert_eq!(sorted, (0..50).collect::<Vec<_>>());
    }
}
