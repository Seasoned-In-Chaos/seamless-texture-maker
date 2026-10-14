//! A bounded least-recently-used cache for processing results.
//!
//! Results can be full-resolution float images (hundreds of megabytes each),
//! so the cache is limited by total bytes as well as by entry count: whichever
//! limit is reached first evicts the oldest entries.

use std::collections::{HashMap, VecDeque};

/// Limits for an [`LruCache`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheLimits {
    pub max_entries: usize,
    pub max_bytes: usize,
}

/// A value the cache can account for.
pub trait Weighted {
    /// Approximate memory held by this value, in bytes.
    fn weight_bytes(&self) -> usize;
}

impl Weighted for crate::Image {
    fn weight_bytes(&self) -> usize {
        self.byte_len()
    }
}

impl<T: Weighted> Weighted for std::sync::Arc<T> {
    fn weight_bytes(&self) -> usize {
        self.as_ref().weight_bytes()
    }
}

/// An LRU cache keyed by `u64`, bounded by entry count and total bytes.
#[derive(Debug)]
pub struct LruCache<V> {
    limits: CacheLimits,
    entries: HashMap<u64, V>,
    order: VecDeque<u64>,
    total_bytes: usize,
}

impl<V: Weighted + Clone> LruCache<V> {
    pub fn new(limits: CacheLimits) -> Self {
        Self { limits, entries: HashMap::new(), order: VecDeque::new(), total_bytes: 0 }
    }

    /// Returns a clone of the value and marks it as most recently used.
    pub fn get(&mut self, key: u64) -> Option<V> {
        let value = self.entries.get(&key)?.clone();
        self.touch(key);
        Some(value)
    }

    /// Stores a value, evicting older entries until both limits hold.
    ///
    /// A value larger than the whole byte budget is not stored.
    pub fn insert(&mut self, key: u64, value: V) {
        let incoming = value.weight_bytes();
        self.remove(key);
        if incoming > self.limits.max_bytes || self.limits.max_entries == 0 {
            return;
        }
        self.evict_to_fit(incoming);
        self.total_bytes += incoming;
        self.entries.insert(key, value);
        self.order.push_back(key);
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.total_bytes = 0;
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    fn remove(&mut self, key: u64) {
        if let Some(old) = self.entries.remove(&key) {
            self.total_bytes = self.total_bytes.saturating_sub(old.weight_bytes());
            self.order.retain(|k| *k != key);
        }
    }

    fn touch(&mut self, key: u64) {
        self.order.retain(|k| *k != key);
        self.order.push_back(key);
    }

    fn evict_to_fit(&mut self, incoming: usize) {
        while self.entries.len() >= self.limits.max_entries
            || self.total_bytes + incoming > self.limits.max_bytes
        {
            let Some(oldest) = self.order.pop_front() else { break };
            if let Some(old) = self.entries.remove(&oldest) {
                self.total_bytes = self.total_bytes.saturating_sub(old.weight_bytes());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct Blob(usize);

    impl Weighted for Blob {
        fn weight_bytes(&self) -> usize {
            self.0
        }
    }

    fn cache(max_entries: usize, max_bytes: usize) -> LruCache<Blob> {
        LruCache::new(CacheLimits { max_entries, max_bytes })
    }

    #[test]
    fn get_returns_what_was_inserted() {
        let mut cache = cache(4, 1000);

        cache.insert(1, Blob(10));

        assert_eq!(cache.get(1), Some(Blob(10)));
        assert_eq!(cache.get(2), None);
    }

    #[test]
    fn entry_limit_evicts_least_recently_used() {
        let mut cache = cache(2, 1000);
        cache.insert(1, Blob(1));
        cache.insert(2, Blob(1));

        cache.get(1);
        cache.insert(3, Blob(1));

        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_none());
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn byte_limit_evicts_until_the_new_entry_fits() {
        let mut cache = cache(10, 100);
        cache.insert(1, Blob(40));
        cache.insert(2, Blob(40));

        cache.insert(3, Blob(50));

        assert!(cache.get(1).is_none());
        assert_eq!(cache.total_bytes(), 90);
    }

    #[test]
    fn oversized_value_is_not_stored() {
        let mut cache = cache(10, 100);

        cache.insert(1, Blob(500));

        assert!(cache.is_empty());
        assert_eq!(cache.total_bytes(), 0);
    }

    #[test]
    fn reinserting_a_key_replaces_its_weight() {
        let mut cache = cache(10, 100);
        cache.insert(1, Blob(60));

        cache.insert(1, Blob(20));

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.total_bytes(), 20);
    }
}
