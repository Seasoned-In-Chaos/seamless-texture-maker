//! Device health: notices a lost wgpu device or an uncaptured wgpu error, so
//! callers stop issuing GPU work and fall back to the CPU instead of panicking.
//!
//! wgpu's default uncaptured-error handler panics, and `Device::poll` panics
//! on a lost device. [`DeviceHealth::watch`] replaces the handler with one
//! that records the fault and registers the device-lost callback;
//! [`DeviceHealth::wait`] polls only while the device is healthy and catches
//! the poll's panic as a last resort. Every GPU entry point checks
//! [`DeviceHealth::fault`] first.
//!
//! The approach follows PhotoCraft's `photocraft-gpu::health`.

use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

/// Longest fault description kept, in characters.
const MAX_DETAIL_CHARS: usize = 200;

/// Why the GPU can no longer be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    /// The device was lost: a driver reset, or the system ended a submission.
    Lost(String),
    /// An uncaptured wgpu error (out of memory, validation, internal). The
    /// output can no longer be trusted.
    Error(String),
}

impl Fault {
    pub fn is_lost(&self) -> bool {
        matches!(self, Fault::Lost(_))
    }
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fault::Lost(detail) if detail.is_empty() => f.write_str("GPU device lost"),
            Fault::Lost(detail) => write!(f, "GPU device lost: {detail}"),
            Fault::Error(detail) => write!(f, "GPU error: {detail}"),
        }
    }
}

#[derive(Default)]
struct Inner {
    faulted: AtomicBool,
    fault: Mutex<Option<Fault>>,
}

/// The shared health flag of one wgpu device. Cheap to clone; every clone
/// sees the same state.
#[derive(Clone, Default)]
pub struct DeviceHealth(Arc<Inner>);

impl fmt::Debug for DeviceHealth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceHealth").field("fault", &self.fault()).finish()
    }
}

impl DeviceHealth {
    /// A healthy flag that is not attached to a device.
    pub fn new() -> Self {
        Self::default()
    }

    /// Watches `device`: a device loss or an uncaptured error marks the
    /// returned flag instead of panicking. Replaces any handlers set before,
    /// because wgpu keeps one of each per device.
    pub fn watch(device: &wgpu::Device) -> Self {
        let health = Self::new();
        let lost = health.clone();
        // wgpu calls this with its own locks held: record, never call back in.
        device.set_device_lost_callback(move |reason, message| {
            let detail = match (reason, message.is_empty()) {
                (wgpu::DeviceLostReason::Destroyed, true) => "device destroyed".to_owned(),
                (_, true) => "lost by the driver".to_owned(),
                (_, false) => message,
            };
            log::error!("wgpu device lost ({reason:?}): {detail}");
            lost.mark(Fault::Lost(detail));
        });
        let errors = health.clone();
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| errors.uncaptured(&error)));
        health
    }

    fn uncaptured(&self, error: &wgpu::Error) {
        let text = error.to_string();
        log::error!("uncaptured wgpu error: {text}");
        // Errors about invalid objects follow a device loss whose own
        // callback may not have run yet; it fires on the next poll.
        let lower = text.to_ascii_lowercase();
        let lost = lower.contains("lost") || lower.contains("destroyed");
        let detail = first_line(&text);
        self.mark(if lost { Fault::Lost(detail) } else { Fault::Error(detail) });
    }

    /// Marks the device unusable. The first fault is kept, except that a
    /// later device loss replaces an earlier error.
    pub fn mark(&self, fault: Fault) {
        let mut slot = self.0.fault.lock().unwrap_or_else(PoisonError::into_inner);
        let replace = slot.as_ref().is_none_or(|old| !old.is_lost() && fault.is_lost());
        if replace {
            *slot = Some(fault);
        }
        self.0.faulted.store(true, Ordering::Release);
    }

    /// Whether the device may still be used.
    pub fn is_ok(&self) -> bool {
        !self.0.faulted.load(Ordering::Acquire)
    }

    /// Why the device cannot be used, if it cannot.
    pub fn fault(&self) -> Option<Fault> {
        if self.is_ok() {
            return None;
        }
        self.0.fault.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// Blocks until all submitted work has finished, unless the device is
    /// unusable. Returns whether the device is still healthy afterwards.
    pub fn wait(&self, device: &wgpu::Device) -> bool {
        if !self.is_ok() {
            return false;
        }
        let wait = wgpu::PollType::Wait { submission_index: None, timeout: None };
        match catch_unwind(AssertUnwindSafe(|| device.poll(wait))) {
            Ok(Ok(_)) => {}
            Ok(Err(error)) => log::warn!("wgpu poll: {error}"),
            Err(_) => self.mark(Fault::Lost("the device stopped responding".to_owned())),
        }
        self.is_ok()
    }
}

/// The first non-empty line of a message, shortened to a readable length.
fn first_line(text: &str) -> String {
    let line = text.lines().map(str::trim).find(|line| !line.is_empty()).unwrap_or("");
    let mut out: String = line.chars().take(MAX_DETAIL_CHARS).collect();
    if line.chars().count() > MAX_DETAIL_CHARS {
        out.push('…');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_flag_is_healthy() {
        let health = DeviceHealth::new();

        assert!(health.is_ok());
        assert_eq!(health.fault(), None);
    }

    #[test]
    fn the_first_fault_is_kept_and_seen_by_every_clone() {
        let health = DeviceHealth::new();
        let clone = health.clone();

        clone.mark(Fault::Error("out of memory".to_owned()));
        clone.mark(Fault::Error("second".to_owned()));

        assert!(!health.is_ok());
        assert_eq!(health.fault(), Some(Fault::Error("out of memory".to_owned())));
    }

    #[test]
    fn a_device_loss_replaces_an_error_but_not_an_earlier_loss() {
        let health = DeviceHealth::new();
        health.mark(Fault::Error("out of memory".to_owned()));

        health.mark(Fault::Lost("reset".to_owned()));
        health.mark(Fault::Lost("later".to_owned()));

        assert_eq!(health.fault(), Some(Fault::Lost("reset".to_owned())));
    }

    #[test]
    fn faults_read_as_plain_sentences() {
        assert_eq!(Fault::Lost(String::new()).to_string(), "GPU device lost");
        assert_eq!(Fault::Lost("reset".to_owned()).to_string(), "GPU device lost: reset");
        assert_eq!(Fault::Error("x".to_owned()).to_string(), "GPU error: x");
    }

    #[test]
    fn fault_details_are_one_trimmed_line_of_bounded_length() {
        assert_eq!(first_line("\n  Buffer is invalid\nmore"), "Buffer is invalid");
        assert_eq!(first_line(&"a".repeat(300)).chars().count(), MAX_DETAIL_CHARS + 1);
    }
}
