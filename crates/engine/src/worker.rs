//! Background worker threads fed through a mailbox.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

/// What happens to requests that are still waiting when a new one arrives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Only the newest request matters: waiting ones are discarded.
    LatestOnly,
    /// Every request is processed, in order.
    Queue,
}

/// The requests waiting for a worker.
pub struct Mailbox<T> {
    policy: Policy,
    waiting: Mutex<VecDeque<T>>,
    arrived: Condvar,
    closed: AtomicBool,
}

impl<T> Mailbox<T> {
    fn new(policy: Policy) -> Self {
        Self {
            policy,
            waiting: Mutex::new(VecDeque::new()),
            arrived: Condvar::new(),
            closed: AtomicBool::new(false),
        }
    }

    /// A poisoned lock only means another thread panicked mid-update of a
    /// plain queue; the queue itself is still usable.
    fn lock(&self) -> MutexGuard<'_, VecDeque<T>> {
        self.waiting.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn post(&self, request: T) {
        let mut waiting = self.lock();
        if self.policy == Policy::LatestOnly {
            waiting.clear();
        }
        waiting.push_back(request);
        self.arrived.notify_one();
    }

    /// Blocks until a request arrives; `None` once the mailbox is closed.
    fn take(&self) -> Option<T> {
        let mut waiting = self.lock();
        loop {
            if self.closed.load(Ordering::Acquire) {
                return None;
            }
            if let Some(request) = waiting.pop_front() {
                return Some(request);
            }
            waiting = self.arrived.wait(waiting).unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.lock().clear();
        self.arrived.notify_all();
    }

    /// True when a newer request is waiting, so the current one is outdated.
    pub fn has_waiting(&self) -> bool {
        !self.lock().is_empty()
    }
}

/// A named thread that handles requests one at a time.
///
/// Dropping the worker closes its mailbox; the thread exits after the request
/// it is handling. It is not joined, so shutdown never waits on a long render.
pub struct Worker<T> {
    mailbox: Arc<Mailbox<T>>,
}

impl<T: Send + 'static> Worker<T> {
    /// Starts the thread. `handle` receives each request and the mailbox, so a
    /// long job can stop early when [`Mailbox::has_waiting`] turns true.
    pub fn spawn(
        name: &str,
        policy: Policy,
        mut handle: impl FnMut(T, &Mailbox<T>) + Send + 'static,
    ) -> std::io::Result<Self> {
        let mailbox = Arc::new(Mailbox::new(policy));
        let inbox = Arc::clone(&mailbox);
        std::thread::Builder::new().name(name.to_owned()).spawn(move || {
            while let Some(request) = inbox.take() {
                handle(request, &inbox);
            }
        })?;
        Ok(Self { mailbox })
    }

    pub fn post(&self, request: T) {
        self.mailbox.post(request);
    }
}

impl<T> Drop for Worker<T> {
    fn drop(&mut self) {
        self.mailbox.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    const WAIT: Duration = Duration::from_secs(5);

    #[test]
    fn queue_policy_handles_every_request_in_order() {
        let (sender, receiver) = mpsc::channel();
        let worker = Worker::spawn("test-queue", Policy::Queue, move |n: u32, _| {
            sender.send(n).unwrap();
        })
        .unwrap();

        (1..=5).for_each(|n| worker.post(n));
        let handled: Vec<u32> = (0..5).map(|_| receiver.recv_timeout(WAIT).unwrap()).collect();

        assert_eq!(handled, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn latest_only_policy_skips_requests_that_were_overtaken() {
        let (started, wait_started) = mpsc::channel();
        let (release, wait_release) = mpsc::channel::<()>();
        let (sender, receiver) = mpsc::channel();
        let worker = Worker::spawn("test-latest", Policy::LatestOnly, move |n: u32, _| {
            if n == 1 {
                started.send(()).unwrap();
                wait_release.recv_timeout(WAIT).unwrap();
            }
            sender.send(n).unwrap();
        })
        .unwrap();

        worker.post(1);
        wait_started.recv_timeout(WAIT).unwrap();
        (2..=9).for_each(|n| worker.post(n));
        release.send(()).unwrap();

        assert_eq!(receiver.recv_timeout(WAIT).unwrap(), 1);
        assert_eq!(receiver.recv_timeout(WAIT).unwrap(), 9);
        assert!(receiver.recv_timeout(Duration::from_millis(100)).is_err());
    }

    #[test]
    fn handler_can_see_that_it_was_superseded() {
        let (started, wait_started) = mpsc::channel();
        let (release, wait_release) = mpsc::channel::<()>();
        let (sender, receiver) = mpsc::channel();
        let worker =
            Worker::spawn("test-superseded", Policy::LatestOnly, move |n: u32, mailbox| {
                if n == 1 {
                    started.send(()).unwrap();
                    wait_release.recv_timeout(WAIT).unwrap();
                }
                sender.send((n, mailbox.has_waiting())).unwrap();
            })
            .unwrap();

        worker.post(1);
        wait_started.recv_timeout(WAIT).unwrap();
        worker.post(2);
        release.send(()).unwrap();

        assert_eq!(receiver.recv_timeout(WAIT).unwrap(), (1, true));
        assert_eq!(receiver.recv_timeout(WAIT).unwrap(), (2, false));
    }

    #[test]
    fn dropping_the_worker_stops_its_thread() {
        let (sender, receiver) = mpsc::channel::<u32>();
        let worker = Worker::spawn("test-drop", Policy::Queue, move |n, _| {
            let _ = sender.send(n);
        })
        .unwrap();

        drop(worker);

        assert!(receiver.recv_timeout(WAIT).is_err(), "sender should be dropped with the thread");
    }
}
