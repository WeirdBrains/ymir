//! Device fairness (M6): one device's batch work can't fill the node's queue and leave the
//! user's other devices with `Busy`. Interactive requests are never held back (the admission
//! queue already serves them first); each device may have at most `cap` batch or background
//! requests running or waiting here, and a request past that is answered `Busy`, which a batch
//! client treats as "try again shortly".

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use weirdnet_agent::remote::RemoteError;

#[derive(Default)]
struct Inner {
    cap: AtomicUsize,
    held: Mutex<HashMap<String, usize>>,
}

#[derive(Clone, Default)]
pub struct Fair(Arc<Inner>);

/// A batch request counted against its device until dropped.
pub struct Hold {
    fair: Fair,
    device: String,
}

impl Fair {
    /// No cap until [`Fair::set_cap`].
    pub fn new() -> Self {
        let f = Self::default();
        f.set_cap(usize::MAX);
        f
    }

    pub fn set_cap(&self, cap: usize) {
        self.0.cap.store(cap.max(1), Ordering::SeqCst);
    }

    /// Counts a batch request from `device`, or `Busy` if it already has `cap`.
    pub fn enter(&self, device: &str) -> Result<Hold, RemoteError> {
        let mut held = self.0.held.lock().unwrap_or_else(|e| e.into_inner());
        let n = held.entry(device.to_string()).or_default();
        if *n >= self.0.cap.load(Ordering::SeqCst) {
            return Err(RemoteError::Busy);
        }
        *n += 1;
        Ok(Hold { fair: self.clone(), device: device.to_string() })
    }

    pub fn held(&self, device: &str) -> usize {
        self.0.held.lock().unwrap_or_else(|e| e.into_inner()).get(device).copied().unwrap_or(0)
    }
}

impl Drop for Hold {
    fn drop(&mut self) {
        let mut held = self.fair.0.held.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(n) = held.get_mut(&self.device) {
            *n -= 1;
            if *n == 0 {
                held.remove(&self.device);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_device_is_capped_alone_and_the_slot_returns_on_drop() {
        let f = Fair::new();
        f.set_cap(2);
        let a1 = f.enter("a").unwrap();
        let _a2 = f.enter("a").unwrap();
        assert!(matches!(f.enter("a"), Err(RemoteError::Busy)));
        let _b = f.enter("b").expect("another device is not held back");
        drop(a1);
        assert_eq!(f.held("a"), 1);
        f.enter("a").unwrap();
    }
}
