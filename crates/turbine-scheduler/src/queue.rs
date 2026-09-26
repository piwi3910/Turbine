//! The waiting queue: ordered by the [`AdmissionKey`] the scheduling policy assigns (smallest
//! first). Under the `default` policy that is priority (lower first) then arrival, with
//! preempted requests ahead of every never-admitted request.

use std::collections::{BTreeMap, HashMap};

use turbine_core::types::RequestId;

use crate::policy::AdmissionKey;

/// Waiting requests. The bound (`scheduler.max_queued_requests`) is enforced by the scheduler
/// at submission; this type only orders.
#[derive(Debug, Default)]
pub struct WaitingQueue {
    ordered: BTreeMap<AdmissionKey, RequestId>,
    slots: HashMap<RequestId, AdmissionKey>,
    next_order: u64,
}

impl WaitingQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn contains(&self, id: RequestId) -> bool {
        self.slots.contains_key(&id)
    }

    /// The submission / push counter: a fresh value on every call, for the policy's
    /// `AdmissionInfo::submit_no` and `push_no`.
    pub fn next_order(&mut self) -> u64 {
        let n = self.next_order;
        self.next_order += 1;
        n
    }

    /// Queue `id` at `key` (requeueing it when already queued). Keys are unique per queued
    /// request; a key equal to another request's replaces that request's position.
    pub fn push(&mut self, id: RequestId, key: AdmissionKey) {
        self.remove(id);
        if let Some(displaced) = self.ordered.insert(key, id) {
            tracing::error!(event = "scheduler_bug", request_id = %displaced.0, "duplicate admission key");
            debug_assert!(false, "duplicate admission key {key:?}");
            self.slots.remove(&displaced);
        }
        self.slots.insert(id, key);
    }

    /// The request that would be admitted next.
    pub fn peek(&self) -> Option<RequestId> {
        self.ordered.values().next().copied()
    }

    pub fn pop(&mut self) -> Option<RequestId> {
        let id = self.peek()?;
        self.remove(id);
        Some(id)
    }

    /// Removes `id` wherever it sits; false when it is not queued.
    pub fn remove(&mut self, id: RequestId) -> bool {
        match self.slots.remove(&id) {
            None => false,
            Some(key) => {
                self.ordered.remove(&key);
                true
            }
        }
    }

    /// Queued requests in admission order.
    pub fn iter(&self) -> impl Iterator<Item = RequestId> + '_ {
        self.ordered.values().copied()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use turbine_core::types::Priority;

    use super::*;
    use crate::policy::{AdmissionInfo, DefaultPolicy, SchedulingPolicy};

    fn id(n: u128) -> RequestId {
        RequestId(uuid::Uuid::from_u128(n))
    }

    /// Queue `n` as the default policy does: a new submission, or a preemption.
    fn push(q: &mut WaitingQueue, n: u128, priority: i32, arrival_s: u64, preempted: bool) {
        let order = q.next_order();
        let key = DefaultPolicy.admission_key(&AdmissionInfo {
            priority: Priority(priority),
            arrival: Duration::from_secs(arrival_s),
            submit_no: order,
            preempted,
            push_no: order,
        });
        q.push(id(n), key);
    }

    #[test]
    fn priority_then_arrival_with_preempted_first() {
        let mut q = WaitingQueue::new();
        push(&mut q, 1, 0, 1, false);
        push(&mut q, 2, -1, 2, false);
        push(&mut q, 3, 0, 0, false);
        push(&mut q, 4, 0, 1, false); // tie: submission order
        assert_eq!(q.len(), 4);
        assert_eq!(q.iter().collect::<Vec<_>>(), [id(2), id(3), id(1), id(4)]);

        push(&mut q, 9, 5, 0, true);
        push(&mut q, 8, 5, 0, true);
        assert_eq!(q.peek(), Some(id(8)));
        assert_eq!(q.pop(), Some(id(8)));
        assert_eq!(q.pop(), Some(id(9)));
        assert!(q.remove(id(3)));
        assert!(!q.remove(id(3)));
        assert_eq!(q.iter().collect::<Vec<_>>(), [id(2), id(1), id(4)]);
        assert!(q.contains(id(1)));
        assert_eq!(q.pop(), Some(id(2)));
        assert_eq!(q.len(), 2);
    }
}
