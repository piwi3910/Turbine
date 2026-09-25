//! The waiting queue: ordered by priority (lower first) then arrival; preempted requests go
//! to the front, ahead of every never-admitted request.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::Duration;

use turbine_core::types::{Priority, RequestId};

/// Ordering key of a never-admitted request: priority, arrival, then submission order.
type Key = (Priority, Duration, u64);

/// Where a queued request sits.
#[derive(Clone, Copy, Debug)]
enum Slot {
    Front,
    Ordered(Key),
}

/// Waiting requests. The bound (`scheduler.max_queued_requests`) is enforced by the scheduler
/// at submission; this type only orders.
#[derive(Debug, Default)]
pub struct WaitingQueue {
    /// Preempted requests, most recently pushed first.
    front: VecDeque<RequestId>,
    ordered: BTreeMap<Key, RequestId>,
    slots: HashMap<RequestId, Slot>,
    next_seq_no: u64,
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

    /// Queue a new request by priority then arrival (ties: submission order).
    pub fn push(&mut self, id: RequestId, priority: Priority, arrival: Duration) {
        let key = (priority, arrival, self.next_seq_no);
        self.next_seq_no += 1;
        self.ordered.insert(key, id);
        self.slots.insert(id, Slot::Ordered(key));
    }

    /// Queue a preempted request ahead of everything else.
    pub fn push_front(&mut self, id: RequestId) {
        self.front.push_front(id);
        self.slots.insert(id, Slot::Front);
    }

    /// The request that would be admitted next.
    pub fn peek(&self) -> Option<RequestId> {
        self.front
            .front()
            .copied()
            .or_else(|| self.ordered.values().next().copied())
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
            Some(Slot::Front) => {
                self.front.retain(|x| *x != id);
                true
            }
            Some(Slot::Ordered(key)) => {
                self.ordered.remove(&key);
                true
            }
        }
    }

    /// Queued requests in admission order.
    pub fn iter(&self) -> impl Iterator<Item = RequestId> + '_ {
        self.front
            .iter()
            .copied()
            .chain(self.ordered.values().copied())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn id(n: u128) -> RequestId {
        RequestId(uuid::Uuid::from_u128(n))
    }

    #[test]
    fn priority_then_arrival_with_preempted_first() {
        let mut q = WaitingQueue::new();
        q.push(id(1), Priority(0), Duration::from_secs(1));
        q.push(id(2), Priority(-1), Duration::from_secs(2));
        q.push(id(3), Priority(0), Duration::from_secs(0));
        q.push(id(4), Priority(0), Duration::from_secs(1)); // tie: submission order
        assert_eq!(q.len(), 4);
        assert_eq!(q.iter().collect::<Vec<_>>(), [id(2), id(3), id(1), id(4)]);

        q.push_front(id(9));
        q.push_front(id(8));
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
