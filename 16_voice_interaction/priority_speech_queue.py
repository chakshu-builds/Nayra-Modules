"""Thread-safe priority speech queue with preemption support."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import logging
import threading
import time
from typing import Any, List, Optional, Tuple

from shared.schemas import SpeechPriority, SpeechRequest

logger = logging.getLogger("NAYRA-PrioritySpeechQueue")


class PrioritySpeechQueue:
    """Heap-based priority queue for speech synthesis requests with size bounding."""

    def __init__(self, max_size: int = 100) -> None:
        self.max_size = max_size
        self._heap: List[Tuple[int, float, int, SpeechRequest]] = []
        self._lock = threading.RLock()
        self._counter = 0
        self._current_priority: Optional[SpeechPriority] = None

    def push(self, request: SpeechRequest) -> bool:
        """Enqueue request. Returns True if this request should preempt currently playing speech."""
        with self._lock:
            # Enforce max queue capacity to prevent unbounded memory growth
            if len(self._heap) >= self.max_size:
                # Find worst (highest numerical value) priority item
                worst_idx = max(range(len(self._heap)), key=lambda i: (self._heap[i][0], -self._heap[i][1]))
                worst_prio = self._heap[worst_idx][0]
                if int(request.priority.value) < worst_prio:
                    # Drop the worst item to accommodate higher priority
                    logger.warning(
                        f"Speech queue full ({self.max_size}). Dropping lower priority item: {self._heap[worst_idx][3].text[:20]}"
                    )
                    self._heap.pop(worst_idx)
                    heapq.heapify(self._heap)
                else:
                    logger.warning(
                        f"Speech queue full ({self.max_size}). Dropping new incoming item: {request.text[:20]}"
                    )
                    return False

            self._counter += 1
            # Priority tuple: (priority_value, request_timestamp, counter, request)
            heapq.heappush(self._heap, (int(request.priority.value), request.created_at, self._counter, request))
            logger.debug(f"Queued speech request: '{request.text[:30]}...' with priority {request.priority.name}")

            # Preemption check: if new priority is strictly higher than currently playing
            if self._current_priority is not None and request.priority.value < self._current_priority.value:
                logger.info(f"Preemption triggered: {request.priority.name} preempts {self._current_priority.name}")
                return True
            return False

    def pop(self) -> Optional[SpeechRequest]:
        """Pop the highest priority pending request."""
        with self._lock:
            if not self._heap:
                return None
            _, _, _, req = heapq.heappop(self._heap)
            self._current_priority = req.priority
            return req

    def peek(self) -> Optional[SpeechRequest]:
        """Inspect highest priority item without popping."""
        with self._lock:
            if not self._heap:
                return None
            return self._heap[0][3]

    def clear(self) -> None:
        """Purge all pending requests."""
        with self._lock:
            self._heap.clear()
            self._current_priority = None

    def cancel_matching(self, filter_fn: Any) -> int:
        """Remove queued requests satisfying filter_fn(request). Returns count of removed items."""
        with self._lock:
            initial_count = len(self._heap)
            self._heap = [item for item in self._heap if not filter_fn(item[3])]
            if len(self._heap) < initial_count:
                heapq.heapify(self._heap)
            return initial_count - len(self._heap)

    def mark_completed(self) -> None:
        """Mark currently playing item as finished."""
        with self._lock:
            self._current_priority = None

    @property
    def current_priority(self) -> Optional[SpeechPriority]:
        with self._lock:
            return self._current_priority

    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)
