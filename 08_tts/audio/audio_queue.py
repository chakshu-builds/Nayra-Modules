"""NAYRA Module 08 — Thread-Safe Bounded Priority Audio Queue.

Manages speech request scheduling with strict priority levels (0=Critical, 1=Perception, 2=Normal),
FIFO ordering within priority tiers, duplicate request suppression, and safety preemption.
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

try:
    from ..request_schema import PriorityLevel, RequestStatus, SpeechRequest
except (ImportError, ValueError):
    from request_schema import PriorityLevel, RequestStatus, SpeechRequest

logger = logging.getLogger("NAYRA-AudioQueue")


class AudioQueue:
    """Thread-safe priority request queue with preemption and duplicate suppression.

    Guarantees:
    1. Critical safety alerts (P0) strictly precede perception (P1) and OCR (P2).
    2. FIFO execution order is preserved within identical priority levels.
    3. Duplicate requests within a configurable time window are suppressed.
    4. Critical requests with interrupt=True purge pending lower-priority OCR requests.
    5. The queue is bounded; overflow drops lowest-priority pending tasks first.
    """

    def __init__(
        self,
        max_size: int = 50,
        duplicate_window_sec: float = 3.0,
    ) -> None:
        self.max_size = max_size
        self.duplicate_window_sec = duplicate_window_sec

        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)

        # Internal heap: entries are tuples of (priority, sequence_id, SpeechRequest)
        self._heap: List[Tuple[int, int, SpeechRequest]] = []
        self._sequence_counter: int = 0

        # Duplicate tracking: text_hash -> timestamp
        self._recent_requests: Dict[str, float] = {}

        self._is_shutdown: bool = False

    def enqueue(self, request: SpeechRequest) -> bool:
        """Enqueue a validated SpeechRequest according to priority and preemption rules.

        Args:
            request: SpeechRequest instance.

        Returns:
            True if successfully enqueued, False if rejected (e.g. duplicate or overflow).
        """
        with self._lock:
            if self._is_shutdown:
                logger.warning("Rejecting request %s: AudioQueue is shut down.", request.request_id)
                request.status = RequestStatus.FAILED
                request.error_message = "AudioQueue is shut down."
                return False

            now = time.time()
            self._purge_stale_duplicates(now)

            # Check duplicate suppression
            text_key = f"{request.priority}:{request.text.strip()}"
            last_seen = self._recent_requests.get(text_key)
            if last_seen is not None and (now - last_seen) < self.duplicate_window_sec:
                logger.info(
                    "Suppressed duplicate request (priority=%d, text='%s').",
                    request.priority,
                    request.text[:30],
                )
                request.status = RequestStatus.CANCELLED
                request.error_message = "Suppressed as duplicate request within time window."
                return False

            # If Critical request (P0) with interrupt=True arrives, purge pending OCR requests (P2)
            if request.priority == PriorityLevel.CRITICAL and request.interrupt:
                purged_count = self._purge_lower_priority_locked(threshold_priority=PriorityLevel.NORMAL)
                if purged_count > 0:
                    logger.info("Critical alert purged %d pending lower-priority requests.", purged_count)

            # Handle queue overflow
            if len(self._heap) >= self.max_size:
                if not self._evict_lowest_priority_locked(request.priority):
                    logger.warning("Queue full (%d items); dropping request %s.", self.max_size, request.request_id)
                    request.status = RequestStatus.FAILED
                    request.error_message = "AudioQueue capacity exceeded."
                    return False

            # Insert into heap
            self._sequence_counter += 1
            heapq.heappush(self._heap, (request.priority, self._sequence_counter, request))
            self._recent_requests[text_key] = now
            request.status = RequestStatus.PENDING

            logger.debug(
                "Enqueued request %s (priority=%d, seq=%d, queue_size=%d).",
                request.request_id,
                request.priority,
                self._sequence_counter,
                len(self._heap),
            )

            self._not_empty.notify()
            return True

    def get(self, timeout: Optional[float] = None) -> Optional[SpeechRequest]:
        """Pop the highest-priority SpeechRequest from the queue.

        Args:
            timeout: Maximum wait time in seconds (None blocks indefinitely).

        Returns:
            SpeechRequest instance, or None if timeout expired or queue is shut down.
        """
        with self._lock:
            start_time = time.time()
            while not self._heap and not self._is_shutdown:
                if timeout is not None:
                    remaining = timeout - (time.time() - start_time)
                    if remaining <= 0:
                        return None
                    self._not_empty.wait(timeout=remaining)
                else:
                    self._not_empty.wait()

            if self._is_shutdown or not self._heap:
                return None

            priority, seq, request = heapq.heappop(self._heap)
            return request

    def _purge_lower_priority_locked(self, threshold_priority: int) -> int:
        """Discard all pending requests with priority >= threshold_priority."""
        retained: List[Tuple[int, int, SpeechRequest]] = []
        purged = 0

        for prio, seq, req in self._heap:
            if prio >= threshold_priority:
                req.status = RequestStatus.CANCELLED
                req.error_message = "Preempted by critical higher-priority alert."
                purged += 1
            else:
                retained.append((prio, seq, req))

        self._heap = retained
        heapq.heapify(self._heap)
        return purged

    def _evict_lowest_priority_locked(self, incoming_priority: int) -> bool:
        """Evict one lowest-priority item if incoming has strictly higher priority."""
        if not self._heap:
            return False

        # Find maximum priority integer (lowest urgency)
        max_prio = max(item[0] for item in self._heap)
        if max_prio <= incoming_priority:
            return False  # Cannot evict; all items are higher or equal priority

        # Remove the latest lowest-priority item
        for i in reversed(range(len(self._heap))):
            if self._heap[i][0] == max_prio:
                evicted = self._heap.pop(i)
                evicted[2].status = RequestStatus.FAILED
                evicted[2].error_message = "Evicted due to queue overflow."
                heapq.heapify(self._heap)
                logger.debug("Evicted request %s to make room for priority %d.", evicted[2].request_id, incoming_priority)
                return True

        return False

    def _purge_stale_duplicates(self, now: float) -> None:
        """Remove duplicate keys older than duplicate_window_sec."""
        stale = [k for k, ts in self._recent_requests.items() if (now - ts) >= self.duplicate_window_sec]
        for k in stale:
            del self._recent_requests[k]

    def clear(self) -> int:
        """Clear all pending requests from the queue."""
        with self._lock:
            count = len(self._heap)
            for _, _, req in self._heap:
                req.status = RequestStatus.CANCELLED
                req.error_message = "Cancelled via queue clear."
            self._heap.clear()
            return count

    def qsize(self) -> int:
        """Return the number of currently pending requests."""
        with self._lock:
            return len(self._heap)

    def is_empty(self) -> bool:
        """Return True if no requests are pending."""
        with self._lock:
            return len(self._heap) == 0

    def empty(self) -> bool:
        """Alias for is_empty() for queue interface compatibility."""
        return self.is_empty()

    def shutdown(self) -> None:
        """Shut down the queue and unblock all waiting workers."""
        with self._lock:
            self._is_shutdown = True
            self.clear()
            self._not_empty.notify_all()
            logger.info("AudioQueue shutdown completed.")
