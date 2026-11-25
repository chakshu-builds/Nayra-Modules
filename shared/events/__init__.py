"""NAYRA Lightweight Thread-Safe Event Bus."""

from __future__ import annotations

import collections
import logging
import threading
from typing import Any, Callable, Dict, List

logger = logging.getLogger("NAYRA-EventBus")


class EventBus:
    """Thread-safe publish/subscribe event dispatcher."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._subscribers = collections.defaultdict(list)
                cls._instance._sub_lock = threading.RLock()
            return cls._instance

    def subscribe(self, event_type: str, callback: Callable[[Any], None]) -> None:
        with self._sub_lock:
            if callback not in self._subscribers[event_type]:
                self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable[[Any], None]) -> None:
        with self._sub_lock:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)

    def publish(self, event_type: str, payload: Any) -> None:
        with self._sub_lock:
            callbacks = list(self._subscribers.get(event_type, []))

        for cb in callbacks:
            try:
                cb(payload)
            except Exception as exc:
                logger.error("Error executing subscriber for event '%s': %s", event_type, exc)
