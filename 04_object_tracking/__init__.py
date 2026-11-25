"""
NAYRA - Assistive System
Module 04: Real-Time Object Tracking

Exposes the public multi-object tracking interface, Track structures,
and motion state representations.
"""

from .kalman_filter import KalmanBoxTracker, xyah_to_xyxy, xyxy_to_xyah
from .track import MovementDirection, Track, TrackState
from .tracker import ByteTracker, TrackingMetrics

__all__ = [
    "ByteTracker",
    "Track",
    "TrackState",
    "MovementDirection",
    "TrackingMetrics",
    "KalmanBoxTracker",
    "xyxy_to_xyah",
    "xyah_to_xyxy",
]
