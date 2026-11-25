"""
NAYRA - Assistive System
Module 01: Camera Input Package

Exposes stream receiver, HTTP MJPEG client, frame reader, validation, and resolution handling.
"""

from .android_stream import AndroidStreamReceiver, ConnectionState, HttpMjpegClient, StreamInfo
from .frame_reader import (
    FPSCalculator,
    FrameMetadata,
    FrameReader,
    FrameValidationResult,
    FrameValidator,
    LatestFrameBuffer,
    ResolutionHandler,
)

__all__ = [
    "AndroidStreamReceiver",
    "HttpMjpegClient",
    "ConnectionState",
    "StreamInfo",
    "FrameReader",
    "FrameMetadata",
    "FrameValidationResult",
    "FrameValidator",
    "ResolutionHandler",
    "FPSCalculator",
    "LatestFrameBuffer",
]
