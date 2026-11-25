"""NAYRA Unified System Exceptions."""

from __future__ import annotations


class NayraException(Exception):
    """Base exception for all NAYRA subsystems."""
    pass


class PerceptionUnavailableError(NayraException):
    """Raised when camera, detector, or depth pipeline is not initialized."""
    pass


class AudioDeviceError(NayraException):
    """Raised when microphone or audio output fails."""
    pass


class IntentRoutingError(NayraException):
    """Raised when an intent cannot be routed to any registered module."""
    pass
