"""
NAYRA - Assistive System
Module 02: Camera Manager Package

Exposes the unified CameraManager facade and CameraStatus diagnostics.
"""

from .camera_manager import CameraManager, CameraStatus

__all__ = [
    "CameraManager",
    "CameraStatus",
]
