"""Pytest configuration for Module 08 TTS tests."""

import importlib
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Ensure 'tts' alias is loaded
import tts
