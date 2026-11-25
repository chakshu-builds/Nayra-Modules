"""NAYRA — Public Object Detection Module Entrypoint."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_module_dir = Path(__file__).resolve().parent / "03_object_detection"
__path__ = [str(_module_dir)]

if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

_pkg = importlib.import_module("03_object_detection")

if "object_detection" not in sys.modules:
    sys.modules["object_detection"] = sys.modules[__name__]

for attr in dir(_pkg):
    if not attr.startswith("_"):
        globals()[attr] = getattr(_pkg, attr)

if __name__ == "__main__":
    from scripts.test_object_detection import main
    main()

