"""Hardware acceleration detection and Windows CUDA runtime setup."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def resolve_device(requested_device: str = "auto") -> Tuple[bool, str, Optional[str]]:
    """Determine CUDA availability and configure Windows runtime DLL paths.

    Returns:
        (cuda_available, active_device_name, gpu_name)
    """
    cuda_available = False
    gpu_name: Optional[str] = None

    try:
        import torch

        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.exists():
            try:
                os.add_dll_directory(str(torch_lib.resolve()))
                os.environ["PATH"] = f"{torch_lib.resolve()};{os.environ.get('PATH', '')}"
            except (AttributeError, OSError):
                pass

        if torch.cuda.is_available():
            cuda_available = True
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    target = requested_device.lower()
    if target in ("auto", "cuda") and cuda_available:
        active_device = gpu_name or "CUDA"
    else:
        active_device = "CPU"

    return cuda_available, active_device, gpu_name
