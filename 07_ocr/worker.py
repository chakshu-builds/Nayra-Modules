"""NAYRA Module 07 — Asynchronous On-Demand OCR Worker.

Provides thread-safe, non-blocking execution of the unified OCR pipeline
for interactive camera applications (such as main.py).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

try:
    from .pipeline.ocr_engine import OCREngine
    from .pipeline.result_schema import OCRResult, PrimaryLanguage, RegionStatus
except (ImportError, ValueError):
    from pipeline.ocr_engine import OCREngine
    from pipeline.result_schema import OCRResult, PrimaryLanguage, RegionStatus

logger = logging.getLogger("NAYRA-OCRWorker")


class AsyncOCRWorker:
    """Non-blocking background worker for on-demand OCR requests.

    Guarantees:
    - Never blocks the camera acquisition or preview thread (maintains 30 FPS).
    - Takes a snapshot copy (frame.copy()) to prevent memory mutation races.
    - Rejects duplicate concurrent requests while busy.
    - Communicates results back via thread-safe accessors.
    - Retains the latest result for a configurable display duration.
    - Prints full recognized text and stage breakdown to the terminal.
    """

    def __init__(
        self,
        preferred_provider: str = "CUDA",
        display_duration_sec: float = 8.0,
        engine: Optional[OCREngine] = None,
        on_result_callback: Optional[Callable[[OCRResult], None]] = None,
    ) -> None:
        """Initialize the asynchronous OCR worker.

        Args:
            preferred_provider: 'CUDA' or 'CPU'.
            display_duration_sec: Time in seconds to keep results active on HUD.
            engine: Optional pre-initialized OCREngine instance.
            on_result_callback: Optional callback invoked when OCR completes.
        """
        self.preferred_provider = preferred_provider.upper()
        self.display_duration_sec = float(display_duration_sec)
        self.on_result_callback = on_result_callback

        self._engine = engine
        self._lock = threading.Lock()
        self._busy = False
        self._worker_thread: Optional[threading.Thread] = None
        self._latest_result: Optional[OCRResult] = None
        self._result_timestamp: float = 0.0
        self._request_count: int = 0
        self._rejected_count: int = 0

    @property
    def is_busy(self) -> bool:
        """Check whether the worker is actively processing an OCR request."""
        with self._lock:
            return self._busy

    def trigger(self, frame: np.ndarray) -> bool:
        """Trigger an on-demand OCR request on the latest camera frame.

        Args:
            frame: OpenCV BGR uint8 NumPy array.

        Returns:
            True if the request was accepted and dispatched to the background thread.
            False if rejected (worker busy, invalid frame, or engine uninitialized).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.warning("[OCR] Trigger rejected: input frame is invalid or empty.")
            return False

        with self._lock:
            if self._busy:
                self._rejected_count += 1
                logger.warning(
                    "[OCR] Request ignored: worker is currently busy processing a previous frame (rejected=%d).",
                    self._rejected_count,
                )
                return False
            self._busy = True
            self._request_count += 1
            req_id = self._request_count

        # Create an independent snapshot copy to avoid frame-mutation race conditions
        snapshot = frame.copy()

        self._worker_thread = threading.Thread(
            target=self._run_inference_worker,
            args=(snapshot, req_id),
            name=f"NayraOCRWorker-{req_id}",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("[OCR] Dispatched on-demand OCR request #%d (frame: %dx%d).", req_id, snapshot.shape[1], snapshot.shape[0])
        return True

    def _run_inference_worker(self, snapshot: np.ndarray, req_id: int) -> None:
        """Background thread worker executing the OCR pipeline."""
        t_start = time.perf_counter()
        result: Optional[OCRResult] = None

        try:
            # Ensure engine is loaded
            if self._engine is None:
                from . import get_ocr_engine
                self._engine = get_ocr_engine(preferred_provider=self.preferred_provider)

            result = self._engine.process_image(snapshot)

        except Exception as exc:
            logger.error("[OCR] Inference error on request #%d: %s", req_id, exc, exc_info=True)
            result = OCRResult(
                original_shape=snapshot.shape[:2] if snapshot is not None else (0, 0),
                execution_providers={"worker": "Error"},
                success=False,
                error_message=f"Inference exception: {exc}",
                total_time_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        finally:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            if result is not None and result.total_time_ms == 0.0:
                result.total_time_ms = elapsed_ms

            # Print terminal output
            if result is not None:
                self._print_terminal_summary(result, req_id)

            with self._lock:
                self._latest_result = result
                self._result_timestamp = time.monotonic()
                self._busy = False

            if self.on_result_callback and result is not None:
                try:
                    self.on_result_callback(result)
                except Exception as cb_err:
                    logger.warning("[OCR] Result callback error: %s", cb_err)

    def _print_terminal_summary(self, res: OCRResult, req_id: int) -> None:
        """Format and print full recognized text and telemetry to terminal."""
        sep = "=" * 68
        lines = [
            f"\n{sep}",
            f"  [NAYRA ON-DEMAND OCR] Request #{req_id} — {'SUCCESS' if res.success else 'FAILED'}",
            f"{sep}",
            f"  Primary Language : {res.primary_language.value}",
            f"  Regions Detected : {res.detection_count}  |  Recognized: {res.recognition_count}  |  Unresolved: {res.unresolved_count}",
            f"  Total Latency    : {res.total_time_ms:.1f} ms  (Detection: {res.detection_time_ms:.1f} ms | Recognition: {res.recognition_time_ms:.1f} ms)",
            f"  Providers        : {', '.join(f'{k}={v}' for k, v in res.execution_providers.items())}",
        ]
        if not res.success:
            lines.append(f"  Error Message    : {res.error_message}")

        lines.append(f"{'-' * 68}")
        lines.append("  Recognized Text:")

        if res.combined_text.strip():
            for text_line in res.combined_text.splitlines():
                lines.append(f"    >> {text_line}")
        else:
            lines.append("    (No text detected or recognized in scene)")

        lines.append(f"{sep}\n")
        print("\n".join(lines), flush=True)

    def get_active_result(self) -> Optional[OCRResult]:
        """Return the latest OCR result if within the active display duration.

        Returns:
            OCRResult if recent, None if display duration has elapsed.
        """
        with self._lock:
            if self._latest_result is not None:
                if (time.monotonic() - self._result_timestamp) <= self.display_duration_sec:
                    return self._latest_result
            return None

    def get_latest_result(self) -> Optional[OCRResult]:
        """Return the latest OCR result regardless of display expiration."""
        with self._lock:
            return self._latest_result

    def shutdown(self, timeout: float = 2.0) -> None:
        """Gracefully wait for any ongoing background OCR task to complete."""
        thread = self._worker_thread
        if thread is not None and thread.is_alive():
            logger.info("[OCR] Waiting for background OCR worker thread to finish...")
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("[OCR] Worker thread did not terminate within timeout.")
