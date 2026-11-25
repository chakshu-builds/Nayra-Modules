"""NAYRA Module 07 — Unified Multilingual OCR Engine.

Orchestrates:
1. DBNet Text Detection (CUDA on RTX 4050 / CPU fallback)
2. Perspective-Corrected Text Cropping
3. Standalone Script Routing (Devanagari, Latin, Shared Numeric, Unknown)
4. Dedicated Line Recognizers (Hindi PP-OCRv5 & English PP-OCRv5)
5. Reading-Order Sorting & Formatted Text Assembly
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import unicodedata

import cv2
import numpy as np

try:
    from ..detector.dbnet_detector import DBNetDetector, DetectionResult
    from ..poc_english_ocr import EnglishOCRModel
    from ..poc_hindi_ocr import HindiOCRModel
    from ..router.script_router import RouterResult, ScriptRouter, ScriptType
except (ImportError, ValueError):
    from detector.dbnet_detector import DBNetDetector, DetectionResult
    from poc_english_ocr import EnglishOCRModel
    from poc_hindi_ocr import HindiOCRModel
    from router.script_router import RouterResult, ScriptRouter, ScriptType

from .batch_recognizer import predict_batch
from .reading_order import format_combined_text, sort_regions_reading_order
from .result_schema import OCRResult, PrimaryLanguage, RegionStatus, TextRegion
from .text_cropper import extract_perspective_crop

logger = logging.getLogger("NAYRA-OCREngine")


class OCREngine:
    """Unified multilingual OCR engine integrating DBNet, ScriptRouter, and PP-OCRv5 recognizers.

    Attributes:
        preferred_provider: 'CUDA' or 'CPU'.
        detector: Initialized DBNetDetector instance.
        router: Initialized ScriptRouter instance.
        hindi_model: Initialized Hindi PP-OCRv5 recognition model.
        english_model: Initialized English PP-OCRv5 recognition model.
        execution_providers: Active execution providers map.
    """

    def __init__(
        self,
        preferred_provider: str = "CUDA",
        detector_model_path: Optional[str | Path] = None,
        detector_config_path: Optional[str | Path] = None,
        hindi_model_path: Optional[str | Path] = None,
        hindi_dict_path: Optional[str | Path] = None,
        english_model_path: Optional[str | Path] = None,
        english_dict_path: Optional[str | Path] = None,
        det_max_side: int = 960,
        det_thresh: float = 0.3,
        det_box_thresh: float = 0.6,
        det_unclip_ratio: float = 1.5,
        sort_reading_order_enabled: bool = True,
        use_batching: bool = True,
    ) -> None:
        """Initialize the OCR pipeline and warm up persistent ONNX sessions."""
        self.preferred_provider = preferred_provider.upper()
        self.sort_reading_order_enabled = bool(sort_reading_order_enabled)
        self.use_batching = bool(use_batching)
        use_cuda = self.preferred_provider == "CUDA"

        logger.info("Initializing NAYRA Unified OCR Engine (Preferred Provider: %s)...", self.preferred_provider)

        # 1. Text Line Detector
        self.detector = DBNetDetector(
            model_path=detector_model_path,
            config_path=detector_config_path,
            preferred_provider=self.preferred_provider,
            max_side_len=det_max_side,
            thresh=det_thresh,
            box_thresh=det_box_thresh,
            unclip_ratio=det_unclip_ratio,
            sort_reading_order=False,  # Engine handles unified reading order after crop recognition
        )

        # 2. Standalone Script Router
        self.router = ScriptRouter()

        # 3. Hindi Line Recognizer
        self.hindi_model = HindiOCRModel(
            model_path=hindi_model_path or "models/ocr/devanagari_PP-OCRv5_mobile_rec.onnx",
            dict_path=hindi_dict_path or "models/ocr/devanagari_PP-OCRv5_mobile_rec.yml",
            use_cuda=use_cuda,
        )

        # 4. English Line Recognizer
        self.english_model = EnglishOCRModel(
            model_path=english_model_path or "models/ocr/en_PP-OCRv5_mobile_rec.onnx",
            dict_path=english_dict_path or "models/ocr/en_PP-OCRv5_mobile_rec.yml",
            use_cuda=use_cuda,
        )

        self.execution_providers = {
            "detector": self.detector.active_provider,
            "recognizer_hindi": self.hindi_model.active_providers[0] if self.hindi_model.active_providers else "CPUExecutionProvider",
            "recognizer_english": self.english_model.active_providers[0] if self.english_model.active_providers else "CPUExecutionProvider",
        }
        logger.info("NAYRA Unified OCR Engine initialized successfully. Active Providers: %s", self.execution_providers)

    def process_image(self, image: Any) -> OCRResult:
        """Execute full end-to-end OCR pipeline on an input image.

        Args:
            image: OpenCV BGR uint8 numpy array of shape (H, W, 3).

        Returns:
            Structured OCRResult containing all detected regions, recognized text,
            reading-order metadata, and per-stage latency telemetry.
        """
        t_total_start = time.perf_counter()

        # Validate input
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return OCRResult(
                original_shape=(0, 0),
                execution_providers=self.execution_providers,
                success=False,
                error_message="Invalid image input: expected a non-empty numpy.ndarray.",
            )

        if image.ndim != 3 or image.shape[2] != 3:
            return OCRResult(
                original_shape=image.shape[:2],
                execution_providers=self.execution_providers,
                success=False,
                error_message=f"Invalid image format: expected 3-channel BGR image, got shape {image.shape}.",
            )

        orig_h, orig_w = image.shape[:2]

        try:
            # -------------------------------------------------------------
            # Stage 1: Text Line Detection
            # -------------------------------------------------------------
            det_res = self.detector.detect(image)
            det_time_ms = det_res.total_time_ms

            if not det_res.success:
                return OCRResult(
                    original_shape=(orig_h, orig_w),
                    detection_time_ms=det_time_ms,
                    total_time_ms=(time.perf_counter() - t_total_start) * 1000.0,
                    execution_providers=self.execution_providers,
                    success=False,
                    error_message=f"Detection failed: {det_res.error_message}",
                )

            # Fast return on zero detections
            if det_res.num_detections == 0:
                return OCRResult(
                    original_shape=(orig_h, orig_w),
                    primary_language=PrimaryLanguage.NONE,
                    detection_count=0,
                    recognition_count=0,
                    unresolved_count=0,
                    detection_time_ms=det_time_ms,
                    total_time_ms=(time.perf_counter() - t_total_start) * 1000.0,
                    execution_providers=self.execution_providers,
                    success=True,
                )

            # -------------------------------------------------------------
            # Stage 2 & 3: Crop Extraction, Script Routing & Recognition
            # -------------------------------------------------------------
            regions: List[TextRegion] = []
            total_routing_ms = 0.0
            total_recognition_ms = 0.0

            # Queue for recognition
            rec_queue: List[Dict[str, Any]] = []

            for i in range(det_res.num_detections):
                box_4pt = det_res.boxes_4pt[i]
                box_xyxy = det_res.boxes_xyxy[i]
                polygon = det_res.polygons[i] if i < len(det_res.polygons) else box_4pt

                # 2. Extract perspective-corrected crop
                crop, crop_err = extract_perspective_crop(image, box_4pt)
                if crop is None or crop.size == 0:
                    regions.append(
                        TextRegion(
                            region_id=i,
                            polygon=polygon,
                            box_xyxy=box_xyxy,
                            crop_shape=(0, 0),
                            script="UNKNOWN",
                            routing_confidence=0.0,
                            text="",
                            confidence=0.0,
                            status=RegionStatus.FAILED_RECOGNITION,
                            reading_order=i,
                            error_message=crop_err or "Crop extraction failed",
                        )
                    )
                    continue

                crop_h, crop_w = crop.shape[:2]

                # 3. Visual Script Routing
                t_route_0 = time.perf_counter()
                route_res = self.router.route(crop)
                total_routing_ms += (time.perf_counter() - t_route_0) * 1000.0

                routed_script = route_res.script.value
                routing_conf = float(route_res.confidence)

                # Script Routing Policy:
                # - DEVANAGARI -> Hindi recognizer
                # - LATIN -> English recognizer
                # - SHARED_NUMERIC -> English recognizer (explicit policy: ASCII digits 0-9 & symbols)
                # - UNKNOWN -> Do not force recognizer; preserve crop and return UNRESOLVED
                if route_res.script == ScriptType.UNKNOWN:
                    regions.append(
                        TextRegion(
                            region_id=i,
                            polygon=polygon,
                            box_xyxy=box_xyxy,
                            crop_shape=(crop_h, crop_w),
                            script="UNKNOWN",
                            routing_confidence=routing_conf,
                            text="",
                            confidence=0.0,
                            status=RegionStatus.UNRESOLVED_SCRIPT,
                            reading_order=i,
                            error_message=f"Unresolved script: {route_res.decision_reason}",
                        )
                    )
                    continue

                rec_queue.append({
                    "region_id": i,
                    "polygon": polygon,
                    "box_xyxy": box_xyxy,
                    "crop_shape": (crop_h, crop_w),
                    "script": routed_script,
                    "routing_confidence": routing_conf,
                    "crop": crop,
                    "is_hindi": (route_res.script == ScriptType.DEVANAGARI),
                })

            # 4. Text Recognition (batched by script when enabled, with sequential fallback)
            t_rec_start = time.perf_counter()
            hindi_items = [item for item in rec_queue if item["is_hindi"]]
            english_items = [item for item in rec_queue if not item["is_hindi"]]

            def process_recognition_group(items: List[Dict[str, Any]], model: Any) -> None:
                if not items:
                    return
                crops = [it["crop"] for it in items]
                results: Optional[List[Tuple[str, float]]] = None

                if self.use_batching and len(crops) > 1:
                    try:
                        results = predict_batch(model, crops)
                    except Exception as b_err:
                        logger.warning("Batch recognition failed, falling back to sequential: %s", b_err)
                        results = None

                res_tuples: List[Tuple[str, float, Optional[str]]] = []
                if results is None:
                    # Sequential prediction path
                    for c in crops:
                        try:
                            t, cf = model.predict(c)
                            res_tuples.append((t, cf, None))
                        except Exception as s_err:
                            logger.warning("Sequential prediction failure: %s", s_err)
                            res_tuples.append(("", 0.0, f"Recognition exception: {s_err}"))
                else:
                    res_tuples = [(t, cf, None) for (t, cf) in results]

                for it, (text, conf, err_msg) in zip(items, res_tuples):
                    norm_text = unicodedata.normalize("NFC", text.strip()) if text else ""
                    status = RegionStatus.SUCCESS if norm_text else RegionStatus.FAILED_RECOGNITION
                    regions.append(
                        TextRegion(
                            region_id=it["region_id"],
                            polygon=it["polygon"],
                            box_xyxy=it["box_xyxy"],
                            crop_shape=it["crop_shape"],
                            script=it["script"],
                            routing_confidence=it["routing_confidence"],
                            text=norm_text,
                            confidence=float(conf),
                            status=status,
                            reading_order=it["region_id"],
                            error_message=err_msg,
                        )
                    )

            process_recognition_group(hindi_items, self.hindi_model)
            process_recognition_group(english_items, self.english_model)
            total_recognition_ms = (time.perf_counter() - t_rec_start) * 1000.0

            # -------------------------------------------------------------
            # Stage 4: Reading Order Sorting & Document Text Assembly
            # -------------------------------------------------------------
            if self.sort_reading_order_enabled and len(regions) > 1:
                sorted_regions, grouped_lines = sort_regions_reading_order(regions)
                regions = sorted_regions
            else:
                grouped_lines = [regions] if regions else []

            combined_text = format_combined_text(grouped_lines)

            # Frame-level primary language classification
            success_regions = [r for r in regions if r.status == RegionStatus.SUCCESS]
            hindi_count = sum(1 for r in success_regions if r.script == "DEVANAGARI")
            english_count = sum(1 for r in success_regions if r.script in ("LATIN", "SHARED_NUMERIC"))

            if hindi_count > 0 and english_count == 0:
                primary_lang = PrimaryLanguage.HINDI
            elif english_count > 0 and hindi_count == 0:
                primary_lang = PrimaryLanguage.ENGLISH
            elif hindi_count > 0 and english_count > 0:
                primary_lang = PrimaryLanguage.MIXED
            elif len(regions) > 0:
                primary_lang = PrimaryLanguage.UNKNOWN
            else:
                primary_lang = PrimaryLanguage.NONE

            total_ms = (time.perf_counter() - t_total_start) * 1000.0

            return OCRResult(
                original_shape=(orig_h, orig_w),
                regions=regions,
                combined_text=combined_text,
                primary_language=primary_lang,
                detection_count=det_res.num_detections,
                recognition_count=len(success_regions),
                unresolved_count=len(regions) - len(success_regions),
                detection_time_ms=det_time_ms,
                routing_time_ms=total_routing_ms,
                recognition_time_ms=total_recognition_ms,
                total_time_ms=total_ms,
                execution_providers=self.execution_providers,
                success=True,
            )

        except Exception as e:
            logger.exception("Fatal exception in OCR pipeline: %s", e)
            return OCRResult(
                original_shape=(orig_h, orig_w),
                total_time_ms=(time.perf_counter() - t_total_start) * 1000.0,
                execution_providers=self.execution_providers,
                success=False,
                error_message=f"Fatal pipeline exception: {e}",
            )
