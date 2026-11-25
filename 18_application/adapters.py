"""Adapters integrating existing Module 07 (OCR) and Module 10 (Object Memory) into the shared IAssistiveModule architecture."""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional
import numpy as np

from shared.interfaces import IAssistiveModule, IVoiceOutput
from shared.schemas import IntentRequest, IntentResponse, SpeechPriority, UserIntent

logger = logging.getLogger("NAYRA-ApplicationAdapters")


class ObjectMemoryAdapter(IAssistiveModule):
    """Adapter bridging Module 10 ObjectMemoryManager into IAssistiveModule."""

    def __init__(
        self,
        memory_manager: Optional[Any] = None,
        voice_output: Optional[IVoiceOutput] = None,
        is_camera_active: bool = True,
        preferred_language: str = "en",
    ) -> None:
        self.memory = memory_manager
        self.voice_output = voice_output
        self.is_camera_active = is_camera_active
        self.preferred_language = preferred_language
        self._is_active = False

    @property
    def module_name(self) -> str:
        return "10_object_memory"

    def initialize(self) -> bool:
        self._is_active = True
        logger.info("ObjectMemoryAdapter initialized.")
        return True

    def shutdown(self) -> None:
        self._is_active = False
        if self.memory and hasattr(self.memory, "shutdown"):
            try:
                self.memory.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down ObjectMemoryManager: {e}")

    def process_query(self, request: IntentRequest) -> IntentResponse:
        # If memory manager has query(), delegate directly for rich spatial & historical reasoning
        if self.memory and hasattr(self.memory, "query"):
            try:
                pref_lang = "en" if self.preferred_language == "en" else request.detected_language
                res = self.memory.query(request.raw_query, preferred_language=pref_lang)
                if hasattr(res, "spoken_response_hi") and isinstance(res.spoken_response_hi, str):
                    spoken_hi = res.spoken_response_hi
                    spoken_en = res.spoken_response_en
                    data = {
                        "found": bool(getattr(res, "found", False)),
                        "canonical_name": getattr(res, "canonical_name", ""),
                        "is_currently_visible": getattr(res, "is_currently_visible", False),
                        "time_category": getattr(res, "time_category", "OLD"),
                        "status": "never_detected" if not getattr(res, "found", False) else "detected",
                    }
                    if self.voice_output:
                        chosen = spoken_en if (self.preferred_language == "en" or request.detected_language not in ("hi", "hinglish")) else spoken_hi
                        self.voice_output.speak(chosen, priority=SpeechPriority.USER_QUERY_RESPONSE)
                    return IntentResponse(
                        success=True,
                        intent=request.intent,
                        spoken_response_hi=spoken_hi,
                        spoken_response_en=spoken_en,
                        data=data,
                    )
            except Exception as e:
                logger.error(f"Error querying memory_manager via query(): {e}")

        target = (request.target_entity or "object").strip().lower()
        spoken_hi = ""
        spoken_en = ""
        data: dict[str, Any] = {}

        if self.memory:
            try:
                import time

                # 1. Check if currently visible in camera right now
                visible_obs = None
                if hasattr(self.memory, "get_currently_visible_observation"):
                    visible_obs = self.memory.get_currently_visible_observation(target)

                if visible_obs is not None:
                    anchor = getattr(visible_obs, "anchor_object", None)
                    zone = getattr(visible_obs, "horizontal_zone", "CENTER")
                    zone_label_hi = "saamne" if zone == "CENTER" else ("left side mein" if zone == "LEFT" else "right side mein")
                    zone_label_en = "in front of you" if zone == "CENTER" else ("on your left" if zone == "LEFT" else "on your right")

                    if anchor:
                        spoken_hi = f"Aapka {target} abhi {zone_label_hi} {anchor} par dikh raha hai."
                        spoken_en = f"Your {target} is currently visible {zone_label_en} on the {anchor}."
                    else:
                        spoken_hi = f"Aapka {target} abhi {zone_label_hi} dikh raha hai."
                        spoken_en = f"Your {target} is currently visible {zone_label_en}."

                    data = {"status": "currently_visible", "observation": str(visible_obs)}

                else:
                    # 2. Check historical observation from persistent database
                    latest_obs = None
                    if hasattr(self.memory, "get_latest_observation"):
                        latest_obs = self.memory.get_latest_observation(target)

                    if latest_obs is not None:
                        obs_ts = getattr(latest_obs, "timestamp", 0.0)
                        age_sec = max(0.0, time.time() - obs_ts)
                        anchor = getattr(latest_obs, "anchor_object", None)

                        if age_sec < 180.0:
                            # Recently observed (< 3 minutes)
                            if anchor:
                                spoken_hi = f"Aapka {target} abhi camera mein nahi dikh raha, lekin kuch der pehle {anchor} par dekha gaya tha."
                                spoken_en = f"Your {target} is not in camera view right now, but was seen on the {anchor} a moment ago."
                            else:
                                spoken_hi = f"Aapka {target} abhi camera mein nahi dikh raha, lekin kuch der pehle dekha gaya tha."
                                spoken_en = f"Your {target} is not in camera view right now, but was seen a moment ago."
                            data = {"status": "recently_observed", "age_sec": age_sec, "observation": str(latest_obs)}
                        else:
                            # Previously observed (older historical record)
                            minutes_ago = int(age_sec // 60)
                            if anchor:
                                spoken_hi = f"Aapka {target} abhi nahi dikh raha. Pichli baar lagbhag {minutes_ago} minute pehle {anchor} par dekha gaya tha."
                                spoken_en = f"Your {target} is not currently visible. It was last seen about {minutes_ago} minutes ago on the {anchor}."
                            else:
                                spoken_hi = f"Aapka {target} abhi nahi dikh raha. Pichli baar lagbhag {minutes_ago} minute pehle dekha gaya tha."
                                spoken_en = f"Your {target} is not currently visible. It was last seen about {minutes_ago} minutes ago."
                            data = {"status": "previously_observed", "age_sec": age_sec, "observation": str(latest_obs)}

                    else:
                        # 3. Check if target is a known object class
                        supported_classes = [
                            "phone", "cell phone", "mobile", "bottle", "botal", "cup", "mug",
                            "book", "kitaab", "laptop", "keys", "chabi", "chashma", "glasses",
                            "wallet", "remote", "backpack", "chair", "table", "couch", "sofa"
                        ]
                        if target in supported_classes or any(s in target for s in supported_classes):
                            if self.is_camera_active:
                                spoken_hi = f"Mujhe aapka {target} pehle kabhi nahi dikha hai. Abhi camera view mein bhi nahi hai."
                                spoken_en = f"I have never seen your {target} in memory, and it is not currently in camera view."
                            else:
                                spoken_hi = f"Camera band hai. Memory mein aapka {target} pehle kabhi nahi dikha hai."
                                spoken_en = f"Camera is inactive. I have never seen your {target} in memory."
                            data = {"status": "never_detected", "camera_active": self.is_camera_active}
                        else:
                            spoken_hi = f"Main '{target}' ko pehchanne ke liye train nahi hoon, isliye iski location nahi bata sakti."
                            spoken_en = f"I am not trained to detect '{target}', so I cannot provide its location."
                            data = {"status": "unsupported_class"}

            except Exception as e:
                logger.error(f"Error querying object memory: {e}")
                spoken_hi = f"Object memory lookup mein samasya aayi: {e}"
                spoken_en = f"Error during object memory lookup: {e}"
                data = {"status": "error", "error": str(e)}
        else:
            spoken_hi = f"Object memory module active nahi hai. {target} nahi dhundh saki."
            spoken_en = f"Object memory module is not active. Could not search for {target}."
            data = {"status": "inactive"}

        if self.voice_output:
            chosen = spoken_en if (self.preferred_language == "en" or request.detected_language not in ("hi", "hinglish")) else spoken_hi
            self.voice_output.speak(chosen, priority=SpeechPriority.USER_QUERY_RESPONSE)

        return IntentResponse(
            success=True,
            intent=request.intent,
            spoken_response_hi=spoken_hi,
            spoken_response_en=spoken_en,
            data=data,
        )


class OCRAdapter(IAssistiveModule):
    """Adapter bridging Module 07 OCR / Read Aloud into IAssistiveModule."""

    def __init__(
        self,
        ocr_worker: Optional[Any] = None,
        voice_output: Optional[IVoiceOutput] = None,
        preferred_language: str = "en",
        frame_provider: Optional[Any] = None,
    ) -> None:
        self.ocr_worker = ocr_worker
        self.voice_output = voice_output
        self.preferred_language = preferred_language
        self.frame_provider = frame_provider
        self._latest_ocr_text: Optional[str] = None
        self._is_active = False

    @property
    def module_name(self) -> str:
        return "07_ocr"

    def initialize(self) -> bool:
        self._is_active = True
        logger.info("OCRAdapter initialized.")
        return True

    def shutdown(self) -> None:
        self._is_active = False
        if self.ocr_worker and hasattr(self.ocr_worker, "shutdown"):
            try:
                self.ocr_worker.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down OCR worker: {e}")

    def update_text(self, text: str) -> None:
        self._latest_ocr_text = text

    def process_query(self, request: IntentRequest) -> IntentResponse:
        active_text = None
        # 1. On-demand live OCR from latest camera frame
        if self.frame_provider is not None:
            try:
                frame = self.frame_provider()
                if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                    try:
                        from ocr import read_text
                        ocr_res = read_text(frame)
                        if ocr_res and getattr(ocr_res, "success", False):
                            txt = getattr(ocr_res, "combined_text", "")
                            if txt and txt.strip():
                                active_text = txt.strip()
                                self._latest_ocr_text = active_text
                    except Exception as ocr_err:
                        logger.error(f"Error in on-demand OCR: {ocr_err}")
            except Exception as e:
                logger.error(f"Error accessing frame for OCR: {e}")

        # 2. Cached or worker result fallback
        if not active_text and self._latest_ocr_text:
            active_text = self._latest_ocr_text
        elif not active_text and self.ocr_worker and hasattr(self.ocr_worker, "get_active_result"):
            res = self.ocr_worker.get_active_result()
            if res and hasattr(res, "text"):
                active_text = res.text

        if active_text and active_text.strip():
            clean = active_text.strip()
            spoken_hi = f"Saamne likha hai: {clean}."
            spoken_en = f"Text reads: {clean}."
        else:
            spoken_hi = "Saamne koi text ya likhawat nahi dikh rahi hai."
            spoken_en = "No readable text detected in front of you."

        if self.voice_output:
            chosen = spoken_en if (self.preferred_language == "en" or request.detected_language not in ("hi", "hinglish")) else spoken_hi
            self.voice_output.speak(chosen, priority=SpeechPriority.USER_QUERY_RESPONSE)

        return IntentResponse(
            success=True,
            intent=request.intent,
            spoken_response_hi=spoken_hi,
            spoken_response_en=spoken_en,
            data={"text": active_text},
        )
