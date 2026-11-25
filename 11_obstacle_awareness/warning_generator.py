"""Spoken warning formulation with hysteresis and repetition suppression."""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

from shared.schemas import ObstacleHazard

logger = logging.getLogger("NAYRA-ObstacleWarningGenerator")


class ObstacleWarningGenerator:
    """Manages warning cadence, repetition suppression, and multi-hazard summarization."""

    def __init__(self, cooldown_sec: float = 6.0) -> None:
        self.cooldown_sec = cooldown_sec
        # Maps (key, zone) -> last_warning_time
        self._last_warn_time: Dict[Tuple[str, str], float] = {}
        self._lock = threading.Lock()

    def should_announce(self, hazard: ObstacleHazard, now: Optional[float] = None) -> bool:
        """Evaluate whether a warning should be spoken or suppressed."""
        curr_time = now if now is not None else time.time()
        key = str(hazard.track_id) if hazard.track_id is not None else hazard.class_name
        cache_key = (key, hazard.zone)

        with self._lock:
            # Periodic pruning if cache grows large
            if len(self._last_warn_time) > 150:
                self._last_warn_time = {
                    k: v for k, v in self._last_warn_time.items()
                    if (curr_time - v) < 120.0
                }

            last_time = self._last_warn_time.get(cache_key, 0.0)
            elapsed = curr_time - last_time

            # Urgent warnings bypass cooldown if at least 2.5s elapsed
            effective_cooldown = 2.5 if hazard.is_urgent else self.cooldown_sec
            if elapsed >= effective_cooldown:
                self._last_warn_time[cache_key] = curr_time
                return True
            return False

    def generate_scene_warning(self, hazards: List[ObstacleHazard]) -> Optional[Tuple[str, str, bool]]:
        """Return (warning_hi, warning_en, is_urgent) for top candidate hazard in path, or None if suppressed.

        Note: Ambient objects on LEFT or RIGHT are suppressed from automatic speech;
        only obstacles directly in the walking corridor (CENTER / urgent) trigger warnings.
        """
        if not hazards:
            return None

        # Only warn for obstacles blocking the path (turning the corridor RED)
        path_hazards = [h for h in hazards if h.is_urgent or h.zone == "CENTER"]
        for hazard in path_hazards:
            if self.should_announce(hazard):
                cname = hazard.class_name.lower().strip()
                warn_en = f"Move carefully, you have a {cname} in your front."
                return hazard.warning_hi, warn_en, hazard.is_urgent
        return None

    def formulate_query_summary(self, hazards: List[ObstacleHazard]) -> Tuple[str, str]:
        """Synthesize response when user asks 'Is there any obstacle in front of me?'"""
        if not hazards:
            return "Aapke saamne koi rukawat nahi dikh rahi, rasta saaf hai.", "No obstacles detected ahead, your path is clear."

        # Check for urgent or center path obstacles
        center_hazards = [h for h in hazards if h.is_urgent or h.zone == "CENTER"]
        if center_hazards:
            top_obj = center_hazards[0].class_name.lower().strip()
            return (
                f"Savdhaan, dhyan se chaliye, aapke saamne ek {top_obj} hai.",
                f"Move carefully, {top_obj} is in your front.",
            )

        # If only left or right objects exist, the walking path in front is clear
        return (
            "Aapka saamne ka rasta saaf hai.",
            "No obstacles in your path, your front is clear.",
        )
