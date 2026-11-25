"""NAYRA Module 10 — Relational Spatial Context Engine.

Infers spatial relationships between personal objects (e.g., phone, remote, bottle)
and anchor furniture (e.g., dining table, couch, bed, chair, refrigerator).
Provides bilingual (English / Hindi) relational phrases and horizontal zone fallback.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple


# Predefined furniture/structural anchor categories in COCO 80
ANCHOR_CLASSES: set[str] = {
    "dining table",
    "couch",
    "bed",
    "chair",
    "refrigerator",
    "bench",
    "sink",
    "desk",
    "table",
}

# Mapping from COCO class labels to canonical names and Hindi aliases
CANONICAL_MAPPING: Dict[str, Tuple[str, str]] = {
    "cell phone": ("phone", "फ़ोन"),
    "phone": ("phone", "फ़ोन"),
    "remote": ("remote", "रिमोट"),
    "tv": ("tv", "टीवी"),
    "tvmonitor": ("tv", "टीवी"),
    "television": ("tv", "टीवी"),
    "bottle": ("bottle", "बोतल"),
    "cup": ("cup", "कप"),
    "book": ("book", "किताब"),
    "laptop": ("laptop", "लैपटॉप"),
    "backpack": ("backpack", "बैग"),
    "handbag": ("bag", "पर्स"),
    "suitcase": ("suitcase", "सूटकेस"),
    "umbrella": ("umbrella", "छाता"),
    "clock": ("clock", "घड़ी"),
    "scissors": ("scissors", "कैंची"),
    "mouse": ("mouse", "माउस"),
    "keyboard": ("keyboard", "कीबोर्ड"),
    "vase": ("vase", "गुलदस्ता"),
    "bowl": ("bowl", "कटोरा"),
    "dining table": ("dining table", "डाइनिंग टेबल"),
    "table": ("dining table", "डाइनिंग टेबल"),
    "desk": ("dining table", "डाइनिंग टेबल"),
    "couch": ("sofa", "सोफ़ा"),
    "sofa": ("sofa", "सोफ़ा"),
    "bed": ("bed", "बिस्तर"),
    "chair": ("chair", "कुर्सी"),
    "refrigerator": ("fridge", "फ्रिज"),
    "fridge": ("fridge", "फ्रिज"),
}

# Hindi names for anchor objects when formulating spatial phrases
ANCHOR_HINDI_NAMES: Dict[str, str] = {
    "dining table": "dining table",
    "table": "table",
    "couch": "sofa",
    "sofa": "sofa",
    "bed": "bed",
    "chair": "chair",
    "refrigerator": "fridge",
    "fridge": "fridge",
    "bench": "bench",
    "sink": "sink",
    "laptop": "laptop",
    "tv": "TV",
    "monitor": "monitor",
}


class SpatialContextEngine:
    """Computes relational positioning between objects and anchor furniture."""

    def __init__(self, frame_width: int = 640, frame_height: int = 480) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.diag = math.hypot(frame_width, frame_height)

    @staticmethod
    def get_canonical_info(raw_label: str) -> Tuple[str, str]:
        """Return (canonical_name, hindi_alias) for a detector label."""
        cleaned = raw_label.lower().strip()
        if cleaned in CANONICAL_MAPPING:
            return CANONICAL_MAPPING[cleaned]
        # Fallback to cleaned label
        return cleaned, cleaned

    def compute_zone(self, bbox: Tuple[int, int, int, int], width: Optional[int] = None) -> Tuple[str, str, str]:
        """Compute horizontal zone (LEFT, CENTER, RIGHT) and bilingual phrases."""
        w = width or self.frame_width
        cx = (bbox[0] + bbox[2]) / 2.0

        if cx < w * 0.33:
            return "LEFT", "on your left", "baayein taraf"
        elif cx > w * 0.66:
            return "RIGHT", "on your right", "daayein taraf"
        else:
            return "CENTER", "in front of you", "saamne"

    def compute_vertical_zone(self, bbox: Tuple[int, int, int, int], height: Optional[int] = None) -> Tuple[str, str, str]:
        """Compute vertical zone (UPPER, MIDDLE, LOWER) and bilingual phrases."""
        h = height or self.frame_height
        cy = (bbox[1] + bbox[3]) / 2.0

        if cy < h * 0.33:
            return "UPPER", "upper area", "oopar ki taraf"
        elif cy > h * 0.66:
            return "LOWER", "lower area", "neeche ki taraf"
        else:
            return "MIDDLE", "middle area", "beech mein"

    def evaluate_spatial_relation(
        self,
        obj_bbox: Tuple[int, int, int, int],
        anchor_bbox: Tuple[int, int, int, int],
        anchor_name: str,
    ) -> Optional[Tuple[str, float, str, str]]:
        """Evaluate spatial relationship between an object and an anchor furniture bounding box.

        Returns:
            Tuple of (relation_type, distance_score, english_phrase, hindi_phrase) or None.
            relation_type: 'on' or 'near'
        """
        ox1, oy1, ox2, oy2 = obj_bbox
        ax1, ay1, ax2, ay2 = anchor_bbox

        obj_w = max(1, ox2 - ox1)
        obj_h = max(1, oy2 - oy1)
        anc_w = max(1, ax2 - ax1)
        anc_h = max(1, ay2 - ay1)

        # Centroids
        ocx = (ox1 + ox2) / 2.0
        ocy = (oy1 + oy2) / 2.0
        acx = (ax1 + ax2) / 2.0
        acy = (ay1 + ay2) / 2.0

        # Horizontal overlap
        overlap_x = max(0, min(ox2, ax2) - max(ox1, ax1))
        overlap_ratio_obj = overlap_x / obj_w

        # Check 'ON' relation
        # Object bottom rests on or near the anchor top / upper surface
        # e.g., table top: ay1. oy2 should be around ay1 (e.g. ay1 - 0.2*anc_h to ay1 + 0.6*anc_h)
        # and significant horizontal overlap
        is_vertically_supported = (ay1 - 0.25 * anc_h) <= oy2 <= (ay1 + 0.65 * anc_h)
        if overlap_ratio_obj >= 0.4 and is_vertically_supported:
            dist = abs(ocy - ay1)
            en_phrase = f"on the {anchor_name}"
            hi_name = ANCHOR_HINDI_NAMES.get(anchor_name, anchor_name)
            hi_phrase = f"{hi_name} par"
            return "on", dist, en_phrase, hi_phrase

        # Check 'NEAR' relation
        # Euclidean centroid distance
        dist_px = math.hypot(ocx - acx, ocy - acy)
        # Normalize by diagonal
        norm_dist = dist_px / max(1.0, self.diag)
        if norm_dist < 0.35:
            en_phrase = f"near the {anchor_name}"
            hi_name = ANCHOR_HINDI_NAMES.get(anchor_name, anchor_name)
            hi_phrase = f"{hi_name} ke paas"
            return "near", dist_px, en_phrase, hi_phrase

        return None

    def find_best_anchor(
        self,
        obj_bbox: Tuple[int, int, int, int],
        anchors: Sequence[Tuple[str, Tuple[int, int, int, int]]],
    ) -> Optional[Tuple[str, str, str, str]]:
        """Find the most plausible anchor furniture for the given object.

        Args:
            obj_bbox: (x1, y1, x2, y2)
            anchors: list of (anchor_name, (x1, y1, x2, y2))

        Returns:
            (anchor_name, relation_type, en_phrase, hi_phrase) or None
        """
        if not anchors:
            return None

        on_candidates: List[Tuple[float, str, str, str, str]] = []
        near_candidates: List[Tuple[float, str, str, str, str]] = []

        for anc_name, anc_bbox in anchors:
            rel = self.evaluate_spatial_relation(obj_bbox, anc_bbox, anc_name)
            if rel is not None:
                rel_type, score, en_phrase, hi_phrase = rel
                if rel_type == "on":
                    on_candidates.append((score, anc_name, rel_type, en_phrase, hi_phrase))
                else:
                    near_candidates.append((score, anc_name, rel_type, en_phrase, hi_phrase))

        # Prioritize 'on' over 'near'
        if on_candidates:
            on_candidates.sort(key=lambda x: x[0])
            best = on_candidates[0]
            return best[1], best[2], best[3], best[4]

        if near_candidates:
            near_candidates.sort(key=lambda x: x[0])
            best = near_candidates[0]
            return best[1], best[2], best[3], best[4]

        return None
