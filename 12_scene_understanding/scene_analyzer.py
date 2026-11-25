"""Scene Analyzer synthesizing multi-modal detections into coherent verbal scene descriptions."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("NAYRA-SceneAnalyzer")

HINDI_OBJECT_NAMES: Dict[str, str] = {
    "person": "vyakti",
    "chair": "kursi",
    "couch": "sofa",
    "sofa": "sofa",
    "table": "table",
    "dining table": "dining table",
    "desk": "desk",
    "laptop": "laptop",
    "tv": "television",
    "bottle": "botal",
    "cup": "cup",
    "book": "kitaab",
    "cell phone": "phone",
    "backpack": "bag",
    "bed": "bed",
    "door": "darwaza",
    "keyboard": "keyboard",
    "mouse": "mouse",
}


@dataclass
class SceneDescription:
    """Structured scene description container."""
    spoken_hi: str
    spoken_en: str
    objects_by_zone: Dict[str, List[str]] = field(default_factory=dict)
    people_count: int = 0
    spatial_relations: List[str] = field(default_factory=list)
    ocr_text: Optional[str] = None


class SceneAnalyzer:
    """Analyzes raw visual detections, tracks, spatial layout, and OCR text to construct scene descriptions."""

    def __init__(self) -> None:
        pass

    def classify_zone(self, box_xyxy: Sequence[int | float], frame_width: int) -> str:
        """Classify horizontal zone: LEFT, CENTER, RIGHT."""
        cx = (float(box_xyxy[0]) + float(box_xyxy[2])) / 2.0
        one_third = float(frame_width) / 3.0
        two_thirds = 2.0 * one_third
        if cx < one_third:
            return "LEFT"
        elif cx > two_thirds:
            return "RIGHT"
        return "CENTER"

    def analyze(
        self,
        detections: Sequence[Any],
        frame_width: int = 640,
        frame_height: int = 480,
        ocr_text: Optional[str] = None,
        target_zone: Optional[str] = None,
    ) -> SceneDescription:
        """Produce a structured SceneDescription from visible detections and optional OCR text."""
        objects_by_zone: Dict[str, List[str]] = {"LEFT": [], "CENTER": [], "RIGHT": []}
        people_count = 0
        detected_items: List[Dict[str, Any]] = []

        for det in detections:
            cname = getattr(det, "class_name", None) or getattr(det, "name", "object")
            box = getattr(det, "box_xyxy", None) or getattr(det, "box", (0, 0, 0, 0))
            conf = getattr(det, "confidence", 1.0)
            if conf < 0.35:
                continue

            zone = self.classify_zone(box, frame_width)
            if cname.lower() == "person":
                people_count += 1
            else:
                objects_by_zone[zone].append(cname.lower())

            detected_items.append({
                "name": cname.lower(),
                "box": box,
                "zone": zone,
            })

        # Find spatial relationships (e.g. object on table/desk/chair)
        relations: List[str] = []
        furniture_candidates = [d for d in detected_items if d["name"] in ("table", "dining table", "desk", "couch", "sofa", "bed")]
        small_items = [d for d in detected_items if d["name"] in ("laptop", "cell phone", "bottle", "cup", "book", "keyboard", "mouse")]

        for item in small_items:
            ix1, iy1, ix2, iy2 = item["box"]
            icx = (ix1 + ix2) / 2.0
            icy = (iy1 + iy2) / 2.0
            for furn in furniture_candidates:
                fx1, fy1, fx2, fy2 = furn["box"]
                if fx1 <= icx <= fx2 and iy2 <= fy2 and iy1 >= fy1 - 50:
                    relations.append(f"{item['name']} on {furn['name']}")

        # Build natural spoken description
        spoken_hi, spoken_en = self._build_spoken_text(
            objects_by_zone=objects_by_zone,
            people_count=people_count,
            relations=relations,
            ocr_text=ocr_text,
            target_zone=target_zone,
        )

        return SceneDescription(
            spoken_hi=spoken_hi,
            spoken_en=spoken_en,
            objects_by_zone=objects_by_zone,
            people_count=people_count,
            spatial_relations=relations,
            ocr_text=ocr_text,
        )

    def _build_spoken_text(
        self,
        objects_by_zone: Dict[str, List[str]],
        people_count: int,
        relations: List[str],
        ocr_text: Optional[str],
        target_zone: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Synthesize natural spoken Hindi and English sentences."""
        if target_zone:
            tz = target_zone.upper()
            zone_objs = objects_by_zone.get(tz, [])
            zone_label_hi = "saamne" if tz == "CENTER" else ("left side mein" if tz == "LEFT" else "right side mein")
            zone_label_en = "in front of you" if tz == "CENTER" else ("on your left" if tz == "LEFT" else "on your right")

            if not zone_objs and people_count == 0:
                return (
                    f"Aapke {zone_label_hi} koi vishesh cheez nahi dikh rahi hai.",
                    f"There is nothing notable {zone_label_en}."
                )

            hi_names = [HINDI_OBJECT_NAMES.get(o, o) for o in list(dict.fromkeys(zone_objs))]
            en_names = list(dict.fromkeys(zone_objs))

            hi_part = f"Aapke {zone_label_hi} " + ", ".join(hi_names) + " dikh raha hai."
            en_part = f"{zone_label_en.capitalize()}, there is " + ", ".join(en_names) + "."
            if ocr_text and ocr_text.strip():
                hi_part += f" Saamne likha hai: {ocr_text.strip()}."
                en_part += f" Text reads: {ocr_text.strip()}."
            return hi_part, en_part

        # Full scene synthesis
        total_objects = sum(len(v) for v in objects_by_zone.values())
        if total_objects == 0 and people_count == 0 and not ocr_text:
            return (
                "Aapke aas paas rasta saaf hai, koi vishesh cheez nahi dikh rahi.",
                "Your surroundings are clear, no notable objects detected."
            )

        hi_parts: List[str] = []
        en_parts: List[str] = []

        # People
        if people_count == 1:
            hi_parts.append("Aapke saamne ek vyakti khade hain.")
            en_parts.append("A person is in front of you.")
        elif people_count > 1:
            hi_parts.append(f"Aapke aas paas {people_count} vyakti hain.")
            en_parts.append(f"There are {people_count} people around you.")

        # Center objects
        center_objs = list(dict.fromkeys(objects_by_zone.get("CENTER", [])))
        if center_objs:
            c_hi = [HINDI_OBJECT_NAMES.get(o, o) for o in center_objs]
            hi_parts.append("Saamne " + ", ".join(c_hi) + " hai.")
            en_parts.append("In front is " + ", ".join(center_objs) + ".")

        # Left objects
        left_objs = list(dict.fromkeys(objects_by_zone.get("LEFT", [])))
        if left_objs:
            l_hi = [HINDI_OBJECT_NAMES.get(o, o) for o in left_objs]
            hi_parts.append("Left side mein " + ", ".join(l_hi) + " hai.")
            en_parts.append("On your left is " + ", ".join(left_objs) + ".")

        # Right objects
        right_objs = list(dict.fromkeys(objects_by_zone.get("RIGHT", [])))
        if right_objs:
            r_hi = [HINDI_OBJECT_NAMES.get(o, o) for o in right_objs]
            hi_parts.append("Right side mein " + ", ".join(r_hi) + " hai.")
            en_parts.append("On your right is " + ", ".join(right_objs) + ".")

        # Relationships
        if relations:
            for rel in relations[:2]:
                parts = rel.split(" on ")
                if len(parts) == 2:
                    sub_hi = HINDI_OBJECT_NAMES.get(parts[0], parts[0])
                    furn_hi = HINDI_OBJECT_NAMES.get(parts[1], parts[1])
                    hi_parts.append(f"{sub_hi.capitalize()} {furn_hi} par rakha hai.")
                    en_parts.append(f"{parts[0].capitalize()} is placed on the {parts[1]}.")

        # OCR text
        if ocr_text and ocr_text.strip():
            clean_text = ocr_text.strip()
            hi_parts.append(f"Saamne likha hai: {clean_text}.")
            en_parts.append(f"Text in front reads: {clean_text}.")

        return " ".join(hi_parts), " ".join(en_parts)
