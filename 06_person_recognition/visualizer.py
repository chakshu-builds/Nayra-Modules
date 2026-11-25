"""Visualizer overlay for face recognition, identities, and confidence indicators."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from recognition_types import PersonRecognitionResult, RecognitionStatus, RecognizedPerson

COLOR_KNOWN = (50, 220, 50)       # Emerald Green
COLOR_UNKNOWN = (180, 180, 180)   # Neutral Gray
COLOR_AMBIGUOUS = (0, 190, 255)   # Amber


class PersonRecognitionVisualizer:
    """Renders face recognition boxes, identity labels, and telemetry overlays."""

    def __init__(self, show_landmarks: bool = True, show_confidence: bool = True) -> None:
        self.show_landmarks = show_landmarks
        self.show_confidence = show_confidence

    def render(
        self,
        canvas: np.ndarray,
        recognition_result: PersonRecognitionResult,
    ) -> np.ndarray:
        """Render face recognition boxes and identity labels onto canvas."""
        output = canvas.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        for person in recognition_result.persons:
            x1, y1, x2, y2 = person.bounding_box

            if person.is_known and person.recognition_status == RecognitionStatus.RECOGNIZED:
                box_color = COLOR_KNOWN
                prefix = "[KNOWN] "
            elif person.recognition_status == RecognitionStatus.LOW_CONFIDENCE:
                box_color = COLOR_AMBIGUOUS
                prefix = "[LOW CONF] "
            else:
                box_color = COLOR_UNKNOWN
                prefix = "[UNKNOWN] "

            cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            # Format label
            track_str = f" #{person.face_track_id}" if person.face_track_id is not None else ""
            conf_str = f" ({int(person.similarity_score * 100)}%)" if self.show_confidence and person.similarity_score > 0 else ""
            rel_str = f" [{person.context.relationship}]" if person.context and person.context.relationship else ""
            label_text = f"{prefix}{person.display_name}{rel_str}{track_str}{conf_str}"

            (tw, th), baseline = cv2.getTextSize(label_text, font, 0.46, 1)
            label_y = max(y1 - 6, th + 8)

            cv2.rectangle(
                output,
                (x1, label_y - th - 4),
                (x1 + tw + 6, label_y + baseline),
                box_color,
                -1,
            )

            text_color = (0, 0, 0) if box_color != COLOR_KNOWN else (10, 10, 10)
            cv2.putText(
                output,
                label_text,
                (x1 + 3, label_y - 2),
                font,
                0.46,
                text_color,
                1,
                cv2.LINE_AA,
            )

            # Landmarks (subtle 2px dots)
            if self.show_landmarks and person.landmarks:
                for pt in person.landmarks:
                    cv2.circle(output, (int(round(pt[0])), int(round(pt[1]))), 2, (0, 255, 255), -1, cv2.LINE_AA)

        return output
