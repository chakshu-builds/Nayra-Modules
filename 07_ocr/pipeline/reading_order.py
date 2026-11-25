"""NAYRA Module 07 — Reading-Order Sorting & Text Assembly.

Arranges detected and recognized text regions into standard natural visual
reading order (top-to-bottom, left-to-right) and joins them into formatted text.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np

from .result_schema import TextRegion


def sort_regions_reading_order(
    regions: List[TextRegion],
    y_threshold_ratio: float = 0.5,
) -> Tuple[List[TextRegion], List[List[TextRegion]]]:
    """Sort text regions into visual reading order.

    Algorithm:
      1. Calculate center-y and height for each region.
      2. Group regions into lines based on vertical proximity:
         |region_cy - line_cy| <= line_height * y_threshold_ratio.
      3. Sort lines by median y ascending (top to bottom).
      4. Within each line, sort regions by x_min ascending (left to right).
      5. Assign sequential reading_order indices (0, 1, ...).

    Args:
        regions: List of TextRegion instances.
        y_threshold_ratio: Multiplier of line height for vertical grouping (default 0.5).

    Returns:
        Tuple of:
            sorted_regions: Flat list of sorted TextRegion instances with updated reading_order.
            grouped_lines: List of line groups (each being a list of TextRegions in that line).
    """
    if len(regions) <= 1:
        for idx, r in enumerate(regions):
            r.reading_order = idx
        return regions, [regions] if regions else []

    # Prepare metadata: (index, xmin, ymin, xmax, ymax, cy, height, region)
    items = []
    for idx, r in enumerate(regions):
        xmin, ymin, xmax, ymax = r.box_xyxy
        cy = (ymin + ymax) / 2.0
        h = max(ymax - ymin, 1.0)
        items.append((idx, xmin, ymin, xmax, ymax, cy, h, r))

    # Sort initially by center y
    items.sort(key=lambda it: it[5])

    # Group into lines
    lines: List[List[Tuple[int, float, float, float, float, float, float, TextRegion]]] = []
    current_line: List[Tuple[int, float, float, float, float, float, float, TextRegion]] = []

    for item in items:
        if not current_line:
            current_line.append(item)
            continue

        line_cy = float(np.median([it[5] for it in current_line]))
        line_h = float(np.median([it[6] for it in current_line]))
        y_threshold = line_h * y_threshold_ratio

        if abs(item[5] - line_cy) <= y_threshold:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]

    if current_line:
        lines.append(current_line)

    # Sort lines by median y, and within each line sort by xmin
    sorted_regions: List[TextRegion] = []
    grouped_lines: List[List[TextRegion]] = []
    order_idx = 0

    for line in lines:
        line.sort(key=lambda it: it[1])  # sort by xmin (left-to-right)
        line_regions: List[TextRegion] = []
        for it in line:
            r = it[7]
            r.reading_order = order_idx
            order_idx += 1
            line_regions.append(r)
            sorted_regions.append(r)
        grouped_lines.append(line_regions)

    return sorted_regions, grouped_lines


def format_combined_text(grouped_lines: List[List[TextRegion]]) -> str:
    """Format combined text by joining words in the same line with spaces and lines with newlines.

    Only includes text from regions that have non-empty recognized text.
    """
    line_strings: List[str] = []
    for line in grouped_lines:
        line_texts = [r.text.strip() for r in line if r.text and r.text.strip()]
        if line_texts:
            line_strings.append(" ".join(line_texts))
    return "\n".join(line_strings)
