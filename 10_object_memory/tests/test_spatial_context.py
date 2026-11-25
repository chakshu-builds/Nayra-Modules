"""NAYRA Module 10 — Unit tests for Relational Spatial Context Engine."""

from __future__ import annotations

import pytest
from object_memory.perception.spatial_context import (
    SpatialContextEngine,
    ANCHOR_CLASSES,
    CANONICAL_MAPPING,
)


def test_canonical_mapping():
    engine = SpatialContextEngine()
    canon, hi = engine.get_canonical_info("cell phone")
    assert canon == "phone"
    assert hi == "फ़ोन"

    canon2, hi2 = engine.get_canonical_info("remote")
    assert canon2 == "remote"
    assert hi2 == "रिमोट"

    canon_unk, hi_unk = engine.get_canonical_info("mysterious_gadget")
    assert canon_unk == "mysterious_gadget"


def test_compute_zone():
    engine = SpatialContextEngine(frame_width=600, frame_height=400)

    # Left object
    zone, en, hi = engine.compute_zone((10, 100, 100, 200))
    assert zone == "LEFT"
    assert "left" in en
    assert "baayein" in hi

    # Center object
    zone_c, en_c, hi_c = engine.compute_zone((250, 100, 350, 200))
    assert zone_c == "CENTER"
    assert "front" in en_c
    assert "saamne" in hi_c

    # Right object
    zone_r, en_r, hi_r = engine.compute_zone((450, 100, 550, 200))
    assert zone_r == "RIGHT"
    assert "right" in en_r
    assert "daayein" in hi_r


def test_evaluate_spatial_relation_on_table():
    engine = SpatialContextEngine(frame_width=640, frame_height=480)

    # Table: [100, 250, 500, 450]
    table_bbox = (100, 250, 500, 450)
    # Phone resting on table top: [200, 220, 280, 270] -> bottom is 270, overlaps table horizontally
    phone_bbox = (200, 220, 280, 270)

    rel = engine.evaluate_spatial_relation(phone_bbox, table_bbox, "dining table")
    assert rel is not None
    rel_type, score, en_phrase, hi_phrase = rel
    assert rel_type == "on"
    assert en_phrase == "on the dining table"
    assert "dining table par" in hi_phrase


def test_evaluate_spatial_relation_near_couch():
    engine = SpatialContextEngine(frame_width=640, frame_height=480)

    # Couch: [50, 150, 300, 350]
    couch_bbox = (50, 150, 300, 350)
    # Remote on a small side area near couch: [310, 200, 350, 230]
    remote_bbox = (310, 200, 350, 230)

    rel = engine.evaluate_spatial_relation(remote_bbox, couch_bbox, "couch")
    assert rel is not None
    rel_type, score, en_phrase, hi_phrase = rel
    assert rel_type == "near"
    assert "near the couch" in en_phrase
    assert "sofa ke paas" in hi_phrase


def test_find_best_anchor_priority():
    engine = SpatialContextEngine(frame_width=640, frame_height=480)

    table_bbox = (100, 200, 500, 400)
    chair_bbox = (50, 180, 120, 350)
    anchors = [
        ("chair", chair_bbox),
        ("dining table", table_bbox),
    ]

    # Object is physically ON the table, but also near chair
    obj_bbox = (200, 180, 260, 220)

    result = engine.find_best_anchor(obj_bbox, anchors)
    assert result is not None
    anc_name, rel_type, en_phrase, hi_phrase = result
    assert anc_name == "dining table"
    assert rel_type == "on"
