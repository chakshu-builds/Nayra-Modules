"""NAYRA Module 10 — Unit tests for Query Parser and Empathetic Response Generator."""

from __future__ import annotations

import time
import pytest

from object_memory.db.models import ObjectObservation
from object_memory.nlp.query_parser import MemoryQueryParser
from object_memory.nlp.response_generator import ResponseGenerator


def test_parse_supported_objects():
    parser = MemoryQueryParser()

    # Phone queries
    q1 = parser.parse("Mera phone kahan hai?")
    assert q1.target_object == "phone"
    assert q1.is_supported is True
    assert q1.query_type == "WHERE"

    q2 = parser.parse("Maine mobile kahan chhoda?")
    assert q2.target_object == "phone"
    assert q2.is_supported is True

    q3 = parser.parse("Where is my cellphone?")
    assert q3.target_object == "phone"
    assert q3.is_supported is True

    # Remote queries
    q4 = parser.parse("TV ka remote kahan hai?")
    assert q4.target_object == "remote"
    assert q4.is_supported is True

    # Medicine / Bottle queries
    q5 = parser.parse("Where did I last see my medicine?")
    assert q5.target_object == "bottle"
    assert q5.is_supported is True

    q6 = parser.parse("Pani ki bottle kahan rakhi hai?")
    assert q6.target_object == "bottle"
    assert q6.is_supported is True

    # Book queries
    q7 = parser.parse("Meri kitab kahan hai?")
    assert q7.target_object == "book"
    assert q7.is_supported is True


def test_parse_unsupported_objects():
    parser = MemoryQueryParser()

    # Chashma / Eyeglasses
    q1 = parser.parse("Mera chashma kahan hai?")
    assert q1.target_object == "glasses"
    assert q1.is_supported is False
    assert q1.query_type == "WHERE"

    q2 = parser.parse("Where are my spectacles?")
    assert q2.target_object == "glasses"
    assert q2.is_supported is False

    # Keys / Chabhi
    q3 = parser.parse("Meri keys kahan rakhi thi?")
    assert q3.target_object == "keys"
    assert q3.is_supported is False

    q4 = parser.parse("Almari ki chabhi kahan hai?")
    assert q4.target_object == "keys"
    assert q4.is_supported is False


def test_response_for_unsupported_object():
    parser = MemoryQueryParser()
    q = parser.parse("Mera chashma kahan hai?")
    res = ResponseGenerator.generate_response(q, obs=None)

    assert res.found is False
    assert "chashma" in res.spoken_response_hi.lower() or "चश्मा" in res.spoken_response_hi
    assert "saksham nahi" in res.spoken_response_hi
    assert "cannot currently recognize" in res.spoken_response_en.lower()


def test_response_for_not_found_object():
    parser = MemoryQueryParser()
    q = parser.parse("Mera phone kahan hai?")
    res = ResponseGenerator.generate_response(q, obs=None)

    assert res.found is False
    assert "dikhai nahi diya" in res.spoken_response_hi
    assert "haven't seen your phone yet" in res.spoken_response_en


def test_response_for_recent_observation():
    parser = MemoryQueryParser()
    q = parser.parse("Mera phone kahan hai?")

    # Observed 5 minutes ago on dining table
    t_obs = time.time() - 300
    obs = ObjectObservation(
        id=1,
        object_label="cell phone",
        canonical_name="phone",
        hindi_alias="फ़ोन",
        timestamp=t_obs,
        confidence=0.9,
        bbox=(200, 200, 250, 250),
        spatial_relation="on",
        anchor_object="dining table",
        horizontal_zone="CENTER",
    )

    res = ResponseGenerator.generate_response(q, obs)
    assert res.found is True
    assert res.time_category == "RECENT"
    assert "dining table par" in res.spoken_response_hi
    assert "wahin hona chahiye" in res.spoken_response_hi
    assert "on the dining table" in res.spoken_response_en
    assert "should still be there" in res.spoken_response_en


def test_response_for_earlier_observation():
    parser = MemoryQueryParser()
    q = parser.parse("Where is the TV remote?")

    # Observed 45 minutes ago near couch
    t_obs = time.time() - (45 * 60)
    obs = ObjectObservation(
        id=2,
        object_label="remote",
        canonical_name="remote",
        hindi_alias="रिमोट",
        timestamp=t_obs,
        confidence=0.88,
        bbox=(100, 100, 150, 150),
        spatial_relation="near",
        anchor_object="couch",
        horizontal_zone="LEFT",
    )

    res = ResponseGenerator.generate_response(q, obs)
    assert res.found is True
    assert res.time_category == "EARLIER"
    assert "couch ke paas" in res.spoken_response_hi or "sofa ke paas" in res.spoken_response_hi
    assert "hata diya ho" in res.spoken_response_hi
    assert "might have been moved" in res.spoken_response_en


def test_response_for_visible_now():
    parser = MemoryQueryParser()
    q = parser.parse("Mera phone kahan hai?")

    # Observed 10 seconds ago
    t_obs = time.time() - 10
    obs = ObjectObservation(
        id=3,
        object_label="cell phone",
        canonical_name="phone",
        hindi_alias="फ़ोन",
        timestamp=t_obs,
        confidence=0.95,
        bbox=(200, 200, 250, 250),
        spatial_relation="on",
        anchor_object="bed",
        horizontal_zone="CENTER",
    )

    res = ResponseGenerator.generate_response(q, obs)
    assert res.found is True
    assert res.time_category == "NOW"
    assert "abhi" in res.spoken_response_hi
    assert "currently visible" in res.spoken_response_en
