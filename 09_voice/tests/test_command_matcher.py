"""Unit tests for CommandMatcher intent classification."""

import pytest
from voice import CommandIntent, CommandMatcher


def test_read_commands():
    # Exact and variant phrases
    assert CommandMatcher.match_intent("NAYRA, read the text")[0] == CommandIntent.READ
    assert CommandMatcher.match_intent("read the text")[0] == CommandIntent.READ
    assert CommandMatcher.match_intent("Read this")[0] == CommandIntent.READ
    assert CommandMatcher.match_intent("read this please")[0] == CommandIntent.READ
    assert CommandMatcher.match_intent("Mere liye padho")[0] == CommandIntent.READ
    assert CommandMatcher.match_intent("mere liye parho")[0] == CommandIntent.READ
    assert CommandMatcher.match_intent("My Lee Parho")[0] == CommandIntent.READ
    assert CommandMatcher.match_intent("मेरे लिए पढ़ो")[0] == CommandIntent.READ
    assert CommandMatcher.match_intent("यह पढ़ो")[0] == CommandIntent.READ


def test_stop_commands():
    # Exact and variant phrases
    assert CommandMatcher.match_intent("Stop reading")[0] == CommandIntent.STOP
    assert CommandMatcher.match_intent("stop reading please")[0] == CommandIntent.STOP
    assert CommandMatcher.match_intent("NAYRA stop")[0] == CommandIntent.STOP
    assert CommandMatcher.match_intent("Text reading band karo")[0] == CommandIntent.STOP
    assert CommandMatcher.match_intent("reading band karo")[0] == CommandIntent.STOP
    assert CommandMatcher.match_intent("reading banned karo")[0] == CommandIntent.STOP
    assert CommandMatcher.match_intent("band karo")[0] == CommandIntent.STOP
    assert CommandMatcher.match_intent("पढ़ना बंद करो")[0] == CommandIntent.STOP
    assert CommandMatcher.match_intent("रोक दो")[0] == CommandIntent.STOP


def test_unknown_commands():
    assert CommandMatcher.match_intent("")[0] == CommandIntent.UNKNOWN
    assert CommandMatcher.match_intent("hello how are you")[0] == CommandIntent.UNKNOWN
    assert CommandMatcher.match_intent("weather forecast today")[0] == CommandIntent.UNKNOWN
    assert CommandMatcher.match_intent("1 2 3 4 5")[0] == CommandIntent.UNKNOWN
