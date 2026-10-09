"""Edge-case tests for src.preprocess (Week 3 guide: empty/NaN/emoji/caps)."""
from __future__ import annotations

import math

import pytest

from src.preprocess import clean_display_text, normalize_for_frequency


def test_empty_string_returns_no_tokens():
    assert normalize_for_frequency("") == []


def test_nan_returns_no_tokens():
    assert normalize_for_frequency(math.nan) == []


def test_none_returns_no_tokens():
    assert normalize_for_frequency(None) == []


def test_emoji_mapped_to_word():
    tokens = normalize_for_frequency("Love it 😍")
    assert "love_eyes" in tokens and "😍" not in "".join(tokens)


def test_all_caps_lowercased_for_frequency_view():
    assert "delivery" in normalize_for_frequency("THE DELIVERY WAS SO LATE")


def test_slang_normalised():
    tokens = normalize_for_frequency("plz can u help")
    assert "please" in tokens and "u" not in tokens


def test_stop_words_removed_but_negators_kept():
    # "not" is deliberately NOT a stop-word in this project
    tokens = normalize_for_frequency("This is not the best price")
    assert "the" not in tokens and "is" not in tokens
    assert "not" in tokens and "price" in tokens


def test_repeated_characters_capped():
    assert "sooo" == normalize_for_frequency("soooo slow")[0] or \
           "soo" == normalize_for_frequency("soooo slow")[0]


def test_urls_stripped():
    assert normalize_for_frequency("visit http://spam.co.za now") == \
           normalize_for_frequency("visit now")


def test_clean_display_keeps_casing_and_punctuation():
    assert clean_display_text("GREAT service!!") == "GREAT service!!"


def test_clean_display_collapses_whitespace():
    assert clean_display_text("so   messy \n text") == "so messy text"
