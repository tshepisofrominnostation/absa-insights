"""Tests for clause splitting and aspect extraction (Stage A)."""
from __future__ import annotations

from src.aspects import ASPECT_LEXICON, extract_aspects, get_lexicon, split_clauses


def test_contrast_word_splits_clauses():
    clauses = split_clauses("The battery lasts all day but delivery was late.")
    assert any("battery" in c for c in clauses)
    assert any("delivery" in c for c in clauses)
    assert clauses[0] != clauses[1]  # local context preserved


def test_sentence_punctuation_splits_clauses():
    assert len(split_clauses("Great quality. Shipping took ages.")) == 2


def test_aspect_paired_with_local_clause_not_whole_review():
    hits = dict(extract_aspects("Battery is great but the courier was rude"))
    assert set(hits) == {"quality", "delivery"}
    assert "rude" in hits["delivery"] and "great" in hits["quality"]


def test_multiword_phrase_beats_single_word():
    aspects = [a for a, _ in extract_aspects("the customer service agent was helpful")]
    assert "customer service" in aspects


def test_no_aspect_returns_empty():
    assert extract_aspects("Just want to say thanks for everything") == []


def test_duplicate_aspect_in_same_clause_deduplicated():
    hits = extract_aspects("delivery delivery delivery was quick")
    assert len(hits) == 1


def test_lexicon_has_seven_aspects_with_synonyms():
    lex = get_lexicon()
    assert len(lex) == 7
    assert all(len(kws) >= 3 for kws in lex.values())
    assert ASPECT_LEXICON is not lex  # copy, not the live dict


def test_eight_canonical_examples():
    text = ("The battery lasts all day but delivery was late and support "
            "never replied.")
    aspects = set(a for a, _ in extract_aspects(text))
    assert aspects == {"quality", "delivery", "customer service"}
