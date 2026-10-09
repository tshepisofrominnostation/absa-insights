"""Sentiment engines: overall review polarity + aspect-level polarity.

Baseline engine: VADER (Hutto & Gilbert, 2014 — ``vaderSentiment`` package).
VADER is rule-based, needs no training data, runs in milliseconds, and is
tuned for short social/product text — ideal as the ABSA baseline.

Comparison engine: ``src.compare_llm`` (LLM with strict JSON schema).

Threshold convention (used everywhere in this project):
    compound >= +0.05 -> Positive
    compound <= -0.05 -> Negative
    otherwise         -> Neutral
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

POSITIVE, NEGATIVE, NEUTRAL = "Positive", "Negative", "Neutral"
_THRESHOLD = 0.05


@dataclass(frozen=True)
class AspectOpinion:
    """One ABSA triple plus its evidence clause."""
    aspect: str
    sentiment: str
    score: float
    evidence: str


@dataclass
class ReviewResult:
    """Full per-review output, matching the project JSON schema."""
    review_id: str
    text: str
    overall_sentiment: str
    overall_score: float
    aspects: list[AspectOpinion] = field(default_factory=list)

    @property
    def is_unassigned(self) -> bool:
        """True when no aspect from the taxonomy was detected."""
        return not self.aspects


class SentimentEngine(Protocol):
    """Interface every sentiment engine in this project implements."""

    def overall(self, text: str) -> tuple[str, float]: ...
    def aspect(self, clause: str) -> tuple[str, float]: ...


class VaderEngine:
    """VADER baseline engine (raw text in, label + compound score out)."""

    def __init__(self) -> None:
        self._analyzer = SentimentIntensityAnalyzer()

    def overall(self, text: str) -> tuple[str, float]:
        """Score a full review with VADER on raw (unprocessed) text."""
        compound = self._analyzer.polarity_scores(text)["compound"]
        return classify(compound), compound

    def aspect(self, clause: str) -> tuple[str, float]:
        """Score one clause (the aspect's local context) with VADER."""
        compound = self._analyzer.polarity_scores(clause)["compound"]
        return classify(compound), compound


def classify(compound: float) -> str:
    """Map a VADER compound score to a Positive/Neutral/Negative label."""
    if compound >= _THRESHOLD:
        return POSITIVE
    if compound <= -_THRESHOLD:
        return NEGATIVE
    return NEUTRAL


def analyze_review(review_id: str, text: str, engine: SentimentEngine,
                    aspects: list[tuple[str, str]]) -> ReviewResult:
    """Attach aspect-level opinions to a review using the given engine.

    Args:
        review_id: Stable identifier for the review.
        text: Raw review text (used for the overall score).
        engine: Baseline (VADER) or comparison (LLM) engine.
        aspects: ``(aspect, clause)`` pairs from ``src.aspects.extract_aspects``.

    Returns:
        A ``ReviewResult`` with overall + per-aspect sentiment and evidence.
    """
    overall_label, overall_score = engine.overall(text)
    opinions = [
        AspectOpinion(aspect=aspect, sentiment=label, score=score, evidence=clause)
        for aspect, clause in aspects
        for label, score in [engine.aspect(clause)]
    ]
    return ReviewResult(
        review_id=review_id, text=text,
        overall_sentiment=overall_label, overall_score=overall_score,
        aspects=opinions,
    )
