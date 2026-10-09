"""Aspect extraction: lexicon matching over clause-split reviews.

Stage A of the ABSA pipeline. Reviews are split into clauses on sentence
punctuation AND contrast conjunctions, so each aspect is scored against its
*local* context instead of the whole review ("battery is great but delivery
was late" must not score the battery Negative).

Order matters: multi-word phrases are matched before single keywords.
"""
from __future__ import annotations

import re

# 7-aspect taxonomy for SA e-commerce feedback. Phrases first, then single words.
ASPECT_LEXICON: dict[str, list[str]] = {
    "delivery": [
        "delivery", "shipping", "courier", "dispatch", "dispatched",
        "tracking", "arrived", "arrive", "arrival", "late", "still waiting",
        "parcel", "on its way", "fast", "quick delivery",
    ],
    "price": [
        "price", "prices", "cost", "costs", "expensive", "cheap", "affordable",
        "value", "rip-off", "rip off", "ripoff", "overpriced", "pricey",
        "discount", "worth", "rand",
    ],
    "quality": [
        "quality", "build", "material", "sturdy", "flimsy", "broke",
        "broken", "durable", "defective", "faulty", "battery",
        "stopped working", "lasted", "lasting", "well made",
    ],
    "customer service": [
        "customer service", "service", "support", "help", "helpline",
        "agent", "agents", "call centre", "call center", "response",
        "responded", "replied", "reply", "whatsapp", "hotline", "attentive",
    ],
    "product accuracy": [
        "description", "described", "picture", "photo", "size", "colour",
        "color", "fake", "replica", "genuine", "authentic",
        "not as described", "advertised", "advert", "looks like",
    ],
    "packaging": [
        "packaging", "box", "wrapped", "wrapping", "sealed", "bubble wrap",
        "envelope", "damaged box", "package",
    ],
    "returns & refunds": [
        "refund", "refunds", "return", "returned", "returns", "exchange",
        "money back", "replacement", "warranty",
    ],
}

# Sorted so multi-word phrases win over their single-word components.
_LEXICON_ITEMS: list[tuple[str, str]] = sorted(
    ((kw, aspect) for aspect, kws in ASPECT_LEXICON.items() for kw in kws),
    key=lambda pair: -len(pair[0]),
)

# Word-boundary regexes: substring matching caused false positives such as
# "later" triggering the "late" delivery keyword.
_LEXICON_RES: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"\b{re.escape(kw)}\b"), aspect) for kw, aspect in _LEXICON_ITEMS
]

# Sentence punctuation and contrast conjunctions define clause boundaries.
_SENTENCE_SPLIT_RE = re.compile(r"[.!?;\n]+")
_CONTRAST_SPLIT_RE = re.compile(
    r",?\s+\b(?=but\b|however\b|although\b|though\b|except\b|despite\b|"
    r"while\b|whereas\b|unfortunately\b|luckily\b)",
    re.IGNORECASE,
)


def split_clauses(review: str) -> list[str]:
    """Split a review into sentiment-bearing clauses.

    Splits on sentence punctuation and *before* contrast conjunctions
    (but/however/although/...), so polarity stays local to each clause.

    Args:
        review: Raw review text.

    Returns:
        Non-empty clauses, whitespace-normalised, order preserved.
    """
    if not review:
        return []
    clauses: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(str(review)):
        if not sentence.strip():
            continue
        clauses.extend(_CONTRAST_SPLIT_RE.split(sentence))
    return [re.sub(r"\s+", " ", c).strip(" ,") for c in clauses if c.strip(" ,")]


def extract_aspects(review: str) -> list[tuple[str, str]]:
    """Detect aspects in a review and pair each with its local clause.

    Args:
        review: Raw review text.

    Returns:
        List of ``(aspect, clause)`` tuples, one per aspect mention
        (duplicate aspect mentions in different clauses are kept;
        repeats within the same clause are deduplicated).
    """
    hits: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for clause in split_clauses(review):
        clause_l = clause.lower()
        for pattern, aspect in _LEXICON_RES:
            if pattern.search(clause_l) and (aspect, clause) not in seen:
                hits.append((aspect, clause))
                seen.add((aspect, clause))
                # NOTE: multiple aspects CAN share one clause ("delivery was
                # late and support never replied"); they then share the
                # clause's VADER score — a documented limitation (see README).
    return hits


def get_lexicon() -> dict[str, list[str]]:
    """Return a copy of the aspect lexicon (for the dashboard and docs)."""
    return {aspect: list(kws) for aspect, kws in ASPECT_LEXICON.items()}


def emerging_aspects(reviews: list[str], lexicon_words: set[str] | None = None,
                     top_n: int = 10) -> list[tuple[str, int]]:
    """Surface frequent bigrams that are NOT covered by the aspect lexicon.

    A lightweight, dependency-free stand-in for spaCy noun-phrase extraction:
    frequent adjective/noun bigrams outside the lexicon are reported as
    "emerging aspects" (e.g. a new failure theme the taxonomy missed).

    Args:
        reviews: Raw review texts.
        lexicon_words: Known keywords to exclude (defaults to the lexicon).
        top_n: How many emerging aspects to return.

    Returns:
        ``(bigram, count)`` pairs sorted by count, descending.
    """
    from collections import Counter

    from src.preprocess import STOP_WORDS, normalize_for_frequency

    lexicon_words = lexicon_words or {
        kw for kws in ASPECT_LEXICON.values() for kw in kws
    }
    counter: Counter[str] = Counter()
    for review in reviews:
        tokens = normalize_for_frequency(review)
        for a, b in zip(tokens, tokens[1:]):
            if a in lexicon_words or b in lexicon_words:
                continue
            if a in STOP_WORDS or b in STOP_WORDS or len(a) < 3 or len(b) < 3:
                continue
            counter[f"{a} {b}"] += 1
    return counter.most_common(top_n)
