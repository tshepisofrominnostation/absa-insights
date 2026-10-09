"""Text preprocessing for frequency views ONLY.

IMPORTANT (a classic pitfall this module deliberately avoids):
VADER needs raw casing, punctuation and emojis — they carry sentiment signal.
Lowercasing / stop-word removal happens *only* for the word-frequency views.
``normalize_for_frequency`` is the only place text is aggressively normalised.

South African English / internet-slang normalisation rules are documented
in ``_SLANG_MAP`` and applied only in the frequency pipeline.
"""
from __future__ import annotations

import re

# A compact stop-word list bundled with the project (no NLTK download needed).
# Domain words that matter for reviews ("not", "but", "late", "worth") are
# intentionally NOT here — they would destroy the frequency signal.
STOP_WORDS: frozenset[str] = frozenset("""
a about after all am an and any are as at be been being but by can cant could
did do does doing dont for from get got had has have he her hers him his how i
if in into is it its just me more most my no nor of on or our out over own she
so some than that the their them then there these they this to too up us very
was we were what when where which who will with would you your yours
""".split())

# Internet slang / SA-flavoured spellings -> standard English (frequency view only).
_SLANG_MAP: dict[str, str] = {
    "u": "you", "ur": "your", "plz": "please", "pls": "please",
    "bcoz": "because", "becoz": "because", "coz": "because",
    "da": "the", "dis": "this", "dat": "that", "dnt": "dont",
    "im": "i", "ive": "i", "dont": "dont", "cant": "cant",
    "gr8": "great", "mnc": "disgusting", "yoh": "wow", "eish": "eish",
    "lekker": "lekker", "voetsek": "voetsek", "nxa": "angry",
}

# Common emojis -> sentiment-bearing words for the frequency view.
_EMOJI_MAP: dict[str, str] = {
    "\U0001F44D": " thumbs_up ", "\U0001F44E": " thumbs_down ",
    "\U00002764": " heart ", "\U0001F60D": " love_eyes ",
    "\U0001F621": " rage ", "\U0001F62D": " sob ", "\U0001F600": " smile ",
    "\U0001F44F": " clap ", "\U0001F4B8": " money ",
}

_CONTRACTIONS: dict[str, str] = {
    "won't": "will not", "can't": "can not", "n't": " not",
    "'re": " are", "'ve": " have", "'ll": " will", "'m": " am",
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_REPEAT_RE = re.compile(r"(.)\1{2,}")  # sooo -> soo (keep emphasis, cap length)


def normalize_for_frequency(text: str) -> list[str]:
    """Normalise review text into a token list for frequency charts.

    Steps: URL removal -> emoji mapping -> lowercase -> contraction expansion
    -> slang mapping -> punctuation strip -> stop-word removal.

    Args:
        text: Raw review text (any casing/emoji/NaN-safe).

    Returns:
        List of tokens, order preserved, duplicates kept.
    """
    if text is None or (isinstance(text, float) and text != text):
        return []
    text = str(text)
    text = _URL_RE.sub(" ", text)
    for emoji, word in _EMOJI_MAP.items():
        text = text.replace(emoji, word)
    text = text.lower().strip()
    for contraction, expansion in _CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    text = _REPEAT_RE.sub(r"\1\1", text)
    tokens = re.findall(r"[a-z_']+", text)
    tokens = [_SLANG_MAP.get(t, t) for t in tokens]
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def clean_display_text(text: str) -> str:
    """Light clean-up used when quoting reviews in the dashboard/report.

    Keeps casing and punctuation (readability for humans); only collapses
    runaway whitespace and repeated characters (e.g. 'sooooo' -> 'soo').
    """
    if text is None or (isinstance(text, float) and text != text):
        return ""
    text = str(text)
    text = _URL_RE.sub(" ", text)
    text = _REPEAT_RE.sub(r"\1\1", text)
    return re.sub(r"\s+", " ", text).strip()
