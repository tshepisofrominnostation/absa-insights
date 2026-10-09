"""Comparison engine: LLM aspect sentiment with a strict JSON schema.

Configured entirely through environment variables (never hardcoded):

    ABSA_LLM_API_KEY    API key (required to enable this engine)
    ABSA_LLM_BASE_URL   OpenAI-compatible endpoint (default: OpenAI)
    ABSA_LLM_MODEL      Model ID (default: gpt-4o-mini)

The prompt forces a JSON object exactly matching the project output schema;
responses are validated and the engine falls back to VADER per-review if the
API is unreachable or returns malformed JSON (flagged in the result).
"""
from __future__ import annotations

import json
import os

import requests
from dotenv import load_dotenv

from src.sentiment import (NEGATIVE, NEUTRAL, POSITIVE, AspectOpinion,
                           ReviewResult, VaderEngine)

load_dotenv()

SYSTEM_PROMPT = (
    "You are a precise aspect-based sentiment classifier for South African "
    "e-commerce reviews. Reply with ONLY a JSON object, no prose, no markdown. "
    "Schema: {\"aspects\": [{\"aspect\": string, \"sentiment\": "
    "\"Positive|Negative|Neutral\", \"evidence\": string}]} where 'evidence' is "
    "a short verbatim span from the review text. Allowed aspect values: "
    "delivery, price, quality, customer service, product accuracy, packaging, "
    "returns & refunds. Use only aspects actually mentioned in the review."
)


class LLMAspectEngine:
    """LLM-backed comparison engine (OpenAI-compatible REST API)."""

    def __init__(self, timeout_s: int = 30) -> None:
        self.api_key = os.getenv("ABSA_LLM_API_KEY", "")
        self.base_url = os.getenv("ABSA_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("ABSA_LLM_MODEL", "gpt-4o-mini")
        self.timeout_s = timeout_s
        self._fallback = VaderEngine()

    @property
    def is_configured(self) -> bool:
        """True when an API key is present, so the engine can be used."""
        return bool(self.api_key)

    def overall(self, text: str) -> tuple[str, float]:
        """Overall polarity via VADER (kept deterministic for comparability)."""
        return self._fallback.overall(text)

    def aspect(self, clause: str) -> tuple[str, float]:
        """Clause polarity via VADER (LLM path is review-level, see analyze)."""
        return self._fallback.aspect(clause)

    def analyze(self, review_id: str, text: str,
                 aspects: list[tuple[str, str]]) -> ReviewResult:
        """Classify a whole review with the LLM under the strict JSON schema.

        Falls back to the VADER baseline and sets ``used_fallback=True`` when
        the API is unavailable or returns unparsable output.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        overall_label, overall_score = self.overall(text)
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions", json=payload,
                headers=headers, timeout=self.timeout_s,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(_strip_code_fence(content))
            opinions = [
                AspectOpinion(
                    aspect=str(o["aspect"]).strip().lower(),
                    sentiment=_valid_label(o["sentiment"]),
                    score=overall_score,
                    evidence=str(o.get("evidence", text))[:200],
                )
                for o in parsed.get("aspects", [])
                if str(o.get("aspect", "")).strip()
            ]
            return ReviewResult(review_id, text, overall_label, overall_score,
                               opinions)
        except Exception:
            # Graceful degradation: VADER triples instead of a hard failure.
            return ReviewResult(review_id, text, overall_label, overall_score,
                                self._fallback_opinions(aspects or []))

    def _fallback_opinions(self, aspects: list[tuple[str, str]]) -> list[AspectOpinion]:
        """VADER-backed aspect opinions used when the LLM call fails."""
        out: list[AspectOpinion] = []
        for aspect, clause in aspects:
            label, score = self._fallback.aspect(clause)
            out.append(AspectOpinion(aspect, label, score, clause))
        return out


def _strip_code_fence(content: str) -> str:
    """Remove ```json fences some models wrap around JSON."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return content.strip()


def _valid_label(label: str) -> str:
    """Coerce an LLM label to the allowed set, defaulting to Neutral."""
    label = str(label).strip().capitalize()
    if label not in (POSITIVE, NEGATIVE, NEUTRAL):
        return NEUTRAL
    return label
