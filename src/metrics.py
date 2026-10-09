"""Aggregations and evaluation metrics for the ABSA pipeline.

No sklearn: accuracy and macro-F1 are implemented directly (they are ~30
lines and remove a heavyweight dependency).
"""
from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from src.sentiment import ReviewResult

LABELS = ("Positive", "Neutral", "Negative")


def triples_dataframe(results: list[ReviewResult]) -> pd.DataFrame:
    """Flatten per-review results into one row per ABSA triple."""
    rows = [
        {
            "review_id": r.review_id,
            "overall_sentiment": r.overall_sentiment,
            "overall_score": round(r.overall_score, 4),
            "aspect": op.aspect,
            "aspect_sentiment": op.sentiment,
            "aspect_score": round(op.score, 4),
            "evidence": op.evidence,
            "text": r.text,
        }
        for r in results for op in r.aspects
    ]
    return pd.DataFrame(rows, columns=[
        "review_id", "overall_sentiment", "overall_score", "aspect",
        "aspect_sentiment", "aspect_score", "evidence", "text",
    ])


def sentiment_distribution(results: list[ReviewResult]) -> dict[str, int]:
    """Counts of overall Positive/Neutral/Negative across all reviews."""
    return dict(Counter(r.overall_sentiment for r in results))


def coverage(results: list[ReviewResult]) -> dict[str, float]:
    """Share of reviews with at least one aspect (vs. 'unassigned')."""
    n = len(results) or 1
    assigned = sum(1 for r in results if r.aspects)
    return {"assigned": assigned, "unassigned": n - assigned,
            "coverage_pct": round(100.0 * assigned / n, 1)}


def aspect_mention_counts(triples: pd.DataFrame) -> pd.DataFrame:
    """Mention counts per aspect, split by sentiment (for the stacked bar)."""
    if triples.empty:
        return pd.DataFrame(columns=["aspect", "Positive", "Neutral", "Negative", "total"])
    counts = (triples.groupby(["aspect", "aspect_sentiment"]).size()
              .unstack(fill_value=0)
              .reindex(columns=LABELS, fill_value=0)
              .reset_index()
              .rename_axis(None, axis=1))
    counts["total"] = counts[list(LABELS)].sum(axis=1)
    return counts.sort_values("total", ascending=False).reset_index(drop=True)


def evidence_for(triples: pd.DataFrame, aspect: str, sentiment: str,
                  top_n: int = 5) -> pd.DataFrame:
    """Top evidence clauses for one aspect/sentiment pair (dashboard drill-in)."""
    mask = (triples["aspect"] == aspect) & (triples["aspect_sentiment"] == sentiment)
    return (triples.loc[mask, ["review_id", "evidence"]]
            .drop_duplicates("evidence").head(top_n))


# ── Evaluation on the hand-labeled sample ──────────────────────────────

def _accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Plain accuracy; 0.0 on an empty list (never crashes the report)."""
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def _macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    """Macro-averaged F1 over the three sentiment labels."""
    f1s = []
    for label in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall)
                   if precision + recall else 0.0)
    return sum(f1s) / len(f1s)


def evaluate_overall(gold: list[str], pred: list[str]) -> dict[str, float]:
    """Accuracy and macro-F1 for overall review sentiment."""
    return {"n": len(gold),
            "accuracy": round(_accuracy(gold, pred), 3),
            "macro_f1": round(_macro_f1(gold, pred), 3)}


def evaluate_aspects(gold: list[tuple[str, str, str]],
                     pred: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Per-aspect accuracy and macro-F1 over (review, aspect) sentiment pairs.

    Args:
        gold: ``(review_id, aspect, sentiment)`` gold triples.
        pred: ``(review_id, aspect, sentiment)`` predicted triples.

    Returns:
        One row per aspect with n, accuracy, macro_f1 — plus a pooled row.
    """
    gold_map = {(r, a): s for r, a, s in gold}
    pred_map = {(r, a): s for r, a, s in pred}
    by_aspect: dict[str, list[str]] = defaultdict(lambda: ([], []))
    for key in set(gold_map) | set(pred_map):
        aspect = key[1]
        by_aspect[aspect][0].append(gold_map.get(key, "None"))
        by_aspect[aspect][1].append(pred_map.get(key, "None"))

    rows = []
    for aspect, (y_true, y_pred) in sorted(by_aspect.items()):
        rows.append({
            "aspect": aspect, "n": len(y_true),
            "accuracy": round(_accuracy(y_true, y_pred), 3),
            "macro_f1": round(_macro_f1(y_true, y_pred), 3),
        })
    all_true = [s for pair in by_aspect.values() for s in pair[0]]
    all_pred = [s for pair in by_aspect.values() for s in pair[1]]
    rows.append({"aspect": "POOLED", "n": len(all_true),
                 "accuracy": round(_accuracy(all_true, all_pred), 3),
                 "macro_f1": round(_macro_f1(all_true, all_pred), 3)})
    return pd.DataFrame(rows)
