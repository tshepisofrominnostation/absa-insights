"""Headless end-to-end run: CSV -> preprocessing -> ABSA -> figures + aggregates.

Writes:
    reports/figures/*.png             (pie, bar, stacked, word frequencies)
    reports/absa_triples.csv          (full ABSA triples)
    reports/aggregates.json           (headline numbers for the report)

Usage: python run_pipeline.py [path/to/reviews.csv]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src import metrics
from src.aspects import ASPECT_LEXICON, emerging_aspects
from src.ingest import load_reviews
from src.preprocess import normalize_for_frequency
from src.sentiment import VaderEngine, analyze_review
from src.viz import report_figures

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "sample_reviews.csv"
REPORTS = ROOT / "reports"


def main(csv_path: Path = DATA) -> None:
    """Run the pipeline and print headline numbers as JSON."""
    df = load_reviews(csv_path)
    engine = VaderEngine()
    results = [analyze_review(r.review_id, r.text, engine,
                              __import__("src.aspects", fromlist=["extract_aspects"]).extract_aspects(r.text))
               for r in df.itertuples()]

    triples = metrics.triples_dataframe(results)
    dist = metrics.sentiment_distribution(results)
    cov = metrics.coverage(results)

    tokens_all = [t for text in df["text"] for t in normalize_for_frequency(text)]
    neg_ids = {r.review_id for r in results if r.overall_sentiment == "Negative"}
    tokens_neg = [t for text in df.loc[df["review_id"].isin(neg_ids), "text"]
                  for t in normalize_for_frequency(text)]

    figures = report_figures(triples, dist, REPORTS / "figures",
                             tokens_all, tokens_neg)
    triples.to_csv(REPORTS / "absa_triples.csv", index=False)

    agg = {
        "n_reviews": len(df),
        "dropped_rows": df.attrs.get("dropped_rows", 0),
        "sentiment_distribution": dist,
        "coverage": cov,
        "aspect_table": metrics.aspect_mention_counts(triples).to_dict("records"),
        "emerging_aspects": emerging_aspects(df["text"].tolist(), top_n=8),
        "figures": [Path(f).name for f in figures],
    }
    (REPORTS / "aggregates.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False))
    print(json.dumps(agg, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DATA)
