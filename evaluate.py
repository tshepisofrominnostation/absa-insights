"""Evaluate engines on the 100 hand-labeled reviews (data/labeled_eval.csv).

Usage:
    python evaluate.py                # VADER baseline
    python evaluate.py --model llm   # comparison LLM (needs .env API key)

Writes reports/evaluation_<engine>.md and prints the table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src import metrics
from src.aspects import extract_aspects
from src.compare_llm import LLMAspectEngine
from src.sentiment import VaderEngine, analyze_review

ROOT = Path(__file__).parent
EVAL_CSV = ROOT / "data" / "labeled_eval.csv"


def load_gold() -> tuple[list, list]:
    """Load gold labels as (overall list, aspect triple list)."""
    df = pd.read_csv(EVAL_CSV)
    gold_overall = df["gold_overall"].tolist()
    gold_aspects = []
    for _, row in df.iterrows():
        for pair in str(row["gold_aspects"]).split(";"):
            if ":" in pair:
                aspect, sentiment = pair.strip().split(":")
                gold_aspects.append((row["review_id"], aspect.strip(), sentiment.strip()))
    return gold_overall, gold_aspects


def main(model: str) -> None:
    """Run the requested engine over the eval set and report metrics."""
    df = pd.read_csv(EVAL_CSV)
    engine = VaderEngine()
    llm = LLMAspectEngine()

    pred_overall, pred_aspects = [], []
    for _, row in df.iterrows():
        text, rid = row["text"], row["review_id"]
        if model == "llm":
            if not llm.is_configured:
                raise SystemExit("ABSA_LLM_API_KEY not set — see .env.example")
            result = llm.analyze(rid, text, extract_aspects(text))
        else:
            result = analyze_review(rid, text, engine, extract_aspects(text))
        pred_overall.append(result.overall_sentiment)
        pred_aspects.extend((rid, op.aspect, op.sentiment) for op in result.aspects)

    gold_overall, gold_aspects = load_gold()
    overall = metrics.evaluate_overall(gold_overall, pred_overall)
    per_aspect = metrics.evaluate_aspects(gold_aspects, pred_aspects)

    name = "llm" if model == "llm" else "vader"
    md = [
        f"# Evaluation — {name} engine (labeled_eval.csv, n={len(df)})",
        "", f"Overall sentiment: accuracy **{overall['accuracy']}**, "
        f"macro-F1 **{overall['macro_f1']}**", "",
        per_aspect.to_markdown(index=False), "",
    ]
    out = ROOT / "reports" / f"evaluation_{name}.md"
    out.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vader", "llm"], default="vader")
    main(parser.parse_args().model)
