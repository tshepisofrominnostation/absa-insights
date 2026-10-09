"""Load and validate unstructured review CSVs.

This module is the first stage of the pipeline: raw CSV -> validated DataFrame.
It never mutates review text; cleaning happens in ``src.preprocess``.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd


class IngestError(ValueError):
    """Raised when the uploaded file cannot be used as a review dataset."""


def load_reviews(source: str | Path | io.StringIO | bytes, text_col: str | None = None) -> pd.DataFrame:
    """Load a CSV of reviews and return a normalised DataFrame.

    Args:
        source: Path to a CSV, raw CSV bytes, or a file-like buffer.
        text_col: Name of the column containing review text. If None, the
            first column whose name suggests free text is used.

    Returns:
        DataFrame with columns ``review_id`` and ``text`` (plus any original
        metadata columns), one row per review, empty/NaN texts removed.

    Raises:
        IngestError: If the file is unreadable, empty, or has no text column.
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)

    try:
        df = pd.read_csv(source)
    except Exception as exc:  # narrow, user-facing failure
        raise IngestError(f"Could not read the file as CSV: {exc}") from exc

    if df.empty:
        raise IngestError("The file contains no rows.")

    text_col = text_col or _guess_text_column(df)
    if text_col not in df.columns:
        raise IngestError(f"Text column '{text_col}' not found. Available: {list(df.columns)}")

    df = df.rename(columns={text_col: "text"})
    df["text"] = df["text"].astype("string")
    before = len(df)
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    if df.empty:
        raise IngestError("No non-empty reviews found in the text column.")

    if "review_id" in df.columns:
        df["review_id"] = df["review_id"].astype(str)
    else:
        df["review_id"] = [f"r{i:05d}" for i in range(len(df))]

    df.attrs["dropped_rows"] = before - len(df)
    return df


def _guess_text_column(df: pd.DataFrame) -> str:
    """Pick the most likely free-text column from common naming conventions."""
    candidates = [c for c in df.columns if str(c).strip().lower() in
                  {"text", "review", "review_text", "feedback", "comment",
                   "comments", "body", "message", "content"}]
    if candidates:
        return candidates[0]
    # fall back to the object-dtype column with the longest average value
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if not obj_cols:
        raise IngestError("No candidate text column found.")
    return max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean())
