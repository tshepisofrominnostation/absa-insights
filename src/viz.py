"""Charts: Plotly for the Streamlit dashboard, matplotlib/seaborn for the report.

Every chart carries a title, axis labels and a legend — a hard guide
requirement, checked by the plotting helpers themselves.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless rendering for report figures

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import seaborn as sns  # noqa: E402

from collections import Counter  # noqa: E402

SENTIMENT_COLORS = {"Positive": "#0e9f6e", "Neutral": "#8b9dc3",
                    "Negative": "#e02d3b"}
LABEL_ORDER = ["Positive", "Neutral", "Negative"]


# ── Plotly (dashboard) ─────────────────────────────────────────────────

def plotly_pie(dist: dict[str, int]):
    """Sentiment pie chart with counts in the hover and legend."""
    labels = [l for l in LABEL_ORDER if l in dist]
    return px.pie(names=labels, values=[dist[l] for l in labels],
                  title="Overall Sentiment Distribution",
                  color=labels,
                  color_discrete_map=SENTIMENT_COLORS,
                  hole=0.35)


def plotly_bar(dist: dict[str, int]):
    """Sentiment bar chart (counts per label)."""
    labels = [l for l in LABEL_ORDER if l in dist]
    fig = px.bar(x=labels, y=[dist[l] for l in labels],
                 title="Review Count by Overall Sentiment",
                 labels={"x": "Sentiment", "y": "Number of reviews"},
                 color=labels, color_discrete_map=SENTIMENT_COLORS)
    fig.update_layout(showlegend=True)
    return fig


def plotly_aspect_stacked(triples: pd.DataFrame):
    """Aspect × sentiment stacked bar of mention counts."""
    counts = (triples.groupby(["aspect", "aspect_sentiment"]).size()
              .reset_index(name="mentions"))
    fig = px.bar(counts, x="aspect", y="mentions", color="aspect_sentiment",
                 title="Mentions per Aspect, Split by Sentiment",
                 labels={"aspect": "Aspect", "mentions": "Mentions",
                         "aspect_sentiment": "Sentiment"},
                 category_orders={"aspect_sentiment": LABEL_ORDER},
                 color_discrete_map=SENTIMENT_COLORS)
    return fig


def plotly_word_freq(tokens: list[str], top_n: int = 20, negative_only: bool = False):
    """Top-N word frequency bar chart (all reviews or negative-only)."""
    freq = Counter(tokens).most_common(top_n)
    words, counts = (list(zip(*freq)) if freq else ([], []))
    subtitle = "negative reviews only" if negative_only else "all reviews"
    fig = px.bar(x=list(words), y=list(counts),
                 title=f"Top {top_n} Words ({subtitle})",
                 labels={"x": "Word", "y": "Frequency"})
    fig.update_xaxes(tickangle=45)
    return fig


# ── Matplotlib / Seaborn (report figures) ────────────────────────────────

def report_figures(triples: pd.DataFrame, dist: dict[str, int],
                   out_dir: str, tokens_all: list[str],
                   tokens_negative: list[str]) -> list[str]:
    """Render all report figures as PNGs; returns the written file paths."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    written: list[str] = []

    labels = [l for l in LABEL_ORDER if l in dist]
    values = [dist[l] for l in labels]

    # 1. Pie
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90,
           colors=[SENTIMENT_COLORS[l] for l in labels],
           wedgeprops={"edgecolor": "white"})
    ax.set_title("Overall Sentiment Distribution (counts in legend)")
    ax.legend([f"{l}: {v}" for l, v in zip(labels, values)], loc="lower right")
    p = out / "sentiment_pie.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); written.append(str(p)); plt.close(fig)

    # 2. Bar
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.barplot(x=labels, y=values, hue=labels, ax=ax, palette=SENTIMENT_COLORS, legend=True)
    ax.set_title("Review Count by Overall Sentiment")
    ax.set_xlabel("Overall sentiment"); ax.set_ylabel("Number of reviews")
    p = out / "sentiment_bar.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); written.append(str(p)); plt.close(fig)

    # 3. Aspect × sentiment stacked
    counts = (triples.groupby(["aspect", "aspect_sentiment"]).size()
              .unstack(fill_value=0).reindex(columns=LABEL_ORDER, fill_value=0)
              .sort_values("Positive", ascending=False))
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = None
    for label in LABEL_ORDER:
        ax.barh(counts.index, counts[label], left=bottom, label=label,
                color=SENTIMENT_COLORS[label])
        bottom = counts[label] if bottom is None else bottom + counts[label]
    ax.set_title("Mentions per Aspect, Split by Sentiment")
    ax.set_xlabel("Number of mentions"); ax.set_ylabel("Aspect")
    ax.legend(title="Sentiment")
    p = out / "aspect_stacked.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); written.append(str(p)); plt.close(fig)

    # 4. Word frequencies (all vs negative-only, side by side)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, toks, subtitle in ((axes[0], tokens_all, "All reviews"),
                               (axes[1], tokens_negative, "Negative reviews only")):
        top = Counter(toks).most_common(15)
        if top:
            words, freqs = zip(*top)
            sns.barplot(x=list(freqs), y=list(words), hue=list(words), ax=ax,
                        palette="crest", legend=False)
        ax.set_title(f"Top 15 Words — {subtitle}")
        ax.set_xlabel("Frequency"); ax.set_ylabel("Word")
    fig.suptitle("Word Frequency: All vs Negative-Only Reviews")
    fig.tight_layout()
    p = out / "word_freq.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); written.append(str(p)); plt.close(fig)

    return written
