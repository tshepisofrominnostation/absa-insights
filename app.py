"""ABSA-Insights — Streamlit dashboard entry point.

UI wiring only: all analysis lives in ``src/``. Run with:
    streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from src import metrics
from src.aspects import ASPECT_LEXICON, extract_aspects
from src.compare_llm import LLMAspectEngine
from src.ingest import IngestError, load_reviews
from src.preprocess import normalize_for_frequency
from src.sentiment import VaderEngine, analyze_review
from src.viz import plotly_aspect_stacked, plotly_bar, plotly_pie, plotly_word_freq

SAMPLE_CSV = Path(__file__).parent / "data" / "sample_reviews.csv"
LABELS = ["Positive", "Neutral", "Negative"]


@st.cache_data(show_spinner=False)
def run_absa(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Run the full ABSA pipeline over a review DataFrame; cached by input."""
    engine = VaderEngine()
    llm_engine = LLMAspectEngine()
    use_llm = model == "LLM (API key)" and llm_engine.is_configured
    results = []
    for _, row in df.iterrows():
        aspects = extract_aspects(row["text"])
        if use_llm:
            results.append(llm_engine.analyze(row["review_id"], row["text"], aspects))
        else:
            results.append(analyze_review(row["review_id"], row["text"], engine, aspects))
    return metrics.triples_dataframe(results), \
        metrics.sentiment_distribution(results), \
        metrics.coverage(results)


def main() -> None:
    """Dashboard layout and interaction wiring."""
    st.set_page_config(page_title="ABSA-Insights", page_icon="📊", layout="wide")
    st.title("ABSA-Insights — Aspect-Based Sentiment Analysis")
    st.caption("VADER baseline vs LLM comparison · South African e-commerce feedback")

    with st.sidebar:
        st.header("1 · Data")
        uploaded = st.file_uploader("Upload reviews CSV", type=["csv"])
        use_sample = st.checkbox("Use bundled sample dataset", value=uploaded is None)
        if uploaded is not None:
            use_sample = False

        st.header("2 · Model")
        model = st.radio("Sentiment engine", ["VADER (baseline)", "LLM (API key)"],
                         help="The LLM engine needs ABSA_LLM_API_KEY in your environment.")
        if model.startswith("LLM") and not LLMAspectEngine().is_configured:
            st.info("No ABSA_LLM_API_KEY found — falling back to VADER. "
                    "See .env.example.")
        st.header("3 · Aspects")
        st.markdown("**Taxonomy:** " + ", ".join(ASPECT_LEXICON))

    # ── Load ──
    try:
        if uploaded is not None:
            df = load_reviews(uploaded.getvalue())
            source_label = "uploaded file"
        else:
            df = load_reviews(SAMPLE_CSV)
            source_label = "bundled sample dataset"
    except (IngestError, FileNotFoundError) as exc:
        st.error(f"Could not load data: {exc}")
        st.stop()

    if "text" not in df.columns or df.empty:
        st.error("No usable review text found. Check the text column.")
        st.stop()

    st.success(f"Loaded **{len(df):,}** reviews from the {source_label}.")

    with st.spinner("Running ABSA pipeline…"):
        triples, dist, cov = run_absa(df, model)

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Aspects", "Word Frequencies", "ABSA Triples"])

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Reviews", f"{len(df):,}")
        col2.metric("Positive", dist.get("Positive", 0))
        col3.metric("Negative", dist.get("Negative", 0))
        col4.metric("Aspect coverage", f"{cov['coverage_pct']}%")
        st.plotly_chart(plotly_pie(dist), use_container_width=True)
        st.plotly_chart(plotly_bar(dist), use_container_width=True)

    with tab2:
        if triples.empty:
            st.info("No aspects detected in this dataset.")
        else:
            st.plotly_chart(plotly_aspect_stacked(triples), use_container_width=True)
            aspect = st.selectbox("Filter by aspect",
                                  sorted(triples["aspect"].unique()))
            colp, coln = st.columns(2)
            for sentiment, col in (("Positive", colp), ("Negative", coln)):
                with col:
                    st.subheader(f"{sentiment} evidence — {aspect}")
                    ev = metrics.evidence_for(triples, aspect, sentiment, top_n=5)
                    if ev.empty:
                        st.write("_none found_")
                    for _, row in ev.iterrows():
                        st.markdown(f"> “{row['evidence']}”  \n> — {row['review_id']}")

    with tab3:
        scope = st.radio("Word frequency scope",
                         ["All reviews", "Negative reviews only"])
        texts = df["text"].tolist()
        if scope == "Negative reviews only":
            neg_ids = set(triples.loc[triples["overall_sentiment"] == "Negative",
                                      "review_id"])
            texts = df.loc[df["review_id"].isin(neg_ids), "text"].tolist()
        tokens = [t for text in texts for t in normalize_for_frequency(text)]
        if not tokens:
            st.info("No tokens to chart for this scope.")
        else:
            st.plotly_chart(plotly_word_freq(tokens, negative_only=scope != "All reviews"),
                            use_container_width=True)

    with tab4:
        if triples.empty:
            st.info("No ABSA triples to download.")
        else:
            st.dataframe(triples.head(500))
            st.download_button(
                "Download ABSA triples (CSV)",
                data=triples.to_csv(index=False).encode("utf-8"),
                file_name="absa_triples.csv", mime="text/csv")


if __name__ == "__main__":
    main()
