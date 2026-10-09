# Technical Write-up — ABSA-Insights
Freddy (Tshepiso) Thosago · October 2026

## 1. Model architecture

Two-stage ABSA, deliberately modular so each stage can be tested and swapped:

**Stage A — aspect extraction (`src/aspects.py`).**
1. Clause splitting: sentences split on `[.!?;\n]`, then *before* contrast
   conjunctions (but, however, although, though, except, despite, while,
   whereas) — so "battery is great but delivery was late" never lets the
   positive clause rescue the negative one.
2. Lexicon matching: 7 aspects with synonyms, compiled to **word-boundary
   regexes**. Substring matching was rejected after testing: it made "later"
   trigger the `late` delivery keyword. Multi-word phrases are matched before
   single words. Several aspects may share a clause (documented limitation:
   they then share the clause's sentiment).
3. Emerging aspects: frequent bigrams outside the lexicon surface unseen
   themes (dependency-free stand-in for spaCy noun phrases).

**Stage B — polarity (`src/sentiment.py`, `src/compare_llm.py`).**
- Baseline: VADER compound on the aspect's clause; `±0.05` thresholds →
  Positive/Negative/Neutral. Overall review sentiment uses VADER on the raw
  full review (casing, punctuation and emoji kept — they are VADER's signal).
- Comparison: an OpenAI-compatible LLM call with a strict JSON schema
  (aspect ∈ 7-value enum, sentiment ∈ 3-value enum, verbatim evidence),
  temperature 0, configured only via `ABSA_LLM_*` environment variables, with
  graceful per-review VADER fallback on API failure.

## 2. Library choices and why

| Library | Role | Why this one |
|---|---|---|
| VADER (`vaderSentiment`) | Baseline polarity | Rule-based, no training/corpus download, millisecond latency, tuned for short informal text; the standard transparent baseline |
| Plotly | Dashboard charts | Interactive legends/hover without JS work; integrates directly with Streamlit |
| matplotlib + seaborn | Report figures | Publication-quality static PNGs for the report |
| Streamlit | Interface | Pure-Python data apps, minimal boilerplate, free/cheap hosting |
| pandas | Data wrangling | Guide requirement; the lingua franca of the analysis pipeline |
| requests + python-dotenv | Comparison engine | Thin, explicit HTTP with env-based secret handling — no vendor SDK lock-in |

No sklearn: accuracy and macro-F1 are implemented in ~30 lines
(`src/metrics.py`), avoiding a heavyweight dependency for two simple formulas.

## 3. Preprocessing — and the one rule that matters

**Raw text is never mutated before VADER.** Lowercasing and stop-word removal
happen only in `normalize_for_frequency()` for the word-frequency views.
This separation is explicit in the code and tested
(`tests/test_preprocess.py`).

Frequency-view normalisation, in order: URL removal → emoji→word mapping →
lowercase → contraction expansion → repeated-character capping
("soooo"→"soo") → tokenisation → SA/internet slang mapping ("plz"→"please",
"u"→"you") → stop-word removal. Domain words like "not" and "but" are
deliberately kept out of the stop list: removing negators would destroy the
frequency signal.

## 4. Evaluation (VADER baseline, 100 hand-labeled reviews)

Gold: 37 Positive / 41 Negative / 22 Neutral; 153 gold aspect mentions.

| Metric | Value |
|---|---|
| Overall accuracy | 0.50 |
| Overall macro-F1 | 0.476 |
| Aspect-level pooled accuracy | 0.383 |
| Aspect-level pooled macro-F1 | 0.419 |
| Aspect detection precision / recall | 0.93 / 0.82 |

Per-aspect table: `reports/evaluation_vader.md`.

**Why so low — and why that is the right result to report.** VADER's lexicon
has real gaps on retail-complaint phrasing: "delivery was two weeks late"
scores 0.0 ("late" is near-neutral in VADER), "late delivery ruined the
birthday gift surprise" scores **+0.226** (joy words "birthday"/"gift"
outweigh "ruined"). The labeled set includes these traps deliberately. The
baseline's honest weaknesses are the argument for the comparison engine;
**LLM comparison row: PENDING — run `python evaluate.py --model llm` with
`ABSA_LLM_API_KEY` set (see `.env.example`).**

## 5. Key findings on the sample dataset

- 94.3% aspect coverage (68/1,200 unassigned) — the lexicon fits the domain.
- Product accuracy is the worst decided-negative share (54.1%); quality is
  near coin-flip (49.8%); delivery is the most-mentioned and healthiest
  (11.3% decided-negative) — see the insights report for the full causal
  story and the four data-derived recommendations.

## 6. Failure modes I can demonstrate

- Sarcasm: "just LOVE waiting three weeks" → misclassified (VADER reads
  "love"); present in the eval set as e099.
- Joy-word spillover: "late delivery ruined the birthday gift surprise".
- Shared-clause aspects: "delivery was late and support never replied"
  assigns both aspects the same clause score.
- Emerging-aspect noise: template artefacts like "impressed start" surface as
  bigrams — evidence the emerging pass needs curation before real data use.
