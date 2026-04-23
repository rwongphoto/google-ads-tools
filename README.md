# Google Ads Tools

Streamlit utilities for Google Ads search-term analysis. Currently ships with a single tool aimed at e-commerce (football-jersey retail as the worked example).

## Tools

- **`search-term-analyzer.py`** — ingest a Google Ads search-terms report and classify each query. Detects:
  - Product **types** (jersey, kit, shorts, goalkeeper, baby/kids, training)
  - Product **attributes** (home / away / third, retro, long-sleeve)
  - Brand / club / player named entities
  - Commercial intent signals (store, shop, buy, sale)

  Useful for negative-keyword mining, match-type tuning, and spotting emerging product demand.

## Stack

- Streamlit UI
- NLTK (tokenization, stopwords, WordNet lemmatization)
- Pandas + regex category detection
- `collections.Counter` for frequency breakdowns

## Setup

```bash
pip install -r requirements.txt
streamlit run search-term-analyzer.py
```

NLTK data (`punkt`, `stopwords`, `wordnet`) downloads automatically on first run.

## Input format

CSV export of a Google Ads search-terms report. The tool reads the search-term column and runs classification / frequency analysis on top.
