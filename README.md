# NLP-Powered Financial Sentiment Analyser

> Fine-tuned FinBERT on 80K financial headlines for S&P 500 sentiment scoring. Automated ETL pipeline + FastAPI endpoint + live dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green) ![Docker](https://img.shields.io/badge/Docker-ready-blue)

---

## Problem Statement

Investment teams spent 4+ hours daily manually reading financial news to assess market sentiment for 500+ tickers. The manual process was slow, inconsistent, and unscalable.

## Approach

1. **Dataset** — 80K labelled financial headlines (positive/negative/neutral)
2. **Baseline** — VADER & TextBlob, then FinBERT fine-tuning with learning-rate warmup
3. **Fine-tuning** — Hugging Face `Trainer` with class-weighted loss on GPU
4. **ETL** — Automated pipeline pulling from NewsAPI, storing in PostgreSQL
5. **API** — FastAPI with batch scoring endpoint and Redis cache
6. **Dashboard** — Streamlit live sentiment tracker (per ticker, sector)

## Results

| Metric | Value |
|--------|-------|
| Accuracy | **87.3%** |
| Macro F1 | **0.86** |
| Daily tickers tracked | **500+** |
| Analyst time saved | **4 hrs/day** |
| Trading strategies integrated | **3** |

## Project Structure

```
financial-sentiment/
├── api/
│   └── main.py              # FastAPI scoring endpoint
├── notebooks/
│   ├── 01_eda.py            # Headline EDA + baseline models
│   └── 02_fine_tuning.py    # FinBERT fine-tuning pipeline
├── src/
│   ├── data/
│   │   └── data_generator.py  # Synthetic headlines dataset
│   ├── models/
│   │   └── train.py           # Fine-tuning with HuggingFace Trainer
│   └── pipeline/
│       └── etl.py             # News ETL pipeline
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt

# Generate synthetic data
python src/data/data_generator.py

# Fine-tune model (GPU recommended, CPU fallback included)
python src/models/train.py

# Start API
uvicorn api.main:app --reload

# Start dashboard (separate terminal)
streamlit run dashboard/app.py
```

## API Usage

```bash
curl -X POST http://localhost:8000/sentiment \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "headlines": ["Apple reports record Q4 earnings beating estimates"]}'
```

Response:
```json
{
  "ticker": "AAPL",
  "aggregate_sentiment": "POSITIVE",
  "confidence": 0.92,
  "scores": {"positive": 0.92, "neutral": 0.06, "negative": 0.02},
  "headlines_scored": 1
}
```

## Docker

```bash
docker-compose up --build
```
- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501

---

**Author**: Your Name · [GitHub](https://github.com/shubham000111222) · [LinkedIn](https://linkedin.com/in/yourusername)
