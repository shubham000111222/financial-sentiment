"""
FastAPI endpoint for financial sentiment scoring.
"""
import json
import time
from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

app = FastAPI(
    title="Financial Sentiment Analysis API",
    description="FinBERT-based sentiment scoring for financial headlines",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_DIR = Path("models/finbert_finetuned")
classifier = None


@app.on_event("startup")
async def load_model():
    global classifier
    model_path = str(MODEL_DIR) if MODEL_DIR.exists() else "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer,
                          device=device, return_all_scores=True)
    print(f"✅ Sentiment model loaded (device={'cuda' if device == 0 else 'cpu'})")


class SentimentRequest(BaseModel):
    ticker: str
    headlines: List[str]


class SentimentResponse(BaseModel):
    ticker: str
    aggregate_sentiment: str
    confidence: float
    scores: dict
    headlines_scored: int
    latency_ms: float


@app.post("/sentiment", response_model=SentimentResponse)
async def sentiment(req: SentimentRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.headlines:
        raise HTTPException(status_code=400, detail="headlines list cannot be empty")

    t0 = time.perf_counter()
    results = classifier(req.headlines, truncation=True, max_length=128)
    latency = round((time.perf_counter() - t0) * 1000, 2)

    # Aggregate across all headlines
    agg = {"NEGATIVE": 0.0, "NEUTRAL": 0.0, "POSITIVE": 0.0}
    label_map = {"negative": "NEGATIVE", "neutral": "NEUTRAL", "positive": "POSITIVE",
                 "NEGATIVE": "NEGATIVE", "NEUTRAL": "NEUTRAL", "POSITIVE": "POSITIVE",
                 "label_0": "NEGATIVE", "label_1": "NEUTRAL", "label_2": "POSITIVE"}
    for r in results:
        for item in r:
            label = label_map.get(item["label"], item["label"].upper())
            agg[label] = agg.get(label, 0) + item["score"]

    n = len(req.headlines)
    agg = {k: round(v / n, 4) for k, v in agg.items()}
    top_label = max(agg, key=agg.get)

    return SentimentResponse(
        ticker=req.ticker,
        aggregate_sentiment=top_label,
        confidence=agg[top_label],
        scores=agg,
        headlines_scored=n,
        latency_ms=latency,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": classifier is not None}


@app.get("/tickers")
async def tickers():
    return {"supported": "all — supply any ticker symbol"}
