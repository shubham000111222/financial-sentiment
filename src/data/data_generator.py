"""
Synthetic financial headlines dataset generator.
"""
import random
import pandas as pd
import numpy as np
from pathlib import Path

random.seed(42)
np.random.seed(42)

POSITIVE = [
    "{ticker} reports record quarterly earnings, beats Wall Street estimates",
    "{ticker} raises full-year guidance after strong revenue growth",
    "{ticker} announces major share buyback programme worth $5B",
    "{ticker} stock surges after FDA approval for flagship product",
    "{ticker} secures $2B government contract, shares jump 12%",
    "{ticker} delivers surprise profit after cost-cutting measures",
    "{ticker} gains market share as competitors struggle with supply chain",
    "{ticker} dividend increased by 15%, reflecting strong cash flows",
    "{ticker} CEO confident in long-term growth trajectory",
    "{ticker} cloud segment revenue grows 40% year-over-year",
]

NEGATIVE = [
    "{ticker} misses Q3 earnings, revenue falls short of expectations",
    "{ticker} issues profit warning amid weakening consumer demand",
    "{ticker} CEO steps down unexpectedly, stock falls 8%",
    "{ticker} faces regulatory probe over pricing practices",
    "{ticker} recalls product line, taking $300M charge in Q4",
    "{ticker} lays off 10% of workforce amid restructuring",
    "{ticker} debt rating downgraded by Moody's to junk status",
    "{ticker} loses key contract worth $1.2B annually",
    "{ticker} faces class-action lawsuit over misleading disclosures",
    "{ticker} reports widening losses as growth investments weigh on margins",
]

NEUTRAL = [
    "{ticker} reports earnings in line with analyst expectations",
    "{ticker} appoints new CFO effective next quarter",
    "{ticker} maintains full-year guidance unchanged",
    "{ticker} to present at industry conference next week",
    "{ticker} completes acquisition of smaller rival",
    "{ticker} files 10-K with SEC, no material changes disclosed",
    "{ticker} board approves $500M capital expenditure plan",
    "{ticker} opens new headquarters in Austin, Texas",
    "{ticker} enters partnership agreement with logistics company",
    "{ticker} updates investor relations website with new data",
]

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK", "JPM",
    "V", "UNH", "JNJ", "WMT", "XOM", "PG", "MA", "HD", "CVX", "MRK", "LLY",
    "ABBV", "PEP", "COST", "KO", "AVGO", "TMO", "ACN", "MCD", "WFC", "CSCO",
]

label_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
rows = []
for _ in range(80_000):
    label = random.choices([0, 1, 2], weights=[0.30, 0.40, 0.30])[0]
    ticker = random.choice(TICKERS)
    templates = [NEGATIVE, NEUTRAL, POSITIVE][label]
    text = random.choice(templates).format(ticker=ticker)
    rows.append({"headline": text, "ticker": ticker, "label": label, "sentiment": label_map[label]})

df = pd.DataFrame(rows)
out = Path("data")
out.mkdir(exist_ok=True)
df.to_csv(out / "headlines.csv", index=False)
print(f"Generated {len(df)} headlines")
print(df["sentiment"].value_counts())
