#!/usr/bin/env python3
"""Estimate the accuracy CEILING our current features allow.

If a model's predicted probabilities were perfectly calibrated, the best score ANY
model could reach with these features is E[max(p, 1-p)]: for a passenger whose true
transport probability is p, even an oracle bets the bigger side and is still wrong
min(p, 1-p) of the time. That irreducible miss is 'noise the features can't resolve'.

So:  distance from our score to E[max(p,1-p)] = signal we're still leaving on the table.
     distance from E[max(p,1-p)] to 1.0      = irreducible ambiguity in these features.

Caveat: predicted p is an imperfect, noisy estimate of the true probability, so this
slightly OVER-estimates the ceiling. Treat it as a soft upper bound, not gospel.

Run from the repo root:  python src/bayes_ceiling.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from features import build_features

ROOT = Path(__file__).resolve().parent.parent
train = pd.read_csv(ROOT / "data" / "train.csv")
y = train["Transported"].astype(int)
X = build_features(train)
for c in ["Deck", "Side", "HomePlanet", "Destination"]:
    X[c] = X[c].astype("category")

model = LGBMClassifier(
    n_estimators=400, learning_rate=0.03, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=0, n_jobs=-1, verbose=-1,
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

oof_p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
actual = ((oof_p > 0.5).astype(int) == y).mean()
ceiling = np.maximum(oof_p, 1 - oof_p).mean()

print(f"Actual OOF accuracy:                  {actual:.4f}")
print(f"Estimated ceiling  E[max(p, 1-p)]:    {ceiling:.4f}")
print(f"  -> signal still extractable:        {ceiling - actual:+.4f}")
print(f"  -> irreducible ambiguity to 1.0:    {1 - ceiling:.4f}\n")

conf = np.maximum(oof_p, 1 - oof_p)
print("How decidable is each passenger? (model confidence in its own call)")
for lo, hi, label in [(0.9, 1.01, "near-certain (>0.9)"),
                      (0.7, 0.9, "leaning   (0.7-0.9)"),
                      (0.5, 0.7, "coin-tossy (<0.7)")]:
    print(f"  {label:<22}: {((conf >= lo) & (conf < hi)).mean():.1%} of passengers")
