#!/usr/bin/env python3
"""Lever B: randomized hyperparameter search for LightGBM under HONEST CV.

Samples N param sets, scores each with StratifiedGroupKFold, prints the leaderboard.
No Optuna dependency — transparent random search. Run from the repo root:

    python src/tune.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

from features import build_features

ROOT = Path(__file__).resolve().parent.parent
train = pd.read_csv(ROOT / "data" / "train.csv")
y = train["Transported"].astype(int)
groups = train["PassengerId"].str.split("_").str[0]
X = build_features(train)
for c in ["Deck", "Side", "HomePlanet", "Destination"]:
    X[c] = X[c].astype("category")

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
rng = np.random.default_rng(42)

SPACE = {
    "num_leaves": [15, 31, 63, 127],
    "min_child_samples": [10, 20, 40, 80, 120],
    "learning_rate": [0.01, 0.02, 0.03, 0.05],
    "n_estimators": [300, 500, 800, 1200],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0, 10.0],
    "reg_alpha": [0.0, 1.0, 5.0],
}
N_TRIALS = 30
CURRENT = dict(num_leaves=31, min_child_samples=20, learning_rate=0.03,
               n_estimators=400, subsample=0.8, colsample_bytree=0.8,
               reg_lambda=1.0, reg_alpha=0.0)


def score(params: dict) -> tuple[float, float]:
    m = LGBMClassifier(random_state=0, n_jobs=-1, verbose=-1, subsample_freq=1, **params)
    s = cross_val_score(m, X, y, cv=cv, groups=groups, scoring="accuracy")
    return s.mean(), s.std()


results = []
cm, cs = score(CURRENT)
results.append(("current", cm, cs, CURRENT))
print(f"current config: {cm:.4f} +/- {cs:.4f}")

for i in range(N_TRIALS):
    params = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
    m, sd = score(params)
    results.append((f"trial{i}", m, sd, params))
    print(f"trial {i:>2}: {m:.4f} +/- {sd:.4f}")

results.sort(key=lambda r: r[1], reverse=True)
print("\n=== TOP 5 ===")
for name, m, sd, params in results[:5]:
    print(f"{name:<9} {m:.4f} +/- {sd:.4f}")
print("\nbest params:")
print(results[0][3])
