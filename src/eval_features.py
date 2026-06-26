#!/usr/bin/env python3
"""Measure each feature block (Lever A) under the HONEST CV.

Each block is added to the base in isolation, scored with StratifiedGroupKFold,
and compared to the base. Honest error bar is ~0.006, so a block only "counts"
if its gain clearly exceeds that. Run from the repo root:

    python src/eval_features.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

from features_v2 import CAT_COLS, build_features_v2

ROOT = Path(__file__).resolve().parent.parent
train = pd.read_csv(ROOT / "data" / "train.csv")
y = train["Transported"].astype(int)
groups = train["PassengerId"].str.split("_").str[0]
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)


def model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=0, n_jobs=-1, verbose=-1,
    )


def score(**cfg) -> tuple[float, float]:
    X = build_features_v2(train, **cfg)
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    s = cross_val_score(model(), X, y, cv=cv, groups=groups, scoring="accuracy")
    return s.mean(), s.std()


configs = {
    "base (v1 equiv)":   dict(smart_impute=False, blocks=()),
    "+A1 smart impute":  dict(smart_impute=True, blocks=()),
    "+A2 family (Name)": dict(smart_impute=False, blocks=("A2",)),
    "+A3 group aggs":    dict(smart_impute=False, blocks=("A3",)),
    "+A4 cabin region":  dict(smart_impute=False, blocks=("A4",)),
    "+A5 spend struct":  dict(smart_impute=False, blocks=("A5",)),
    "FULL (A1+all)":     dict(smart_impute=True, blocks=("A2", "A3", "A4", "A5")),
}

base_mean = None
print(f"{'config':<20} {'CV':>8} {'std':>7} {'vs base':>9}")
print("-" * 47)
for name, cfg in configs.items():
    m, sd = score(**cfg)
    if base_mean is None:
        base_mean = m
    flag = "  <- real" if (m - base_mean) > 0.006 else ""
    print(f"{name:<20} {m:.4f}  {sd:.4f}  {m - base_mean:+.4f}{flag}")
