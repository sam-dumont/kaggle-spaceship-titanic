#!/usr/bin/env python3
"""Leave-one-out ablation: which features actually MOVE THE NEEDLE?

Feature importance tells you how much a feature is USED. Ablation tells you how
much it's NEEDED: drop it, re-run CV, see the damage. They diverge when features
are redundant — a heavily-used feature can be nearly free to drop if a twin covers
for it. This is the honest test of "does Age matter more than CryoSleep?".

Run from the repo root:  python src/ablation.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from features import SPEND, build_features

ROOT = Path(__file__).resolve().parent.parent
train = pd.read_csv(ROOT / "data" / "train.csv")
y = train["Transported"].astype(int)
X = build_features(train)
for c in ["Deck", "Side", "HomePlanet", "Destination"]:
    X[c] = X[c].astype("category")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def cv_score(cols: list[str]) -> float:
    model = LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=0, n_jobs=-1, verbose=-1,
    )
    return cross_val_score(model, X[cols], y, cv=cv, scoring="accuracy").mean()


ALL = list(X.columns)
full = cv_score(ALL)
print(f"FULL model ({len(ALL)} features): CV = {full:.4f}\n")

drop_groups = {
    "Age": ["Age"],
    "IsChild": ["IsChild"],
    "CryoSleep": ["CryoSleep"],
    "ALL spend (5 + Total)": SPEND + ["TotalSpend"],
    "Cabin (Deck/Side/Num)": ["Deck", "Side", "CabinNum"],
    "HomePlanet": ["HomePlanet"],
    "GroupSize": ["GroupSize"],
}

print(f"{'dropped':<24} {'CV':>8} {'damage':>9}")
print("-" * 44)
for name, cols in drop_groups.items():
    keep = [c for c in ALL if c not in cols]
    s = cv_score(keep)
    print(f"{name:<24} {s:.4f}  {s - full:+.4f}")
