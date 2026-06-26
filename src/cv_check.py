#!/usr/bin/env python3
"""Honest CV: shuffled StratifiedKFold vs StratifiedGroupKFold.

Groups (gggg from PassengerId) were split entirely into train OR test by Kaggle
(we verified 0% overlap), so the real task is predicting UNSEEN groups. A shuffled
CV leaks group-mates across folds and flatters the score; StratifiedGroupKFold keeps
each group whole (and still balances the target), mirroring the real test condition.

Run from the repo root:  python src/cv_check.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
)

from features import build_features

ROOT = Path(__file__).resolve().parent.parent
train = pd.read_csv(ROOT / "data" / "train.csv")
y = train["Transported"].astype(int)
groups = train["PassengerId"].str[:4]
X = build_features(train)
for c in ["Deck", "Side", "HomePlanet", "Destination"]:
    X[c] = X[c].astype("category")


def model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=0, n_jobs=-1, verbose=-1,
    )


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
s1 = cross_val_score(model(), X, y, cv=skf, scoring="accuracy")
print(f"StratifiedKFold (shuffled, leaky):  {s1.mean():.4f} +/- {s1.std():.4f}")

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
s2 = cross_val_score(model(), X, y, cv=sgkf, groups=groups, scoring="accuracy")
print(f"StratifiedGroupKFold (honest):      {s2.mean():.4f} +/- {s2.std():.4f}")

print(f"\noptimism baked into the old CV:     {s1.mean() - s2.mean():+.4f}")
