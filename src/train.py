#!/usr/bin/env python3
"""Phase 2, Step 1: logistic regression on the FULL feature set.

Same model as the baseline (logistic regression) but now fed the independent
signals — HomePlanet, Cabin (deck/side/num), GroupSize, child-age. Holding the
model constant lets us read off the gain from features ALONE.

All learned preprocessing stays inside the Pipeline (fit per CV fold). Run from
the repo root:  python src/train.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from features import SPEND, build_features

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SUB = ROOT / "submissions"
SUB.mkdir(exist_ok=True)

BASELINE_CV = 0.7674  # from src/baseline.py, to compare against

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
y = train["Transported"].astype(int)

X_train = build_features(train)
X_test = build_features(test)

# Column groups get different treatment:
SPEND_COLS = SPEND + ["TotalSpend"]                       # skewed money -> log1p + scale
NUM_COLS = ["GroupSize", "CabinNum", "Age", "CryoSleep", "VIP", "IsChild"]
CAT_COLS = ["Deck", "Side", "HomePlanet", "Destination"]  # one-hot

spend_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("log1p", FunctionTransformer(np.log1p)),
    ("scale", StandardScaler()),
])
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

pre = ColumnTransformer([
    ("spend", spend_pipe, SPEND_COLS),
    ("num", num_pipe, NUM_COLS),
    ("cat", cat_pipe, CAT_COLS),
])
model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(model, X_train, y, cv=cv, scoring="accuracy")
print(f"CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
print(f"   folds:    {np.round(scores, 4)}")
print(f"   baseline: {BASELINE_CV:.4f}   ->   gain from features: {scores.mean() - BASELINE_CV:+.4f}")

model.fit(X_train, y)
pred = model.predict(X_test).astype(bool)
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": pred})
out = SUB / "submission_features_lr.csv"
submission.to_csv(out, index=False)
print(f"\nWrote {out}  ({len(submission)} rows)")
