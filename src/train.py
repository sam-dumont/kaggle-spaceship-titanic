#!/usr/bin/env python3
"""Phase 2: hold the features constant, swap the model.

    python src/train.py logreg   # linear model (Step 1)
    python src/train.py lgbm     # gradient-boosted trees (Step 2, default)

Same features both ways, so the CV difference is the *model* lever in isolation.
LightGBM needs no scaling/encoding/imputation: it handles NaN natively and uses
categorical columns directly (we just mark them as pandas 'category' dtype).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
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

MODEL = sys.argv[1] if len(sys.argv) > 1 else "lgbm"
SPEND_COLS = SPEND + ["TotalSpend"]
NUM_COLS = ["GroupSize", "CabinNum", "Age", "CryoSleep", "VIP", "IsChild"]
CAT_COLS = ["Deck", "Side", "HomePlanet", "Destination"]
PREV = {"baseline (activity, LR)": 0.7674, "full features, LR": 0.7761}

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
y = train["Transported"].astype(int)
X_train = build_features(train)
X_test = build_features(test)


def build_logreg():
    spend_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p)),
        ("scale", StandardScaler()),
    ])
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("spend", spend_pipe, SPEND_COLS),
        ("num", num_pipe, NUM_COLS),
        ("cat", cat_pipe, CAT_COLS),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))]), X_train, X_test


def build_lgbm():
    Xtr, Xte = X_train.copy(), X_test.copy()
    for c in CAT_COLS:                      # native categorical handling
        Xtr[c] = Xtr[c].astype("category")
        Xte[c] = Xte[c].astype("category")
    model = LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=0, n_jobs=-1, verbose=-1, importance_type="gain",
    )
    return model, Xtr, Xte


model, Xtr, Xte = build_logreg() if MODEL == "logreg" else build_lgbm()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(model, Xtr, y, cv=cv, scoring="accuracy")
print(f"MODEL = {MODEL}")
print(f"CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}   folds {np.round(scores, 4)}")
for name, s in PREV.items():
    print(f"   vs {name:<24}: {scores.mean() - s:+.4f}")

model.fit(Xtr, y)
if MODEL == "lgbm":
    imp = pd.Series(model.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
    print("\nLGBM feature importance (share of total gain):")
    print((imp / imp.sum()).round(3).to_string())

pred = model.predict(Xte).astype(bool)
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": pred})
out = SUB / f"submission_{MODEL}.csv"
submission.to_csv(out, index=False)
print(f"\nWrote {out}  ({len(submission)} rows)")
