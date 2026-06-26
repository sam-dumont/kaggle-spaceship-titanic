#!/usr/bin/env python3
"""Lever C: blend LightGBM + XGBoost + CatBoost.

Honest out-of-fold (StratifiedGroupKFold) probabilities for each model, then the
averaged blend. We also print how correlated the models' predictions are: low
correlation = diverse errors = blending can help; high = they're redundant.

Run from the repo root:  python src/ensemble.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from features import build_features

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SUB = ROOT / "submissions"
CAT_COLS = ["Deck", "Side", "HomePlanet", "Destination"]

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
y = train["Transported"].astype(int)
groups = train["PassengerId"].str.split("_").str[0]


def frames(df):
    """Two views: category-dtype for the GBMs, string for CatBoost."""
    X = build_features(df)
    X_tree, X_cat = X.copy(), X.copy()
    for c in CAT_COLS:
        X_tree[c] = X[c].fillna("NA").astype("category")
        X_cat[c] = X[c].fillna("NA").astype(str)
    return X_tree, X_cat


Xtr_tree, Xtr_cat = frames(train)
Xte_tree, Xte_cat = frames(test)

def make_lgb():
    return LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                          random_state=0, n_jobs=-1, verbose=-1)


def make_xgb():
    return XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         enable_categorical=True, tree_method="hist",
                         eval_metric="logloss", random_state=0, n_jobs=-1)


def make_cat():
    return CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                              l2_leaf_reg=3.0, cat_features=CAT_COLS,
                              random_seed=0, verbose=0, allow_writing_files=False)


# (name, factory, which feature view). "tree" = category dtype, "cat" = string.
MODELS = [("LightGBM", make_lgb, "tree"), ("XGBoost", make_xgb, "tree"), ("CatBoost", make_cat, "cat")]
VIEW_TR = {"tree": Xtr_tree, "cat": Xtr_cat}
VIEW_TE = {"tree": Xte_tree, "cat": Xte_cat}

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)


def oof(make_model, X):
    """Manual fold loop: fresh model per fold, no sklearn clone()."""
    p = np.zeros(len(y))
    for tr, va in cv.split(X, y, groups):
        m = make_model()
        m.fit(X.iloc[tr], y.iloc[tr])
        p[va] = m.predict_proba(X.iloc[va])[:, 1]
    return p


print("Computing out-of-fold predictions (honest GroupKFold)...")
probs = {name: oof(mk, VIEW_TR[view]) for name, mk, view in MODELS}

print("\nIndividual OOF accuracy:")
for name, p in probs.items():
    print(f"   {name:<10}: {((p > 0.5).astype(int) == y).mean():.4f}")

blend = sum(probs.values()) / len(probs)
print(f"\n   BLEND (avg) : {((blend > 0.5).astype(int) == y).mean():.4f}")

print("\nPrediction correlation (lower = more diverse = blend helps more):")
print(pd.DataFrame(probs).corr().round(3).to_string())

# ── refit on full train, predict test, blend, write submission ───────────────
test_probs = []
for name, mk, view in MODELS:
    m = mk()
    m.fit(VIEW_TR[view], y)
    test_probs.append(m.predict_proba(VIEW_TE[view])[:, 1])
pred = (sum(test_probs) / len(test_probs)) > 0.5

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": pred})
SUB.mkdir(exist_ok=True)
out = SUB / "submission_ensemble.csv"
submission.to_csv(out, index=False)
print(f"\nWrote {out}  ({len(submission)} rows)")
