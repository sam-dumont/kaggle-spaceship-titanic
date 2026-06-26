#!/usr/bin/env python3
"""Diverse stacking: trees + a linear model + a neural net, combined by a meta-model.

The blend failed before because 3 GBDTs are ~98% correlated. Here we add genuinely
different families (logistic regression, MLP) that should make DIFFERENT mistakes,
then learn a logistic meta-model over their out-of-fold predictions. Everything is
scored under honest StratifiedGroupKFold. Run from the repo root:

    python src/stack.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from features import SPEND, build_features

ROOT = Path(__file__).resolve().parent.parent
train = pd.read_csv(ROOT / "data" / "train.csv")
y = train["Transported"].astype(int)
groups = train["PassengerId"].str.split("_").str[0]
CAT = ["Deck", "Side", "HomePlanet", "Destination"]

X_raw = build_features(train)
X_tree = X_raw.copy()
X_cat = X_raw.copy()
for c in CAT:
    X_tree[c] = X_raw[c].fillna("NA").astype("category")
    X_cat[c] = X_raw[c].fillna("NA").astype(str)

# preprocessing for the non-tree models (they can't take NaN/categoricals)
SPEND_COLS = SPEND + ["TotalSpend"]
NUM = ["GroupSize", "CabinNum", "Age", "CryoSleep", "VIP", "IsChild"]
pre = ColumnTransformer([
    ("spend", Pipeline([("i", SimpleImputer(strategy="median")),
                        ("l", FunctionTransformer(np.log1p)), ("s", StandardScaler())]), SPEND_COLS),
    ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NUM),
    ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                      ("o", OneHotEncoder(handle_unknown="ignore"))]), CAT),
])

LGB = dict(num_leaves=15, min_child_samples=10, learning_rate=0.03, n_estimators=1200,
           subsample=0.9, colsample_bytree=1.0, reg_lambda=5.0, reg_alpha=5.0, subsample_freq=1)

# (name, factory, which X view)
BASES = [
    ("LGBM", lambda: LGBMClassifier(random_state=0, n_jobs=-1, verbose=-1, **LGB), X_tree),
    ("CatBoost", lambda: CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                 l2_leaf_reg=3.0, cat_features=CAT, random_seed=0, verbose=0,
                 allow_writing_files=False), X_cat),
    ("XGBoost", lambda: XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=6,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, enable_categorical=True,
                 tree_method="hist", eval_metric="logloss", random_state=0, n_jobs=-1), X_tree),
    ("LogReg", lambda: Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=2000))]), X_raw),
    ("MLP", lambda: Pipeline([("pre", pre), ("m", MLPClassifier(hidden_layer_sizes=(64, 32),
              max_iter=500, early_stopping=True, random_state=0))]), X_raw),
]

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)


def oof(make_model, X):
    p = np.zeros(len(y))
    for tr, va in cv.split(X, y, groups):
        m = make_model()
        m.fit(X.iloc[tr], y.iloc[tr])
        p[va] = m.predict_proba(X.iloc[va])[:, 1]
    return p


print("Computing out-of-fold predictions for each base model...")
oof_probs = {name: oof(mk, X) for name, mk, X in BASES}

print("\nIndividual OOF accuracy:")
for name, p in oof_probs.items():
    print(f"   {name:<9}: {((p > 0.5).astype(int) == y).mean():.4f}")

P = pd.DataFrame(oof_probs)
print("\nPrediction correlation (look for the LOW numbers — that's the diversity):")
print(P.corr().round(3).to_string())

avg = P.mean(axis=1)
print(f"\nsimple average of all 5:   {((avg > 0.5).astype(int) == y).mean():.4f}")

meta_acc = cross_val_score(LogisticRegression(max_iter=2000), P, y,
                           cv=cv, groups=groups, scoring="accuracy")
print(f"stacked (logistic meta):   {meta_acc.mean():.4f} +/- {meta_acc.std():.4f}")
print(f"\nbest single base:          {max(((p>0.5).astype(int)==y).mean() for p in oof_probs.values()):.4f}")
