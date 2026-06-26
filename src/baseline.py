#!/usr/bin/env python3
"""Baseline: logistic regression on the 'activity' axis only.

Activity axis = CryoSleep + the 5 spending columns. This is the single strongest
concept we found; our scan predicted it caps around 0.74. We submit this both to
test that prediction against the real (hidden) test set and to set a number to beat.

Every preprocessing step lives inside a sklearn Pipeline, so it is re-fit on each
CV fold's training data only — no information leaks from validation rows. That's
the discipline that makes the CV score trustworthy.

Run from the repo root:  python src/baseline.py
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
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SUB = ROOT / "submissions"
SUB.mkdir(exist_ok=True)

SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the activity-axis columns. CryoSleep -> 1.0/0.0/NaN."""
    X = df[SPEND].copy()
    X["CryoSleep"] = df["CryoSleep"].map({True: 1.0, False: 0.0})  # NaN stays NaN
    return X


train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
y = train["Transported"].astype(int)

X_train = make_features(train)
X_test = make_features(test)

# Spending is heavily right-skewed (0 to thousands) and logistic regression is
# scale-sensitive, so: impute median -> log1p (compress the tail) -> standardize.
# CryoSleep just needs missing values filled with its most common value.
spend_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("log1p", FunctionTransformer(np.log1p)),
    ("scale", StandardScaler()),
])
cryo_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])

pre = ColumnTransformer([
    ("spend", spend_pipe, SPEND),
    ("cryo", cryo_pipe, ["CryoSleep"]),
])
model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])

# Honest signal: 5-fold stratified CV. Pipeline is re-fit per fold.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(model, X_train, y, cv=cv, scoring="accuracy")
print(f"CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
print(f"   folds: {np.round(scores, 4)}")

# Refit on ALL training data, predict the test set, write the submission.
model.fit(X_train, y)
pred = model.predict(X_test).astype(bool)  # Kaggle wants True/False
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": pred})
out = SUB / "submission_baseline.csv"
submission.to_csv(out, index=False)

print(f"\nWrote {out}  ({len(submission)} rows)")
print("Predicted class balance:")
print(submission["Transported"].value_counts().to_string())
