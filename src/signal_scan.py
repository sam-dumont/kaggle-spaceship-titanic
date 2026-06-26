#!/usr/bin/env python3
"""Comprehensive single-feature signal scan across ALL axes.

Part A: detail transport-rate tables for the axes we hadn't looked at yet.
Part B: entanglement cross-tabs (are signals independent or echoes?).
Part C: one ranked table — each feature scored by the accuracy it reaches
        ALONE (a depth-4 decision tree, 5-fold stratified CV). Same yardstick
        as "CryoSleep alone = 0.72", applied to everything.

Run from the repo root:  python src/signal_scan.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

df = pd.read_csv(DATA_DIR / "train.csv")
y = df["Transported"].astype(int)
BASE = y.mean()

# ── light feature engineering, just enough to scan every axis ────────────────
df["Group"] = df["PassengerId"].str[:4]
df["GroupSize"] = df.groupby("Group")["PassengerId"].transform("count")
cabin = df["Cabin"].str.split("/", expand=True)
df["Deck"], df["CabinNum"], df["Side"] = cabin[0], pd.to_numeric(cabin[1], errors="coerce"), cabin[2]
df["TotalSpend"] = df[SPEND].sum(axis=1)          # NaN treated as 0 for the scan
df["AnySpend"] = (df["TotalSpend"] > 0).astype(int)


def rate(col: str) -> pd.DataFrame:
    return df.groupby(col)["Transported"].agg(["mean", "count"]).round(4)


# ── PART A: the axes we hadn't checked ───────────────────────────────────────
print("#" * 72)
print(f"# PART A — untested axes      (overall base rate = {BASE:.4f})")
print("#" * 72)

print("\n-- Spending: did they spend ANYTHING at all? --")
print(rate("AnySpend").to_string())
print("\nMedian TotalSpend by outcome:")
print(df.groupby("Transported")["TotalSpend"].median().round(1).to_string())
print("\nTransport rate among people who spent >0 on each amenity:")
for c in SPEND:
    sub = df[df[c] > 0]
    print(f"   {c:<13} spent>0: n={len(sub):>4}  transport rate={sub['Transported'].mean():.4f}")

print("\n-- HomePlanet --");   print(rate("HomePlanet").to_string())
print("\n-- Destination --");  print(rate("Destination").to_string())
print("\n-- VIP --");          print(rate("VIP").to_string())


# ── PART B: entanglement — are big signals independent, or echoes? ───────────
print("\n" + "#" * 72)
print("# PART B — entanglement (is a signal real, or just another in disguise?)")
print("#" * 72)

print("\nCryoSleep  x  AnySpend  (counts) — are these two the same thing?")
print(pd.crosstab(df["CryoSleep"], df["AnySpend"]).to_string())

print("\nHomePlanet x Deck (counts) — is 'Deck B = high' really 'Europa = high'?")
print(pd.crosstab(df["HomePlanet"], df["Deck"]).to_string())


# ── PART C: unified single-feature accuracy ranking ──────────────────────────
print("\n" + "#" * 72)
print("# PART C — every feature scored ALONE (depth-4 tree, 5-fold CV accuracy)")
print("#" * 72)

CAT = ["CryoSleep", "VIP", "HomePlanet", "Destination", "Deck", "Side", "AnySpend"]
NUM = ["Age", "GroupSize", "CabinNum", "TotalSpend", *SPEND]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def single_feature_acc(col: str, kind: str) -> float:
    if kind == "cat":
        X = df[[col]].astype("object").fillna("NA").astype(str)   # uniform strings; constant fill = no leakage
        pre = OneHotEncoder(handle_unknown="ignore")
    else:
        X = df[[col]].apply(pd.to_numeric, errors="coerce")
        pre = SimpleImputer(strategy="median")               # fit per-fold via the pipeline
    pipe = Pipeline([("pre", pre), ("clf", DecisionTreeClassifier(max_depth=4, random_state=0))])
    return cross_val_score(pipe, X, y, cv=cv, scoring="accuracy").mean()


scores = {c: single_feature_acc(c, "cat") for c in CAT}
scores.update({c: single_feature_acc(c, "num") for c in NUM})

ranked = pd.Series(scores).sort_values(ascending=False)
print(f"\n(base rate / zero-skill floor = {BASE:.4f})\n")
print("feature        alone-accuracy   lift over base")
print("-" * 48)
for feat, acc in ranked.items():
    print(f"{feat:<14} {acc:>10.4f}      {acc - BASE:+.4f}")
