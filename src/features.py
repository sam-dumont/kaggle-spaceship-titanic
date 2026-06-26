"""Feature engineering for Spaceship Titanic.

`build_features(df)` returns a model-ready feature frame. It does ONLY deterministic,
row-wise / structural transforms — nothing here looks at the target, so it's safe to
apply identically to train and test with no fold leakage. The learned steps
(imputation, encoding, scaling) live in the model Pipeline, not here.
"""
from __future__ import annotations

import pandas as pd

SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # --- activity axis (the baseline's signal) ---
    for col in SPEND:
        out[col] = df[col]
    out["TotalSpend"] = df[SPEND].sum(axis=1)               # NaN treated as 0
    out["CryoSleep"] = df["CryoSleep"].map({True: 1.0, False: 0.0})

    # --- group structure, from PassengerId 'gggg_pp' (safe: no target) ---
    group = df["PassengerId"].str.split("_").str[0]
    out["GroupSize"] = group.map(group.value_counts())

    # --- cabin: 'deck/num/side' -> three features ---
    cabin = df["Cabin"].str.split("/", expand=True)
    out["Deck"] = cabin[0]
    out["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    out["Side"] = cabin[2]

    # --- planet / destination / vip ---
    out["HomePlanet"] = df["HomePlanet"]
    out["Destination"] = df["Destination"]
    out["VIP"] = df["VIP"].map({True: 1.0, False: 0.0})

    # --- age + the 'young child' signal we isolated (rate ~77% under 5) ---
    out["Age"] = df["Age"]
    out["IsChild"] = (df["Age"] < 12).astype("float")      # NaN age -> 0

    return out
