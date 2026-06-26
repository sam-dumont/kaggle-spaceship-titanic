"""Enhanced feature engineering (Lever A), built in toggleable blocks.

build_features_v2(df, smart_impute=, blocks=) lets us measure each block in
isolation:

  A1  smart_impute=True  -> structure-aware NaN filling (logic, not medians)
  A2  "A2" in blocks     -> family size from Name surname
  A3  "A3" in blocks     -> group aggregates (group spend, mean age, is-solo)
  A4  "A4" in blocks     -> cabin-number binned into ship regions
  A5  "A5" in blocks     -> spend structure (count, luxury vs basic)

Everything is still target-free and computed within a single frame, so it stays
leakage-safe under GroupKFold (groups don't span train/test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CAT_COLS = ["Deck", "Side", "HomePlanet", "Destination"]
# decks that belong to exactly one home planet (from the H4 x-tab)
DECK_TO_PLANET = {"A": "Europa", "B": "Europa", "C": "Europa", "T": "Europa", "G": "Earth"}


def _group_mode_fill(s: pd.Series, group: pd.Series) -> pd.Series:
    mode_by_group = s.groupby(group).agg(lambda x: x.mode().iloc[0] if len(x.mode()) else np.nan)
    return s.fillna(group.map(mode_by_group))


def _group_mean_fill(s: pd.Series, group: pd.Series) -> pd.Series:
    return s.fillna(group.map(s.groupby(group).mean()))


def build_features_v2(df: pd.DataFrame, smart_impute: bool = False, blocks=()) -> pd.DataFrame:
    blocks = set(blocks)
    out = pd.DataFrame(index=df.index)
    group = df["PassengerId"].str.split("_").str[0]

    # raw pulls
    cryo = df["CryoSleep"].map({True: 1.0, False: 0.0})
    spend = df[SPEND].copy()
    home, dest = df["HomePlanet"].copy(), df["Destination"].copy()
    vip = df["VIP"].map({True: 1.0, False: 0.0})
    age = df["Age"].copy()
    cabin = df["Cabin"].str.split("/", expand=True)
    deck, cnum, side = cabin[0].copy(), pd.to_numeric(cabin[1], errors="coerce"), cabin[2].copy()

    # ── A1: structure-aware imputation ──────────────────────────────────────
    if smart_impute:
        total_raw = spend.sum(axis=1)
        cryo = cryo.fillna((total_raw == 0).astype(float))            # spend>0 => awake
        home = _group_mode_fill(home, group)                          # group shares planet
        home = home.fillna(deck.map(DECK_TO_PLANET)).fillna(home.mode().iloc[0])
        deck = _group_mode_fill(deck, group).fillna(deck.mode().iloc[0])
        side = _group_mode_fill(side, group).fillna(side.mode().iloc[0])
        cnum = _group_mean_fill(cnum, group).fillna(cnum.median())
        for c in SPEND:                                               # frozen => 0, else median
            col = spend[c].mask(spend[c].isna() & (cryo == 1), 0.0)
            spend[c] = col.fillna(col.median())
        age = _group_mean_fill(age, group).fillna(age.median())
        vip = vip.fillna(vip.mode().iloc[0])
        dest = _group_mode_fill(dest, group).fillna(dest.mode().iloc[0])

    # ── base columns ────────────────────────────────────────────────────────
    for c in SPEND:
        out[c] = spend[c]
    out["TotalSpend"] = spend.sum(axis=1)
    out["CryoSleep"] = cryo
    out["GroupSize"] = group.map(group.value_counts())
    out["Deck"], out["CabinNum"], out["Side"] = deck, cnum, side
    out["HomePlanet"], out["Destination"], out["VIP"] = home, dest, vip
    out["Age"] = age
    out["IsChild"] = (age < 12).astype(float)

    # ── A2: family from Name ──────────────────────────────────────────────────
    if "A2" in blocks:
        surname = df["Name"].str.split().str[-1]
        out["FamilySize"] = surname.map(surname.value_counts()).fillna(1)

    # ── A3: group aggregates ──────────────────────────────────────────────────
    if "A3" in blocks:
        out["GroupSpend"] = group.map(out["TotalSpend"].groupby(group).sum())
        out["GroupAgeMean"] = group.map(age.groupby(group).mean())
        out["IsSolo"] = (out["GroupSize"] == 1).astype(float)

    # ── A4: cabin-number regions ──────────────────────────────────────────────
    if "A4" in blocks:
        out["CabinRegion"] = (out["CabinNum"] // 300)

    # ── A5: spend structure ───────────────────────────────────────────────────
    if "A5" in blocks:
        out["NumAmenities"] = (spend > 0).sum(axis=1).astype(float)
        out["LuxurySpend"] = spend[["Spa", "VRDeck", "RoomService"]].sum(axis=1)
        out["BasicSpend"] = spend[["FoodCourt", "ShoppingMall"]].sum(axis=1)

    return out
