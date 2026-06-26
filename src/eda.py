#!/usr/bin/env python3
"""Quick exploratory summary of the Spaceship Titanic data.

Prints shape, target balance, missingness, and a couple of sanity signals so we
know what we're modeling before we model it. Run from the repo root:

    python src/eda.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    print(f"train: {train.shape[0]} rows x {train.shape[1]} cols")
    print(f"test:  {test.shape[0]} rows x {test.shape[1]} cols")

    print("\n--- target balance (Transported) ---")
    print(train["Transported"].value_counts(normalize=True).round(4).to_string())

    print("\n--- missingness (% of rows, train) ---")
    miss = (train.isna().mean() * 100).round(2)
    print(miss[miss > 0].sort_values(ascending=False).to_string())

    # Sanity signal: cryosleep passengers shouldn't spend anything.
    cryo = train[train["CryoSleep"] == True]  # noqa: E712 (pandas boolean mask)
    spend_when_cryo = cryo[SPEND_COLS].sum(axis=1)
    print(
        f"\nCryoSleep rows: {len(cryo)}; "
        f"with zero total spend: {(spend_when_cryo == 0).mean():.1%}"
    )

    # Sanity signal: transport rate by the strongest raw feature.
    print("\n--- Transported rate by CryoSleep ---")
    print(train.groupby("CryoSleep")["Transported"].mean().round(4).to_string())


if __name__ == "__main__":
    main()
