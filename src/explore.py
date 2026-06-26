#!/usr/bin/env python3
"""Hypothesis-testing exploration. Driven by the user's hunches, run by us.

Each section tests ONE hypothesis about where signal lives, and prints numbers
plain enough to reason about. Run from the repo root:

    python src/explore.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
train = pd.read_csv(DATA_DIR / "train.csv")
BASE_RATE = train["Transported"].mean()
print(f"Base rate (overall P(Transported)): {BASE_RATE:.4f}   n={len(train)}\n")


# ── Hypothesis 1: CryoSleep is a different "human state" ──────────────────────
print("=" * 70)
print("H1: CryoSleep — different human state")
print("=" * 70)
g = train.groupby("CryoSleep")["Transported"].agg(["mean", "count"])
print(g.round(4).to_string(), "\n")

# How far does a SINGLE rule get us? Predict Transported = CryoSleep value.
known = train[train["CryoSleep"].notna()].copy()
known["CryoSleep"] = known["CryoSleep"].astype(bool)
rule_acc = (known["CryoSleep"] == known["Transported"]).mean()
null_frac = train["CryoSleep"].isna().mean()
print(f"1-rule model 'predict Transported = CryoSleep':")
print(f"   accuracy on the {1 - null_frac:.1%} of rows where CryoSleep is known: {rule_acc:.4f}")
print(f"   (CryoSleep is missing on {null_frac:.1%} of rows)\n")


# ── Hypothesis 2: PassengerId groups share a fate ─────────────────────────────
print("=" * 70)
print("H2: PassengerId group (gggg) — do group-mates share fate?")
print("=" * 70)
train["Group"] = train["PassengerId"].str[:4]
gsize = train.groupby("Group").size()
print("Group-size distribution (how many groups have N members):")
print(gsize.value_counts().sort_index().to_string(), "\n")

solo = (gsize == 1).sum()
print(f"Groups: {gsize.size} total — {solo} are solo travellers, "
      f"{gsize.size - solo} have 2+ members.\n")

# For 2-person groups: how often do BOTH members share the same outcome?
# If fate were independent (a coin flip each), they'd match 50% of the time.
pairs = train[train["Group"].isin(gsize[gsize == 2].index)]
same = pairs.groupby("Group")["Transported"].apply(lambda s: s.nunique() == 1)
print(f"2-person groups: {same.size}")
print(f"   both members share the same outcome: {same.mean():.4f}")
print(f"   (independent coin-flips would give ~0.50)\n")

# Stronger framing: P(you transported | a groupmate was transported)
# vs the base rate. Built only from OTHER members to avoid trivial self-leak.
multi = train[train["Group"].isin(gsize[gsize >= 2].index)].copy()
grp_sum = multi.groupby("Group")["Transported"].transform("sum")
grp_cnt = multi.groupby("Group")["Transported"].transform("count")
others_rate = (grp_sum - multi["Transported"]) / (grp_cnt - 1)  # leave-one-out
multi["any_other_transported"] = others_rate > 0
cond = multi.groupby("any_other_transported")["Transported"].mean()
print("P(Transported) split by 'was any OTHER group member transported?':")
print(cond.round(4).to_string(), "\n")


# ── Hypothesis 3: Age — red herring? ──────────────────────────────────────────
print("=" * 70)
print("H3: Age — red herring, or is there a hidden trend?")
print("=" * 70)
bins = [0, 5, 12, 18, 25, 35, 50, 65, 200]
train["AgeBin"] = pd.cut(train["Age"], bins, right=False)
ab = train.groupby("AgeBin", observed=True)["Transported"].agg(["mean", "count"])
print(ab.round(4).to_string())
print(f"\n(Overall base rate for reference: {BASE_RATE:.4f})\n")


# ── Hypothesis 4: Cabin location — did the anomaly "hit" part of the ship? ─────
print("=" * 70)
print("H4: Cabin = deck/num/side — does physical ship location matter?")
print("=" * 70)
# Split "F/0/S" -> deck=F, num=0, side=S. expand=True makes 3 columns.
cabin = train["Cabin"].str.split("/", expand=True)
train["Deck"] = cabin[0]
train["Side"] = cabin[2]

print("Transport rate by Deck:")
deck = train.groupby("Deck")["Transported"].agg(["mean", "count"])
print(deck.round(4).to_string(), "\n")

print("Transport rate by Side (P=Port, S=Starboard):")
side = train.groupby("Side")["Transported"].agg(["mean", "count"])
print(side.round(4).to_string())
print(f"\n(Overall base rate for reference: {BASE_RATE:.4f})")
