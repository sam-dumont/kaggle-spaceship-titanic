# Spaceship Titanic — Learning the Kaggle Loop

**Date:** 2026-06-26
**Goal:** Get hands-on familiar with *how Kaggle works* end-to-end, using the
Spaceship Titanic getting-started competition as the vehicle. The modeling is the
means; the **workflow loop** is the deliverable.

## Context

- User is experienced in AI-assisted ML; the novel part is the Kaggle mechanics, not
  the modeling.
- Run **locally** on macOS using the **existing Anaconda Python**. Use `uv` only if a
  Python/env install is actually needed. Install the `kaggle` CLI when we reach that step.
- Artifacts are plain **Python scripts** (reproducible, git-friendly).
- Ambition: **loop + one solid model** — baseline first to prove the loop, then ONE
  proper iteration (feature engineering + gradient boosting + cross-validation).

## The problem

- Binary classification: predict `Transported` (True/False) for ~4,300 test passengers.
- Metric: **classification accuracy** against a hidden test set.
- Getting-started competition: rolling leaderboard, ~10 submissions/day, no prizes.
- Realistic "good" score: **~0.80–0.81**. Hard ceiling ~0.83.

## Guiding principles (the real lessons)

1. **Trust local CV, not the leaderboard.** The public LB is a subset of the test set
   and is easy to overfit. Our honest signal is stratified K-fold CV accuracy
   (mean ± std). We watch whether CV and LB move together.
2. **No leakage.** Every transform (imputation, encoding) is fit *inside* each CV fold,
   never on train+test together and never on the full training set before splitting.
3. **Baseline before cleverness.** Submit a trivial model first so every later change is
   measured against a known number.
4. **YAGNI.** No ensembling, no hyperparameter search, no experiment tracking in this
   phase. Add later only if Phase 2 hooks us (becomes Phase 3).

## Project layout

```
spaceship-titanic/
├── data/            # raw train.csv, test.csv, sample_submission.csv (git-ignored)
├── submissions/     # generated submission_*.csv (git-ignored)
├── src/
│   ├── download.py  # kaggle CLI wrapper: pull + unzip competition data
│   ├── eda.py       # quick exploratory summary (missingness, target balance, signals)
│   ├── features.py  # feature engineering: one importable, fold-safe function
│   ├── baseline.py  # dumbest viable model → first submission (prove the loop)
│   └── train.py     # the real model: CV + fit + predict + write submission
├── docs/superpowers/specs/   # this spec
├── README.md        # workflow cheat-sheet ("how it works" notes)
└── requirements.txt # pandas, scikit-learn, lightgbm, kaggle
```

## Workflow phases (the deliverable)

### Phase 0 — Plumbing
- User: create Kaggle account, **accept the competition rules** (downloads 403 without
  this), generate API token → `~/.kaggle/kaggle.json` (chmod 600).
- Install `kaggle` CLI.
- `download.py` fetches and unzips the competition data into `data/`.

### Phase 1 — First submission (baseline)
- `eda.py` prints a quick summary (shape, missingness, target balance).
- `baseline.py`: minimal cleaned-feature logistic regression → `submission.csv`.
- Submit via `kaggle competitions submit`. Goal: **see a real leaderboard score**.

### Phase 2 — One solid iteration
- `features.py`: engineered features from the dataset's structure —
  `PassengerId` group (`gggg_pp`), `Cabin` split (`deck/num/side`), spending columns
  (total + per-service), booleans for CryoSleep/VIP, etc. Fold-safe.
- `train.py`: stratified K-fold CV with **LightGBM**, report out-of-fold accuracy,
  refit on full train, predict test, write `submissions/submission_*.csv`.
- Submit. Compare CV vs LB vs baseline.

## Data flow

`download.py` → `data/*.csv` → `features.transform()` applied per-fold inside
`train.py` → out-of-fold predictions → CV score → refit on full train → predict test →
`submissions/submission_*.csv` → `kaggle competitions submit` → leaderboard score.

## Gotchas we'll hit on purpose (learning moments)

- 403 when downloading before accepting competition rules.
- Submission format: must be `PassengerId,Transported` with `Transported` as boolean
  True/False; wrong columns/types are rejected.
- Daily submission cap (~10/day).
- CV-vs-LB divergence and how to read it.

## Validation

Stratified K-fold (e.g. 5 folds). Transforms fit inside folds only. Report mean ± std
CV accuracy. Baseline first; every later change measured against it.

## Explicitly out of scope (YAGNI)

Ensembling, hyperparameter search, deep nets, MLflow/experiment tracking. Deferred to a
possible Phase 3 only if the user wants to go deeper.
