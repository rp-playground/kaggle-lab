# home-data-for-ml-course

Experiment log for the Kaggle competition
[*Home Data for ML Course*](https://www.kaggle.com/competitions/home-data-for-ml-course)
— Ames Housing regression on the *House Prices* dataset (learning
track). Part of the [`kaggle-lab`](../../README.md) framework.

## Result

| Metric | Value |
|---|---|
| Best public score (RMSE on log-price) | **12438.54** (run 36, 2026-05-12) |
| Baseline (run 1) | 16704.57 |
| Reduction | ~26% relative |
| Best leaderboard position | **98** |
| Documented submissions | 37 over 10 days (2026-05-03 → 2026-05-12) |

## Layout

```
home-data-for-ml-course/
├── config.yaml                 competition metadata (slug, metric, daily limit)
├── experiments/                one notebook per submission (timestamp-named)
├── utils/                      shared transformers and helpers
│                               (AmesNAImputer, AmesEncoder, TargetEncodeColumn,
│                                GaragePLSTransformer, SeedBag, FE helpers)
└── runs.jsonl                  canonical run log (see below)
```

## Run protocol

Every submission writes one JSON line to `runs.jsonl`:

```json
{
  "run_id": "20260509_185237_7b512dcb",
  "parent_run_id": "20260508_190255_8ba3b592",
  "notebook": "experiments/20260509_184714_c2fd0b6b.ipynb",
  "artifact": "artifacts/20260509_185237_7b512dcb.csv",
  "git_sha": "...", "git_dirty": false,
  "changelog": {
    "change": "condition1 and condition2 found quite redundant. replaced by nearness scores",
    "hypothesis": "should have a positive impact."
  },
  "kaggle": { "public_score": 12454.9233, "status": "complete", ... },
  "parent_public_score": 12504.28743,
  "delta_vs_parent": -49.36
}
```

Key properties:

- **`parent_run_id`** turns the log into a tree, not a flat list —
  you can read the experiment graph, ablate one change at a time,
  and recompute lifts retroactively.
- **`changelog.{change, hypothesis}`** makes every score a
  falsifiable claim — failed runs stay in the log alongside
  successful ones.
- **`git_sha` + `git_dirty`** stamp code provenance per submission.
- **`delta_vs_parent`** auto-computes the lift vs the chosen parent,
  not vs the best-so-far.

This makes the run log greppable
(`jq 'select(.kaggle.public_score < 12500)' runs.jsonl`),
scriptable, and self-describing.

## Trajectory (compressed)

| Phase | Runs | Best | Key change |
|---|---|---|---|
| Baseline | 1–2 | 16704.57 | RandomForest, semantic NA, log-price target |
| Model swap + TE | 3–8 | 15635.61 | GradientBoosting + Neighborhood target encoding |
| Outlier + size FE | 9–10 | 14719.79 | TotalSF aggregate + garage block via 3 PLS components |
| Diagnostic + Ames-aware | 11–16 | 14121.44 | AmesNAImputer + AmesEncoder + GBM grid search |
| XGBoost native-cat | 17–18 | 13854.66 | XGBoost(reg_lambda, tree_method='hist') + bath features |
| Ensemble blend | 19–20 | 12745.96 | 4-way equal-weight blend (XGB + Lasso + Ridge + KRR) |
| Interactions + stacking | 21–25 | 12600.44 | Quality×Size interactions + Ridge meta-learner over OOF |
| 5-model stack + holdout | 26–29 | 12504.29 | + HistGBR + SeedBag + 20% holdout oracle |
| Geography FE | 30–31 | **12454.92** | Condition1/Condition2 → proximity-to-amenity nearness scores |
| FE exploration (falsified) | 32–34 | 12454.92 | Three additions all regressed +48 to +57 LB vs best — LivLotRatio+Spaciousness, BldgType×GrLivArea, Neighborhood-median GrLivArea |
| Skew compression on lot dims | 35–36 | **12438.54** | log1p(LotArea) + log1p(LotFrontage) put lot dims into the same log-space as y; an extra-outlier-rules variant (run 35) regressed +123 and was rolled back |
| MI-based feature drop (falsified) | 37 | 12438.54 | Drop 8 low-MI features (5 near-constant categoricals + Misc{Feature,Val} + MoSold) regressed +93 LB — confirms the historical prior from run 11 that MI-based selection is bad on this dataset, even at narrow 8-feature scale |

Falsified runs are kept in the log, not pruned: run 11 (top-20 MI
baseline regression, +2369 LB), run 22 (`drop_originals=False`
variant of quality_interactions, +156), run 29 (TotalSF>4000
outlier removal, +431), run 31 (MiscVal outlier removal, +121),
run 32 (LivLotRatio + Spaciousness ratio features, +48), run 33
(BldgType × GrLivArea interactions, +51), run 34 (MedNhbdArea
neighborhood-median GrLivArea, +57), run 35 (extra outlier rules
on OverallCond==2 & SalePrice>300k and GrLivArea>4000 &
SalePrice<300k, +123 — rolled back in the run 36 best), run 37
(drop 8 low-MI features: 5 near-constant categoricals + Misc{Feature,Val}
+ MoSold, +93 — replays the run 11 anti-pattern at narrower scope).

## Key learnings

- **CV-LB gap on small tabular data.** 1455 train rows / 5 folds is
  small enough that fold-level structure leaks. Multiple CV-LB
  inversions led to a fixed 20% holdout split as a third oracle in
  run 29.
- **Stacking > equal-weight blending.** `Ridge(alpha=1,
  positive=True)` over OOF predictions beat equal-weight `expm1`
  average by ~88 LB on identical base learners (run 25).
- **Interaction features over raw collinear columns.** Explicit
  `OverallQual × TotalSF` made raw `{Exter,Bsmt,Garage}{Qual,Cond}`
  redundant for tree models (run 24 v2).

## How to reproduce

1. Drop the competition data into `data/`:

    ```bash
    kaggle competitions download -c home-data-for-ml-course -p data/
    ```

2. Open the notebook referenced in any `runs.jsonl` row to inspect
   that experiment. Each notebook is self-contained for its run.

3. The shared transformers live in `utils/`:
   - `AmesNAImputer` — distinguishes "feature absent" NaN (semantic
     zero for basement / garage / pool / fireplace / fence / veneer)
     from "missing value" NaN (imputed with leak-safe per-column
     fitted lookups).
   - `AmesEncoder` — one-hot for nominal categoricals with
     `handle_unknown='ignore'`.
   - `TargetEncodeColumn` — wraps sklearn's `TargetEncoder` per
     column, internal CV for leak safety.
   - `GaragePLSTransformer` — compresses 23 garage-block columns
     into 3 PLS components on log-price.
   - `SeedBag` — sklearn-compatible meta-estimator that fits `K`
     models over distinct seeds and averages predictions in
     log-space.

## Status

**Active**. Next iteration thread (when picked back up):

- **Base-learner diversity** — currently 4 of 5 base models are
  gradient-boosted. A genuinely non-tree, non-linear-kernel model
  could decorrelate further.
- **Private LB sanity check** at competition close — confirm the
  ~25% RMSE reduction holds out of sample.
- **Geography FE depth** — Condition1/Condition2 → nearness scores
  was the last lift (run 30). Other geographic signals
  (Neighborhood lat/long if joinable, school-district overlays)
  remain unexplored.
