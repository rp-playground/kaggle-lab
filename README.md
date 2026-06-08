# kaggle-lab

I built this so my Kaggle work would be reproducible. Every submission is a notebook
run with a mandatory changelog — a *change* and a *hypothesis*, written before I see
the score. The framework runs the notebook with papermill, submits the output to
Kaggle, polls for the score, and appends a parent→child record to `runs.jsonl`. I
never edit a record: corrections go in as new rows carrying `supersedes`, and the
current state is the log collapsed to the latest record per run.

Installed as a CLI:

```bash
uv sync --extra dev
uv run kaggle-lab --help
```

## Why this exists

I don't care about the leaderboard score here; I care about the discipline around
it. The changelog is written before the score is known, so a run is a claim to be proved correct. 
Each run pins its code by git SHA, dedupes submissions by SHA1 so I can't re-submit the same CSV, recovers from
a polling timeout, and keeps the regressions in the log rather than only the wins.

## Worked example — home-data (Ames House Prices)

Take a look at `examples/home-data-for-ml-course/`. It
is my 37-run work on the Ames regression problem (metric: RMSE on log-price, lower
is better). `runs.jsonl` keeps all 37 records and `experiments/` ships every run's
notebook — the winners and the 15 regressions — so the whole thing is reproducible.

The table below lists only the **new-best milestones**. The real path wasn't
this easy, and some of my better-sounding ideas made the score worse: keeping only
the top-20 mutual-information features cost +2369 LB, tightening the outlier rule to
`TotalSF>4000` cost +431, and a cluster of late feature-engineering ideas cost +48
to +123. They are all still in the log. I read the full graph with
`kaggle-lab tree examples/home-data-for-ml-course`.

| # | Change | Public RMSE | Best so far |
|---|---|---|---|
| 1 | Baseline pipeline — semantic NA encoding, RandomForest | 16704.57 | ✅ |
| 2 | Replace RandomForest with GradientBoosting | 16218.88 | ✅ |
| 3 | Add Neighborhood target encoding | 15635.61 | ✅ |
| 4 | Remove mega-house outliers, add `TotalSF` | 15192.61 | ✅ |
| 5 | Replace garage block with 3 PLS components | 14719.79 | ✅ |
| 6 | Minimal preprocessing pipeline (median impute + ordinal) | 14495.67 | ✅ |
| 7 | Swap to AmesNAImputer + Ames feature steps | 14436.20 | ✅ |
| 8 | Re-add three FE steps on top of the Ames imputer | 14384.54 | ✅ |
| 9 | Revert garage-PLS (counter-productive) | 14121.44 | ✅ |
| 10 | Switch model to XGBoost (`hist`, native categorical) | 13982.72 | ✅ |
| 11 | Add bath features | 13854.66 | ✅ |
| 12 | Equal-weight blend: XGBoost + Lasso | 12879.91 | ✅ |
| 13 | Extend blend with ElasticNet | 12745.96 | ✅ |
| 14 | Add `quality_size` interaction step | 12688.29 | ✅ |
| 15 | Ridge-weighted blend (replace equal-weight mean) | 12600.44 | ✅ |
| 16 | Wrap XGBoost in `SeedBag(seeds=[42,1,7])` | 12562.82 | ✅ |
| 17 | Add a fixed 20% holdout split | 12504.29 | ✅ |
| 18 | Merge redundant Condition1/Condition2 | 12454.92 | ✅ |
| 19 | log1p(LotArea) + log1p(LotFrontage) skew compression (best) | **12438.54** | ✅ |

Baseline → best: **16704.57 → 12438.54** (~26% relative reduction), best
leaderboard position **98**, over 37 documented submissions in 10 days.

## `runs.jsonl` schema (one JSON object per line)

| Field | Meaning |
|---|---|
| `run_id` | timestamped unique id |
| `competition` | competition slug |
| `notebook` | path to the executed notebook |
| `submission_sha1` | SHA1 of the submission file (dedup key) |
| `parent_run_id` | the run this one was derived from |
| `changelog` | `{change, hypothesis}` written before scoring |
| `git_sha`, `git_dirty` | code provenance |
| `kaggle` | `{status, public_score, private_score, submission_ref, ...}` |
| `parent_public_score`, `delta_vs_parent` | improvement vs parent |
| `supersedes` | run_id this record corrects (append-only) |

Inspect with `jq`, e.g. the score progression:

```bash
jq -r '[.run_id, (.kaggle.public_score|tostring), .changelog.change] | @tsv' \
  examples/home-data-for-ml-course/runs.jsonl
```

## Repo layout

```
kaggle-lab/
├── src/kaggle_lab/     framework: changelog, runner, submitter, poller, tracker, lab, cli
├── src/eda/            reusable plotting helpers
├── tests/              pytest suite + fixtures
└── examples/
    └── home-data-for-ml-course/
        ├── config.yaml     competition metadata (metric: rmse, direction: minimize)
        ├── runs.jsonl      all 37 tracked runs
        ├── experiments/    every run's notebook (winners + regressions)
        └── utils/          Ames-specific transformers
```

## Kaggle data

Competition CSVs are not committed. Download them into
`examples/home-data-for-ml-course/data/` with the Kaggle CLI:

```bash
kaggle competitions download -c home-data-for-ml-course \
  -p examples/home-data-for-ml-course/data && \
  (cd examples/home-data-for-ml-course/data && unzip -o '*.zip')
```
