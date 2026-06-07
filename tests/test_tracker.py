from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from kaggle_lab.tracker import (
    RunRecord, KaggleRecord,
    append_run, read_runs, get_run, latest_runs,
)


def _sample(
    run_id: str = "20260422_131500_f09d",
    parent: str | None = None,
    score: float | None = 0.763,
) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        competition="titanic",
        created_at="2026-04-22T13:15:00Z",
        notebook="experiments/demo.ipynb",
        artifact="artifacts/demo.csv" if score is not None else None,
        submission_sha1="deadbeef" if score is not None else None,
        parent_run_id=parent,
        changelog={"change": "baseline", "hypothesis": "it runs"},
        git_sha="abc1234",
        git_dirty=False,
        submitted=score is not None,
        kaggle=KaggleRecord(
            submission_type="csv",
            submission_ref="1",
            submitted_at="2026-04-22T13:15:05Z",
            scored_at="2026-04-22T13:16:00Z",
            status="complete",
            public_score=score,
            private_score=None,
            message="run_id=20260422_131500_f09d | baseline",
        ) if score is not None else None,
        metrics={"cv_mean": 0.81, "cv_std": 0.02},
        parent_public_score=None,
        delta_vs_parent=None,
        supersedes=None,
    )


def test_append_and_read_round_trip(tmp_path):
    jsonl = tmp_path / "runs.jsonl"
    append_run(jsonl, _sample(run_id="a", score=0.7))
    append_run(jsonl, _sample(run_id="b", parent="a", score=0.75))

    runs = read_runs(jsonl)
    assert [r.run_id for r in runs] == ["a", "b"]
    assert runs[1].parent_run_id == "a"
    assert runs[0].kaggle is not None
    assert runs[0].kaggle.public_score == 0.7


def test_get_run(tmp_path):
    jsonl = tmp_path / "runs.jsonl"
    append_run(jsonl, _sample(run_id="a"))
    append_run(jsonl, _sample(run_id="b", parent="a"))
    assert get_run(jsonl, "b").parent_run_id == "a"
    assert get_run(jsonl, "missing") is None


def test_jsonl_is_one_object_per_line(tmp_path):
    jsonl = tmp_path / "runs.jsonl"
    append_run(jsonl, _sample(run_id="a"))
    append_run(jsonl, _sample(run_id="b", parent="a"))
    lines = jsonl.read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # each line is valid JSON


def test_latest_runs_keeps_last_per_run_id(tmp_path):
    jsonl = tmp_path / "runs.jsonl"
    pending = _sample(run_id="r1", score=None)
    corrected = RunRecord(
        **{
            **asdict(pending),
            "kaggle": {
                "submission_type": "csv", "submission_ref": "1",
                "submitted_at": "2026-04-22T13:15:05Z",
                "scored_at": "2026-04-22T13:17:00Z",
                "status": "complete",
                "public_score": 0.8, "private_score": None,
                "message": "run_id=r1 | baseline",
            },
            "submitted": True,
            "supersedes": "r1",
        }
    )
    # _from_dict-style rehydrate: build KaggleRecord explicitly.
    corrected_dict = asdict(corrected)
    corrected_dict["kaggle"] = KaggleRecord(**corrected_dict["kaggle"])
    corrected = RunRecord(**corrected_dict)

    append_run(jsonl, pending)
    append_run(jsonl, corrected)
    runs = read_runs(jsonl)
    latest = latest_runs(runs)
    assert [r.run_id for r in latest] == ["r1"]
    assert latest[0].supersedes == "r1"
    assert latest[0].kaggle is not None and latest[0].kaggle.public_score == 0.8


def test_get_run_returns_latest_after_supersede(tmp_path):
    jsonl = tmp_path / "runs.jsonl"
    append_run(jsonl, _sample(run_id="r1", score=None))
    # Append a corrected row that shares the run_id and carries supersedes=r1.
    corrected = _sample(run_id="r1", score=0.81)
    d = asdict(corrected) | {"supersedes": "r1"}
    append_run(jsonl, RunRecord(**{**d, "kaggle": KaggleRecord(**d["kaggle"])}))

    rec = get_run(jsonl, "r1")
    assert rec is not None
    assert rec.supersedes == "r1"
    assert rec.kaggle is not None and rec.kaggle.public_score == 0.81


def test_unsubmitted_run_round_trip(tmp_path):
    jsonl = tmp_path / "runs.jsonl"
    r = _sample(run_id="x", score=None)
    append_run(jsonl, r)
    back = read_runs(jsonl)[0]
    assert back.submitted is False
    assert back.kaggle is None
    assert back.artifact is None
    assert back.submission_sha1 is None


from kaggle_lab.tracker import resolve_parent, compute_delta


def test_resolve_parent_by_exact_id(tmp_path):
    jsonl = tmp_path / "runs.jsonl"
    append_run(jsonl, _sample(run_id="20260422_131500_f09d"))
    runs = read_runs(jsonl)
    assert resolve_parent(runs, "20260422_131500_f09d").run_id == "20260422_131500_f09d"


def test_resolve_parent_by_notebook_alias(tmp_path):
    jsonl = tmp_path / "runs.jsonl"
    r = _sample(run_id="20260422_131500_f09d")
    # Replace notebook path to include the v1 alias marker.
    d = asdict(r) | {"notebook": "experiments/20260422_131500_v1_logreg.ipynb"}
    append_run(jsonl, RunRecord(**{**d, "kaggle": KaggleRecord(**d["kaggle"]) if d["kaggle"] else None}))
    runs = read_runs(jsonl)
    assert resolve_parent(runs, "v1").run_id == "20260422_131500_f09d"


def test_resolve_parent_picks_most_recent_alias(tmp_path):
    jsonl = tmp_path / "runs.jsonl"
    older = _sample(run_id="old")
    d_older = asdict(older) | {
        "notebook": "experiments/20260101_100000_v1_a.ipynb",
        "created_at": "2026-01-01T10:00:00Z",
    }
    newer = _sample(run_id="new")
    d_newer = asdict(newer) | {
        "notebook": "experiments/20260401_100000_v1_b.ipynb",
        "created_at": "2026-04-01T10:00:00Z",
    }
    for d in (d_older, d_newer):
        k = KaggleRecord(**d["kaggle"]) if d["kaggle"] else None
        append_run(jsonl, RunRecord(**{**d, "kaggle": k}))
    runs = read_runs(jsonl)
    assert resolve_parent(runs, "v1").run_id == "new"


def test_resolve_parent_not_found(tmp_path):
    jsonl = tmp_path / "runs.jsonl"
    append_run(jsonl, _sample(run_id="a"))
    with pytest.raises(ValueError, match="no run"):
        resolve_parent(read_runs(jsonl), "zzz")


def test_resolve_parent_root_rejected(tmp_path):
    jsonl = tmp_path / "runs.jsonl"
    append_run(jsonl, _sample(run_id="a"))
    with pytest.raises(ValueError, match="root"):
        resolve_parent(read_runs(jsonl), "root")


def test_compute_delta():
    assert compute_delta(0.779, 0.763) == pytest.approx(0.016)
    assert compute_delta(None, 0.763) is None
    assert compute_delta(0.779, None) is None
    assert compute_delta(None, None) is None
