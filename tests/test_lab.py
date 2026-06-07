"""Tests for the Lab facade."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kaggle_lab.lab import Lab
from kaggle_lab.poller import PollerTimeout
from kaggle_lab.submitter import SubmitterError
from kaggle_lab.tracker import KaggleRecord, RunRecord, append_run, read_runs


def _write_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "slug: titanic\nmetric: accuracy\ndirection: maximize\n"
        "submission_type: csv\nsubmission_file: submission.csv\n"
        "kernel_slug: null\ndaily_limit: 10\n"
    )
    return cfg


def _git_init(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / ".gitkeep").write_text("")
    subprocess.run(["git", "add", ".gitkeep"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)


def _notebook_with_changelog(parent: str, path: Path) -> None:
    nb = {
        "cells": [
            {"cell_type": "markdown", "id": "ch", "metadata": {},
             "source": [f"## Changelog\n- parent: {parent}\n- change: baseline\n- hypothesis: it works\n"]},
            {"cell_type": "code", "id": "c", "metadata": {},
             "execution_count": None, "outputs": [],
             "source": [
                 "from pathlib import Path\n",
                 "Path('submission.csv').write_text('PassengerId,Survived\\n1,0\\n2,1\\n')\n",
             ]},
        ],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb))


def test_from_config_loads_paths(tmp_path):
    cfg = _write_config(tmp_path)
    lab = Lab.from_config(cfg)
    assert lab.config.slug == "titanic"
    assert lab.competition_dir == tmp_path
    assert lab.runs_jsonl == tmp_path / "runs.jsonl"
    assert lab.experiments_dir == tmp_path / "experiments"
    assert lab.artifacts_dir == tmp_path / "artifacts"
    assert lab.logs_dir == tmp_path / "logs"
    assert lab.data_dir == tmp_path / "data"


def test_run_end_to_end_with_mocked_api(tmp_path, monkeypatch):
    _git_init(tmp_path)
    cfg_path = _write_config(tmp_path)
    lab = Lab.from_config(cfg_path)
    lab.experiments_dir.mkdir(parents=True)
    nb = lab.experiments_dir / "20260422_140000_0000.ipynb"
    _notebook_with_changelog("root", nb)

    submission_obj = SimpleNamespace(
        ref="42", status="complete",
        publicScore=0.78, privateScore=None,
        description="run_id=<placeholder> | baseline",
    )
    api = MagicMock()
    api.authenticate.return_value = None
    api.competition_submit.side_effect = lambda **kw: (
        setattr(submission_obj, "description", kw["message"])
    )
    api.competition_submissions.return_value = [submission_obj]

    monkeypatch.chdir(tmp_path)
    record = lab.run(nb, api=api, allow_dirty=True)

    assert record.submitted is True
    assert record.kaggle is not None
    assert record.kaggle.public_score == 0.78
    assert record.kaggle.submission_ref == "42"
    assert record.parent_run_id is None  # parent: root -> None
    assert record.changelog == {"change": "baseline", "hypothesis": "it works"}
    assert record.artifact == f"artifacts/{record.run_id}.csv"
    assert (lab.artifacts_dir / f"{record.run_id}.csv").exists()
    assert (lab.logs_dir / f"{record.run_id}.log").exists()
    assert len(read_runs(lab.runs_jsonl)) == 1


def test_run_no_submit_path(tmp_path, monkeypatch):
    _git_init(tmp_path)
    cfg_path = _write_config(tmp_path)
    lab = Lab.from_config(cfg_path)
    lab.experiments_dir.mkdir(parents=True)
    nb = lab.experiments_dir / "nb.ipynb"
    _notebook_with_changelog("root", nb)

    api = MagicMock()
    monkeypatch.chdir(tmp_path)
    record = lab.run(nb, api=api, submit=False, allow_dirty=True)

    assert record.submitted is False
    assert record.kaggle is None
    assert record.artifact is None
    assert record.submission_sha1 is None
    assert (lab.logs_dir / f"{record.run_id}.log").exists()
    api.competition_submit.assert_not_called()
    api.competition_submissions.assert_not_called()


def test_run_dry_run_does_not_execute(tmp_path, monkeypatch):
    _git_init(tmp_path)
    cfg_path = _write_config(tmp_path)
    lab = Lab.from_config(cfg_path)
    lab.experiments_dir.mkdir(parents=True)
    nb = lab.experiments_dir / "nb.ipynb"
    _notebook_with_changelog("root", nb)

    api = MagicMock()
    monkeypatch.chdir(tmp_path)
    result = lab.run(nb, api=api, dry_run=True, allow_dirty=True)

    assert result is None
    assert not lab.runs_jsonl.exists() or read_runs(lab.runs_jsonl) == []
    assert not lab.artifacts_dir.exists() or list(lab.artifacts_dir.iterdir()) == []
    api.competition_submit.assert_not_called()


def test_run_fails_if_changelog_missing(tmp_path, monkeypatch):
    _git_init(tmp_path)
    cfg_path = _write_config(tmp_path)
    lab = Lab.from_config(cfg_path)
    lab.experiments_dir.mkdir(parents=True)
    nb = lab.experiments_dir / "nb.ipynb"
    nb.write_text(json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))

    monkeypatch.chdir(tmp_path)
    with pytest.raises(Exception):  # ChangelogError subclasses ValueError
        lab.run(nb, api=MagicMock(), allow_dirty=True)


def test_run_refuses_dirty_without_flag(tmp_path, monkeypatch):
    _git_init(tmp_path)
    cfg_path = _write_config(tmp_path)
    lab = Lab.from_config(cfg_path)
    lab.experiments_dir.mkdir(parents=True)
    nb = lab.experiments_dir / "nb.ipynb"
    _notebook_with_changelog("root", nb)
    # The nb file is untracked -> dirty.

    monkeypatch.chdir(tmp_path)
    with pytest.raises(Exception, match="dirty"):
        lab.run(nb, api=MagicMock())


def test_run_duplicate_rejected(tmp_path, monkeypatch):
    _git_init(tmp_path)
    cfg_path = _write_config(tmp_path)
    lab = Lab.from_config(cfg_path)
    lab.experiments_dir.mkdir(parents=True)
    nb = lab.experiments_dir / "nb.ipynb"
    _notebook_with_changelog("root", nb)

    import hashlib
    expected_content = 'PassengerId,Survived\n1,0\n2,1\n'
    expected_sha = hashlib.sha1(expected_content.encode()).hexdigest()
    existing = RunRecord(
        run_id="prior", competition="titanic", created_at="2026-04-22T10:00:00Z",
        notebook="experiments/prior.ipynb", artifact="artifacts/prior.csv",
        submission_sha1=expected_sha, parent_run_id=None,
        changelog={"change": "prior", "hypothesis": "..."},
        git_sha="a", git_dirty=False, submitted=True,
        kaggle=KaggleRecord(
            submission_type="csv", submission_ref="1",
            submitted_at="2026-04-22T10:00:00Z", scored_at="2026-04-22T10:01:00Z",
            status="complete", public_score=0.5, private_score=None,
            message="run_id=prior | prior"),
    )
    append_run(lab.runs_jsonl, existing)

    monkeypatch.chdir(tmp_path)
    with pytest.raises(Exception, match="duplicate"):
        lab.run(nb, api=MagicMock(), allow_dirty=True)

    api = MagicMock()
    submission_obj = SimpleNamespace(ref="99", status="complete",
                                     publicScore=0.6, privateScore=None,
                                     description="")
    api.competition_submit.side_effect = lambda **kw: setattr(submission_obj, "description", kw["message"])
    api.competition_submissions.return_value = [submission_obj]
    rec = lab.run(nb, api=api, allow_dirty=True, force_duplicate=True)
    assert rec.submitted is True


def test_refresh_updates_pending_status(tmp_path, monkeypatch):
    cfg_path = _write_config(tmp_path)
    lab = Lab.from_config(cfg_path)
    lab.runs_jsonl.parent.mkdir(parents=True, exist_ok=True)

    pending = RunRecord(
        run_id="r1", competition="titanic", created_at="2026-04-22T10:00:00Z",
        notebook="experiments/r1.ipynb", artifact="artifacts/r1.csv",
        submission_sha1="abc", parent_run_id=None,
        changelog={"change": "c", "hypothesis": "h"},
        git_sha="x", git_dirty=False, submitted=True,
        kaggle=KaggleRecord(
            submission_type="csv", submission_ref="7",
            submitted_at="2026-04-22T10:00:00Z", scored_at=None,
            status="pending", public_score=None, private_score=None,
            message="run_id=r1 | c"),
    )
    append_run(lab.runs_jsonl, pending)

    api = MagicMock()
    api.competition_submissions.return_value = [
        SimpleNamespace(ref="7", status="complete", publicScore=0.77, privateScore=None),
    ]

    new_records = lab.refresh(api=api)
    assert len(new_records) == 1
    rec = new_records[0]
    assert rec.supersedes == "r1"
    assert rec.kaggle.status == "complete"
    assert rec.kaggle.public_score == 0.77
    assert len(read_runs(lab.runs_jsonl)) == 2


def test_run_records_pending_on_poll_timeout(tmp_path, monkeypatch):
    _git_init(tmp_path)
    cfg_path = _write_config(tmp_path)
    lab = Lab.from_config(cfg_path)
    lab.experiments_dir.mkdir(parents=True)
    nb = lab.experiments_dir / "nb.ipynb"
    _notebook_with_changelog("root", nb)

    api = MagicMock()
    api.authenticate.return_value = None
    api.competition_submit.return_value = None
    # After submit, poll returns a "pending" submission forever -> wait_for_score will time out.
    pending_sub = SimpleNamespace(
        ref="42", status="pending", publicScore=None, privateScore=None,
        description="run_id=placeholder | baseline",
    )

    def _submit(**kw):
        pending_sub.description = kw["message"]

    api.competition_submit.side_effect = _submit
    api.competition_submissions.return_value = [pending_sub]

    monkeypatch.chdir(tmp_path)
    # Small timeout so the test is fast. Pass it via the new poll_timeout kwarg OR via monkeypatching the default.
    record = lab.run(nb, api=api, allow_dirty=True, poll_timeout=0.01, poll_interval=0.0)

    assert record is not None
    assert record.submitted is True
    assert record.kaggle is not None
    assert record.kaggle.status == "pending"
    assert record.kaggle.public_score is None
    assert record.kaggle.submission_ref == "42"
    assert record.kaggle.scored_at is None
    # runs.jsonl has one row - the pending one - ready for `refresh` to update later.
    assert len(read_runs(lab.runs_jsonl)) == 1


def test_run_warns_on_submitter_error(tmp_path, monkeypatch, capsys):
    _git_init(tmp_path)
    cfg_path = _write_config(tmp_path)
    lab = Lab.from_config(cfg_path)
    lab.experiments_dir.mkdir(parents=True)
    nb = lab.experiments_dir / "nb.ipynb"
    _notebook_with_changelog("root", nb)

    api = MagicMock()
    api.authenticate.return_value = None
    api.competition_submit.return_value = None
    api.competition_submissions.return_value = []  # ref can't be resolved

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SubmitterError):
        lab.run(nb, api=api, allow_dirty=True, list_retries=1, list_retry_delay=0)

    captured = capsys.readouterr()
    assert ("orphan" in captured.err.lower() or
            "check" in captured.err.lower() or
            "kaggle competitions submissions" in captured.err.lower())
    # No record written
    assert read_runs(lab.runs_jsonl) == []


def test_run_detects_dirty_when_invoked_from_outside_repo(tmp_path_factory, monkeypatch):
    # competition_dir is its own git repo.
    comp_dir = tmp_path_factory.mktemp("comp")
    _git_init(comp_dir)
    cfg_path = _write_config(comp_dir)
    lab = Lab.from_config(cfg_path)
    lab.experiments_dir.mkdir(parents=True)
    nb = lab.experiments_dir / "nb.ipynb"
    _notebook_with_changelog("root", nb)

    # Invoke from a directory that is NOT inside the competition's git repo,
    # to ensure Lab.run anchors git on competition_dir rather than the process cwd.
    # Use a sibling tmp dir (not nested under comp_dir) and make it its own
    # clean git repo so cwd-based detection would wrongly report "not dirty"
    # and silently skip the dirty-tree check.
    outside = tmp_path_factory.mktemp("outside")
    subprocess.run(["git", "init", "-q"], cwd=outside, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=outside, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=outside, check=True)
    (outside / ".gitkeep").write_text("")
    subprocess.run(["git", "add", ".gitkeep"], cwd=outside, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=outside, check=True)
    monkeypatch.chdir(outside)

    # Without allow_dirty, the untracked notebook in competition_dir should still be detected as dirty.
    with pytest.raises(Exception, match="dirty"):
        lab.run(nb, api=MagicMock())


def test_refresh_is_idempotent(tmp_path):
    cfg_path = _write_config(tmp_path)
    lab = Lab.from_config(cfg_path)
    lab.runs_jsonl.parent.mkdir(parents=True, exist_ok=True)

    pending = RunRecord(
        run_id="r1", competition="titanic", created_at="2026-04-22T10:00:00Z",
        notebook="experiments/r1.ipynb", artifact="artifacts/r1.csv",
        submission_sha1="abc", parent_run_id=None,
        changelog={"change": "c", "hypothesis": "h"},
        git_sha="x", git_dirty=False, submitted=True,
        kaggle=KaggleRecord(
            submission_type="csv", submission_ref="7",
            submitted_at="2026-04-22T10:00:00Z", scored_at=None,
            status="pending", public_score=None, private_score=None,
            message="run_id=r1 | c"),
    )
    append_run(lab.runs_jsonl, pending)

    api = MagicMock()
    api.competition_submissions.return_value = [
        SimpleNamespace(ref="7", status="complete", publicScore=0.77, privateScore=None),
    ]

    first = lab.refresh(api=api)
    assert len(first) == 1
    second = lab.refresh(api=api)
    assert second == []
    # One original + one correction = 2 rows; no third row on second refresh.
    assert len(read_runs(lab.runs_jsonl)) == 2


def test_refresh_noop_when_all_terminal(tmp_path):
    cfg_path = _write_config(tmp_path)
    lab = Lab.from_config(cfg_path)
    lab.runs_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = RunRecord(
        run_id="r1", competition="titanic", created_at="2026-04-22T10:00:00Z",
        notebook="experiments/r1.ipynb", artifact="artifacts/r1.csv",
        submission_sha1="abc", parent_run_id=None,
        changelog={"change": "c", "hypothesis": "h"},
        git_sha="x", git_dirty=False, submitted=True,
        kaggle=KaggleRecord(
            submission_type="csv", submission_ref="7",
            submitted_at="2026-04-22T10:00:00Z", scored_at="2026-04-22T10:01:00Z",
            status="complete", public_score=0.77, private_score=None,
            message="run_id=r1 | c"),
    )
    append_run(lab.runs_jsonl, done)

    api = MagicMock()
    new_records = lab.refresh(api=api)
    assert new_records == []
    assert len(read_runs(lab.runs_jsonl)) == 1
