"""Tests for the kaggle-lab click CLI."""
from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from click.testing import CliRunner

from kaggle_lab.cli import main
from kaggle_lab.tracker import RunRecord, KaggleRecord, append_run


def _comp(tmp_path: Path) -> Path:
    d = tmp_path / "titanic"
    d.mkdir()
    (d / "config.yaml").write_text(
        "slug: titanic\nmetric: accuracy\ndirection: maximize\n"
        "submission_type: csv\nsubmission_file: submission.csv\n"
        "kernel_slug: null\ndaily_limit: 10\n"
    )
    return d


def _record(run_id: str, score: float | None, change: str = "change") -> RunRecord:
    k = KaggleRecord(
        submission_type="csv", submission_ref="1",
        submitted_at="2026-04-22T13:15:05Z", scored_at="2026-04-22T13:16:00Z",
        status="complete", public_score=score, private_score=None,
        message=f"run_id={run_id} | {change}",
    ) if score is not None else None
    return RunRecord(
        run_id=run_id, competition="titanic", created_at="2026-04-22T13:15:00Z",
        notebook=f"experiments/{run_id}.ipynb",
        artifact=f"artifacts/{run_id}.csv" if score is not None else None,
        submission_sha1="deadbeef" if score is not None else None,
        parent_run_id=None,
        changelog={"change": change, "hypothesis": "..."},
        git_sha="a1b2c3d", git_dirty=False,
        submitted=score is not None, kaggle=k,
    )


def test_list_empty(tmp_path):
    comp = _comp(tmp_path)
    result = CliRunner().invoke(main, ["list", str(comp)])
    assert result.exit_code == 0
    assert "no runs yet" in result.output.lower()


def test_list_sorts_desc_for_maximize(tmp_path):
    comp = _comp(tmp_path)
    append_run(comp / "runs.jsonl", _record("a", 0.7))
    append_run(comp / "runs.jsonl", _record("b", 0.9))
    append_run(comp / "runs.jsonl", _record("c", 0.8))

    result = CliRunner().invoke(main, ["list", str(comp)])
    assert result.exit_code == 0
    lines = [l for l in result.output.splitlines() if l.strip()]
    ordered_ids = [l.split()[0] for l in lines]
    assert ordered_ids == ["b", "c", "a"]


def test_list_top(tmp_path):
    comp = _comp(tmp_path)
    append_run(comp / "runs.jsonl", _record("a", 0.7))
    append_run(comp / "runs.jsonl", _record("b", 0.9))
    append_run(comp / "runs.jsonl", _record("c", 0.8))
    result = CliRunner().invoke(main, ["list", str(comp), "--top", "2"])
    assert result.exit_code == 0
    ids = [l.split()[0] for l in result.output.splitlines() if l.strip()]
    assert ids == ["b", "c"]


def test_list_ignores_unsubmitted(tmp_path):
    comp = _comp(tmp_path)
    append_run(comp / "runs.jsonl", _record("a", None))  # no score
    append_run(comp / "runs.jsonl", _record("b", 0.8))
    result = CliRunner().invoke(main, ["list", str(comp)])
    ids = [l.split()[0] for l in result.output.splitlines() if l.strip() and "no runs" not in l]
    assert ids == ["b"]


def test_show_existing_run(tmp_path):
    comp = _comp(tmp_path)
    append_run(comp / "runs.jsonl", _record("abc", 0.8, change="hello"))
    result = CliRunner().invoke(main, ["show", str(comp), "abc"])
    assert result.exit_code == 0
    assert '"run_id": "abc"' in result.output
    assert '"change": "hello"' in result.output


def test_show_missing_run(tmp_path):
    comp = _comp(tmp_path)
    append_run(comp / "runs.jsonl", _record("abc", 0.8))
    result = CliRunner().invoke(main, ["show", str(comp), "zzz"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_tree_empty(tmp_path):
    comp = _comp(tmp_path)
    result = CliRunner().invoke(main, ["tree", str(comp)])
    assert result.exit_code == 0
    assert "no runs" in result.output.lower() or result.output.strip() == "root"


def test_tree_structure(tmp_path):
    comp = _comp(tmp_path)
    a = _record("a", 0.70, change="baseline")
    append_run(comp / "runs.jsonl", a)
    append_run(comp / "runs.jsonl", RunRecord(
        **{**a.__dict__, "run_id": "b", "parent_run_id": "a",
           "changelog": {"change": "tweak", "hypothesis": "..."},
           "kaggle": KaggleRecord(**{**a.kaggle.__dict__, "public_score": 0.75}),
           "notebook": "experiments/b.ipynb", "artifact": "artifacts/b.csv"}))
    append_run(comp / "runs.jsonl", RunRecord(
        **{**a.__dict__, "run_id": "c", "parent_run_id": "a",
           "changelog": {"change": "alt", "hypothesis": "..."},
           "kaggle": KaggleRecord(**{**a.kaggle.__dict__, "public_score": 0.72}),
           "notebook": "experiments/c.ipynb", "artifact": "artifacts/c.csv"}))
    result = CliRunner().invoke(main, ["tree", str(comp)])
    assert result.exit_code == 0
    out = result.output
    # Roots appear, children are nested under their parent, deltas shown.
    assert "a" in out and "b" in out and "c" in out
    assert "baseline" in out and "tweak" in out and "alt" in out
    # Some indentation used for children.
    assert any(line.lstrip() != line for line in out.splitlines())


def test_new_creates_notebook_with_changelog(tmp_path):
    comp = _comp(tmp_path)
    result = CliRunner().invoke(main, ["new", str(comp)])
    assert result.exit_code == 0
    nb_paths = list((comp / "experiments").glob("*.ipynb"))
    assert len(nb_paths) == 1
    content = nb_paths[0].read_text()
    assert "## Changelog" in content
    assert "- parent: root" in content
    assert "- change:" in content
    assert "- hypothesis:" in content


def test_new_with_from_sets_parent(tmp_path):
    comp = _comp(tmp_path)
    append_run(comp / "runs.jsonl", _record("abc", 0.8))
    parent_nb = comp / "experiments" / "abc.ipynb"
    parent_nb.parent.mkdir(parents=True, exist_ok=True)
    parent_nb.write_text(json.dumps({
        "cells": [
            {"cell_type": "markdown", "id": "ch", "metadata": {},
             "source": ["## Changelog\n", "- parent: root\n"]},
            {"cell_type": "code", "id": "bp", "metadata": {},
             "execution_count": None, "outputs": [], "source": ["x = 1\n"]},
        ],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
    }))
    result = CliRunner().invoke(main, ["new", str(comp), "--from", "abc"])
    assert result.exit_code == 0
    nbs = sorted((comp / "experiments").glob("*.ipynb"))
    new_nb = next(p for p in nbs if p != parent_nb)
    assert "- parent: abc" in new_nb.read_text()


def _comp_with_sample(tmp_path: Path, sample_name: str | None = None) -> Path:
    d = tmp_path / "titanic"
    d.mkdir()
    (d / "data").mkdir()
    cfg_text = (
        "slug: titanic\nmetric: accuracy\ndirection: maximize\n"
        "submission_type: csv\nsubmission_file: submission.csv\n"
        "kernel_slug: null\ndaily_limit: 10\n"
    )
    if sample_name is not None:
        cfg_text += f"sample_submission: {sample_name}\n"
    (d / "config.yaml").write_text(cfg_text)
    return d


def _code_cell_source(nb_path: Path, cell_id: str = "im") -> str:
    data = json.loads(nb_path.read_text())
    code = next(c for c in data["cells"] if c.get("id") == cell_id)
    return "".join(code["source"])


def test_new_generates_template_with_sample_reference(tmp_path):
    comp = _comp_with_sample(tmp_path, sample_name="gender_submission.csv")
    (comp / "data" / "gender_submission.csv").write_text(
        "PassengerId,Survived\n1,0\n2,1\n"
    )

    result = CliRunner().invoke(main, ["new", str(comp)])
    assert result.exit_code == 0
    nb = list((comp / "experiments").glob("*.ipynb"))[0]
    src = _code_cell_source(nb)
    assert "data_dir / 'gender_submission.csv'" in src
    assert 'submission.to_csv(output_dir / "submission.csv"' in src
    assert "'Survived'" in src  # target column inferred from sample (repr-quoted)


def test_new_auto_detects_sample_when_config_null(tmp_path):
    comp = _comp_with_sample(tmp_path, sample_name=None)
    (comp / "data" / "gender_submission.csv").write_text(
        "PassengerId,Survived\n1,0\n"
    )

    result = CliRunner().invoke(main, ["new", str(comp)])
    assert result.exit_code == 0
    nb = list((comp / "experiments").glob("*.ipynb"))[0]
    src = _code_cell_source(nb)
    assert "data_dir / 'gender_submission.csv'" in src
    assert 'submission.to_csv(output_dir / "submission.csv"' in src


def test_new_falls_back_when_sample_missing(tmp_path):
    comp = _comp_with_sample(tmp_path, sample_name=None)
    # data/ exists but has no submission-y CSV.

    result = CliRunner().invoke(main, ["new", str(comp)])
    assert result.exit_code == 0
    nb = list((comp / "experiments").glob("*.ipynb"))[0]
    src = _code_cell_source(nb)
    # Generic template: no specific sample file reference, just a TODO hint.
    assert 'gender_submission.csv' not in src
    # Still mentions the output filename so the user is reminded.
    assert 'submission.csv' in src


def test_new_skips_autodetect_when_output_file_is_the_only_match(tmp_path):
    # If data/ contains only our own output filename, auto-detect must not
    # pick it as a sample.
    comp = _comp_with_sample(tmp_path, sample_name=None)
    (comp / "data" / "submission.csv").write_text("PassengerId,Survived\n1,0\n")

    result = CliRunner().invoke(main, ["new", str(comp)])
    assert result.exit_code == 0
    nb = list((comp / "experiments").glob("*.ipynb"))[0]
    src = _code_cell_source(nb)
    assert "data_dir / 'submission.csv'" not in src  # fell back to generic


def test_new_falls_back_when_autodetect_is_ambiguous(tmp_path):
    comp = _comp_with_sample(tmp_path, sample_name=None)
    (comp / "data" / "gender_submission.csv").write_text("a,b\n1,2\n")
    (comp / "data" / "other_submission.csv").write_text("a,b\n1,2\n")

    result = CliRunner().invoke(main, ["new", str(comp)])
    assert result.exit_code == 0
    nb = list((comp / "experiments").glob("*.ipynb"))[0]
    src = _code_cell_source(nb)
    # Ambiguous -> no specific file pre-wired, generic template.
    assert "data_dir / 'gender_submission.csv'" not in src
    assert "data_dir / 'other_submission.csv'" not in src


def test_new_includes_boilerplate_cell_with_slug(tmp_path):
    comp = _comp(tmp_path)
    result = CliRunner().invoke(main, ["new", str(comp)])
    assert result.exit_code == 0
    nb = list((comp / "experiments").glob("*.ipynb"))[0]
    data = json.loads(nb.read_text())
    # 1st cell: changelog (markdown), 2nd cell: boilerplate (code)
    assert data["cells"][0]["cell_type"] == "markdown"
    assert data["cells"][1].get("id") == "bp"
    bp_src = "".join(data["cells"][1]["source"])
    assert 'Path("/kaggle/input/titanic")' in bp_src
    assert "train_data = pd.read_csv(data_dir / \"train.csv\")" in bp_src
    assert "test_data  = pd.read_csv(data_dir / \"test.csv\")" in bp_src
    assert "from src.eda import (" in bp_src
    for fn in (
        "numeric_columns", "plot_histograms", "plot_kde", "plot_boxplots",
        "plot_distributions", "describe_df", "get_missing_values",
        "count_duplicates", "plot_categorical_vs_target",
        "plot_numerical_vs_target", "target_rate_table", "grouped_median",
    ):
        assert fn in bp_src


def test_init_creates_scaffold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(main, ["init", "titanic"])
    assert result.exit_code == 0
    root = tmp_path / "competitions" / "titanic"
    for sub in ("data", "experiments", "artifacts", "logs", "utils"):
        assert (root / sub).is_dir()
    cfg_text = (root / "config.yaml").read_text()
    assert "slug: titanic" in cfg_text
    assert "submission_type: csv" in cfg_text
    assert "sample_submission: null" in cfg_text


def test_init_refuses_to_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "competitions" / "titanic").mkdir(parents=True)
    (tmp_path / "competitions" / "titanic" / "config.yaml").write_text("existing: true\n")
    result = CliRunner().invoke(main, ["init", "titanic"])
    assert result.exit_code != 0
    assert (tmp_path / "competitions" / "titanic" / "config.yaml").read_text() == "existing: true\n"


def _superseded_pair(run_id: str, final_score: float) -> tuple[RunRecord, RunRecord]:
    pending = RunRecord(
        run_id=run_id, competition="titanic", created_at="2026-04-22T10:00:00Z",
        notebook=f"experiments/{run_id}.ipynb", artifact=f"artifacts/{run_id}.csv",
        submission_sha1="abc", parent_run_id=None,
        changelog={"change": "c", "hypothesis": "h"},
        git_sha="x", git_dirty=False, submitted=True,
        kaggle=KaggleRecord(
            submission_type="csv", submission_ref="7",
            submitted_at="2026-04-22T10:00:00Z", scored_at=None,
            status="pending", public_score=None, private_score=None,
            message=f"run_id={run_id} | c"),
    )
    corrected = RunRecord(
        **{
            **pending.__dict__,
            "kaggle": KaggleRecord(
                submission_type="csv", submission_ref="7",
                submitted_at="2026-04-22T10:00:00Z",
                scored_at="2026-04-22T10:05:00Z",
                status="complete", public_score=final_score, private_score=None,
                message=f"run_id={run_id} | c"),
            "supersedes": run_id,
        }
    )
    return pending, corrected


def test_show_returns_latest_after_supersede(tmp_path):
    comp = _comp(tmp_path)
    pending, corrected = _superseded_pair("r1", 0.88)
    append_run(comp / "runs.jsonl", pending)
    append_run(comp / "runs.jsonl", corrected)

    result = CliRunner().invoke(main, ["show", str(comp), "r1"])
    assert result.exit_code == 0
    assert '"status": "complete"' in result.output
    assert '"public_score": 0.88' in result.output
    assert '"supersedes": "r1"' in result.output


def test_list_dedups_superseded_rows(tmp_path):
    comp = _comp(tmp_path)
    pending, corrected = _superseded_pair("r1", 0.65)
    append_run(comp / "runs.jsonl", pending)
    append_run(comp / "runs.jsonl", corrected)
    append_run(comp / "runs.jsonl", _record("r2", 0.9))

    result = CliRunner().invoke(main, ["list", str(comp)])
    assert result.exit_code == 0
    ids = [l.split()[0] for l in result.output.splitlines() if l.strip()]
    assert ids == ["r2", "r1"]  # each run_id once, sorted desc


def test_tree_dedups_superseded_rows(tmp_path):
    comp = _comp(tmp_path)
    pending, corrected = _superseded_pair("r1", 0.65)
    append_run(comp / "runs.jsonl", pending)
    append_run(comp / "runs.jsonl", corrected)

    result = CliRunner().invoke(main, ["tree", str(comp)])
    assert result.exit_code == 0
    # r1 appears exactly once, not duplicated.
    r1_lines = [l for l in result.output.splitlines() if "r1" in l]
    assert len(r1_lines) == 1


def _fake_download_api(files: dict[str, str]) -> MagicMock:
    """Return a MagicMock whose competition_download_files writes a ZIP in path."""
    api = MagicMock()
    api.authenticate.return_value = None

    def _download(*, competition, path, **kwargs):
        zip_path = Path(path) / f"{competition}.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            for name, content in files.items():
                z.writestr(name, content)

    api.competition_download_files.side_effect = _download
    return api


def test_pull_data_downloads_extracts_and_cleans_up(tmp_path, monkeypatch):
    comp = _comp(tmp_path)
    (comp / "data").mkdir()

    api = _fake_download_api({
        "train.csv": "PassengerId,Survived\n1,0\n",
        "test.csv": "PassengerId\n1\n",
    })
    monkeypatch.setattr("kaggle_lab.cli._make_api", lambda: api)

    result = CliRunner().invoke(main, ["pull-data", str(comp)])
    assert result.exit_code == 0, result.output
    assert (comp / "data" / "train.csv").read_text().startswith("PassengerId")
    assert (comp / "data" / "test.csv").exists()
    assert not (comp / "data" / "titanic.zip").exists()
    api.competition_download_files.assert_called_once()
    kw = api.competition_download_files.call_args.kwargs
    assert kw["competition"] == "titanic"
    assert Path(kw["path"]) == comp / "data"


def test_pull_data_no_zip_is_no_op(tmp_path, monkeypatch):
    # If the SDK writes extracted files directly (no zip), the command must
    # still succeed without trying to unzip anything.
    comp = _comp(tmp_path)
    (comp / "data").mkdir()

    api = MagicMock()
    api.authenticate.return_value = None
    def _download(*, competition, path, **kwargs):
        (Path(path) / "train.csv").write_text("a,b\n1,2\n")
    api.competition_download_files.side_effect = _download
    monkeypatch.setattr("kaggle_lab.cli._make_api", lambda: api)

    result = CliRunner().invoke(main, ["pull-data", str(comp)])
    assert result.exit_code == 0, result.output
    assert (comp / "data" / "train.csv").exists()


def test_init_download_flag_pulls_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    api = _fake_download_api({"train.csv": "a\n1\n"})
    monkeypatch.setattr("kaggle_lab.cli._make_api", lambda: api)

    result = CliRunner().invoke(main, ["init", "titanic", "--download"])
    assert result.exit_code == 0, result.output
    root = tmp_path / "competitions" / "titanic"
    assert (root / "data" / "train.csv").read_text().startswith("a")
    assert not (root / "data" / "titanic.zip").exists()
    api.competition_download_files.assert_called_once()


def test_init_without_download_does_not_call_api(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    api = MagicMock()
    monkeypatch.setattr("kaggle_lab.cli._make_api", lambda: api)

    result = CliRunner().invoke(main, ["init", "titanic"])
    assert result.exit_code == 0
    api.competition_download_files.assert_not_called()


def test_quota_reports_remaining(tmp_path, monkeypatch):
    comp = _comp(tmp_path)
    today = datetime.now(timezone.utc).replace(hour=10, minute=0).isoformat()
    subs = [
        SimpleNamespace(date=today, status="complete"),
        SimpleNamespace(date=today, status="complete"),
        SimpleNamespace(date="2020-01-01T00:00:00Z", status="complete"),  # old
    ]
    fake_api = SimpleNamespace(
        authenticate=lambda: None,
        competition_submissions=lambda competition: subs,
    )
    monkeypatch.setattr("kaggle_lab.cli._make_api", lambda: fake_api)

    result = CliRunner().invoke(main, ["quota", str(comp)])
    assert result.exit_code == 0
    # daily_limit=10 in fixture, 2 used today -> 8 remaining
    assert "used today: 2" in result.output.lower() or "2/10" in result.output
    assert "8" in result.output
