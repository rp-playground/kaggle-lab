from __future__ import annotations

import re
from pathlib import Path

from kaggle_lab.runner import execute_notebook, new_run_id, sha1_file


def test_run_id_format():
    rid = new_run_id()
    assert re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{8}", rid), rid


def test_run_id_unique_within_second():
    ids = {new_run_id() for _ in range(50)}
    assert len(ids) == 50


def test_sha1_file(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("hello")
    # sha1("hello") = aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d
    assert sha1_file(p) == "aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d"


def test_execute_notebook_produces_submission(tmp_path):
    nb_src = Path("tests/fixtures/notebooks/tiny_pipeline.ipynb")
    workdir = tmp_path / "work"
    workdir.mkdir()
    log = tmp_path / "run.log"

    executed_path = execute_notebook(nb_src, workdir=workdir, log_path=log)

    assert executed_path.exists()
    assert (workdir / "submission.csv").exists()
    assert log.exists() and log.stat().st_size >= 0  # log created, may be empty
