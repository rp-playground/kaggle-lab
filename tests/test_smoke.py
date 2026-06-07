"""End-to-end smoke test: init + new + run --no-submit."""
from __future__ import annotations

import json
import subprocess

from click.testing import CliRunner

from kaggle_lab.cli import main
from kaggle_lab.tracker import read_runs


def test_smoke_init_and_no_submit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # We need git to exist so Lab.run's git probe succeeds (even --allow-dirty still queries).
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)

    runner = CliRunner()

    # 1. init
    result = runner.invoke(main, ["init", "smoke"])
    assert result.exit_code == 0, result.output
    comp = tmp_path / "competitions" / "smoke"
    assert (comp / "config.yaml").exists()

    # 2. new -> creates a starter notebook
    result = runner.invoke(main, ["new", str(comp)])
    assert result.exit_code == 0, result.output
    nbs = list((comp / "experiments").glob("*.ipynb"))
    assert len(nbs) == 1
    nb_path = nbs[0]

    # 3. Replace the TODO fields so the changelog parses cleanly
    nb = json.loads(nb_path.read_text())
    nb["cells"][0]["source"] = [
        "## Changelog\n",
        "- parent: root\n",
        "- change: smoke\n",
        "- hypothesis: end-to-end cli works\n",
    ]
    # Make the notebook trivially pass (no submission.csv needed since --no-submit)
    nb["cells"][1]["source"] = ["print('smoke ok')\n"]
    nb_path.write_text(json.dumps(nb))

    # 4. run --no-submit
    result = runner.invoke(main, ["run", str(nb_path), "--no-submit", "--allow-dirty"])
    assert result.exit_code == 0, result.output

    # 5. runs.jsonl has one row with submitted=False, no artifact
    rows = read_runs(comp / "runs.jsonl")
    assert len(rows) == 1
    assert rows[0].submitted is False
    assert rows[0].artifact is None
    assert rows[0].changelog["change"] == "smoke"

    # 6. list works
    result = runner.invoke(main, ["list", str(comp)])
    assert result.exit_code == 0
    # no submitted runs -> "no runs yet"
    assert "no runs yet" in result.output.lower()

    # 7. tree shows the one run
    result = runner.invoke(main, ["tree", str(comp)])
    assert result.exit_code == 0
    assert rows[0].run_id in result.output
