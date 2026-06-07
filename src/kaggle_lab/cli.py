"""Click entry-point for the kaggle-lab command."""
from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import click

from .config import Config
from .lab import Lab
from .runner import new_run_id
from .tracker import get_run, latest_runs, read_runs, resolve_parent


STUB_CONFIG = """\
slug: {slug}
metric: accuracy
direction: maximize
submission_type: csv
submission_file: submission.csv
sample_submission: null
kernel_slug: null
daily_limit: 10
"""


def _make_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _resolve_sample_submission(
    competition_dir: Path, cfg: Config
) -> Path | None:
    """Locate the sample-submission CSV for the notebook template.

    Precedence: explicit config > glob auto-detect (exactly one match wins,
    excluding the framework's own output filename).
    """
    data_dir = competition_dir / "data"
    if cfg.sample_submission:
        candidate = data_dir / cfg.sample_submission
        return candidate if candidate.exists() else None
    if not data_dir.is_dir():
        return None
    output_name = cfg.submission_file
    matches = [
        p for p in sorted(data_dir.glob("*submission*.csv"))
        if p.name != output_name
    ]
    return matches[0] if len(matches) == 1 else None


def _sample_columns(path: Path) -> list[str]:
    with path.open() as f:
        reader = csv.reader(f)
        return next(reader, [])


def _submission_cell_source(sample: Path | None) -> list[str]:
    if sample is None:
        return [
            "# TODO: train model, compute predictions.\n",
            "\n",
            "# TODO: build `submission` matching the competition's schema\n",
            "# (see data/ for a sample-submission CSV), then:\n",
            "# submission.to_csv(output_dir / \"submission.csv\", index=False)\n",
        ]
    cols = _sample_columns(sample)
    target = cols[-1] if len(cols) >= 2 else "<target_col>"
    return [
        "# TODO: train model, compute predictions.\n",
        "\n",
        f"sample = pd.read_csv(data_dir / {sample.name!r})\n",
        "submission = sample.copy()\n",
        f"# TODO: submission[{target!r}] = predictions\n",
        "submission.to_csv(output_dir / \"submission.csv\", index=False)\n",
    ]


def _changelog_cell(parent: str) -> dict:
    return {
        "cell_type": "markdown", "id": "ch", "metadata": {},
        "source": [
            "## Changelog\n",
            f"- parent: {parent}\n",
            "- change: TODO describe the change\n",
            "- hypothesis: TODO why this should help\n",
        ],
    }


def _boilerplate_cell(slug: str) -> dict:
    return {
        "cell_type": "code", "id": "bp", "metadata": {},
        "execution_count": None, "outputs": [],
        "source": [
            "import sys\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "from pathlib import Path\n",
            "\n",
            "IS_KAGGLE = Path(\"/kaggle/input\").exists()\n",
            "\n",
            "if IS_KAGGLE:\n",
            f"    data_dir   = Path(\"/kaggle/input/{slug}\")\n",
            "    output_dir = Path(\"/kaggle/working\")\n",
            "else:\n",
            "    def _find_competition_dir(start: Path) -> Path:\n",
            "        for p in [start, *start.parents]:\n",
            "            if (p / \"config.yaml\").exists() and (p / \"data\").is_dir():\n",
            "                return p\n",
            "        raise RuntimeError(\"competition dir not found (no ancestor has config.yaml + data/)\")\n",
            "\n",
            "    comp_dir   = _find_competition_dir(Path.cwd())\n",
            "    data_dir   = comp_dir / \"data\"\n",
            "    output_dir = Path.cwd()\n",
            "\n",
            "for _p in [Path.cwd(), *Path.cwd().parents]:\n",
            "    if (_p / \"src\" / \"eda\").is_dir():\n",
            "        if str(_p) not in sys.path:\n",
            "            sys.path.insert(0, str(_p))\n",
            "        break\n",
            "\n",
            "from src.eda import (\n",
            "    numeric_columns,\n",
            "    plot_histograms,\n",
            "    plot_kde,\n",
            "    plot_boxplots,\n",
            "    plot_distributions,\n",
            "    describe_df,\n",
            "    get_missing_values,\n",
            "    count_duplicates,\n",
            "    plot_categorical_vs_target,\n",
            "    plot_numerical_vs_target,\n",
            "    target_rate_table,\n",
            "    grouped_median,\n",
            ")\n",
            "\n",
            "train_data = pd.read_csv(data_dir / \"train.csv\")\n",
            "test_data  = pd.read_csv(data_dir / \"test.csv\")\n",
        ],
    }


def _starter_notebook(parent: str, slug: str, sample: Path | None = None) -> dict:
    return {
        "cells": [
            _changelog_cell(parent),
            _boilerplate_cell(slug),
            {"cell_type": "code", "id": "im", "metadata": {},
             "execution_count": None, "outputs": [],
             "source": _submission_cell_source(sample)},
        ],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
    }


def _is_changelog_cell(cell: dict) -> bool:
    if cell.get("cell_type") != "markdown":
        return False
    if cell.get("id") == "ch":
        return True
    return "".join(cell.get("source", [])).lstrip().startswith("## Changelog")


def _clean_cell(cell: dict) -> dict:
    out = dict(cell)
    if out.get("cell_type") == "code":
        out["outputs"] = []
        out["execution_count"] = None
    return out


def _derived_notebook(parent_run_id: str, parent_nb: dict) -> dict:
    cells = [_changelog_cell(parent_run_id)]
    for c in parent_nb.get("cells", []):
        if _is_changelog_cell(c):
            continue
        cells.append(_clean_cell(c))
    return {
        "cells": cells,
        "metadata": parent_nb.get("metadata", {}),
        "nbformat": parent_nb.get("nbformat", 4),
        "nbformat_minor": parent_nb.get("nbformat_minor", 5),
    }


@click.group()
def main() -> None:
    """kaggle-lab: reproducible experiment tracking for Kaggle competitions.

    Scaffold competitions and notebooks, run and submit experiments, and inspect
    the append-only run log (list / show / tree).
    """


@main.command("list")
@click.argument("competition_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--top", type=int, default=None, help="Show only the top N by public_score.")
def list_cmd(competition_dir: Path, top: int | None) -> None:
    """List scored runs, best public score first."""
    lab = Lab.from_config(competition_dir / "config.yaml")
    runs = latest_runs(read_runs(lab.runs_jsonl))
    scored = [r for r in runs if r.kaggle and r.kaggle.public_score is not None]
    if not scored:
        click.echo("no runs yet")
        return
    scored.sort(
        key=lambda r: r.kaggle.public_score,
        reverse=(lab.config.direction == "maximize"),
    )
    if top:
        scored = scored[:top]
    for r in scored:
        click.echo(f"{r.run_id}  {r.kaggle.public_score:.5f}  {r.changelog['change']}")


@main.command("show")
@click.argument("competition_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("run_id")
def show_cmd(competition_dir: Path, run_id: str) -> None:
    """Print the full JSON record of a single run."""
    lab = Lab.from_config(competition_dir / "config.yaml")
    record = get_run(lab.runs_jsonl, run_id)
    if record is None:
        click.echo(f"run not found: {run_id}", err=True)
        raise click.exceptions.Exit(code=1)
    click.echo(json.dumps(asdict(record), indent=2))


@main.command("tree")
@click.argument("competition_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def tree_cmd(competition_dir: Path) -> None:
    """Render the parent->child run tree with score deltas."""
    lab = Lab.from_config(competition_dir / "config.yaml")
    runs = latest_runs(read_runs(lab.runs_jsonl))
    if not runs:
        click.echo("no runs yet")
        return

    by_parent: dict[str | None, list] = {}
    by_id = {r.run_id: r for r in runs}
    for r in runs:
        by_parent.setdefault(r.parent_run_id, []).append(r)

    lines: list[str] = ["root"]

    def _walk(parent_id: str | None, prefix: str) -> None:
        children = sorted(by_parent.get(parent_id, []), key=lambda r: r.created_at)
        for i, r in enumerate(children):
            is_last = i == len(children) - 1
            connector = "└── " if is_last else "├── "
            score = (
                r.kaggle.public_score
                if r.kaggle and r.kaggle.public_score is not None
                else None
            )
            parent = by_id.get(r.parent_run_id) if r.parent_run_id else None
            parent_score = (
                parent.kaggle.public_score
                if parent and parent.kaggle and parent.kaggle.public_score is not None
                else None
            )
            delta = (
                score - parent_score
                if score is not None and parent_score is not None
                else None
            )
            score_str = f"{score:.3f}" if score is not None else "   -"
            delta_str = f"({delta:+.3f})" if delta is not None else ""
            change = r.changelog.get("change", "")
            lines.append(
                f"{prefix}{connector}{r.run_id}  {change:40}  {score_str}  {delta_str}"
            )
            next_prefix = prefix + ("    " if is_last else "│   ")
            _walk(r.run_id, next_prefix)

    _walk(None, "")
    click.echo("\n".join(lines))


@main.command("new")
@click.argument("competition_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--from", "from_run", default="root", help="Parent run id.")
def new_cmd(competition_dir: Path, from_run: str) -> None:
    """Scaffold a new experiment notebook, optionally derived from a parent run."""
    lab = Lab.from_config(competition_dir / "config.yaml")
    lab.experiments_dir.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id()
    nb_path = lab.experiments_dir / f"{run_id}.ipynb"

    if from_run == "root":
        sample = _resolve_sample_submission(lab.competition_dir, lab.config)
        nb_path.write_text(json.dumps(_starter_notebook(from_run, lab.config.slug, sample), indent=1))
    else:
        try:
            parent_record = resolve_parent(read_runs(lab.runs_jsonl), from_run)
        except ValueError as e:
            raise click.ClickException(str(e))
        parent_nb_path = lab.competition_dir / parent_record.notebook
        if not parent_nb_path.is_file():
            raise click.ClickException(
                f"parent notebook not found: {parent_nb_path}"
            )
        parent_nb = json.loads(parent_nb_path.read_text())
        new_nb = _derived_notebook(parent_record.run_id, parent_nb)
        nb_path.write_text(json.dumps(new_nb, indent=1))
    click.echo(str(nb_path))


def _download_competition_data(api, slug: str, data_dir: Path) -> None:
    """Fetch competition data into data_dir, unzip bundle, drop the archive."""
    data_dir.mkdir(parents=True, exist_ok=True)
    api.competition_download_files(
        competition=slug, path=str(data_dir), quiet=False
    )
    bundle = data_dir / f"{slug}.zip"
    if bundle.exists():
        with zipfile.ZipFile(bundle) as z:
            z.extractall(data_dir)
        bundle.unlink()


@main.command("init")
@click.argument("slug")
@click.option("--download", is_flag=True, help="Fetch the competition data into data/ after scaffolding.")
def init_cmd(slug: str, download: bool) -> None:
    """Scaffold a new competition directory (config + folder layout)."""
    root = Path("competitions") / slug
    cfg = root / "config.yaml"
    if cfg.exists():
        click.echo(f"refusing to overwrite existing config: {cfg}", err=True)
        raise click.exceptions.Exit(code=1)
    for sub in ("data", "experiments", "artifacts", "logs", "utils"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    cfg.write_text(STUB_CONFIG.format(slug=slug))
    click.echo(str(root))
    if download:
        _download_competition_data(_make_api(), slug, root / "data")
        click.echo(f"data downloaded to {root / 'data'}")


@main.command("pull-data")
@click.argument("competition_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def pull_data_cmd(competition_dir: Path) -> None:
    """Download the competition data into its data/ directory."""
    lab = Lab.from_config(competition_dir / "config.yaml")
    _download_competition_data(_make_api(), lab.config.slug, lab.data_dir)
    click.echo(f"data downloaded to {lab.data_dir}")


@main.command("run")
@click.argument("notebook", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--dry-run", is_flag=True,
              help="Resolve the parent and report the run without executing the notebook.")
@click.option("--no-submit", is_flag=True,
              help="Execute and record the run but skip the Kaggle submission.")
@click.option("--allow-dirty", is_flag=True,
              help="Run even when the git tree has uncommitted changes.")
@click.option("--force-duplicate", is_flag=True,
              help="Submit even if an identical submission (same SHA1) already exists.")
def run_cmd(
    notebook: Path,
    dry_run: bool,
    no_submit: bool,
    allow_dirty: bool,
    force_duplicate: bool,
) -> None:
    """Execute a notebook, submit it to Kaggle, and record the run."""
    competition_dir = notebook.resolve().parent.parent
    lab = Lab.from_config(competition_dir / "config.yaml")
    record = lab.run(
        notebook=notebook,
        submit=not no_submit,
        dry_run=dry_run,
        allow_dirty=allow_dirty,
        force_duplicate=force_duplicate,
    )
    if record is None:
        return
    if record.submitted and record.kaggle:
        score = record.kaggle.public_score
        click.echo(f"{record.run_id}  score={score}  delta={record.delta_vs_parent}")
    else:
        click.echo(f"{record.run_id}  (no submit)")


@main.command("refresh")
@click.argument("competition_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def refresh_cmd(competition_dir: Path) -> None:
    """Poll Kaggle for pending runs and update their scores."""
    lab = Lab.from_config(competition_dir / "config.yaml")
    api = _make_api()
    updated = lab.refresh(api=api)
    if not updated:
        click.echo("no pending runs updated")
        return
    for r in updated:
        score = r.kaggle.public_score if r.kaggle else None
        status = r.kaggle.status if r.kaggle else "?"
        click.echo(f"{r.run_id} -> {status} (score={score})")


@main.command("quota")
@click.argument("competition_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def quota_cmd(competition_dir: Path) -> None:
    """Show today's submission count against the daily limit."""
    lab = Lab.from_config(competition_dir / "config.yaml")
    api = _make_api()
    subs = api.competition_submissions(competition=lab.config.slug) or []
    today = datetime.now(timezone.utc).date()

    def _on_today(s) -> bool:
        try:
            return datetime.fromisoformat(str(s.date).replace("Z", "+00:00")).date() == today
        except Exception:
            return False

    used = sum(1 for s in subs if _on_today(s))
    remaining = max(lab.config.daily_limit - used, 0)
    click.echo(f"used today: {used}/{lab.config.daily_limit}; remaining: {remaining}")


if __name__ == "__main__":
    main()
