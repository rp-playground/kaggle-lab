"""Append-only runs.jsonl read/write."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KaggleRecord:
    submission_type: str
    submission_ref: str | None
    submitted_at: str
    scored_at: str | None
    status: str
    public_score: float | None
    private_score: float | None
    message: str


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    competition: str
    created_at: str
    notebook: str
    artifact: str | None
    submission_sha1: str | None
    parent_run_id: str | None
    changelog: dict[str, str]
    git_sha: str
    git_dirty: bool
    submitted: bool
    kaggle: KaggleRecord | None
    metrics: dict[str, float] = field(default_factory=dict)
    parent_public_score: float | None = None
    delta_vs_parent: float | None = None
    supersedes: str | None = None


def append_run(jsonl: Path, record: RunRecord) -> None:
    jsonl = Path(jsonl)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("a") as f:
        f.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")


def read_runs(jsonl: Path) -> list[RunRecord]:
    jsonl = Path(jsonl)
    if not jsonl.exists():
        return []
    records: list[RunRecord] = []
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        records.append(_from_dict(json.loads(line)))
    return records


def latest_runs(runs: list[RunRecord]) -> list[RunRecord]:
    """Collapse an append-only list to the latest row per run_id.

    Why: `runs.jsonl` may contain multiple rows for the same run_id when a
    correction is appended (e.g. poll-timeout pending -> complete). Callers that
    render the current state (list, tree, show) must see only the latest row,
    and refresh() must not re-process stale rows.
    """
    by_id: dict[str, RunRecord] = {}
    for r in runs:
        by_id[r.run_id] = r
    return list(by_id.values())


def get_run(jsonl: Path, run_id: str) -> RunRecord | None:
    latest: RunRecord | None = None
    for r in read_runs(jsonl):
        if r.run_id == run_id:
            latest = r
    return latest


def _from_dict(d: dict[str, Any]) -> RunRecord:
    k = d.get("kaggle")
    return RunRecord(**{**d, "kaggle": KaggleRecord(**k) if k else None})


import re

_ALIAS_RE = re.compile(r"^v(\d+)$")


def resolve_parent(runs: list[RunRecord], parent: str) -> RunRecord:
    if parent == "root":
        raise ValueError("'root' is not a run; use None as parent_run_id instead")

    for r in runs:
        if r.run_id == parent:
            return r

    alias = _ALIAS_RE.match(parent)
    if alias:
        needle = f"_v{alias.group(1)}_"
        matches = [r for r in runs if needle in Path(r.notebook).name]
        if matches:
            matches.sort(key=lambda r: r.created_at, reverse=True)
            return matches[0]

    raise ValueError(f"no run matches parent={parent!r}")


def compute_delta(score: float | None, parent_score: float | None) -> float | None:
    if score is None or parent_score is None:
        return None
    return score - parent_score
