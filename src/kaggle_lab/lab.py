"""Top-level Lab facade: wires config + path conventions."""
from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .changelog import parse_changelog
from .config import Config, load_config
from .poller import PollerTimeout, wait_for_score
from .runner import execute_notebook, new_run_id, sha1_file
from .submitter import SubmitterError, submit_csv
from .tracker import (
    KaggleRecord,
    RunRecord,
    append_run,
    compute_delta,
    latest_runs,
    read_runs,
    resolve_parent,
)


def _default_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_state(cwd: Path | None = None) -> tuple[str, bool]:
    try:
        sha_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
            cwd=str(cwd) if cwd is not None else None,
        )
        dirty_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=False,
            cwd=str(cwd) if cwd is not None else None,
        )
    except FileNotFoundError:
        return "", False
    if sha_proc.returncode != 0:
        return "", False
    return sha_proc.stdout.strip(), bool(dirty_proc.stdout.strip())


@dataclass(frozen=True)
class Lab:
    config: Config
    competition_dir: Path

    @classmethod
    def from_config(cls, path: Path | str) -> "Lab":
        cfg = load_config(path)
        return cls(config=cfg, competition_dir=cfg.path.parent)

    @property
    def runs_jsonl(self) -> Path:
        return self.competition_dir / "runs.jsonl"

    @property
    def experiments_dir(self) -> Path:
        return self.competition_dir / "experiments"

    @property
    def artifacts_dir(self) -> Path:
        return self.competition_dir / "artifacts"

    @property
    def logs_dir(self) -> Path:
        return self.competition_dir / "logs"

    @property
    def data_dir(self) -> Path:
        return self.competition_dir / "data"

    def run(
        self,
        notebook: Path,
        *,
        api=None,
        submit: bool = True,
        dry_run: bool = False,
        allow_dirty: bool = False,
        force_duplicate: bool = False,
        poll_interval: float = 10.0,
        poll_timeout: float = 300.0,
        list_retries: int = 3,
        list_retry_delay: float = 2.0,
    ) -> RunRecord | None:
        notebook = Path(notebook)
        run_id = new_run_id()
        changelog = parse_changelog(notebook)

        # Resolve parent.
        parent_run_id: str | None = None
        parent_public_score: float | None = None
        if changelog.parent != "root":
            parent_record = resolve_parent(read_runs(self.runs_jsonl), changelog.parent)
            parent_run_id = parent_record.run_id
            if parent_record.kaggle is not None:
                parent_public_score = parent_record.kaggle.public_score

        git_sha, git_dirty = _git_state(cwd=self.competition_dir)
        if git_dirty and not allow_dirty:
            raise RuntimeError(
                "refusing to run with dirty git tree; use --allow-dirty to override"
            )

        if dry_run:
            print(f"{run_id}: would execute {notebook} (parent={parent_run_id})")
            return None

        workdir = self.logs_dir / run_id
        workdir.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_dir / f"{run_id}.log"
        execute_notebook(notebook, workdir=workdir, log_path=log_path)

        artifact_rel: str | None = None
        submission_sha1: str | None = None
        kaggle_rec: KaggleRecord | None = None
        poll_public_score: float | None = None

        if submit:
            src = workdir / self.config.submission_file
            if not src.exists():
                raise RuntimeError(
                    f"notebook did not produce {self.config.submission_file}"
                )
            submission_sha1 = sha1_file(src)

            existing = [
                r for r in read_runs(self.runs_jsonl)
                if r.submission_sha1 == submission_sha1
            ]
            if existing and not force_duplicate:
                raise RuntimeError(
                    f"duplicate submission: same sha1 as {existing[0].run_id}; "
                    "use --force-duplicate to override"
                )

            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            artifact_abs = self.artifacts_dir / f"{run_id}.csv"
            shutil.copy2(src, artifact_abs)
            artifact_rel = f"artifacts/{run_id}.csv"

            if self.config.submission_type == "kernel":
                raise NotImplementedError(
                    "kernel submission is deferred to milestone 8"
                )

            api = api or _default_api()
            submitted_at = _iso_now()
            try:
                submission_ref = submit_csv(
                    api=api,
                    competition=self.config.slug,
                    artifact=artifact_abs,
                    run_id=run_id,
                    change_message=changelog.change,
                    list_retries=list_retries,
                    list_retry_delay=list_retry_delay,
                )
            except SubmitterError:
                print(
                    f"WARNING: submission may have reached Kaggle but ref could not be resolved. "
                    f"Check: kaggle competitions submissions {self.config.slug}",
                    file=sys.stderr,
                )
                raise
            try:
                poll_result = wait_for_score(
                    api=api,
                    competition=self.config.slug,
                    submission_ref=submission_ref,
                    interval=poll_interval,
                    timeout=poll_timeout,
                )
                scored_at = _iso_now()
                status = poll_result.status
                public_score = poll_result.public_score
                private_score = poll_result.private_score
            except PollerTimeout:
                scored_at = None
                status = "pending"
                public_score = None
                private_score = None
            poll_public_score = public_score
            tail = changelog.change[:400]
            message = f"run_id={run_id} | {tail}"
            kaggle_rec = KaggleRecord(
                submission_type="csv",
                submission_ref=submission_ref,
                submitted_at=submitted_at,
                scored_at=scored_at,
                status=status,
                public_score=public_score,
                private_score=private_score,
                message=message,
            )

        if notebook.is_absolute() and notebook.is_relative_to(self.competition_dir):
            notebook_str = str(notebook.relative_to(self.competition_dir))
        else:
            try:
                notebook_str = str(notebook.resolve().relative_to(self.competition_dir.resolve()))
            except ValueError:
                notebook_str = str(notebook)

        record = RunRecord(
            run_id=run_id,
            competition=self.config.slug,
            created_at=_iso_now(),
            notebook=notebook_str,
            artifact=artifact_rel,
            submission_sha1=submission_sha1,
            parent_run_id=parent_run_id,
            changelog={"change": changelog.change, "hypothesis": changelog.hypothesis},
            git_sha=git_sha,
            git_dirty=git_dirty,
            submitted=submit,
            kaggle=kaggle_rec,
            metrics={},
            parent_public_score=parent_public_score,
            delta_vs_parent=compute_delta(poll_public_score, parent_public_score),
            supersedes=None,
        )
        append_run(self.runs_jsonl, record)
        return record

    def refresh(self, *, api=None) -> list[RunRecord]:
        api = api or _default_api()
        results: list[RunRecord] = []
        runs = latest_runs(read_runs(self.runs_jsonl))
        for r in runs:
            if not r.submitted or not r.kaggle:
                continue
            if r.kaggle.status not in ("pending", "error") and r.kaggle.public_score is not None:
                continue
            if not r.kaggle.submission_ref:
                continue
            try:
                poll = wait_for_score(
                    api=api,
                    competition=self.config.slug,
                    submission_ref=r.kaggle.submission_ref,
                    interval=0,
                    timeout=0,
                )
            except PollerTimeout:
                continue

            if (poll.status == r.kaggle.status
                    and poll.public_score == r.kaggle.public_score):
                continue

            new_kaggle = KaggleRecord(
                submission_type=r.kaggle.submission_type,
                submission_ref=r.kaggle.submission_ref,
                submitted_at=r.kaggle.submitted_at,
                scored_at=_iso_now(),
                status=poll.status,
                public_score=poll.public_score,
                private_score=poll.private_score,
                message=r.kaggle.message,
            )
            parent_score = r.parent_public_score
            new_delta = compute_delta(poll.public_score, parent_score)
            new_rec = RunRecord(
                **{
                    **r.__dict__,
                    "kaggle": new_kaggle,
                    "delta_vs_parent": new_delta,
                    "supersedes": r.run_id,
                }
            )
            append_run(self.runs_jsonl, new_rec)
            results.append(new_rec)
        return results
