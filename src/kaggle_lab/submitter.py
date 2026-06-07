"""Submission transport for Kaggle (CSV path)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any


class SubmitterError(RuntimeError):
    pass


_MAX_CHANGE_CHARS = 400


def _build_message(run_id: str, change: str) -> str:
    tail = change[:_MAX_CHANGE_CHARS]
    return f"run_id={run_id} | {tail}"


def submit_csv(
    *,
    api: Any,
    competition: str,
    artifact: Path,
    run_id: str,
    change_message: str,
    list_retries: int = 3,
    list_retry_delay: float = 2.0,
) -> str:
    msg = _build_message(run_id, change_message)
    api.competition_submit(
        file_name=str(artifact),
        message=msg,
        competition=competition,
    )

    marker = f"run_id={run_id} "
    for _ in range(list_retries):
        subs = api.competition_submissions(competition=competition) or []
        for s in subs:
            if getattr(s, "description", "").startswith(marker):
                return str(s.ref)
        time.sleep(list_retry_delay)

    raise SubmitterError(f"could not resolve submission ref for run_id={run_id}")
