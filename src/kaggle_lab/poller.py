"""Poll Kaggle submissions listing until scoring is done."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


class PollerTimeout(TimeoutError):
    pass


@dataclass(frozen=True)
class PollResult:
    status: str
    public_score: float | None
    private_score: float | None
    error: str | None = None


_TERMINAL = {"complete", "error"}


def _to_float(x: Any) -> float | None:
    if x is None or x == "":
        return None
    return float(x)


def wait_for_score(
    *,
    api: Any,
    competition: str,
    submission_ref: str,
    interval: float = 10.0,
    timeout: float = 300.0,
) -> PollResult:
    deadline = time.monotonic() + timeout
    while True:
        for s in api.competition_submissions(competition=competition) or []:
            if str(getattr(s, "ref", "")) != str(submission_ref):
                continue
            raw_status = getattr(s, "status", "pending")
            status = getattr(raw_status, "name", str(raw_status)).lower()
            if status in _TERMINAL:
                return PollResult(
                    status=status,
                    public_score=_to_float(getattr(s, "publicScore", None)),
                    private_score=_to_float(getattr(s, "privateScore", None)),
                    error=getattr(s, "errorDescription", None) if status == "error" else None,
                )
            break  # matched but not terminal, sleep and retry

        if time.monotonic() >= deadline:
            raise PollerTimeout(f"scoring did not complete within {timeout}s (ref={submission_ref})")
        time.sleep(interval)
