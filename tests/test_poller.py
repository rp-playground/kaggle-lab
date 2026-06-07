from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kaggle_lab.poller import wait_for_score, PollerTimeout, PollResult


def test_returns_complete_submission():
    pending = SimpleNamespace(ref="42", status="pending", publicScore=None, privateScore=None)
    done    = SimpleNamespace(ref="42", status="complete", publicScore=0.763, privateScore=None)
    api = MagicMock()
    api.competition_submissions.side_effect = [[pending], [pending], [done]]

    result = wait_for_score(api=api, competition="titanic", submission_ref="42",
                             interval=0, timeout=5)
    assert isinstance(result, PollResult)
    assert result.status == "complete"
    assert result.public_score == 0.763
    assert result.error is None


def test_timeout_raises():
    pending = SimpleNamespace(ref="42", status="pending", publicScore=None, privateScore=None)
    api = MagicMock()
    api.competition_submissions.return_value = [pending]
    with pytest.raises(PollerTimeout):
        wait_for_score(api=api, competition="titanic", submission_ref="42",
                       interval=0, timeout=0.01)


def test_error_status_returns_result_not_raises():
    err = SimpleNamespace(ref="42", status="error", publicScore=None, privateScore=None,
                          errorDescription="bad format")
    api = MagicMock()
    api.competition_submissions.return_value = [err]
    result = wait_for_score(api=api, competition="titanic", submission_ref="42",
                             interval=0, timeout=5)
    assert result.status == "error"
    assert result.public_score is None
    assert result.error == "bad format"


def test_matches_ref_as_string_even_if_api_returns_int():
    done = SimpleNamespace(ref=42, status="complete", publicScore=0.9, privateScore=None)
    api = MagicMock()
    api.competition_submissions.return_value = [done]
    result = wait_for_score(api=api, competition="titanic", submission_ref="42",
                             interval=0, timeout=5)
    assert result.status == "complete"
    assert result.public_score == 0.9
