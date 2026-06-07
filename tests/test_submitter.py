from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kaggle_lab.submitter import submit_csv, SubmitterError


def _mock_api(submissions_after_submit: list[SimpleNamespace]) -> MagicMock:
    api = MagicMock()
    api.authenticate.return_value = None
    api.competition_submit.return_value = None
    api.competition_submissions.return_value = submissions_after_submit
    return api


def test_submit_csv_uses_run_id_prefix(tmp_path):
    artifact = tmp_path / "s.csv"
    artifact.write_text("PassengerId,Survived\n1,0\n")
    sub = SimpleNamespace(ref="42", description="run_id=20260422_131500_abcd | change")
    api = _mock_api([sub])

    ref = submit_csv(
        api=api,
        competition="titanic",
        artifact=artifact,
        run_id="20260422_131500_abcd",
        change_message="change",
    )

    assert ref == "42"
    api.competition_submit.assert_called_once()
    kwargs = api.competition_submit.call_args.kwargs
    assert kwargs["competition"] == "titanic"
    assert kwargs["file_name"] == str(artifact)
    assert kwargs["message"].startswith("run_id=20260422_131500_abcd | ")
    assert kwargs["message"].endswith("change")


def test_submit_csv_raises_when_ref_not_found(tmp_path):
    artifact = tmp_path / "s.csv"; artifact.write_text("x")
    api = _mock_api([SimpleNamespace(ref="9", description="unrelated message")])
    with pytest.raises(SubmitterError, match="ref"):
        submit_csv(
            api=api, competition="titanic", artifact=artifact,
            run_id="20260422_131500_abcd", change_message="c",
            list_retries=1, list_retry_delay=0,
        )


def test_submit_csv_truncates_long_change_message(tmp_path):
    artifact = tmp_path / "s.csv"; artifact.write_text("x")
    huge = "x" * 600
    sub = SimpleNamespace(ref="1", description="run_id=r1 " + huge[:400])
    api = _mock_api([sub])
    submit_csv(
        api=api, competition="titanic", artifact=artifact,
        run_id="r1", change_message=huge,
    )
    msg = api.competition_submit.call_args.kwargs["message"]
    # run_id prefix ("run_id=r1 | ") + at most 400 chars of the change message
    assert len(msg) <= len("run_id=r1 | ") + 400


def test_submit_csv_retries_listing_then_succeeds(tmp_path, mocker):
    artifact = tmp_path / "s.csv"; artifact.write_text("x")
    api = MagicMock()
    api.authenticate.return_value = None
    api.competition_submit.return_value = None
    # First listing returns empty, second has the submission.
    sub = SimpleNamespace(ref="7", description="run_id=rid | change")
    api.competition_submissions.side_effect = [[], [sub]]

    ref = submit_csv(
        api=api, competition="titanic", artifact=artifact,
        run_id="rid", change_message="change",
        list_retries=3, list_retry_delay=0,
    )
    assert ref == "7"
    assert api.competition_submissions.call_count == 2
