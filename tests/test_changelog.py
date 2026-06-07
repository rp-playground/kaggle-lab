from __future__ import annotations

from pathlib import Path

import pytest

from kaggle_lab.changelog import Changelog, ChangelogError, parse_changelog

FX = Path(__file__).parent / "fixtures/notebooks"


def test_parse_valid():
    c = parse_changelog(FX / "valid_changelog.ipynb")
    assert c.parent == "20260422_131500_f09d"
    assert c.change == "RF baseline"
    assert c.hypothesis == "bagging beats LR"


def test_missing_cell_raises():
    with pytest.raises(ChangelogError, match="no .*Changelog"):
        parse_changelog(FX / "missing_changelog.ipynb")


def test_missing_key_raises():
    with pytest.raises(ChangelogError, match="hypothesis"):
        parse_changelog(FX / "malformed_changelog.ipynb")


def test_continuation_lines_folded():
    c = parse_changelog(FX / "multiline_changelog.ipynb")
    assert c.parent == "20260422_131500_f09d"
    assert c.change == "switched LR to RF with n_estimators=300 and tuned max_depth"
    assert c.hypothesis == "bagging beats LR"


def test_unknown_keys_ignored():
    c = parse_changelog(FX / "extra_keys_changelog.ipynb")
    assert c.parent == "20260422_131500_f09d"
    assert c.change == "RF baseline"
    assert c.hypothesis == "bagging beats LR"


def test_stray_unindented_line_rejected(tmp_path):
    nb = tmp_path / "stray.ipynb"
    nb.write_text(
        '{"cells": [{"cell_type": "markdown", "id": "x", "metadata": {}, '
        '"source": ["## Changelog\\n", "- parent: x\\n", "- change: y\\n", '
        '"stray paragraph\\n", "- hypothesis: z\\n"]}], '
        '"metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
    )
    with pytest.raises(ChangelogError, match="unexpected line"):
        parse_changelog(nb)
