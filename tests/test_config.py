from __future__ import annotations

from pathlib import Path

import pytest

from kaggle_lab.config import Config, load_config, ConfigError

FIXTURE = Path(__file__).parent / "fixtures/titanic/config.yaml"


def test_load_valid_config():
    cfg = load_config(FIXTURE)
    assert cfg.slug == "titanic"
    assert cfg.submission_type == "csv"
    assert cfg.submission_file == "submission.csv"
    assert cfg.direction == "maximize"
    assert cfg.daily_limit == 10
    assert cfg.kernel_slug is None
    assert cfg.sample_submission is None


def test_sample_submission_is_loaded_when_present(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "slug: titanic\nmetric: accuracy\ndirection: maximize\n"
        "submission_type: csv\nsubmission_file: submission.csv\n"
        "sample_submission: gender_submission.csv\n"
        "kernel_slug: null\ndaily_limit: 10\n"
    )
    cfg = load_config(p)
    assert cfg.sample_submission == "gender_submission.csv"


def test_missing_slug_fails(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("metric: accuracy\nsubmission_type: csv\n")
    with pytest.raises(ConfigError, match="slug"):
        load_config(p)


def test_invalid_submission_type_fails(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "slug: x\nmetric: acc\ndirection: maximize\n"
        "submission_type: magic\nsubmission_file: s.csv\n"
        "kernel_slug: null\ndaily_limit: 5\n"
    )
    with pytest.raises(ConfigError, match="submission_type"):
        load_config(p)


def test_kernel_type_requires_kernel_slug(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "slug: x\nmetric: acc\ndirection: maximize\n"
        "submission_type: kernel\nsubmission_file: s.csv\n"
        "kernel_slug: null\ndaily_limit: 5\n"
    )
    with pytest.raises(ConfigError, match="kernel_slug"):
        load_config(p)


def test_nonpositive_daily_limit_fails(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "slug: x\nmetric: acc\ndirection: maximize\n"
        "submission_type: csv\nsubmission_file: s.csv\n"
        "kernel_slug: null\ndaily_limit: 0\n"
    )
    with pytest.raises(ConfigError, match="daily_limit"):
        load_config(p)


def test_non_integer_daily_limit_fails(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "slug: x\nmetric: acc\ndirection: maximize\n"
        "submission_type: csv\nsubmission_file: s.csv\n"
        "kernel_slug: null\ndaily_limit: abc\n"
    )
    with pytest.raises(ConfigError, match="daily_limit"):
        load_config(p)
