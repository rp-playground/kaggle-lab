"""Load and validate per-competition config.yaml."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    slug: str
    metric: str
    direction: Literal["maximize", "minimize"]
    submission_type: Literal["csv", "kernel"]
    submission_file: str
    kernel_slug: str | None
    daily_limit: int
    path: Path
    sample_submission: str | None = None


_REQUIRED = ("slug", "metric", "direction", "submission_type", "submission_file", "daily_limit")


def load_config(path: Path | str) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}

    for k in _REQUIRED:
        if k not in raw:
            raise ConfigError(f"missing required key: {k}")

    if raw["submission_type"] not in ("csv", "kernel"):
        raise ConfigError(f"submission_type must be 'csv' or 'kernel', got {raw['submission_type']!r}")

    if raw["direction"] not in ("maximize", "minimize"):
        raise ConfigError("direction must be 'maximize' or 'minimize'")

    if raw["submission_type"] == "kernel" and not raw.get("kernel_slug"):
        raise ConfigError("kernel_slug is required when submission_type is 'kernel'")

    try:
        daily_limit_int = int(raw["daily_limit"])
    except (TypeError, ValueError) as e:
        raise ConfigError(
            f"daily_limit must be an integer, got {raw['daily_limit']!r}"
        ) from e
    if daily_limit_int <= 0:
        raise ConfigError(f"daily_limit must be > 0, got {raw['daily_limit']!r}")

    return Config(
        slug=raw["slug"],
        metric=raw["metric"],
        direction=raw["direction"],
        submission_type=raw["submission_type"],
        submission_file=raw["submission_file"],
        kernel_slug=raw.get("kernel_slug"),
        daily_limit=daily_limit_int,
        path=path,
        sample_submission=raw.get("sample_submission"),
    )
