"""Contract test: pin kaggle_lab's SDK method calls to the real KaggleApi surface."""
from __future__ import annotations

import pytest

# kaggle/__init__.py calls authenticate() at import time and exits when
# ~/.kaggle/kaggle.json is missing; catch SystemExit to skip gracefully.
try:
    from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: F401
    _IMPORTABLE = True
except (ImportError, SystemExit):
    _IMPORTABLE = False


@pytest.mark.skipif(
    not _IMPORTABLE,
    reason="kaggle SDK is not importable without valid credentials",
)
def test_methods_we_call_exist_on_real_api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    # Bypass __init__ so we don't re-trigger authentication.
    api = KaggleApi.__new__(KaggleApi)
    for name in ("authenticate", "competition_submit", "competition_submissions"):
        assert hasattr(api, name), f"KaggleApi is missing method {name!r}"
