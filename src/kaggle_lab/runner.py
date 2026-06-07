"""Execute notebooks via papermill and produce run identity + artifact metadata."""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from pathlib import Path

import papermill as pm


def new_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone(timedelta(hours=2)))
    return f"{now.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"


def sha1_file(path: Path | str) -> str:
    h = hashlib.sha1()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def execute_notebook(notebook: Path | str, *, workdir: Path, log_path: Path) -> Path:
    notebook = Path(notebook)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    executed = workdir / notebook.name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        pm.execute_notebook(
            input_path=str(notebook),
            output_path=str(executed),
            cwd=str(workdir),
            stdout_file=log,
            stderr_file=log,
            progress_bar=False,
            kernel_name="python3",
        )
    return executed
