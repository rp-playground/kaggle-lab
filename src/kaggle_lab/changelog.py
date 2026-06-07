"""Parse the required ## Changelog cell in an experiment notebook."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


class ChangelogError(ValueError):
    pass


@dataclass(frozen=True)
class Changelog:
    parent: str
    change: str
    hypothesis: str


_HEADER = re.compile(r"^##\s+Changelog\s*$", re.MULTILINE)
_ENTRY = re.compile(r"^-\s*(\w+)\s*:\s*(.+?)\s*$", re.MULTILINE)


def parse_changelog(notebook_path: Path | str) -> Changelog:
    nb = json.loads(Path(notebook_path).read_text())
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        src = "".join(cell.get("source", []))
        if _HEADER.search(src):
            return _parse_body(src, notebook_path)
    raise ChangelogError(f"no ## Changelog cell found in {notebook_path}")


def _parse_body(src: str, notebook_path: Path | str) -> Changelog:
    fields: dict[str, str] = {}
    last_key: str | None = None
    for line in src.splitlines():
        if not line.strip():
            continue
        if _HEADER.match(line):
            continue
        m = _ENTRY.match(line)
        if m:
            last_key = m.group(1)
            fields[last_key] = m.group(2).strip()
            continue
        if line[0].isspace() and last_key is not None:
            fields[last_key] = f"{fields[last_key]} {line.strip()}"
            continue
        raise ChangelogError(
            f"{notebook_path}: unexpected line in changelog: {line!r}"
        )
    for key in ("parent", "change", "hypothesis"):
        if key not in fields or not fields[key]:
            raise ChangelogError(
                f"{notebook_path}: changelog missing required key: {key}"
            )
    return Changelog(parent=fields["parent"], change=fields["change"], hypothesis=fields["hypothesis"])
