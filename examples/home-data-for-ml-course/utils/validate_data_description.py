"""Independent validator for data/data_description.json against data_description.txt.

Re-parses the source .txt with a different strategy than parse_data_description.py
(so a bug shared between parser and validator is less likely) and cross-checks:
    * every header in the .txt has a corresponding entry in the .json
      (modulo NAME_MAP and EXTRAS);
    * descriptions match exactly;
    * for categorical features, code sets and meanings match;
    * the `type` field (numerical/object) is consistent with the column lists.

Run as a script: exits non-zero if any check fails.
"""

import json
import re
import sys
from pathlib import Path

from parse_data_description import (
    EXTRA_FEATURES,
    NAME_MAP,
    NUMERICAL_COLS,
    OBJECT_COLS,
)

HEADER_PAT = re.compile(r"^([A-Za-z0-9]+):\s*(.*)$")
VALUE_SPLIT = re.compile(r"\t+|\s{2,}")


def _extract_txt_headers(lines: list[str]) -> list[tuple[int, str, str]]:
    """Return (lineno, name, description) for each feature header in the txt."""
    headers = []
    for i, ln in enumerate(lines, 1):
        if not ln or ln[0] in (" ", "\t"):
            continue
        m = HEADER_PAT.match(ln)
        if m:
            headers.append((i, m.group(1), m.group(2).strip()))
    return headers


def _extract_value_block(lines: list[str], start_li: int, end_li: int) -> dict[str, str]:
    """Parse the indented value lines between two header line numbers (1-based, exclusive)."""
    values: dict[str, str] = {}
    for ln in lines[start_li:end_li - 1]:
        if not ln.strip() or ln[0] not in (" ", "\t"):
            continue
        parts = VALUE_SPLIT.split(ln.strip(), maxsplit=1)
        if len(parts) == 2:
            code, meaning = parts[0].strip(), parts[1].strip()
            if code:
                values[code] = meaning
    return values


def validate(txt_path: Path, json_path: Path) -> list[str]:
    """Run all consistency checks; return a list of error messages (empty == OK)."""
    data = json.loads(json_path.read_text())
    lines = txt_path.read_text().splitlines()
    headers = _extract_txt_headers(lines)
    errors: list[str] = []

    # 1. Name coverage
    txt_names_mapped = {NAME_MAP.get(n, n) for _, n, _ in headers}
    json_names = set(data.keys())
    extras = set(EXTRA_FEATURES.keys())

    missing_in_json = txt_names_mapped - json_names
    extra_in_json = json_names - txt_names_mapped - extras
    if missing_in_json:
        errors.append(f"Features in txt but NOT in JSON: {sorted(missing_in_json)}")
    if extra_in_json:
        errors.append(f"Features in JSON but NOT in txt (and not in EXTRAS): {sorted(extra_in_json)}")
    missing_extras = extras - json_names
    if missing_extras:
        errors.append(f"Expected EXTRAS missing from JSON: {sorted(missing_extras)}")

    # 2. Descriptions
    for _, name, desc in headers:
        key = NAME_MAP.get(name, name)
        if key not in data:
            continue
        if data[key]["description"] != desc:
            errors.append(
                f"Description mismatch for {key}: txt={desc!r} vs json={data[key]['description']!r}"
            )

    # 3. Value blocks (walk pairs of consecutive headers; sentinel covers the last)
    extended = headers + [(len(lines) + 1, "__END__", "")]
    for idx in range(len(extended) - 1):
        start_li, name, _ = extended[idx]
        end_li, _, _ = extended[idx + 1]
        key = NAME_MAP.get(name, name)
        if key not in data:
            continue

        txt_values = _extract_value_block(lines, start_li, end_li)
        json_values = data[key]["values"]

        if not txt_values:
            if json_values is not None:
                errors.append(f"{key}: txt has no values, JSON has {list(json_values)}")
            continue
        if json_values is None:
            errors.append(f"{key}: txt has {len(txt_values)} values, JSON is None")
            continue

        txt_codes = set(txt_values)
        json_codes = set(json_values)
        if txt_codes != json_codes:
            errors.append(
                f"{key}: code-set mismatch. only-in-txt={sorted(txt_codes - json_codes)} "
                f"only-in-json={sorted(json_codes - txt_codes)}"
            )
            continue
        for code in txt_codes:
            if txt_values[code] != json_values[code]:
                errors.append(
                    f"{key}/{code}: meaning mismatch. txt={txt_values[code]!r} json={json_values[code]!r}"
                )

    # 4. Type assignment
    num_set, obj_set = set(NUMERICAL_COLS), set(OBJECT_COLS)
    for k, v in data.items():
        expected = "numerical" if k in num_set else ("object" if k in obj_set else "unknown")
        if v["type"] != expected:
            errors.append(f"{k}: type={v['type']!r} expected={expected!r}")

    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    txt = root / "data" / "data_description.txt"
    js = root / "data" / "data_description.json"

    data = json.loads(js.read_text())
    print(f"[INFO] Entries in JSON: {len(data)}")
    errors = validate(txt, js)

    if errors:
        print(f"[FAIL] {len(errors)} error(s):")
        for e in errors:
            print(" -", e)
        return 1

    print("[OK] All checks passed.")
    print(
        f"  Numerical: {sum(1 for v in data.values() if v['type']=='numerical')} | "
        f"Object: {sum(1 for v in data.values() if v['type']=='object')} | "
        f"With values: {sum(1 for v in data.values() if v['values'])} | "
        f"Without: {sum(1 for v in data.values() if v['values'] is None)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
