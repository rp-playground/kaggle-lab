"""Parse data/data_description.txt into a structured JSON dict.

Output schema (keyed by feature name):
    {
      "<FeatureName>": {
        "description": str,
        "type": "numerical" | "object",
        "values": {code: meaning, ...} | None
      },
      ...
    }

Run as a script to (re)generate data/data_description.json.
"""

import json
import re
from pathlib import Path

NUMERICAL_COLS = [
    'Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
    'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
    'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
    'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
    'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
    'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
    'MiscVal', 'MoSold', 'YrSold', 'SalePrice',
]

OBJECT_COLS = [
    'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
    'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
    'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
    'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
    'SaleType', 'SaleCondition',
]

# Description file uses short names for these; map to real column names.
NAME_MAP = {"Bedroom": "BedroomAbvGr", "Kitchen": "KitchenAbvGr"}

# Features absent from data_description.txt — added with synthetic descriptions.
EXTRA_FEATURES = {
    "Id": {"description": "Unique identifier for each property", "type": "numerical", "values": None},
    "SalePrice": {"description": "Sale price of the property in dollars (target variable)", "type": "numerical", "values": None},
}

# Matches a feature header line like "MSZoning: Identifies the general zoning ...".
# Anchored at start of line (no leading whitespace -> distinguishes headers from
# indented value lines). Group 1 = feature name (alnum only, e.g. "1stFlrSF"),
# group 2 = free-text description after the colon.
HEADER_RE = re.compile(r"^([A-Za-z0-9]+):\s*(.*)$")

# Splits a value line "<code><sep><meaning>" into [code, meaning]. The separator
# is either a tab run (\t+) or 2+ whitespace chars (\s{2,}) — never a single
# space, since meanings themselves contain single spaces
# (e.g. "1-STORY 1946 & NEWER ALL STYLES").
VALUE_SPLIT_RE = re.compile(r"\t+|\s{2,}")


def _feature_type(name: str) -> str:
    if name in NUMERICAL_COLS:
        return "numerical"
    if name in OBJECT_COLS:
        return "object"
    return "unknown"


def parse_description(src: Path) -> dict:
    """Parse the Ames data_description.txt file into a feature dict."""
    features: dict = {}
    current = None  # (name, description, values_dict)

    for raw in src.read_text().splitlines():
        if not raw.strip():
            continue
        # Header lines have no leading whitespace.
        if not raw.startswith((" ", "\t")):
            m = HEADER_RE.match(raw)
            if m:
                if current is not None:
                    name, desc, vals = current
                    mapped = NAME_MAP.get(name, name)
                    features[mapped] = {
                        "description": desc.strip(),
                        "type": _feature_type(mapped),
                        "values": vals if vals else None,
                    }
                current = (m.group(1), m.group(2), {})
                continue
        # Value line: "  code<whitespace>meaning"
        if current is not None:
            parts = VALUE_SPLIT_RE.split(raw.strip(), maxsplit=1)
            if len(parts) == 2:
                code, meaning = parts[0].strip(), parts[1].strip()
                if code:
                    current[2][code] = meaning

    # Flush last feature.
    if current is not None:
        name, desc, vals = current
        mapped = NAME_MAP.get(name, name)
        features[mapped] = {
            "description": desc.strip(),
            "type": _feature_type(mapped),
            "values": vals if vals else None,
        }

    for k, v in EXTRA_FEATURES.items():
        features.setdefault(k, v)

    return features


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "data" / "data_description.txt"
    dst = root / "data" / "data_description.json"

    features = parse_description(src)

    all_known = set(NUMERICAL_COLS + OBJECT_COLS)
    parsed = set(features.keys())
    missing = all_known - parsed
    extra = parsed - all_known
    print(f"Total features: {len(features)}")
    print(f"Missing from parsed: {sorted(missing)}")
    print(f"Extra (unknown type): {sorted(extra)}")

    dst.write_text(json.dumps(features, indent=2, ensure_ascii=False))
    print(f"Written to {dst}")


if __name__ == "__main__":
    main()
