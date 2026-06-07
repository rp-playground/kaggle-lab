"""
Encoders for Ames Housing columns where NaN means 'feature absent', not 'missing'.

Per the data description, the following columns use "NA" as a meaningful category:
    Alley, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2,
    FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond,
    PoolQC, Fence, MiscFeature

When read with pandas, "NA" becomes NaN and is indistinguishable from a real
missing value. These functions restore the intended meaning.

Columns are grouped by the encoding pattern they share, so one function
handles all members of a group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Group 1: Ex/Gd/TA/Fa/Po/NA quality ladder  ->  ordinal 0..5
# ---------------------------------------------------------------------------
QUALITY_COLS = [
    "BsmtQual", "BsmtCond",
    "FireplaceQu",
    "GarageQual", "GarageCond",
    "PoolQC",
    # Note: ExterQual, ExterCond, HeatingQC, KitchenQual share the same
    # vocabulary but have NO "NA" level in the description, so a NaN there
    # IS missing. Add them here only if you want to map their non-null values
    # on the same scale; do not use this function to fillna on them.
]
QUALITY_MAP = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}


def encode_quality(df: pd.DataFrame, cols: list[str] = QUALITY_COLS) -> pd.DataFrame:
    """Map Ex/Gd/TA/Fa/Po -> 5..1 and NaN -> 0 (feature absent)."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(QUALITY_MAP).fillna(0).astype("int8")
    return out


# ---------------------------------------------------------------------------
# Group 2: Other ordinal scales where NA means 'absent'
# ---------------------------------------------------------------------------
# Each entry: column -> {category: rank}.  NaN is filled with 0.
ORDINAL_MAPS: dict[str, dict[str, int]] = {
    "BsmtExposure":  {"No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1":  {"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2":  {"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish":  {"Unf": 1, "RFn": 2, "Fin": 3},
    "Fence":         {"MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
}


def encode_ordinal_absent(df: pd.DataFrame,
                          mappings: dict[str, dict[str, int]] = ORDINAL_MAPS,
                          ) -> pd.DataFrame:
    """Apply per-column ordinal maps; NaN -> 0 (feature absent)."""
    out = df.copy()
    for col, mp in mappings.items():
        if col in out.columns:
            out[col] = out[col].map(mp).fillna(0).astype("int8")
    return out


# ---------------------------------------------------------------------------
# Group 3: Nominal 'absent' categories — keep as string, no order implied
# ---------------------------------------------------------------------------
NOMINAL_ABSENT_COLS = {
    "Alley":       "NoAlley",
    "GarageType":  "NoGarage",
    "MiscFeature": "None",
    # MasVnrType: description lists "None" as a literal category, but ~60% of
    # rows are NaN in both train and test — too high and too symmetric to be
    # random missingness. Cross-checks against MasVnrArea==0 confirm NaN means
    # "no veneer". Collapse NaN into the existing "None" label.
    "MasVnrType":  "None",
}


def encode_nominal_absent(df: pd.DataFrame,
                          fillers: dict[str, str] = NOMINAL_ABSENT_COLS,
                          ) -> pd.DataFrame:
    """Replace NaN with an explicit 'absent' label, keeping the column nominal.

    Use one-hot or target encoding downstream; do NOT treat as ordinal.
    """
    out = df.copy()
    for col, label in fillers.items():
        if col in out.columns:
            out[col] = out[col].fillna(label).astype("category")
    return out


# ---------------------------------------------------------------------------
# Group 4: Companion numeric columns — NaN/blank means 0 when feature absent
# ---------------------------------------------------------------------------
# These aren't in the "NA" list above, but they ride along with the absent
# feature: if there's no garage, GarageArea/Cars are 0, GarageYrBlt is undefined.
NUMERIC_ZERO_FILL = [
    "GarageArea", "GarageCars",
    "MasVnrArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath",
]


def encode_numeric_companions(df: pd.DataFrame,
                              cols: list[str] = NUMERIC_ZERO_FILL,
                              ) -> pd.DataFrame:
    """Fill NaN with 0 for numeric features that are 0 when the parent is absent."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].fillna(0)
    return out


def encode_garage_year(df: pd.DataFrame, col: str = "GarageYrBlt") -> pd.DataFrame:
    """GarageYrBlt has no meaningful 0; align it with YearBuilt when no garage.

    This avoids creating a misleading outlier year (like 0) while still encoding
    'no garage' implicitly through the matching GarageType/GarageQual = absent.
    """
    out = df.copy()
    if col in out.columns and "YearBuilt" in out.columns:
        out[col] = out[col].fillna(out["YearBuilt"])
    return out


def enforce_masvnr_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Force MasVnrArea to 0 wherever MasVnrType says 'None'.

    A handful of rows (~5 in train, ~3 in test) have NaN type with a non-zero
    area — almost certainly data-entry noise (e.g. area=1.0). After folding
    NaN-type into 'None', those rows would otherwise claim 'no veneer, area=288'
    which is internally contradictory. Zeroing the area removes the conflict.
    """
    out = df.copy()
    if {"MasVnrType", "MasVnrArea"}.issubset(out.columns):
        mask = out["MasVnrType"].astype(str).eq("None")
        out.loc[mask, "MasVnrArea"] = 0.0
    return out


# ---------------------------------------------------------------------------
# One-shot convenience
# ---------------------------------------------------------------------------
def encode_all_absent_as_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every group above in the right order. Returns a new DataFrame."""
    out = df
    out = encode_quality(out)
    out = encode_ordinal_absent(out)
    out = encode_nominal_absent(out)
    out = encode_numeric_companions(out)
    out = encode_garage_year(out)
    out = enforce_masvnr_consistency(out)
    return out


# ---------------------------------------------------------------------------
# LotFrontage: NOT 'absent', genuinely missing. Impute by neighborhood median.
# ---------------------------------------------------------------------------
# ~16-17% missing in both train and test, symmetric across splits. Frontage
# can't be zero (every parcel touches a street), so the NaNs are real missing
# values that need imputation, not encoding. Lots within the same neighborhood
# share subdivision plat / zoning / era, so neighborhood-median is much
# tighter than a global median.
#
# Fit the lookup ONCE on training data, then apply the same lookup to test —
# otherwise you leak test-distribution information into the imputed values.

def fit_lot_frontage_lookup(train_df: pd.DataFrame) -> dict:
    """Build {neighborhood: median_frontage} from training data only.

    Stores a global fallback for any neighborhood that appears in test but
    not in train (or that is entirely NaN within its training group).
    """
    by_nbhd = train_df.groupby("Neighborhood")["LotFrontage"].median()
    fallback = train_df["LotFrontage"].median()
    return {"by_nbhd": by_nbhd.to_dict(), "fallback": float(fallback)}


def apply_lot_frontage_lookup(df: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    """Fill NaN LotFrontage using the fitted neighborhood-median lookup."""
    out = df.copy()
    if "LotFrontage" not in out.columns or "Neighborhood" not in out.columns:
        return out
    nbhd_map = lookup["by_nbhd"]
    fallback = lookup["fallback"]
    imputed = out["Neighborhood"].map(nbhd_map).fillna(fallback)
    out["LotFrontage"] = out["LotFrontage"].fillna(imputed)
    return out


# ---------------------------------------------------------------------------
# Electrical: exactly one missing row in the canonical Ames train set.
# No "NA" category in the description, so it's genuine missingness.
# Mode-impute: SBrkr dominates (~91% of rows), and the single missing row
# is a 2007 build, where SBrkr is essentially universal.
# ---------------------------------------------------------------------------

def fit_electrical_mode(train_df: pd.DataFrame) -> str:
    """Return the most common Electrical value from training data."""
    return train_df["Electrical"].mode().iloc[0]


def apply_electrical_mode(df: pd.DataFrame, mode_value: str) -> pd.DataFrame:
    """Fill NaN Electrical with the fitted training mode."""
    out = df.copy()
    if "Electrical" in out.columns:
        out["Electrical"] = out["Electrical"].fillna(mode_value)
    return out


# ---------------------------------------------------------------------------
# Long-tail categorical NaNs in test (1-4 rows each, none in train).
# Mode imputation is the right move when one category overwhelmingly dominates.
# ---------------------------------------------------------------------------
SIMPLE_MODE_COLS = [
    "Utilities",     # ~99.96% AllPub — column is near-degenerate, often dropped
    "Functional",    # description literally says "Assume typical"
    "SaleType",      # ~87% WD
    "Exterior1st",   # mode VinylSd (~35%); long tail but only 1 row missing
    "Exterior2nd",   # mode VinylSd (~35%); same reasoning
    "KitchenQual",   # mode TA (~50%)
]


def fit_simple_modes(train_df: pd.DataFrame,
                     cols: list[str] = SIMPLE_MODE_COLS) -> dict[str, str]:
    """Compute the training mode for each column. Skip columns not present."""
    modes: dict[str, str] = {}
    for c in cols:
        if c in train_df.columns:
            m = train_df[c].mode(dropna=True)
            if len(m) > 0:
                modes[c] = m.iloc[0]
    return modes


def apply_simple_modes(df: pd.DataFrame, modes: dict[str, str]) -> pd.DataFrame:
    """Fill NaN with the fitted training mode for each column."""
    out = df.copy()
    for c, m in modes.items():
        if c in out.columns:
            out[c] = out[c].fillna(m)
    return out


# ---------------------------------------------------------------------------
# MSZoning: conditional mode by (MSSubClass, Neighborhood).
#
# Plain mode would give RL to all 4 missing test rows (RL is ~79% of train),
# but zoning is mechanically tied to subdivision and dwelling type. Older
# 1-story homes (MSSubClass=30) in IDOTRR are almost always RM, not RL.
# Conditioning preserves that structure; falling back through MSSubClass alone
# and then global mode handles unseen combinations.
# ---------------------------------------------------------------------------

def fit_mszoning_lookup(train_df: pd.DataFrame) -> dict:
    """Build a 3-level fallback for MSZoning imputation.

    Levels: (MSSubClass, Neighborhood) -> MSSubClass -> global mode.
    """
    if "MSZoning" not in train_df.columns:
        return {}
    by_pair = (
        train_df.dropna(subset=["MSZoning"])
                .groupby(["MSSubClass", "Neighborhood"])["MSZoning"]
                .agg(lambda s: s.mode().iloc[0])
                .to_dict()
    )
    by_subclass = (
        train_df.dropna(subset=["MSZoning"])
                .groupby("MSSubClass")["MSZoning"]
                .agg(lambda s: s.mode().iloc[0])
                .to_dict()
    )
    global_mode = train_df["MSZoning"].mode().iloc[0]
    return {"by_pair": by_pair, "by_subclass": by_subclass, "global": global_mode}


def apply_mszoning_lookup(df: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    """Impute MSZoning using the 3-level fallback."""
    if not lookup or "MSZoning" not in df.columns:
        return df
    out = df.copy()
    pair_map = lookup["by_pair"]
    sub_map  = lookup["by_subclass"]
    fallback = lookup["global"]

    def _impute(row):
        if pd.notna(row["MSZoning"]):
            return row["MSZoning"]
        key = (row.get("MSSubClass"), row.get("Neighborhood"))
        if key in pair_map:
            return pair_map[key]
        if row.get("MSSubClass") in sub_map:
            return sub_map[row["MSSubClass"]]
        return fallback

    out["MSZoning"] = out.apply(_impute, axis=1)
    return out


# ---------------------------------------------------------------------------
# Unified pipeline: stateless encodings + fitted imputers, in one boundary.
# ---------------------------------------------------------------------------

def fit_imputers(train_df: pd.DataFrame) -> dict:
    """Fit every learned-from-training imputer and return the parameters.

    Call once, on the training set only. The returned dict is the only state
    that needs to flow from train to test.
    """
    return {
        "lot_frontage":  fit_lot_frontage_lookup(train_df),
        "electrical":    fit_electrical_mode(train_df),
        "simple_modes":  fit_simple_modes(train_df),
        "mszoning":      fit_mszoning_lookup(train_df),
    }


def transform(df: pd.DataFrame, imputers: dict) -> pd.DataFrame:
    """Run the full pipeline: stateless encodings, then fitted imputations.

    Use the SAME `imputers` dict for both train and test.
    """
    out = encode_all_absent_as_zero(df)
    out = apply_lot_frontage_lookup(out, imputers["lot_frontage"])
    out = apply_electrical_mode(out, imputers["electrical"])
    out = apply_simple_modes(out, imputers["simple_modes"])
    out = apply_mszoning_lookup(out, imputers["mszoning"])
    return out


if __name__ == "__main__":
    sample = pd.DataFrame({
        "BsmtQual":     ["Ex", "Gd", np.nan, "TA"],
        "PoolQC":       [np.nan, np.nan, "Gd", np.nan],
        "BsmtExposure": ["Gd", np.nan, "No", "Av"],
        "GarageFinish": ["Fin", np.nan, "Unf", "RFn"],
        "Fence":        [np.nan, "MnPrv", np.nan, "GdPrv"],
        "Alley":        [np.nan, "Pave", np.nan, "Grvl"],
        "GarageType":   ["Attchd", np.nan, "Detchd", np.nan],
        "MiscFeature":  [np.nan, "Shed", np.nan, np.nan],
        "GarageArea":   [400.0, np.nan, 250.0, 500.0],
        "GarageYrBlt":  [1990.0, np.nan, 1985.0, 2001.0],
        "YearBuilt":    [1990, 1960, 1985, 2001],
        "MasVnrArea":   [0.0, np.nan, 150.0, np.nan],
    })

    encoded = encode_all_absent_as_zero(sample)
    print(encoded.to_string())
    print("\nAny remaining NaN in handled columns?",
          encoded.isna().any().any())
