"""
Feature engineering for Ames Housing.

Stateless transformations only — every feature defined here depends solely on
the row it's computed from, so the same function is safe to apply to train
and test independently. Anything that needs to be fitted on training data
(scalers, target encoders, etc.) should live in a separate fit/apply module,
following the same pattern as ames_na_encoding.py.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from .feature_contracts import feature_contract


@feature_contract(
    requires={"YrSold", "YearBuilt", "YearRemodAdd"},
    produces={"HouseAge", "YearsSinceRemod", "WasRemodeled", "IsNewHouse"},
    conditional_drops={("drop_originals", True): {"YearBuilt", "YearRemodAdd", "YrSold"}},
)
def add_temporal_features(df: pd.DataFrame, *, drop_originals: bool = False) -> pd.DataFrame:
    """Add age-based features derived from YrSold, YearBuilt, YearRemodAdd.

    Features added:
      HouseAge        — years between construction and sale, clipped at 0
      YearsSinceRemod — years between last remodel/addition and sale, clipped at 0
      WasRemodeled    — 1 if YearRemodAdd > YearBuilt, else 0
                        (per the data dictionary, YearRemodAdd == YearBuilt
                        when no remodel has occurred)
      IsNewHouse      — 1 if HouseAge == 0, else 0
                        (often more useful than HouseAge near zero, since
                        new construction is its own market segment)

    Negative ages from data-entry noise (sale before remodel finished, etc.)
    are clipped to 0 rather than left as nonsensical negatives.

    Parameters
    ----------
    drop_originals : bool
        If True, drop YearBuilt, YearRemodAdd, and YrSold after deriving
        the new features. Useful for linear models where keeping both
        creates redundant signal; safe to leave False for tree models.
    """
    out = df.copy()

    out["HouseAge"]        = (out["YrSold"] - out["YearBuilt"]).clip(lower=0)
    out["YearsSinceRemod"] = (out["YrSold"] - out["YearRemodAdd"]).clip(lower=0)
    # Strict inequality: YearRemodAdd >= YearBuilt by definition, and the
    # data dictionary says they're equal when there's no remodel.
    out["WasRemodeled"]    = (out["YearRemodAdd"] > out["YearBuilt"]).astype("int8")
    out["IsNewHouse"]      = (out["HouseAge"] == 0).astype("int8")

    if drop_originals:
        out = out.drop(columns=["YearBuilt", "YearRemodAdd", "YrSold"])

    return out


_TOTAL_SF_PARTS = ("TotalBsmtSF", "1stFlrSF", "2ndFlrSF")


@feature_contract(
    requires=set(_TOTAL_SF_PARTS),
    produces={"TotalSF"},
    conditional_drops={("drop_originals", True): set(_TOTAL_SF_PARTS)},
)
def add_size_features(df: pd.DataFrame, *, drop_originals: bool = False) -> pd.DataFrame:
    """Add aggregate size features derived from per-floor square footage.

    Features added:
      TotalSF — TotalBsmtSF + 1stFlrSF + 2ndFlrSF, the total finished
                square footage across basement and above-grade floors.
                A handful of Ames rows with TotalSF beyond ~7000 are
                well-known outliers (mostly the Edwards mega-houses);
                row-level filtering on this is a training-time decision
                and lives outside this stateless transform.

    Assumes upstream NA imputation has already run (basement-less rows
    should have TotalBsmtSF == 0, not NaN); we don't fillna here so that
    a missed imputation surfaces as a visible NaN rather than a silently
    wrong sum.

    Parameters
    ----------
    drop_originals : bool
        If True, drop TotalBsmtSF, 1stFlrSF, and 2ndFlrSF after deriving
        TotalSF. Useful for linear models where keeping both creates
        redundant signal; safe to leave False for tree models.
    """
    out = df.copy()
    out["TotalSF"] = sum(out[c] for c in _TOTAL_SF_PARTS)

    if drop_originals:
        out = out.drop(columns=list(_TOTAL_SF_PARTS))

    return out


_TOTAL_BATH_FULL_PARTS = ("FullBath", "BsmtFullBath")
_TOTAL_BATH_HALF_PARTS = ("HalfBath", "BsmtHalfBath")


@feature_contract(
    requires=set(_TOTAL_BATH_FULL_PARTS) | set(_TOTAL_BATH_HALF_PARTS),
    produces={"TotalBath"},
    conditional_drops={
        ("drop_originals", True): set(_TOTAL_BATH_FULL_PARTS) | set(_TOTAL_BATH_HALF_PARTS),
    },
)
def add_bath_features(df: pd.DataFrame, *, drop_originals: bool = False) -> pd.DataFrame:
    """Add an aggregate TotalBath = full + 0.5 * half across above-grade and basement.

    Features added:
      TotalBath — FullBath + BsmtFullBath + 0.5 * (HalfBath + BsmtHalfBath).
                  Trees with bounded depth can't sum four columns in a single
                  split, so handing them the aggregate captures the
                  "bathroom equivalents" signal in one threshold.

    Half-baths are weighted 0.5 by realtor convention (a half-bath has
    roughly half the market premium of a full bath). Basement vs above-grade
    bathrooms also carry different market value — buyers price a 2nd-floor
    master bath differently than a basement guest bath — so drop_originals
    defaults to False, leaving the four components in place for trees to
    split on alongside the aggregate.

    Assumes upstream NA imputation has already run (BsmtFullBath /
    BsmtHalfBath should be 0, not NaN, for basement-less rows); we do not
    fillna here so a missed imputation surfaces as a visible NaN rather
    than a silently wrong sum.

    Parameters
    ----------
    drop_originals : bool
        If True, drop FullBath, HalfBath, BsmtFullBath, BsmtHalfBath after
        deriving TotalBath. Default False — for tree models the aggregate
        and the components together usually beat either alone.
    """
    out = df.copy()
    full = sum(out[c] for c in _TOTAL_BATH_FULL_PARTS)
    half = sum(out[c] for c in _TOTAL_BATH_HALF_PARTS)
    out["TotalBath"] = full + 0.5 * half

    if drop_originals:
        cols = list(_TOTAL_BATH_FULL_PARTS) + list(_TOTAL_BATH_HALF_PARTS)
        out = out.drop(columns=cols)

    return out


# Columns Ames stores as integer codes that are nominal (no order). The codes
# look numeric but have no meaningful magnitude relationship — e.g. MSSubClass
# 60 is "2-STORY 1946 & NEWER" and 30 is "1-STORY 1945 & OLDER", but
# 60 - 30 = 30 means nothing. Leaving them numeric forces tree models to
# discover arbitrary thresholds between unrelated dwelling types and silently
# treats one-hot encoders as numeric passthrough.
NOMINAL_CODE_COLS = ("MSSubClass",)


@feature_contract(
    requires={
        "OverallQual", "OverallCond", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",
        "GrLivArea", "ExterQual", "KitchenQual", "BsmtQual", "GarageQual",
        "FireplaceQu", "PoolQC", "GarageArea", "Fireplaces", "PoolArea",
    },
    produces={
        "QualSF", "QualGrLivSF", "CondSF",
        "ExterQual_x_GrLivArea", "KitchenQual_x_GrLivArea",
        "BsmtQual_x_TotalBsmtSF", "GarageQual_x_GarageArea",
        "FireplaceQu_x_Fireplaces", "PoolQC_x_PoolArea",
    },
    # NB: function accepts drop_originals as a no-op (kept for API parity).
)
def add_quality_size_features(df, drop_originals=False):
    out = df.copy()
    total_sf = out["TotalBsmtSF"].fillna(0) + out["1stFlrSF"] + out["2ndFlrSF"]
    grliv    = out["GrLivArea"]

    # Headline: total quality-weighted footprint
    out["QualSF"]      = out["OverallQual"] * total_sf
    out["QualGrLivSF"] = out["OverallQual"] * grliv
    out["CondSF"]      = out["OverallCond"] * total_sf

    # Component-level "quality of X × size of X". Letter grades Ex..Po -> 5..1
    # with missing -> 0. Note: AmesNAImputer already ordinal-encodes
    # BsmtQual/GarageQual/FireplaceQu/PoolQC to int8 0..5 (see ames_na_encoding
    # QUALITY_MAP), so for those columns we use the value directly. Only
    # ExterQual/KitchenQual reach this step as strings and need the map.
    qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
    for col, area_col in [
        ("ExterQual",    "GrLivArea"),       # exterior quality scales with house size
        ("KitchenQual",  "GrLivArea"),
        ("BsmtQual",     "TotalBsmtSF"),
        ("GarageQual",   "GarageArea"),
        ("FireplaceQu",  "Fireplaces"),      # count, not area, but same logic
        ("PoolQC",       "PoolArea"),
    ]:
        if pd.api.types.is_numeric_dtype(out[col]):
            score = out[col].fillna(0)
        else:
            score = out[col].map(qual_map).fillna(0)
        area = out[area_col].fillna(0)
        out[f"{col}_x_{area_col}"] = score * area

    return out


@feature_contract(
    requires={"OverallQual", "OverallCond"},
    produces={"OverallQual_sq", "QualCond", "QualCond_gap"},
    conditional_drops={("drop_originals", True): {"OverallQual", "OverallCond"}},
)
def add_quality_interactions(df, drop_originals=False):
    out = df.copy()

    # Quality² — captures the convexity (rich buyers pay disproportionately for top quality)
    out["OverallQual_sq"] = out["OverallQual"] ** 2

    # Quality × Condition interaction — a 9/3 (great house, poor condition) is NOT a 6/6
    out["QualCond"] = out["OverallQual"] * out["OverallCond"]

    # Asymmetric penalty: condition matters more when quality is high
    out["QualCond_gap"] = out["OverallQual"] - out["OverallCond"]

    if drop_originals:
        out = out.drop(columns=["OverallQual", "OverallCond"])

    return out


_PORCH_COLS = ("OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF")


@feature_contract(
    requires=set(_PORCH_COLS),
    produces={"TotalPorchSF", "HasPorch"},
    conditional_drops={("drop_originals", True): set(_PORCH_COLS)},
)
def add_porch_features(df: pd.DataFrame, *, drop_originals: bool = False) -> pd.DataFrame:
    """Add aggregate porch/deck square footage and a HasPorch indicator.

    TotalPorchSF — sum of OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch
                   + WoodDeckSF. Same logic as TotalSF / TotalBath: trees with
                   bounded depth can't sum five columns in a single split.
    HasPorch     — 1 if TotalPorchSF > 0, else 0. Linear models can't reach the
                   absent-vs-present threshold from the continuous columns alone.
    """
    out = df.copy()
    out["TotalPorchSF"] = sum(out[c] for c in _PORCH_COLS)
    out["HasPorch"] = (out["TotalPorchSF"] > 0).astype("int8")
    if drop_originals:
        out = out.drop(columns=list(_PORCH_COLS))
    return out


@feature_contract(
    # Intentionally tolerant: function uses .get(col, 0) with defaults, so
    # missing source columns don't crash — they just yield zero indicators.
    requires=set(),
    produces={"HasGarage", "HasBsmt", "Has2ndFloor", "HasFireplace", "HasPool", "HasMasVnr"},
)
def add_presence_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add Has* binary flags for components that may be absent.

    Trees can recover these via a single split on the parent area column, but
    linear models cannot — the 0 vs >0 transition is a different signal than
    the linear slope. Adding the indicator is cheap and helps the linear/KRR
    branch of the blend.
    """
    out = df.copy()
    out["HasGarage"]    = (out.get("GarageArea", 0).fillna(0) > 0).astype("int8") \
                          if "GarageArea" in out.columns else 0
    out["HasBsmt"]      = (out.get("TotalBsmtSF", 0).fillna(0) > 0).astype("int8") \
                          if "TotalBsmtSF" in out.columns else 0
    out["Has2ndFloor"]  = (out["2ndFlrSF"] > 0).astype("int8") if "2ndFlrSF" in out.columns else 0
    out["HasFireplace"] = (out["Fireplaces"] > 0).astype("int8") if "Fireplaces" in out.columns else 0
    out["HasPool"]      = (out["PoolArea"] > 0).astype("int8") if "PoolArea" in out.columns else 0
    out["HasMasVnr"]    = (out["MasVnrArea"].fillna(0) > 0).astype("int8") \
                          if "MasVnrArea" in out.columns else 0
    return out


@feature_contract(
    # Mandatory inputs; the others (TotalBath, LotFrontage, LotArea,
    # 1stFlrSF, 2ndFlrSF) are consumed opportunistically — declared in
    # `produces` is the maximal set, conditional on those being present.
    requires={"GrLivArea", "TotRmsAbvGrd", "BedroomAbvGr"},
    produces={
        "RoomDensity", "SFperBedroom",
        "BathPerBedroom", "LotEfficiency", "BuildingFootprint",
    },
)
def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row ratios that linear models / shallow trees can't compute internally.

    RoomDensity        — GrLivArea / TotRmsAbvGrd, "size per room".
    SFperBedroom       — GrLivArea / BedroomAbvGr.
    BathPerBedroom     — TotalBath / BedroomAbvGr (only if TotalBath already
                         computed by add_bath_features).
    LotEfficiency      — LotFrontage / sqrt(LotArea). Frontage relative to
                         total lot is what realtors call "street appeal" capacity.
    BuildingFootprint  — (1stFlrSF + 2ndFlrSF) / LotArea. The fraction of the
                         lot covered by the building. Distinct from TotalSF
                         because it normalises by lot size.

    Assumes AmesNAImputer has already filled LotFrontage; LotArea is never
    NaN in the canonical Ames data. Denominators are clipped at 1 to avoid
    divide-by-zero on degenerate rows.
    """
    out = df.copy()
    out["RoomDensity"]  = out["GrLivArea"]    / out["TotRmsAbvGrd"].clip(lower=1)
    out["SFperBedroom"] = out["GrLivArea"]    / out["BedroomAbvGr"].clip(lower=1)
    if "TotalBath" in out.columns:
        out["BathPerBedroom"] = out["TotalBath"] / out["BedroomAbvGr"].clip(lower=1)
    if "LotFrontage" in out.columns and "LotArea" in out.columns:
        out["LotEfficiency"] = out["LotFrontage"] / np.sqrt(out["LotArea"].clip(lower=1))
    if "1stFlrSF" in out.columns and "2ndFlrSF" in out.columns and "LotArea" in out.columns:
        out["BuildingFootprint"] = (out["1stFlrSF"] + out["2ndFlrSF"]) / out["LotArea"].clip(lower=1)
    return out


def cast_nominal_codes(df: pd.DataFrame,
                       cols: tuple[str, ...] = NOMINAL_CODE_COLS) -> pd.DataFrame:
    """Cast nominal numeric-coded columns to string so the encoder treats
    them as categories.

    Stateless: the cast depends only on the column itself, no fitting needed.
    Returns a new DataFrame; the input is not modified.
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(str)
    return out


@feature_contract(
    requires={"Condition1", "Condition2"},
    produces={
        "HasTwoConditions", "ConditionScore",
        "NearPositive", "NearRailroad", "NearArtery",
    },
    conditional_drops={("drop_originals", True): {"Condition1", "Condition2"}},
)
def add_condition_features(df, drop_originals=False):
    out = df.copy()

    # Has a genuine second condition?
    out["HasTwoConditions"] = (out["Condition1"] != out["Condition2"]).astype(int)

    # Net positive/negative off-site exposure
    POS = {"PosN", "PosA"}
    NEG = {"Artery", "Feedr", "RRNn", "RRAn", "RRNe", "RRAe"}

    def score(c):
        if c in POS: return 1
        if c in NEG: return -1
        return 0  # Norm

    out["ConditionScore"] = (
            out["Condition1"].map(score).fillna(0)
            + out["Condition2"].map(score).fillna(0)
    )

    # Specific high-signal flags (cheap, interpretable)
    out["NearPositive"] = ((out["Condition1"].isin(POS)) | (out["Condition2"].isin(POS))).astype(int)
    out["NearRailroad"] = (
            out["Condition1"].str.startswith("RR", na=False)
            | out["Condition2"].str.startswith("RR", na=False)
    ).astype(int)
    out["NearArtery"] = (
            (out["Condition1"] == "Artery") | (out["Condition2"] == "Artery")
    ).astype(int)

    if drop_originals:
        out = out.drop(columns=["Condition1", "Condition2"])
    return out