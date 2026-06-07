"""
Random Forest baseline for Ames Housing.

Pipeline:
  1. Encode the four quality columns (Ex/Gd/TA/Fa/Po) without an NA level.
  2. Encode CentralAir as binary.
  3. One-hot encode all remaining string/category columns, FITTED ON TRAIN ONLY.
  4. Fit a RandomForestRegressor on log1p(SalePrice).

This module assumes train_data and test_data have already been through:
  - encode_all_absent_as_zero / transform (from ames_na_encoding)
  - add_temporal_features (from ames_feature_engineering)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score


# ---------------------------------------------------------------------------
# Stateless numeric encodings (safe to apply per-split)
# ---------------------------------------------------------------------------
QUALITY_NO_NA_COLS = ["ExterQual", "ExterCond", "HeatingQC", "KitchenQual"]
QUALITY_MAP = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}


def encode_quality_no_na(df: pd.DataFrame) -> pd.DataFrame:
    """Map Ex/Gd/TA/Fa/Po to 5..1 for columns that have no NA level.

    Unlike the columns in ames_na_encoding.encode_quality, these four are
    mandatory features (every house has an exterior, heating, kitchen).
    Any NaN here is genuine missingness — we leave it as NaN so it surfaces
    as an error rather than silently becoming 0.
    """
    out = df.copy()
    for c in QUALITY_NO_NA_COLS:
        if c in out.columns:
            out[c] = out[c].map(QUALITY_MAP).astype("Int8")  # nullable, preserves NaN
    return out


def encode_central_air(df: pd.DataFrame) -> pd.DataFrame:
    """CentralAir is just Y/N — encode as 0/1."""
    out = df.copy()
    if "CentralAir" in out.columns:
        out["CentralAir"] = (out["CentralAir"] == "Y").astype("int8")
    return out


# ---------------------------------------------------------------------------
# Fitted one-hot encoding for everything else
# ---------------------------------------------------------------------------

def fit_onehot(train_df: pd.DataFrame) -> tuple[OneHotEncoder, list[str]]:
    """Fit a OneHotEncoder on every remaining string/category column in train.

    Returns the fitted encoder plus the list of columns it was fitted on,
    so the same columns are encoded at apply time even if test happens to
    have a column with all-numeric values that pandas auto-typed differently.
    """
    cat_cols = train_df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    encoder = OneHotEncoder(
        handle_unknown="ignore",  # silently zero out categories never seen in train
        sparse_output=False,      # dense ndarray — RF doesn't benefit from sparse
        dtype=np.int8,
    )
    encoder.fit(train_df[cat_cols])
    return encoder, cat_cols
  

def apply_onehot(df: pd.DataFrame,
                 encoder: OneHotEncoder,
                 cat_cols: list[str]) -> pd.DataFrame:
    """Apply the fitted encoder, returning a numeric-only DataFrame.

    The non-categorical columns pass through unchanged; the categorical
    columns are replaced with their one-hot expansion using train's vocabulary.
    """
    encoded_array = encoder.transform(df[cat_cols])
    encoded_cols = encoder.get_feature_names_out(cat_cols)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)

    numeric_part = df.drop(columns=cat_cols)
    return pd.concat([numeric_part, encoded_df], axis=1)


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def fit_encoders(train_df: pd.DataFrame) -> dict:
    """Fit every encoder that needs to be learned on train. Returns a dict
    of parameters, mirroring the fit_imputers pattern from ames_na_encoding."""
    # First apply the stateless steps so the encoder sees the final column set
    train_prepped = encode_central_air(encode_quality_no_na(train_df))
    encoder, cat_cols = fit_onehot(train_prepped)
    return {"onehot": encoder, "cat_cols": cat_cols}


def transform(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Apply the full encoding pipeline. Use the SAME encoders dict for
    train and test."""
    out = encode_quality_no_na(df)
    out = encode_central_air(out)
    out = apply_onehot(out, encoders["onehot"], encoders["cat_cols"])
    return out


# ---------------------------------------------------------------------------
# Convenience: train + evaluate a Random Forest
# ---------------------------------------------------------------------------

def train_random_forest(train_X: pd.DataFrame,
                        train_y: pd.Series,
                        n_estimators: int = 300,
                        random_state: int = 42,
                        cv_folds: int = 5) -> RandomForestRegressor:
    """Cross-validate, then fit a Random Forest on the full training set.

    Target is assumed to already be on the log scale.
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=random_state,
    )

    scores = cross_val_score(
        rf, train_X, train_y,
        cv=cv_folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    rmse = -scores
    print(f"CV RMSE (log-price): {rmse.mean():.4f} ± {rmse.std():.4f}")
    print(f"Per-fold:            {np.round(rmse, 4).tolist()}")

    rf.fit(train_X, train_y)
    return rf
