"""
Sklearn-compatible Pipeline for the Ames Housing baseline.

Wraps the existing fit/transform functions from
  - ames_na_encoding (fit_imputers / transform)
  - ames_random_forest_baseline (fit_encoders / transform)
into BaseEstimator+TransformerMixin classes so they can be composed in a
sklearn Pipeline and refit per CV fold (no preprocessing leakage).

The functional API in those modules is unchanged.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, TargetEncoder

from . import ames_na_encoding
from . import ames_random_forest_baseline as ames_rf
from .ames_feature_engineering import add_temporal_features


class AmesNAImputer(BaseEstimator, TransformerMixin):
    """Sklearn wrapper around ames_na_encoding.fit_imputers / transform.

    Fits all training-derived imputers (LotFrontage neighborhood-median,
    Electrical mode, simple modes, MSZoning conditional mode) on the
    fold's training portion, then applies the same parameters at transform.
    """

    def fit(self, X, y=None):
        self.params_ = ames_na_encoding.fit_imputers(X)
        return self

    def transform(self, X):
        return ames_na_encoding.transform(X, self.params_)


class AmesEncoder(BaseEstimator, TransformerMixin):
    """Sklearn wrapper around ames_random_forest_baseline.fit_encoders / transform.

    Fits the OneHotEncoder vocabulary plus the stateless quality/CentralAir
    encodings on the fold's training portion. Returns a dense numeric DataFrame
    ready for any sklearn regressor.
    """

    def fit(self, X, y=None):
        self.encoders_ = ames_rf.fit_encoders(X)
        return self

    def transform(self, X):
        return ames_rf.transform(X, self.encoders_)


class TargetEncodeColumn(BaseEstimator, TransformerMixin):
    """Replace a single high-cardinality categorical column with sklearn's
    TargetEncoder output, in place.

    Wraps sklearn.preprocessing.TargetEncoder so we can sit inside a
    DataFrame-shaped Pipeline without using ColumnTransformer (which would
    reorder columns and prefix names). The wrapper preserves the input
    DataFrame layout: same columns, same order; only the named column's
    dtype changes from string/category to float.

    fit_transform is overridden to call sklearn TargetEncoder's fit_transform,
    which uses internal cross-validation to produce out-of-fold encoding.
    Skipping that override would route through TransformerMixin's default
    fit().transform() and silently disable the CV protection.
    """

    def __init__(self, column: str = "Neighborhood",
                 target_type: str = "continuous"):
        self.column = column
        self.target_type = target_type

    def fit(self, X, y):
        self.encoder_ = TargetEncoder(target_type=self.target_type)
        self.encoder_.fit(X[[self.column]], y)
        return self

    def transform(self, X):
        out = X.copy()
        out[self.column] = self.encoder_.transform(X[[self.column]]).ravel()
        return out

    def fit_transform(self, X, y):
        self.encoder_ = TargetEncoder(target_type=self.target_type)
        encoded = self.encoder_.fit_transform(X[[self.column]], y)
        out = X.copy()
        out[self.column] = encoded.ravel()
        return out


class GaragePLSTransformer(BaseEstimator, TransformerMixin):
    """Compress the garage feature block into a few PLS components.

    Selects every column whose name starts with ``Garage`` (the post-encoder
    block: ordinal qualities, numerics, and one-hot ``GarageType_*`` dummies),
    appends an internal ``HasGarage`` flag derived from ``GarageArea``, scales
    the block with StandardScaler, then fits PLSRegression supervised on the
    target. At transform time appends ``GaragePLS{1..n}`` columns. The
    StandardScaler / PLS state is fitted per fold by sklearn's Pipeline, so no
    leakage.

    Mirrors the design from experiments/eda-garage.ipynb: PLS captures the
    dominant garage-size+quality signal in a small number of components,
    letting downstream tree models split on a denser representation. The
    ``HasGarage`` flag is internal to the transformer — it is used as input to
    PLS but not added to the output frame, since its information is already
    folded into the components.

    Parameters
    ----------
    n_components : int
        Number of PLS components to retain.
    drop_originals : bool
        If True, drop every ``Garage*`` column from the output frame after
        appending the components. Use when the components are intended to
        replace the original block (true dimensionality reduction).
    """

    def __init__(self, n_components: int = 3, drop_originals: bool = False):
        self.n_components = n_components
        self.drop_originals = drop_originals

    def _build_block(self, X: pd.DataFrame) -> pd.DataFrame:
        garage_cols = [c for c in X.columns if c.startswith("Garage")]
        block = X[garage_cols].copy()
        block["HasGarage"] = (X["GarageArea"] > 0).astype("int8")
        return block

    def fit(self, X, y):
        block = self._build_block(X)
        self.garage_cols_ = [c for c in block.columns if c != "HasGarage"]
        self.feature_names_ = block.columns.tolist()
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(block)
        self.pls_ = PLSRegression(n_components=self.n_components)
        self.pls_.fit(scaled, y)
        return self

    def transform(self, X):
        block = self._build_block(X)
        # Align to the fitted feature set: any one-hot column unseen in this
        # split (rare GarageType levels) is filled with 0, matching how
        # OneHotEncoder(handle_unknown='ignore') would treat it.
        block = block.reindex(columns=self.feature_names_, fill_value=0)
        scaled = self.scaler_.transform(block)
        components = self.pls_.transform(scaled)
        comp_cols = [f"GaragePLS{i+1}" for i in range(self.n_components)]
        comp_df = pd.DataFrame(components, columns=comp_cols, index=X.index)
        out = pd.concat([X, comp_df], axis=1)
        if self.drop_originals:
            out = out.drop(columns=self.garage_cols_)
        return out


def build_pipeline(model) -> Pipeline:
    """Compose the full Ames preprocessing + model pipeline.

    Steps:
      1. AmesNAImputer        — learned NA imputation
      2. add_temporal_features (FunctionTransformer, stateless)
      3. AmesEncoder          — quality/CentralAir/OneHot encoding
      4. model                — any sklearn-compatible regressor

    Parameters
    ----------
    model : estimator
        Any object implementing fit/predict (e.g. RandomForestRegressor).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline that takes raw Ames features (post-CSV-load) and the
        target on whatever scale the caller chose (typically log1p).
    """
    return Pipeline([
        ("na",      AmesNAImputer()),
        ("fe",      FunctionTransformer(
                        add_temporal_features,
                        kw_args={"drop_originals": True},
                    )),
        ("encoder", AmesEncoder()),
        ("model",   model),
    ])
