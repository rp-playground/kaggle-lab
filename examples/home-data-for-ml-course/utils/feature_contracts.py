"""Level-1 contract validator for the Ames FE pipeline.

Each FE function (or stateful transformer) optionally declares a
FeatureContract — sets of `requires` / `produces` / `drops` columns,
plus conditional drops keyed on a kwarg (e.g. `drop_originals=True`).

`validate_pipeline()` symbolically walks an sklearn Pipeline and checks:
  - every step's `requires` columns are present at that point
  - no step `produces` a column name that already exists
  - every step's effective `drops` are present (else they're a no-op signal)

Steps without an attached contract are skipped silently, so the validator
works incrementally — contract a few functions today, the rest later, and
coverage grows monotonically.

Usage from a notebook:

    from utils.feature_contracts import validate_pipeline
    issues = validate_pipeline(xgb_pipe, X.columns, strict=False)
    for i in issues:
        print(i)

Also see `find_producer(col, pipe)` for "does this feature already exist?"
spot checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional


@dataclass(frozen=True)
class FeatureContract:
    """What a step needs, what it adds, what it removes.

    `conditional_drops` is keyed on (kwarg_name, kwarg_value) tuples. At
    validation time the runtime kw_args (read from FunctionTransformer
    or instance attributes) are matched against these keys to decide
    which drops fire.

    Example — `add_size_features` drops three SF columns only when called
    with `drop_originals=True`:

        conditional_drops={
            ("drop_originals", True): {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"},
        }
    """

    requires: frozenset[str] = frozenset()
    produces: frozenset[str] = frozenset()
    drops:    frozenset[str] = frozenset()
    conditional_drops: dict[tuple[str, Any], frozenset[str]] = field(default_factory=dict)


def feature_contract(
    *,
    requires: Iterable[str] = (),
    produces: Iterable[str] = (),
    drops:    Iterable[str] = (),
    conditional_drops: Optional[dict[tuple[str, Any], Iterable[str]]] = None,
) -> Callable[[Callable], Callable]:
    """Attach a FeatureContract to a feature function. Metadata only —
    never modifies function behavior. The contract is stored on the
    function object as `.contract` and is read at validation time.
    """
    cd = {k: frozenset(v) for k, v in (conditional_drops or {}).items()}
    contract = FeatureContract(
        requires=frozenset(requires),
        produces=frozenset(produces),
        drops=frozenset(drops),
        conditional_drops=cd,
    )

    def deco(fn: Callable) -> Callable:
        fn.contract = contract  # type: ignore[attr-defined]
        return fn

    return deco


def _resolve_drops(contract: FeatureContract, kw_args: dict) -> frozenset[str]:
    eff = set(contract.drops)
    for (key, val), cols in contract.conditional_drops.items():
        if kw_args.get(key) == val:
            eff |= cols
    return frozenset(eff)


def _step_contract(step) -> Optional[tuple[FeatureContract, dict]]:
    """Extract (contract, runtime_kw_args) from a pipeline step, or None."""
    from sklearn.preprocessing import FunctionTransformer

    if isinstance(step, FunctionTransformer):
        contract = getattr(step.func, "contract", None)
        if contract is None:
            return None
        return contract, dict(step.kw_args or {})

    # Generic estimator — contract may live as a class or instance attribute.
    contract = getattr(step, "contract", None)
    if contract is None:
        return None
    # Pull kw-like state off the instance for conditional-drop matching.
    kw_args: dict = {}
    for (key, _val) in contract.conditional_drops:
        if hasattr(step, key):
            kw_args[key] = getattr(step, key)
    return contract, kw_args


def _iter_steps(pipe):
    """Flatten nested Pipelines into a (name, step) generator."""
    if not hasattr(pipe, "steps"):
        return
    for name, step in pipe.steps:
        if hasattr(step, "steps"):
            yield from _iter_steps(step)
        else:
            yield name, step


def validate_pipeline(
    pipe,
    initial_columns: Iterable[str],
    *,
    strict: bool = True,
) -> list[str]:
    """Walk `pipe` step-by-step and check each contract against the
    evolving column set. Returns issues (empty list = pass).

    If `strict=True` (default), raises ValueError on any issue. Steps
    without an attached contract are skipped silently.
    """
    cols: set[str] = set(initial_columns)
    seen_producers: dict[str, str] = {}
    issues: list[str] = []

    for name, step in _iter_steps(pipe):
        info = _step_contract(step)
        if info is None:
            continue
        contract, kw_args = info

        missing = contract.requires - cols
        if missing:
            issues.append(f"[{name}] missing required columns: {sorted(missing)}")

        collisions = contract.produces & cols
        if collisions:
            for c in sorted(collisions):
                prior = seen_producers.get(c, "<input>")
                issues.append(
                    f"[{name}] would overwrite existing column {c!r} "
                    f"(previously produced by [{prior}])"
                )

        effective_drops = _resolve_drops(contract, kw_args)
        ghost_drops = effective_drops - cols
        if ghost_drops:
            issues.append(
                f"[{name}] declares drop of columns not present: {sorted(ghost_drops)}"
            )

        cols = (cols - effective_drops) | contract.produces
        for c in contract.produces:
            seen_producers[c] = name

    if issues and strict:
        raise ValueError(
            "Pipeline contract validation failed:\n  - " + "\n  - ".join(issues)
        )
    return issues


def simulate_columns(pipe, initial_columns: Iterable[str]) -> set[str]:
    """Replay the symbolic walk and return the final column set."""
    cols: set[str] = set(initial_columns)
    for _name, step in _iter_steps(pipe):
        info = _step_contract(step)
        if info is None:
            continue
        contract, kw_args = info
        cols = (cols - _resolve_drops(contract, kw_args)) | contract.produces
    return cols


def find_producer(col: str, pipe) -> Optional[str]:
    """Return the step-name that produces `col`, or None.

    Spot-check helper for "does this feature already exist somewhere in
    the pipeline?" before adding a duplicate.
    """
    for name, step in _iter_steps(pipe):
        info = _step_contract(step)
        if info is None:
            continue
        contract, _ = info
        if col in contract.produces:
            return name
    return None


if __name__ == "__main__":
    # Symbolic dry-run of the best-run pipeline (no model, no data) — prints
    # any contract issues and the resulting feature set.
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    from utils.ames_feature_engineering import (
        add_size_features, add_bath_features, add_porch_features,
        add_temporal_features, add_quality_size_features,
        add_presence_indicators, add_ratio_features, add_condition_features,
    )

    # Raw Ames columns post-AmesNAImputer (target/Id already dropped upstream).
    initial_cols = {
        "MSSubClass", "MSZoning", "LotFrontage", "LotArea", "Street", "Alley",
        "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope",
        "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle",
        "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "RoofStyle",
        "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "MasVnrArea",
        "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond",
        "BsmtExposure", "BsmtFinType1", "BsmtFinSF1", "BsmtFinType2",
        "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "Heating", "HeatingQC",
        "CentralAir", "Electrical", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
        "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath",
        "BedroomAbvGr", "KitchenAbvGr", "KitchenQual", "TotRmsAbvGrd",
        "Functional", "Fireplaces", "FireplaceQu", "GarageType", "GarageYrBlt",
        "GarageFinish", "GarageCars", "GarageArea", "GarageQual", "GarageCond",
        "PavedDrive", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
        "ScreenPorch", "PoolArea", "PoolQC", "Fence", "MiscFeature", "MiscVal",
        "MoSold", "YrSold", "SaleType", "SaleCondition",
    }

    pipe = Pipeline([
        ("quality_size", FunctionTransformer(add_quality_size_features, kw_args={"drop_originals": False})),
        ("bath",         FunctionTransformer(add_bath_features,         kw_args={"drop_originals": False})),
        ("presence",     FunctionTransformer(add_presence_indicators)),
        ("porch",        FunctionTransformer(add_porch_features,        kw_args={"drop_originals": False})),
        ("ratio",        FunctionTransformer(add_ratio_features)),
        ("size",         FunctionTransformer(add_size_features,         kw_args={"drop_originals": True})),
        ("fe",           FunctionTransformer(add_temporal_features,     kw_args={"drop_originals": True})),
        ("condition",    FunctionTransformer(add_condition_features,    kw_args={"drop_originals": True})),
    ])

    issues = validate_pipeline(pipe, initial_cols, strict=False)
    if issues:
        print("CONTRACT ISSUES:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("OK — contract validation passed for the best-run pipeline")

    final = simulate_columns(pipe, initial_cols)
    added   = sorted(final - initial_cols)
    dropped = sorted(initial_cols - final)
    print(f"\nFinal column count: {len(final)}  (was {len(initial_cols)})")
    print(f"Added   ({len(added):2d}): {added}")
    print(f"Dropped ({len(dropped):2d}): {dropped}")

    # Sanity check on the spot-check helper.
    print(f"\nWho produces TotalSF?     -> {find_producer('TotalSF', pipe)}")
    print(f"Who produces TotalPorchSF? -> {find_producer('TotalPorchSF', pipe)}")
    print(f"Who produces NoSuchThing?  -> {find_producer('NoSuchThing', pipe)}")
