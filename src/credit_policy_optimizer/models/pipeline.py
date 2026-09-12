"""Credit risk modeling pipeline with reproducible preprocessing and LightGBM classifier."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from polars.datatypes import DataTypeClass
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# Canonical feature column names
DEFAULT_NUMERIC_FEATURES: list[str] = [
    "monthly_income",
    "debt_to_income",
    "historical_delinquencies",
    "revolving_utilization",
    "loan_amount",
    "interest_rate",
    "cost_of_funds",
]

DEFAULT_CATEGORICAL_FEATURES: list[str] = [
    "loan_term_months",
]

COLUMN_DTYPES: dict[str, DataTypeClass | pl.DataType] = {
    "monthly_income": pl.Float64,
    "debt_to_income": pl.Float64,
    "historical_delinquencies": pl.Float64,
    "revolving_utilization": pl.Float64,
    "loan_amount": pl.Float64,
    "loan_term_months": pl.Int64,
    "interest_rate": pl.Float64,
    "cost_of_funds": pl.Float64,
}


class DataFrameAligner(BaseEstimator, TransformerMixin):
    """Transformer that normalizes inputs and guarantees feature column alignment.

    Accepts Polars DataFrames, dictionaries, lists of dictionaries, Pydantic models,
    or NumPy arrays. Fills any missing expected columns with null/NaN to allow
    downstream imputers to handle them gracefully.
    """

    def __init__(self, expected_columns: Sequence[str] | None = None) -> None:
        """Initialize aligner with optional list of expected feature names."""
        self.expected_columns = list(expected_columns) if expected_columns is not None else None

    def fit(self, X: Any, y: Any = None) -> "DataFrameAligner":
        """Record expected columns from training data if not explicitly provided."""
        if self.expected_columns is None:
            if isinstance(X, pl.DataFrame):
                self.expected_columns = list(X.columns)
            elif isinstance(X, dict):
                self.expected_columns = list(X.keys())
            elif isinstance(X, Sequence) and len(X) > 0:
                first = X[0]
                if hasattr(first, "model_dump"):
                    self.expected_columns = list(first.model_dump().keys())
                elif isinstance(first, dict):
                    self.expected_columns = list(first.keys())
                else:
                    msg = (
                        f"Cannot infer expected columns from sequence element of type {type(first)}"
                    )
                    raise TypeError(msg)
            elif hasattr(X, "columns"):
                self.expected_columns = list(X.columns)
            else:
                msg = f"Cannot infer expected columns from input of type {type(X)}"
                raise TypeError(msg)
        return self

    def transform(self, X: Any) -> pl.DataFrame:
        """Align input into a Polars DataFrame matching expected columns."""
        if isinstance(X, pl.DataFrame):
            df = X.clone()
        elif hasattr(X, "model_dump"):
            # Single Pydantic model
            df = pl.DataFrame([X.model_dump()])
        elif isinstance(X, dict):
            # Single dictionary: wrap in list to form 1-row DataFrame
            df = pl.DataFrame([X])
        elif isinstance(X, Sequence) and len(X) > 0 and hasattr(X[0], "model_dump"):
            # List of Pydantic models
            df = pl.DataFrame([item.model_dump() for item in X])
        elif isinstance(X, Sequence) and len(X) > 0 and isinstance(X[0], dict):
            # List of dicts
            df = pl.DataFrame(list(X))
        elif hasattr(X, "to_dict"):
            # Pandas DataFrame or Series fallback
            try:
                df = pl.DataFrame(X.to_dict("list"))
            except Exception:
                df = pl.from_pandas(X)
        elif isinstance(X, np.ndarray):
            if self.expected_columns is None:
                msg = "Expected columns must be defined when passing a numpy array."
                raise ValueError(msg)
            df = pl.DataFrame(X, schema=self.expected_columns)
        else:
            msg = f"Unsupported input type for DataFrameAligner: {type(X)}"
            raise TypeError(msg)

        if self.expected_columns is not None:
            # Ensure every expected column is present and safely cast
            exprs = []
            for col in self.expected_columns:
                target_dtype = COLUMN_DTYPES.get(col, pl.Float64)
                if col not in df.columns:
                    exprs.append(pl.lit(None).cast(target_dtype).alias(col))
                else:
                    exprs.append(pl.col(col).cast(target_dtype, strict=False).alias(col))
            df = df.with_columns(exprs).select(self.expected_columns)

        return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain-specific feature engineering transformer for credit risk applications."""

    def fit(self, X: Any, y: Any = None) -> "FeatureEngineer":
        """Fit transformer (stateless)."""
        return self

    def transform(self, X: Any) -> pl.DataFrame:
        """Derive financial risk indicators."""
        if not isinstance(X, pl.DataFrame):
            df = pl.DataFrame(X)
        else:
            df = X.clone()

        exprs = []

        # 1. Ratio: loan amount to monthly income (handles non-positive income safely)
        if "loan_amount" in df.columns and "monthly_income" in df.columns:
            exprs.append(
                pl.when((pl.col("monthly_income") > 0) & pl.col("monthly_income").is_not_null())
                .then(pl.col("loan_amount") / pl.col("monthly_income"))
                .otherwise(None)
                .cast(pl.Float64)
                .alias("loan_to_income")
            )
        else:
            exprs.append(pl.lit(None).cast(pl.Float64).alias("loan_to_income"))

        # 2. Net Interest Spread (active interest rate - cost of funds)
        if "interest_rate" in df.columns and "cost_of_funds" in df.columns:
            exprs.append(
                (pl.col("interest_rate") - pl.col("cost_of_funds"))
                .cast(pl.Float64)
                .alias("net_interest_spread")
            )
        else:
            exprs.append(pl.lit(None).cast(pl.Float64).alias("net_interest_spread"))

        # 3. Estimated installment-to-income burden
        if (
            "loan_amount" in df.columns
            and "loan_term_months" in df.columns
            and "monthly_income" in df.columns
        ):
            exprs.append(
                pl.when(
                    (pl.col("loan_term_months") > 0)
                    & (pl.col("monthly_income") > 0)
                    & pl.col("loan_term_months").is_not_null()
                    & pl.col("monthly_income").is_not_null()
                )
                .then(
                    (pl.col("loan_amount") / pl.col("loan_term_months")) / pl.col("monthly_income")
                )
                .otherwise(None)
                .cast(pl.Float64)
                .alias("installment_to_income")
            )
        else:
            exprs.append(pl.lit(None).cast(pl.Float64).alias("installment_to_income"))

        return df.with_columns(exprs)


def build_preprocessor(
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
) -> ColumnTransformer:
    """Build a reproducible ColumnTransformer for numerical and categorical features.

    Parameters
    ----------
    numeric_features : Sequence[str] | None
        Names of continuous and discrete numeric features.
    categorical_features : Sequence[str] | None
        Names of categorical features.

    Returns
    -------
    ColumnTransformer
        Configured scikit-learn transformer.
    """
    num_cols = list(numeric_features) if numeric_features is not None else DEFAULT_NUMERIC_FEATURES
    cat_cols = (
        list(categorical_features)
        if categorical_features is not None
        else DEFAULT_CATEGORICAL_FEATURES
    )

    # Derived numeric features generated by FeatureEngineer
    derived_numeric = ["loan_to_income", "net_interest_spread", "installment_to_income"]
    all_numeric = num_cols + derived_numeric

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, all_numeric),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )


@dataclass
class LightGBMConfig:
    """Hyperparameter configuration for LightGBM classifier optimized for log-loss."""

    objective: str = "binary"
    metric: str = "binary_logloss"
    learning_rate: float = 0.05
    n_estimators: int = 150
    num_leaves: int = 31
    max_depth: int = 5
    min_child_samples: int = 25
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_alpha: float = 0.01
    reg_lambda: float = 0.1
    random_state: int = 42
    verbosity: int = -1
    extra_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary passed to LGBMClassifier."""
        params: dict[str, Any] = {
            "objective": self.objective,
            "metric": self.metric,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "verbosity": self.verbosity,
        }
        params.update(self.extra_params)
        return params


def build_credit_pipeline(
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    model_config: LightGBMConfig | None = None,
) -> Pipeline:
    """Build an end-to-end reproducible credit risk estimation Pipeline.

    Parameters
    ----------
    numeric_features : Sequence[str] | None
        List of numeric column names.
    categorical_features : Sequence[str] | None
        List of categorical column names.
    model_config : LightGBMConfig | None
        Hyperparameter configuration for LightGBM.

    Returns
    -------
    Pipeline
        Unfitted Scikit-Learn Pipeline ready to train.
    """
    num_cols = list(numeric_features) if numeric_features is not None else DEFAULT_NUMERIC_FEATURES
    cat_cols = (
        list(categorical_features)
        if categorical_features is not None
        else DEFAULT_CATEGORICAL_FEATURES
    )
    expected_cols = num_cols + cat_cols

    config = model_config or LightGBMConfig()
    classifier = lgb.LGBMClassifier(**config.to_dict())

    preprocessor = build_preprocessor(
        numeric_features=num_cols,
        categorical_features=cat_cols,
    )

    return Pipeline(
        steps=[
            ("aligner", DataFrameAligner(expected_columns=expected_cols)),
            ("engineer", FeatureEngineer()),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_credit_pipeline(
    X_train: Any,
    y_train: Any,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    model_config: LightGBMConfig | None = None,
) -> Pipeline:
    """Instantiate and fit the credit risk pipeline on training data.

    Parameters
    ----------
    X_train : Any
        Training features (Polars DataFrame, Pandas DataFrame, or sequence of dicts).
    y_train : Any
        Binary default targets (0 or 1).
    numeric_features : Sequence[str] | None
        Numeric feature names.
    categorical_features : Sequence[str] | None
        Categorical feature names.
    model_config : LightGBMConfig | None
        LightGBM configuration.

    Returns
    -------
    Pipeline
        Fitted Scikit-Learn Pipeline.
    """
    pipeline = build_credit_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_config=model_config,
    )

    # Convert y_train to 1D numpy array
    if isinstance(y_train, pl.Series):
        y_arr = y_train.to_numpy()
    elif hasattr(y_train, "to_numpy"):
        y_arr = y_train.to_numpy()
    else:
        y_arr = np.asarray(y_train)

    pipeline.fit(X_train, y_arr)
    return pipeline
