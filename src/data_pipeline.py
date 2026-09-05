"""Telemetry Preprocessing & Stratified Causal Data Pipeline.

Handles feature transformations, log-scaling for skewed metrics, one-hot categorical
encoding, and leakage-free stratified splitting across joint (W, Y) distributions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class CausalSplitDataset:
    """Container for processed training and evaluation datasets in causal uplift modeling."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    w_train: np.ndarray
    w_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    latent_train: pd.DataFrame | None = None
    latent_test: pd.DataFrame | None = None
    feature_names: list[str] | None = None


class TelemetryDataPipeline(BaseEstimator, TransformerMixin):
    """Preprocessing transformer for B2B SaaS telemetry data.

    Applies log1p transforms to skewed revenue and ticket distributions,
    standardizes continuous metrics, and one-hot encodes categorical dimensions.
    Guarantees zero data leakage by computing all statistics strictly on the
    training set.
    """

    NUMERIC_COLS: ClassVar[list[str]] = [
        "account_age_months",
        "monthly_recurring_revenue",
        "active_users_ratio",
        "login_frequency_trend_30d",
        "feature_usage_diversity",
        "support_tickets_90d",
        "unresolved_p1_tickets",
        "csat_score",
        "payment_failure_events",
    ]

    LOG_TRANSFORM_COLS: ClassVar[list[str]] = [
        "monthly_recurring_revenue",
        "support_tickets_90d",
    ]

    CATEGORICAL_COLS: ClassVar[list[str]] = [
        "plan_tier",
        "contract_type",
    ]

    TREATMENT_COL: ClassVar[str] = "treatment_assigned"
    OUTCOME_COL: ClassVar[str] = "retained"
    ID_COL: ClassVar[str] = "user_id"

    LATENT_COLS: ClassVar[list[str]] = [
        "y_control_latent",
        "y_treated_latent",
        "tau_true",
        "archetype",
    ]

    def __init__(self, scale_numeric: bool = True) -> None:
        self.scale_numeric = scale_numeric
        self.encoder_: OneHotEncoder | None = None
        self.scaler_: StandardScaler | None = None
        self.encoded_cat_names_: list[str] = []
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TelemetryDataPipeline:
        """Fit encoding and scaling transformers strictly on training data."""
        X_df = X.copy()

        # 1. Fit categorical One-Hot Encoder
        self.encoder_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.encoder_.fit(X_df[self.CATEGORICAL_COLS])
        self.encoded_cat_names_ = self.encoder_.get_feature_names_out(self.CATEGORICAL_COLS).tolist()

        # 2. Compute numeric features with log-transforms applied
        num_features = self._transform_numerics(X_df)

        # 3. Fit scaler if enabled
        if self.scale_numeric:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(num_features)

        self.feature_names_ = self.NUMERIC_COLS + self.encoded_cat_names_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform raw covariates using fitted transformers."""
        if self.encoder_ is None:
            raise RuntimeError("Pipeline must be fitted before transforming data.")

        X_df = X.copy()

        # 1. Numeric transformation & scaling
        num_features = self._transform_numerics(X_df)
        if self.scale_numeric and self.scaler_ is not None:
            num_scaled = self.scaler_.transform(num_features)
        else:
            num_scaled = num_features.values

        num_df = pd.DataFrame(
            num_scaled,
            columns=self.NUMERIC_COLS,
            index=X_df.index,
        )

        # 2. Categorical transformation
        cat_encoded = self.encoder_.transform(X_df[self.CATEGORICAL_COLS])
        cat_df = pd.DataFrame(
            cat_encoded,
            columns=self.encoded_cat_names_,
            index=X_df.index,
        )

        # 3. Combine into final feature matrix
        transformed_df = pd.concat([num_df, cat_df], axis=1)
        return transformed_df[self.feature_names_]

    def _transform_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply safe log1p transformations to skewed numeric columns."""
        num_df = df[self.NUMERIC_COLS].copy()
        for col in self.LOG_TRANSFORM_COLS:
            if col in num_df.columns:
                num_df[col] = np.log1p(np.maximum(num_df[col].values, 0.0))
        return num_df


def load_telemetry(file_path: str | Path) -> pd.DataFrame:
    """Load telemetry dataset from CSV."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Telemetry dataset not found at: {path}")
    logger.info("Loading telemetry dataset from %s", path)
    return pd.read_csv(path)


def split_causal_dataset(
    df: pd.DataFrame,
    test_size: float = 0.30,
    random_state: int = 42,
    scale_numeric: bool = True,
) -> CausalSplitDataset:
    """Perform leakage-free stratified split on joint (W, Y) and apply feature pipeline.

    Stratification on joint key (2 * W + Y) guarantees exact representation
    of treatment/control assignment and factual retention rates across train
    and held-out evaluation sets.

    Parameters
    ----------
    df : pd.DataFrame
        Raw telemetry dataframe.
    test_size : float, default=0.30
        Fraction of data to allocate to held-out test split.
    random_state : int, default=42
        Seed for reproducibility.
    scale_numeric : bool, default=True
        Whether to standardize continuous covariates.

    Returns
    -------
    CausalSplitDataset
        Clean, preprocessed feature matrices and target/treatment vectors.
    """
    treatment_col = TelemetryDataPipeline.TREATMENT_COL
    outcome_col = TelemetryDataPipeline.OUTCOME_COL
    latent_cols = [c for c in TelemetryDataPipeline.LATENT_COLS if c in df.columns]

    # Create joint stratification key: 2*W + Y in {0, 1, 2, 3}
    stratify_key = 2 * df[treatment_col].values + df[outcome_col].values

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_key,
    )

    # Initialize and fit pipeline STRICTLY on train_df
    pipeline = TelemetryDataPipeline(scale_numeric=scale_numeric)
    pipeline.fit(train_df)

    X_train = pipeline.transform(train_df)
    X_test = pipeline.transform(test_df)

    w_train = train_df[treatment_col].to_numpy(dtype=int)
    w_test = test_df[treatment_col].to_numpy(dtype=int)

    y_train = train_df[outcome_col].to_numpy(dtype=int)
    y_test = test_df[outcome_col].to_numpy(dtype=int)

    latent_train = train_df[latent_cols].copy().reset_index(drop=True) if latent_cols else None
    latent_test = test_df[latent_cols].copy().reset_index(drop=True) if latent_cols else None

    logger.info(
        "Causal split complete: Train shape=%s, Test shape=%s | Features=%d",
        X_train.shape,
        X_test.shape,
        len(pipeline.feature_names_),
    )

    return CausalSplitDataset(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        w_train=w_train,
        w_test=w_test,
        y_train=y_train,
        y_test=y_test,
        latent_train=latent_train,
        latent_test=latent_test,
        feature_names=pipeline.feature_names_,
    )
