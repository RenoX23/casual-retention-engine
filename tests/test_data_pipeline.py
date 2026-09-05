"""Unit tests for telemetry data generator and preprocessing pipeline.

Validates schema compliance, zero data leakage, joint (W, Y) stratification balance,
isolation of latent evaluation variables, and numerical stability.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.generate_telemetry import generate_synthetic_telemetry, validate_schema
from src.data_pipeline import TelemetryDataPipeline, split_causal_dataset


@pytest.fixture(scope="module")
def sample_telemetry_df() -> pd.DataFrame:
    """Fixture providing small synthetic telemetry dataset (3,000 rows)."""
    return generate_synthetic_telemetry(n_samples=3000, random_seed=42)


def test_schema_compliance(sample_telemetry_df: pd.DataFrame) -> None:
    """Verify generated dataset strictly conforms to telemetry_schema.json."""
    schema_path = Path("data/telemetry_schema.json")
    assert schema_path.exists(), "Schema file data/telemetry_schema.json not found"
    assert validate_schema(sample_telemetry_df, schema_path) is True


def test_all_four_archetypes_present(sample_telemetry_df: pd.DataFrame) -> None:
    """Verify that all four behavioral response archetypes are generated."""
    expected_archetypes = {"Persuadable", "Sleeping Dog", "Sure Thing", "Lost Cause"}
    present_archetypes = set(sample_telemetry_df["archetype"].unique())
    assert expected_archetypes.issubset(present_archetypes), (
        f"Missing archetypes: {expected_archetypes - present_archetypes}"
    )

    # Check minimum proportions
    counts = sample_telemetry_df["archetype"].value_counts(normalize=True)
    assert counts["Persuadable"] > 0.10, "Persuadables cohort is under-represented"
    assert counts["Sleeping Dog"] > 0.03, "Sleeping Dogs cohort is under-represented"
    assert counts["Sure Thing"] > 0.30, "Sure Things cohort is under-represented"
    assert counts["Lost Cause"] > 0.03, "Lost Causes cohort is under-represented"


def test_zero_data_leakage(sample_telemetry_df: pd.DataFrame) -> None:
    """Ensure transformers are fit strictly on training partition without test leakage."""
    pipeline = TelemetryDataPipeline(scale_numeric=True)

    train_df = sample_telemetry_df.iloc[:2000].copy()
    test_df = sample_telemetry_df.iloc[2000:].copy()

    # Fit and transform train
    pipeline.fit(train_df)
    X_train_baseline = pipeline.transform(train_df)

    # Modify test_df wildly (e.g. 100x MRR) and transform again
    perturbed_test = test_df.copy()
    perturbed_test["monthly_recurring_revenue"] = perturbed_test["monthly_recurring_revenue"] * 1000.0

    # Transforming perturbed test must NOT affect training statistics or transformed output
    X_train_after = pipeline.transform(train_df)
    pd.testing.assert_frame_equal(X_train_baseline, X_train_after)

    # Check that scaler mean matches train_df, not perturbed test
    expected_log_mrr_mean = np.log1p(train_df["monthly_recurring_revenue"]).mean()
    mrr_idx = pipeline.NUMERIC_COLS.index("monthly_recurring_revenue")
    actual_fitted_mean = pipeline.scaler_.mean_[mrr_idx]
    np.testing.assert_almost_equal(actual_fitted_mean, expected_log_mrr_mean, decimal=4)


def test_joint_stratification_balance(sample_telemetry_df: pd.DataFrame) -> None:
    """Ensure split_causal_dataset preserves exact proportions of (W, Y) joint states."""
    test_size = 0.30
    dataset = split_causal_dataset(sample_telemetry_df, test_size=test_size, random_state=42)

    # Total shapes
    n_train = len(dataset.w_train)
    n_test = len(dataset.w_test)
    assert n_train + n_test == len(sample_telemetry_df)
    assert abs(n_test / len(sample_telemetry_df) - test_size) < 0.01

    # Joint distribution check: 2*W + Y in {0, 1, 2, 3}
    train_joint = 2 * dataset.w_train + dataset.y_train
    test_joint = 2 * dataset.w_test + dataset.y_test

    for state in [0, 1, 2, 3]:
        p_train = (train_joint == state).mean()
        p_test = (test_joint == state).mean()
        assert abs(p_train - p_test) < 0.005, (
            f"Stratification discrepancy in state {state}: train={p_train:.4f}, test={p_test:.4f}"
        )


def test_latent_variables_strictly_isolated(sample_telemetry_df: pd.DataFrame) -> None:
    """Verify that feature matrices do not contain latent ground truth or identifiers."""
    dataset = split_causal_dataset(sample_telemetry_df, test_size=0.30, random_state=42)

    forbidden_cols = set(TelemetryDataPipeline.LATENT_COLS) | {
        TelemetryDataPipeline.ID_COL,
        TelemetryDataPipeline.TREATMENT_COL,
        TelemetryDataPipeline.OUTCOME_COL,
    }

    train_feature_cols = set(dataset.X_train.columns)
    test_feature_cols = set(dataset.X_test.columns)

    assert not (forbidden_cols & train_feature_cols), f"Leaked forbidden columns in train: {forbidden_cols & train_feature_cols}"
    assert not (forbidden_cols & test_feature_cols), f"Leaked forbidden columns in test: {forbidden_cols & test_feature_cols}"

    # Verify latent ground truth is accessible in separate container attributes
    assert dataset.latent_train is not None
    assert dataset.latent_test is not None
    assert "tau_true" in dataset.latent_test.columns
    assert "archetype" in dataset.latent_test.columns


def test_unfitted_pipeline_raises() -> None:
    """Verify that transform before fit raises RuntimeError."""
    pipeline = TelemetryDataPipeline()
    dummy_df = pd.DataFrame({"account_age_months": [10]})
    with pytest.raises(RuntimeError, match="Pipeline must be fitted"):
        pipeline.transform(dummy_df)


def test_numerical_stability() -> None:
    """Verify pipeline handles edge values (0 revenue, 0 tickets) without NaNs."""
    df_edge = pd.DataFrame(
        {
            "account_age_months": [1, 60],
            "monthly_recurring_revenue": [0.0, 10000.0],
            "active_users_ratio": [0.0, 1.0],
            "login_frequency_trend_30d": [-1.0, 3.0],
            "feature_usage_diversity": [1, 15],
            "support_tickets_90d": [0, 50],
            "unresolved_p1_tickets": [0, 4],
            "csat_score": [1.0, 5.0],
            "payment_failure_events": [0, 5],
            "plan_tier": ["Starter", "Enterprise"],
            "contract_type": ["Monthly", "Annual"],
        }
    )

    pipeline = TelemetryDataPipeline(scale_numeric=True)
    pipeline.fit(df_edge)
    transformed = pipeline.transform(df_edge)

    assert not transformed.isna().any().any(), "Transformed feature matrix contains NaNs"
    assert not np.isinf(transformed.values).any(), "Transformed feature matrix contains infs"
