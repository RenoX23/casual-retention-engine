"""Unit test suite for Causal ML Evaluation Metrics (Qini, AUUC, and Decile Analysis).

Validates:
1. Analytical verification against manual hand-computed toy cohorts.
2. Perfect causal ranking achieving Q_norm = 1.0.
3. Inverted/adversarial ranking yielding Q_norm < 0.0.
4. Theoretical random baseline integration properties (AUUC_random = 0.5 * Q(N)).
5. Strict input validation and boundary condition exception guards.
6. Decile uplift table structure, incremental conversions, and monotonicity testing.
7. Sleeping Dogs detection and isolation in lower deciles.
8. pd.Series vs np.ndarray input interoperability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.generate_telemetry import generate_synthetic_telemetry
from src.data_pipeline import split_causal_dataset
from src.evaluation.decile_analysis import DecileAnalysisResult, compute_decile_analysis
from src.evaluation.qini_metric import compute_qini_curve
from src.models.t_learner import TLearner

# -------------------------------------------------------------------------
# 1. Manual Math & Analytical Invariant Tests
# -------------------------------------------------------------------------

def test_qini_hand_calculated_toy_cohort() -> None:
    """Verify Qini and cumulative gain against hand-calculated 4-sample cohort.

    Data setup:
    Index 0: Y=1, W=1 (Treated responder)   - Pred Uplift = 0.9
    Index 1: Y=0, W=0 (Control non-responder) - Pred Uplift = 0.5
    Index 2: Y=0, W=1 (Treated non-responder) - Pred Uplift = 0.2
    Index 3: Y=1, W=0 (Control responder)     - Pred Uplift = -0.5

    Sorted by Pred Uplift: [0, 1, 2, 3]
    Total N_t = 2, Total N_c = 2 => Q-factor = N_t / N_c = 1.0

    Cutoff k=1 (idx 0): Y_t=1, Y_c=0 => Q(1) = 1.0, Gain(1) = (1/1 - 0) * 1 = 1.0
    Cutoff k=2 (idx 1): Y_t=1, Y_c=0 => Q(2) = 1.0, Gain(2) = (1/1 - 0/1) * 2 = 2.0
    Cutoff k=3 (idx 2): Y_t=1, Y_c=0 => Q(3) = 1.0, Gain(3) = (1/2 - 0/1) * 3 = 1.5
    Cutoff k=4 (idx 3): Y_t=1, Y_c=1 => Q(4) = 1 - 1*1.0 = 0.0, Gain(4) = (1/2 - 1/2) * 4 = 0.0
    """
    y = np.array([1, 0, 0, 1])
    w = np.array([1, 0, 1, 0])
    preds = np.array([0.9, 0.5, 0.2, -0.5])

    res = compute_qini_curve(y, preds, w, n_points=4)

    # Cutoff k=1: n_t=1, n_c=0 => Gain guarded to 0.0 (cannot compute control rate with n_c=0)
    expected_qini = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    expected_gain = np.array([0.0, 0.0, 2.0, 1.5, 0.0])

    np.testing.assert_allclose(res.qini_model, expected_qini, rtol=1e-5)
    np.testing.assert_allclose(res.cum_gain_model, expected_gain, rtol=1e-5)


def test_qini_perfect_ranking_score() -> None:
    """Verify that predictions matching optimal ranking produce Q_norm = 1.0."""
    rng = np.random.default_rng(42)
    n = 200
    w = rng.integers(0, 2, size=n)
    tau_latent = rng.normal(0.2, 0.4, size=n)
    # Binary outcome based on treatment effect
    y = np.where(w == 1, (tau_latent > 0.1).astype(int), (tau_latent < -0.1).astype(int))

    # Perfect prediction: exactly proportional to true latent ordering
    preds_perfect = tau_latent.copy()

    res = compute_qini_curve(y, preds_perfect, w, tau_true=tau_latent, n_points=50)

    # When model equals optimal oracle, score must be exactly 1.0
    assert pytest.approx(res.qini_score, rel=1e-4) == 1.0
    assert pytest.approx(res.auuc_model, rel=1e-4) == res.auuc_optimal


def test_qini_inverted_ranking_negative_score() -> None:
    """Verify that inverted/worst-possible ranking produces negative Q_norm."""
    rng = np.random.default_rng(42)
    n = 200
    w = rng.integers(0, 2, size=n)
    tau_latent = rng.normal(0.2, 0.3, size=n)
    y = np.where(w == 1, (tau_latent > 0).astype(int), 0)

    # Invert the ranking completely (prioritize sleeping dogs / negative uplift)
    preds_inverted = -tau_latent

    res = compute_qini_curve(y, preds_inverted, w, tau_true=tau_latent, n_points=50)
    assert res.qini_score < 0.0
    assert res.auuc_model < res.auuc_random


def test_random_baseline_integration_property() -> None:
    """Verify that AUUC_random satisfies the analytical identity: 0.5 * Q(N)."""
    rng = np.random.default_rng(42)
    n = 300
    w = rng.integers(0, 2, size=n)
    y = rng.integers(0, 2, size=n)
    preds = rng.standard_normal(n)

    res = compute_qini_curve(y, preds, w, n_points=100)

    # Q(N) is the endpoint of the Qini curve
    q_end = res.qini_model[-1]
    expected_random_auuc = 0.5 * q_end

    assert pytest.approx(res.auuc_random, rel=1e-4) == expected_random_auuc


# -------------------------------------------------------------------------
# 2. Boundary Conditions & Exception Guards
# -------------------------------------------------------------------------

def test_qini_input_validation_length_mismatch() -> None:
    """Verify that length mismatches raise ValueError."""
    y = np.array([1, 0, 1])
    w = np.array([1, 0])
    preds = np.array([0.5, 0.2, 0.1])
    with pytest.raises(ValueError, match="Array length mismatch"):
        compute_qini_curve(y, preds, w)


def test_qini_input_validation_non_binary() -> None:
    """Verify that non-binary y or w vectors raise ValueError."""
    y = np.array([1, 0, 2])
    w = np.array([1, 0, 1])
    preds = np.array([0.5, 0.2, 0.1])
    with pytest.raises(ValueError, match="y_true must be strictly binary"):
        compute_qini_curve(y, preds, w)

    y_valid = np.array([1, 0, 1])
    w_invalid = np.array([1, 0, 3])
    with pytest.raises(ValueError, match="treatment must be strictly binary"):
        compute_qini_curve(y_valid, preds, w_invalid)


def test_qini_single_arm_cohort_raises() -> None:
    """Verify that all-treated or all-control cohorts raise ValueError."""
    y = np.array([1, 0, 1, 0])
    preds = np.array([0.5, 0.2, 0.1, 0.0])

    # All treated (N_c = 0)
    w_all_t = np.array([1, 1, 1, 1])
    with pytest.raises(ValueError, match="must contain both treated and control"):
        compute_qini_curve(y, preds, w_all_t)

    # All control (N_t = 0)
    w_all_c = np.array([0, 0, 0, 0])
    with pytest.raises(ValueError, match="must contain both treated and control"):
        compute_qini_curve(y, preds, w_all_c)


# -------------------------------------------------------------------------
# 3. Interoperability & Container Invariance
# -------------------------------------------------------------------------

def test_series_numpy_interoperability() -> None:
    """Verify that compute_qini_curve produces identical results for pd.Series and np.ndarray."""
    rng = np.random.default_rng(42)
    n = 100
    y_arr = rng.integers(0, 2, size=n)
    w_arr = rng.integers(0, 2, size=n)
    preds_arr = rng.uniform(-0.5, 0.5, size=n)

    res_arr = compute_qini_curve(y_arr, preds_arr, w_arr, n_points=20)

    y_ser = pd.Series(y_arr)
    w_ser = pd.Series(w_arr)
    preds_ser = pd.Series(preds_arr)

    res_ser = compute_qini_curve(y_ser, preds_ser, w_ser, n_points=20)

    np.testing.assert_allclose(res_arr.qini_model, res_ser.qini_model)
    np.testing.assert_allclose(res_arr.qini_random, res_ser.qini_random)
    assert res_arr.qini_score == res_ser.qini_score
    assert res_arr.auuc_model == res_ser.auuc_model


# -------------------------------------------------------------------------
# 4. Decile Uplift Analysis Tests
# -------------------------------------------------------------------------

def test_decile_analysis_table_structure() -> None:
    """Verify decile table schema, decile bin counts, and monotonicity properties."""
    rng = np.random.default_rng(42)
    n = 1000
    w = rng.binomial(1, 0.5, size=n)
    # Synthetic uplift: top deciles have high retention when treated
    pred_uplift = rng.normal(0, 1, size=n)
    y = np.where(w == 1, (pred_uplift > -0.5).astype(int), (pred_uplift > 0.5).astype(int))

    result: DecileAnalysisResult = compute_decile_analysis(y, pred_uplift, w, n_deciles=10)

    table = result.decile_table
    assert len(table) == 10
    assert list(table["decile"]) == list(range(1, 11))
    assert table["n_total"].sum() == n

    # Required metric columns
    required_cols = {
        "decile",
        "n_total",
        "n_treated",
        "n_control",
        "rate_treated",
        "rate_control",
        "empirical_uplift",
        "mean_predicted_uplift",
        "incremental_conversions",
        "cum_incremental_conversions",
    }
    assert required_cols.issubset(table.columns)

    # Monotonicity correlation must be within [-1, 1]
    assert -1.0 <= result.monotonicity_spearman_corr <= 1.0
    assert 0.0 <= result.monotonicity_p_value <= 1.0


def test_decile_analysis_sleeping_dogs_detection() -> None:
    """Verify that negative empirical uplift in Decile 10 flags sleeping_dogs_isolated."""
    n = 500
    w = np.random.default_rng(42).binomial(1, 0.5, size=n)
    # Create scenario where lowest predicted customers suffer from treatment
    pred_uplift = np.linspace(0.8, -0.8, n)
    # If pred_uplift < -0.3, treated churn (Y=0) while control renews (Y=1)
    y = np.where(
        pred_uplift < -0.3,
        np.where(w == 1, 0, 1),  # Sleeping Dogs: treated churn, control retain
        np.where(w == 1, 1, 0),  # Persuadables: treated retain, control churn
    )

    result = compute_decile_analysis(y, pred_uplift, w, n_deciles=10)
    assert result.bottom_decile_uplift < -0.01
    assert result.sleeping_dogs_isolated is True


# -------------------------------------------------------------------------
# 5. Full End-to-End Pipeline & Model Integration Test
# -------------------------------------------------------------------------

def test_metrics_integration_with_t_learner() -> None:
    """Verify end-to-end integration of TLearner with Qini and Decile evaluation engines."""
    raw_df = generate_synthetic_telemetry(n_samples=2000, random_seed=42)
    dataset = split_causal_dataset(raw_df, test_size=0.30, random_state=42)

    learner = TLearner(n_estimators=30, random_state=42)
    learner.fit(dataset.X_train, dataset.w_train, dataset.y_train)

    tau_pred = learner.predict_uplift(dataset.X_test)
    tau_true = dataset.latent_test["tau_true"].values

    qini_res = compute_qini_curve(
        y_true=dataset.y_test,
        uplift_preds=tau_pred,
        treatment=dataset.w_test,
        tau_true=tau_true,
        n_points=50,
    )

    # TLearner must outperform uniform random baseline on synthetic benchmark
    assert qini_res.qini_score > 0.20
    assert qini_res.auuc_model > qini_res.auuc_random

    decile_res = compute_decile_analysis(
        y_true=dataset.y_test,
        uplift_preds=tau_pred,
        treatment=dataset.w_test,
        tau_true=tau_true,
        n_deciles=10,
    )

    # Top decile must exhibit strong positive empirical uplift
    assert decile_res.top_decile_uplift > 0.15
    # Decile analysis must achieve positive monotonicity correlation
    assert decile_res.monotonicity_spearman_corr > 0.50
