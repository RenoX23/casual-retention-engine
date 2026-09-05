"""Unit test suite for Causal Uplift Meta-Learners and Propensity Baseline.

Validates:
1. Abstract base class interface enforcement and instantiation guards.
2. PropensityBaseline traditional churn benchmark behavior and probability constraints.
3. S-Learner treatment feature augmentation and potential outcome estimation.
4. T-Learner two-model architecture, calibration methods (sigmoid/isotonic), and raw model preservation.
5. X-Learner 4-stage counterfactual imputation and propensity weighting.
6. Rigorous input validation (binary constraints, dimension matching, C-contiguous arrays).
7. Uplift theoretical bounds [-1.0, 1.0] across all estimators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.generate_telemetry import generate_synthetic_telemetry
from src.data_pipeline import split_causal_dataset
from src.models.base_learner import BaseUpliftLearner
from src.models.propensity_baseline import PropensityBaseline
from src.models.s_learner import SLearner
from src.models.t_learner import TLearner
from src.models.x_learner import XLearner


@pytest.fixture(scope="module")
def synthetic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Provide a small processed train/test dataset for model testing."""
    raw_df = generate_synthetic_telemetry(n_samples=1500, random_seed=42)
    dataset = split_causal_dataset(raw_df, test_size=0.30, random_state=42)
    return (
        dataset.X_train.to_numpy(),
        dataset.w_train,
        dataset.y_train,
        dataset.X_test.to_numpy(),
        dataset.w_test,
        dataset.y_test,
    )


# -------------------------------------------------------------------------
# 1. BaseUpliftLearner Abstract Contract Tests
# -------------------------------------------------------------------------

def test_base_learner_abc_instantiation_raises() -> None:
    """Verify that BaseUpliftLearner cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseUpliftLearner()  # type: ignore[abstract]


def test_input_validation_binary_constraints() -> None:
    """Verify that non-binary treatment or outcome vectors raise ValueError."""
    learner = SLearner()
    X = np.random.default_rng(42).standard_normal((50, 4))
    w_valid = np.random.default_rng(42).integers(0, 2, size=50)
    y_valid = np.random.default_rng(42).integers(0, 2, size=50)

    # Invalid treatment (contains 2)
    w_invalid = w_valid.copy()
    w_invalid[0] = 2
    with pytest.raises(ValueError, match="Treatment vector w must be strictly binary"):
        learner.fit(X, w_invalid, y_valid)

    # Invalid outcome (contains -1)
    y_invalid = y_valid.copy()
    y_invalid[0] = -1
    with pytest.raises(ValueError, match="Outcome vector y must be strictly binary"):
        learner.fit(X, w_valid, y_invalid)


def test_input_validation_length_mismatch() -> None:
    """Verify that length mismatches between X, w, and y raise ValueError."""
    learner = TLearner()
    X = np.random.default_rng(42).standard_normal((50, 4))
    w_short = np.ones(40, dtype=int)
    y_short = np.ones(40, dtype=int)
    w_valid = np.ones(50, dtype=int)
    y_valid = np.ones(50, dtype=int)

    with pytest.raises(ValueError, match="Length mismatch: X has 50 rows, w has 40"):
        learner.fit(X, w_short, y_valid)

    with pytest.raises(ValueError, match="Length mismatch: X has 50 rows, y has 40"):
        learner.fit(X, w_valid, y_short)


# -------------------------------------------------------------------------
# 2. PropensityBaseline Benchmark Tests
# -------------------------------------------------------------------------

def test_propensity_baseline_unfitted_raises() -> None:
    """Verify calling methods before fit raises RuntimeError."""
    baseline = PropensityBaseline()
    X = np.random.default_rng(42).standard_normal((10, 4))
    with pytest.raises(RuntimeError, match="not fitted yet"):
        baseline.predict_proba(X)
    with pytest.raises(RuntimeError, match="not fitted yet"):
        baseline.predict_uplift(X)


def test_propensity_baseline_fit_and_predict(synthetic_data: tuple) -> None:
    """Verify PropensityBaseline produces valid probabilities and targeting score."""
    X_train, _, y_train, X_test, _, _ = synthetic_data
    baseline = PropensityBaseline(n_estimators=30, random_state=42)
    baseline.fit(X_train, None, y_train)

    assert baseline.is_fitted_ is True

    # Predict proba shape and probability axioms
    proba = baseline.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    # Churn + Retention propensity sum to 1.0
    p_churn = baseline.predict_churn_propensity(X_test)
    p_retain = baseline.predict_retention_propensity(X_test)
    np.testing.assert_allclose(p_churn + p_retain, 1.0, rtol=1e-5)

    # Uplift returns churn propensity as the traditional targeting score
    uplift_score = baseline.predict_uplift(X_test)
    np.testing.assert_allclose(uplift_score, p_churn)

    # Potential outcomes decomposition should raise NotImplementedError
    with pytest.raises(NotImplementedError, match="does not implement explicit potential outcome"):
        baseline.predict_potential_outcomes(X_test)


# -------------------------------------------------------------------------
# 3. S-Learner Tests
# -------------------------------------------------------------------------

def test_s_learner_unfitted_raises() -> None:
    """Verify calling predict_uplift on unfitted SLearner raises RuntimeError."""
    learner = SLearner()
    X = np.random.default_rng(42).standard_normal((10, 4))
    with pytest.raises(RuntimeError, match="not fitted yet"):
        learner.predict_uplift(X)


def test_s_learner_fit_and_predict(synthetic_data: tuple) -> None:
    """Verify SLearner training, potential outcomes, and uplift bounds."""
    X_train, w_train, y_train, X_test, _, _ = synthetic_data
    s_learner = SLearner(n_estimators=30, random_state=42)
    s_learner.fit(X_train, w_train, y_train)

    assert s_learner.is_fitted_ is True

    mu_0, mu_1 = s_learner.predict_potential_outcomes(X_test)
    assert len(mu_0) == len(X_test)
    assert len(mu_1) == len(X_test)
    assert np.all((mu_0 >= 0.0) & (mu_0 <= 1.0))
    assert np.all((mu_1 >= 0.0) & (mu_1 <= 1.0))

    tau_hat = s_learner.predict_uplift(X_test)
    np.testing.assert_allclose(tau_hat, mu_1 - mu_0, rtol=1e-5)
    assert np.all((tau_hat >= -1.0) & (tau_hat <= 1.0))


# -------------------------------------------------------------------------
# 4. T-Learner Tests
# -------------------------------------------------------------------------

def test_t_learner_unfitted_raises() -> None:
    """Verify calling predict_uplift on unfitted TLearner raises RuntimeError."""
    learner = TLearner()
    X = np.random.default_rng(42).standard_normal((10, 4))
    with pytest.raises(RuntimeError, match="not fitted yet"):
        learner.predict_uplift(X)


@pytest.mark.parametrize("calibrate, method", [(True, "sigmoid"), (True, "isotonic"), (False, "sigmoid")])
def test_t_learner_calibration_variants(synthetic_data: tuple, calibrate: bool, method: str) -> None:
    """Verify TLearner operates with sigmoid, isotonic, and disabled calibration."""
    X_train, w_train, y_train, X_test, _, _ = synthetic_data
    t_learner = TLearner(
        n_estimators=30,
        calibrate_probabilities=calibrate,
        calibration_method=method,
        calibration_cv=3,
        random_state=42,
    )
    t_learner.fit(X_train, w_train, y_train)

    assert t_learner.is_fitted_ is True
    assert t_learner.raw_model_0_ is not None
    assert t_learner.raw_model_1_ is not None

    mu_0, mu_1 = t_learner.predict_potential_outcomes(X_test)
    assert np.all((mu_0 >= 0.0) & (mu_0 <= 1.0))
    assert np.all((mu_1 >= 0.0) & (mu_1 <= 1.0))

    tau_hat = t_learner.predict_uplift(X_test)
    np.testing.assert_allclose(tau_hat, mu_1 - mu_0, rtol=1e-5)
    assert np.all((tau_hat >= -1.0) & (tau_hat <= 1.0))


# -------------------------------------------------------------------------
# 5. X-Learner Tests
# -------------------------------------------------------------------------

def test_x_learner_unfitted_raises() -> None:
    """Verify calling predict_uplift on unfitted XLearner raises RuntimeError."""
    learner = XLearner()
    X = np.random.default_rng(42).standard_normal((10, 4))
    with pytest.raises(RuntimeError, match="not fitted yet"):
        learner.predict_uplift(X)


@pytest.mark.parametrize("use_propensity", [True, False])
def test_x_learner_fit_and_predict(synthetic_data: tuple, use_propensity: bool) -> None:
    """Verify XLearner 4-stage fitting, propensity estimation, and uplift bounds."""
    X_train, w_train, y_train, X_test, _, _ = synthetic_data
    x_learner = XLearner(
        n_estimators=30,
        propensity_model=use_propensity,
        random_state=42,
    )
    x_learner.fit(X_train, w_train, y_train)

    assert x_learner.is_fitted_ is True
    assert x_learner.stage1_mu0_ is not None
    assert x_learner.stage1_mu1_ is not None
    assert x_learner.stage2_tau0_ is not None
    assert x_learner.stage2_tau1_ is not None

    # Propensity scores
    propensity = x_learner.predict_propensity(X_test)
    assert np.all((propensity >= 0.0) & (propensity <= 1.0))

    # Potential outcomes from Stage 1
    mu_0, mu_1 = x_learner.predict_potential_outcomes(X_test)
    assert np.all((mu_0 >= 0.0) & (mu_0 <= 1.0))
    assert np.all((mu_1 >= 0.0) & (mu_1 <= 1.0))

    # Uplift estimation
    tau_hat = x_learner.predict_uplift(X_test)
    assert len(tau_hat) == len(X_test)
    assert np.all((tau_hat >= -1.0) & (tau_hat <= 1.0))


# -------------------------------------------------------------------------
# 6. DataFrame vs NumPy Array Interoperability
# -------------------------------------------------------------------------

def test_dataframe_numpy_equivalence(synthetic_data: tuple) -> None:
    """Verify models produce identical outputs whether fed pd.DataFrame or np.ndarray."""
    X_train_arr, w_train, y_train, X_test_arr, _, _ = synthetic_data
    feature_names = [f"feat_{i}" for i in range(X_train_arr.shape[1])]
    X_train_df = pd.DataFrame(X_train_arr, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_arr, columns=feature_names)

    # Train on DF, predict on both
    t_learner = TLearner(n_estimators=20, calibrate_probabilities=False, random_state=42)
    t_learner.fit(X_train_df, w_train, y_train)

    pred_from_df = t_learner.predict_uplift(X_test_df)
    pred_from_arr = t_learner.predict_uplift(X_test_arr)

    np.testing.assert_allclose(pred_from_df, pred_from_arr, rtol=1e-5)
