"""Unit test suite for TreeSHAP Differential Uplift Explainability Engine.

Validates:
1. TreeSHAP explainer initialization, fitted guards, and raw booster validation.
2. Differential SHAP array dimensions and numerical finiteness.
3. Global feature importance generation and rank ordering.
4. Local instance waterfall breakdown and out-of-bounds index handling.
5. Behavioral archetype segmentation (Persuadables vs. Sleeping Dogs).
6. DataFrame and NumPy array interoperability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.generate_telemetry import generate_synthetic_telemetry
from src.data_pipeline import split_causal_dataset
from src.explainability.uplift_shap import UpliftExplanation, UpliftTreeExplainer
from src.models.t_learner import TLearner


@pytest.fixture(scope="module")
def fitted_tl_and_data() -> tuple[TLearner, pd.DataFrame, pd.Series]:
    """Fit a small TLearner and provide test features and archetypes."""
    raw_df = generate_synthetic_telemetry(n_samples=1000, random_seed=42)
    dataset = split_causal_dataset(raw_df, test_size=0.30, random_state=42)

    learner = TLearner(n_estimators=20, random_state=42)
    learner.fit(dataset.X_train, dataset.w_train, dataset.y_train)

    return learner, dataset.X_test, dataset.latent_test["archetype"]


def test_unfitted_learner_raises() -> None:
    """Verify initializing explainer with unfitted TLearner raises RuntimeError."""
    unfitted_learner = TLearner()
    with pytest.raises(RuntimeError, match="must be fitted before initializing"):
        UpliftTreeExplainer(unfitted_learner)


def test_explainer_diff_shap_dimensions(fitted_tl_and_data: tuple) -> None:
    """Verify differential SHAP dimensions, base values, and importance table."""
    learner, X_test, _ = fitted_tl_and_data
    sub_X = X_test.iloc[:25]

    explainer = UpliftTreeExplainer(learner, feature_names=list(sub_X.columns))
    explanation: UpliftExplanation = explainer.explain(sub_X)

    # Dimensional assertions
    assert explanation.shap_values_diff.shape == (25, sub_X.shape[1])
    assert explanation.shap_values_treated.shape == (25, sub_X.shape[1])
    assert explanation.shap_values_control.shape == (25, sub_X.shape[1])

    # Values must be finite
    assert np.all(np.isfinite(explanation.shap_values_diff))
    assert np.all(np.isfinite(explanation.shap_values_treated))
    assert np.all(np.isfinite(explanation.shap_values_control))

    # Feature importance schema and ordering
    fi = explanation.feature_importance
    assert len(fi) == sub_X.shape[1]
    assert list(fi.columns) == ["feature", "mean_abs_uplift_shap", "mean_uplift_shap", "rank"]
    assert fi["mean_abs_uplift_shap"].is_monotonic_decreasing


def test_explain_instance_waterfall(fitted_tl_and_data: tuple) -> None:
    """Verify local instance drill-down structure and index error guards."""
    learner, X_test, _ = fitted_tl_and_data
    sub_X = X_test.iloc[:10]

    explainer = UpliftTreeExplainer(learner, feature_names=list(sub_X.columns))
    explanation = explainer.explain(sub_X)

    # Valid instance
    instance_dict = explainer.explain_instance(explanation, idx=0, top_k=5)
    assert instance_dict["instance_index"] == 0
    assert len(instance_dict["features"]) == 5
    assert len(instance_dict["shap_contributions"]) == 5
    assert len(instance_dict["feature_values"]) == 5
    assert isinstance(instance_dict["total_uplift_shap"], float)

    # Out of bounds index
    with pytest.raises(IndexError, match="out of range"):
        explainer.explain_instance(explanation, idx=999)


def test_explain_archetypes_breakdown(fitted_tl_and_data: tuple) -> None:
    """Verify archetype segmentation produces summaries for all present archetypes."""
    learner, X_test, archetypes = fitted_tl_and_data
    sub_X = X_test.iloc[:60]
    sub_arch = archetypes.iloc[:60]

    explainer = UpliftTreeExplainer(learner, feature_names=list(sub_X.columns))
    explanation = explainer.explain(sub_X)

    arch_summaries = explainer.explain_archetypes(explanation, sub_arch)
    assert len(arch_summaries) >= 2  # At least Persuadables and Sure Things present

    for _arch_name, df_summary in arch_summaries.items():
        assert isinstance(df_summary, pd.DataFrame)
        assert "feature" in df_summary.columns
        assert "mean_uplift_shap" in df_summary.columns
        assert "mean_abs_uplift_shap" in df_summary.columns

    # Archetype length mismatch guard
    with pytest.raises(ValueError, match="does not match explanation sample count"):
        explainer.explain_archetypes(explanation, archetypes.iloc[:10])


def test_explainer_numpy_interoperability(fitted_tl_and_data: tuple) -> None:
    """Verify explainer accepts raw np.ndarray as well as pd.DataFrame."""
    learner, X_test, _ = fitted_tl_and_data
    sub_X_df = X_test.iloc[:15]
    sub_X_arr = sub_X_df.to_numpy()

    explainer = UpliftTreeExplainer(learner, feature_names=list(sub_X_df.columns))

    exp_df = explainer.explain(sub_X_df)
    exp_arr = explainer.explain(sub_X_arr)

    np.testing.assert_allclose(exp_df.shap_values_diff, exp_arr.shap_values_diff, rtol=1e-5)
    assert exp_df.base_value_diff == exp_arr.base_value_diff
