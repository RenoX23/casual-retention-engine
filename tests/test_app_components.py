"""Smoke and integration tests for Streamlit dashboard UI components.

Validates that all UI rendering components execute cleanly without runtime
exceptions given realistic telemetry and model predictions.
"""

from __future__ import annotations

import pytest

from app.components.explainability_view import render_explainability_view
from app.components.model_performance import render_model_performance
from app.components.roi_simulator import render_roi_simulator
from data.generate_telemetry import generate_synthetic_telemetry
from src.data_pipeline import split_causal_dataset
from src.explainability.uplift_shap import UpliftTreeExplainer
from src.models.propensity_baseline import PropensityBaseline
from src.models.t_learner import TLearner


@pytest.fixture(scope="module")
def app_test_data() -> dict:
    """Provide realistic fitted models, predictions, and explanations for UI component testing."""
    raw_df = generate_synthetic_telemetry(n_samples=800, random_seed=42)
    dataset = split_causal_dataset(raw_df, test_size=0.30, random_state=42)

    t_learner = TLearner(n_estimators=15, random_state=42)
    t_learner.fit(dataset.X_train, dataset.w_train, dataset.y_train)

    prop_model = PropensityBaseline(n_estimators=15, random_state=42)
    prop_model.fit(dataset.X_train, None, dataset.y_train)

    tau_pred = t_learner.predict_uplift(dataset.X_test)
    p_churn = prop_model.predict_churn_propensity(dataset.X_test)

    explainer = UpliftTreeExplainer(t_learner, feature_names=list(dataset.X_test.columns))
    explanation = explainer.explain(dataset.X_test.iloc[:30])

    return {
        "dataset": dataset,
        "tau_pred": tau_pred,
        "p_churn": p_churn,
        "explainer": explainer,
        "explanation": explanation,
    }


def test_roi_simulator_renders(app_test_data: dict) -> None:
    """Verify render_roi_simulator executes without raising any exceptions."""
    d = app_test_data["dataset"]
    render_roi_simulator(
        tau_pred=app_test_data["tau_pred"],
        churn_propensity=app_test_data["p_churn"],
        mrr=d.mrr_test,
        archetypes=d.latent_test["archetype"],
    )


def test_model_performance_renders(app_test_data: dict) -> None:
    """Verify render_model_performance executes without raising any exceptions."""
    d = app_test_data["dataset"]
    predictions = {
        "T-Learner": app_test_data["tau_pred"],
        "Propensity": app_test_data["p_churn"],
    }
    render_model_performance(
        y_test=d.y_test,
        w_test=d.w_test,
        model_predictions=predictions,
        tau_true=d.latent_test["tau_true"].values,
    )


def test_explainability_view_renders(app_test_data: dict) -> None:
    """Verify render_explainability_view executes without raising any exceptions."""
    d = app_test_data["dataset"]
    archetypes = d.latent_test["archetype"].iloc[: len(app_test_data["explanation"].shap_values_diff)]
    render_explainability_view(
        explainer=app_test_data["explainer"],
        explanation=app_test_data["explanation"],
        archetypes=archetypes,
    )
