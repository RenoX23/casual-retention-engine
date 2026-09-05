"""Causal-Retain: Uplift Modeling & Customer Revenue Recovery Engine.

Interactive executive dashboard for Causal Machine Learning, featuring:
1. Executive Financial ROI & Revenue Recovery What-If Simulator.
2. Causal Model Performance Leaderboard & Cumulative Qini Benchmarks.
3. TreeSHAP Differential Uplift Explainability & Sleeping Dogs Root-Cause Drill-Down.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

with contextlib.suppress(ImportError):
    import lightgbm  # noqa: F401

import numpy as np
import pandas as pd
import streamlit as st

from app.components.explainability_view import render_explainability_view
from app.components.model_performance import render_model_performance
from app.components.roi_simulator import render_roi_simulator
from data.generate_telemetry import generate_synthetic_telemetry
from src.data_pipeline import CausalSplitDataset, split_causal_dataset
from src.explainability.uplift_shap import UpliftExplanation, UpliftTreeExplainer
from src.models.propensity_baseline import PropensityBaseline
from src.models.s_learner import SLearner
from src.models.t_learner import TLearner
from src.models.x_learner import XLearner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_app")

# Page Configuration
st.set_page_config(
    page_title="Causal-Retain: Uplift & Revenue Recovery",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Loading and partitioning SaaS telemetry dataset...")
def load_and_split_data(n_samples: int = 4000, random_seed: int = 42) -> tuple[pd.DataFrame, CausalSplitDataset]:
    """Generate synthetic telemetry and execute zero-leakage causal split."""
    raw_df = generate_synthetic_telemetry(n_samples=n_samples, random_seed=random_seed)
    dataset = split_causal_dataset(raw_df, test_size=0.30, random_state=random_seed)
    return raw_df, dataset


@st.cache_resource(show_spinner="Training Causal Uplift Meta-Learners & Baseline...")
def train_models(
    X_train: pd.DataFrame,
    w_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, Any]:
    """Train S-Learner, T-Learner, X-Learner, and Propensity Baseline."""
    t_learner = TLearner(n_estimators=40, random_state=42)
    t_learner.fit(X_train, w_train, y_train)

    x_learner = XLearner(n_estimators=40, random_state=42)
    x_learner.fit(X_train, w_train, y_train)

    s_learner = SLearner(n_estimators=40, random_state=42)
    s_learner.fit(X_train, w_train, y_train)

    propensity_model = PropensityBaseline(n_estimators=40, random_state=42)
    propensity_model.fit(X_train, None, y_train)

    return {
        "t_learner": t_learner,
        "x_learner": x_learner,
        "s_learner": s_learner,
        "propensity": propensity_model,
    }


@st.cache_resource(show_spinner="Computing TreeSHAP Differential Attributions...")
def compute_shap_explanations(
    _t_learner: TLearner,
    X_test: pd.DataFrame,
) -> tuple[UpliftTreeExplainer, UpliftExplanation]:
    """Compute TreeSHAP differential attributions on T-Learner."""
    explainer = UpliftTreeExplainer(_t_learner, feature_names=list(X_test.columns))
    # Explain test subset for ultra-responsive dashboard performance
    explanation = explainer.explain(X_test.iloc[:150])
    return explainer, explanation


def main() -> None:
    """Streamlit Application Entry Point."""
    # Top Branding Header
    st.title("📈 Causal-Retain: Uplift Modeling & Revenue Recovery Engine")
    st.caption(
        "B2B SaaS Causal Machine Learning Platform | "
        "Heterogeneous Treatment Effect Estimation & Budget Optimization"
    )

    # Sidebar Information
    with st.sidebar:
        st.header("⚙️ Simulation Settings")
        cohort_size = st.select_slider(
            "Customer Cohort Volume",
            options=[2000, 4000, 6000],
            value=4000,
            help="Total customer accounts in simulated historical observation window.",
        )
        random_seed = st.number_input("Simulation Random Seed", min_value=1, max_value=9999, value=42)

        st.markdown("---")
        st.header("📊 Telemetry Architecture")
        st.markdown(
            """
            - **Framework**: Neyman-Rubin Potential Outcomes
            - **Targeting Policy**: Individual Treatment Effect (CATE)
            - **Primary Estimator**: Dual LightGBM T-Learner
            - **Calibration**: Cross-Validated Platt Scaling
            - **Explainability**: Differential TreeSHAP
            """
        )
        st.markdown("---")
        st.caption("Developed for Advanced Causal ML & Revenue Optimization")

    # Load Data & Models
    raw_df, dataset = load_and_split_data(n_samples=cohort_size, random_seed=int(random_seed))
    models = train_models(dataset.X_train, dataset.w_train, dataset.y_train)

    # Compute predictions on held-out test cohort
    t_learner: TLearner = models["t_learner"]
    x_learner: XLearner = models["x_learner"]
    s_learner: SLearner = models["s_learner"]
    prop_model: PropensityBaseline = models["propensity"]

    tau_t = t_learner.predict_uplift(dataset.X_test)
    tau_x = x_learner.predict_uplift(dataset.X_test)
    tau_s = s_learner.predict_uplift(dataset.X_test)
    p_churn = prop_model.predict_churn_propensity(dataset.X_test)

    model_predictions = {
        "T-Learner (Two Models)": tau_t,
        "X-Learner (Imputation)": tau_x,
        "S-Learner (Single Model)": tau_s,
        "Propensity Baseline": p_churn,
    }

    tau_true = dataset.latent_test["tau_true"].values if dataset.latent_test is not None else None
    archetypes = dataset.latent_test["archetype"] if dataset.latent_test is not None else None
    mrr_test = dataset.mrr_test if dataset.mrr_test is not None else dataset.X_test["monthly_recurring_revenue"].values

    # Precompute TreeSHAP
    explainer, explanation = compute_shap_explanations(t_learner, dataset.X_test)

    # Main Dashboard Tabs
    tab1, tab2, tab3 = st.tabs(
        [
            "💰 Financial ROI Simulator",
            "📊 Causal Performance & Qini",
            "🔍 TreeSHAP Explainability",
        ]
    )

    with tab1:
        render_roi_simulator(
            tau_pred=tau_t,
            churn_propensity=p_churn,
            mrr=mrr_test,
            archetypes=archetypes,
        )

    with tab2:
        render_model_performance(
            y_test=dataset.y_test,
            w_test=dataset.w_test,
            model_predictions=model_predictions,
            tau_true=tau_true,
        )

    with tab3:
        render_explainability_view(
            explainer=explainer,
            explanation=explanation,
            archetypes=archetypes.iloc[: len(explanation.shap_values_diff)] if archetypes is not None else None,
        )


if __name__ == "__main__":
    main()
