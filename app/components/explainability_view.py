"""TreeSHAP Differential Uplift Explainability Component.

Renders global causal feature importance, archetype contrast profiling
(Persuadables vs. Sleeping Dogs), and localized account waterfall attributions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.explainability.uplift_shap import UpliftExplanation, UpliftTreeExplainer


def render_explainability_view(
    explainer: UpliftTreeExplainer,
    explanation: UpliftExplanation,
    archetypes: pd.Series | np.ndarray | None = None,
) -> None:
    """Render the TreeSHAP explainability tab.

    Parameters
    ----------
    explainer : UpliftTreeExplainer
        Initialized explainer instance.
    explanation : UpliftExplanation
        Precomputed differential SHAP explanation container.
    archetypes : pd.Series | np.ndarray | None, default=None
        Optional customer archetype labels.
    """
    st.header("Causal Interpretability & TreeSHAP Differential Attribution")
    st.markdown(
        """
        Unlike standard ML which explains predicted churn $P(\\text{Churn} \\mid X)$,
        **Causal TreeSHAP** decomposes the difference between treatment and control response surfaces:
        $$\\phi_i(\\text{uplift}) = \\phi_i(\\mu_1) - \\phi_i(\\mu_0)$$
        Explaining **why** a customer will renew if intervened, or why they are at risk of being a *Sleeping Dog*.
        """
    )

    # 1. Global Feature Importance Bar Chart
    st.subheader("Global Causal Uplift Drivers")
    st.markdown(
        "Features with high **Mean Absolute Uplift SHAP** generate the strongest "
        "treatment effect heterogeneity across accounts."
    )

    fi = explanation.feature_importance
    fig_global = go.Figure()
    fig_global.add_trace(
        go.Bar(
            y=fi["feature"][::-1],
            x=fi["mean_abs_uplift_shap"][::-1],
            orientation="h",
            marker={
                "color": fi["mean_abs_uplift_shap"][::-1],
                "colorscale": "Viridis",
            },
            text=[f"{v:.3f}" for v in fi["mean_abs_uplift_shap"][::-1]],
            textposition="auto",
        )
    )
    fig_global.update_layout(
        title="Global Feature Attribution on Uplift (Mean |Delta SHAP|)",
        xaxis_title="Mean Absolute Differential SHAP",
        yaxis_title="Covariate",
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        height=450,
    )
    st.plotly_chart(fig_global, use_container_width=True)

    st.markdown("---")

    # 2. Archetype Comparison: Persuadables vs Sleeping Dogs
    if archetypes is not None:
        st.subheader("Archetype Drivers: Persuadables vs. Sleeping Dogs")
        arch_summaries = explainer.explain_archetypes(explanation, archetypes)

        if "Persuadable" in arch_summaries and "Sleeping Dog" in arch_summaries:
            p_df = arch_summaries["Persuadable"].set_index("feature")
            s_df = arch_summaries["Sleeping Dog"].set_index("feature")

            top_feats = fi["feature"].head(6).tolist()
            p_vals = [p_df.loc[f, "mean_uplift_shap"] for f in top_feats]
            s_vals = [s_df.loc[f, "mean_uplift_shap"] for f in top_feats]

            fig_arch = go.Figure()
            fig_arch.add_trace(
                go.Bar(
                    name="Persuadables (Positive Uplift)",
                    x=top_feats,
                    y=p_vals,
                    marker_color="#10B981",
                )
            )
            fig_arch.add_trace(
                go.Bar(
                    name="Sleeping Dogs (Negative Uplift)",
                    x=top_feats,
                    y=s_vals,
                    marker_color="#EF4444",
                )
            )
            fig_arch.update_layout(
                title="Directional Uplift Attribution Across Opposing Archetypes",
                barmode="group",
                xaxis_title="Top Causal Features",
                yaxis_title="Directional Mean Uplift SHAP (Push vs Pull)",
                legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
                height=400,
                margin={"l": 20, "r": 20, "t": 40, "b": 20},
            )
            st.plotly_chart(fig_arch, use_container_width=True)

    st.markdown("---")

    # 3. Individual Customer Account Waterfall Drill-down
    st.subheader("Individual Account Root-Cause Drill-Down")
    n_samples = len(explanation.shap_values_diff)
    selected_idx = st.slider(
        "Select Customer Account Index:",
        min_value=0,
        max_value=min(n_samples - 1, 300),
        value=0,
        help="Explore localized waterfall feature attributions for any customer in the test cohort.",
    )

    waterfall = explainer.explain_instance(explanation, idx=selected_idx, top_k=6)

    # Display customer stats
    col_idx, col_pred, col_base = st.columns(3)
    with col_idx:
        st.write(f"**Account ID:** `AC-{(selected_idx + 1042):05d}`")
    with col_pred:
        pred_val = waterfall["total_uplift_shap"]
        tag = "Persuadable" if pred_val > 0.1 else ("Sleeping Dog" if pred_val < -0.05 else "Neutral")
        st.write(f"**Predicted Uplift:** `{pred_val:+.3f}` ({tag})")
    with col_base:
        st.write(f"**Base Cohort Uplift:** `{waterfall['base_value']:+.3f}`")

    # Plot Waterfall Chart
    fig_wf = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["relative"] * len(waterfall["features"]) + ["relative", "total"],
            x=waterfall["features"] + ["Other Features", "Net Uplift"],
            y=waterfall["shap_contributions"] + [waterfall["other_features_contribution"], 0],
            text=[f"{v:+.2f}" for v in waterfall["shap_contributions"]]
            + [f"{waterfall['other_features_contribution']:+.2f}", f"{waterfall['total_uplift_shap']:+.2f}"],
            connector={"line": {"color": "#6B7280"}},
            decreasing={"marker": {"color": "#EF4444"}},
            increasing={"marker": {"color": "#10B981"}},
            totals={"marker": {"color": "#3B82F6"}},
        )
    )
    fig_wf.update_layout(
        title=f"Root-Cause Uplift Attribution: Account AC-{(selected_idx + 1042):05d}",
        yaxis_title="Uplift Contribution",
        height=450,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    st.plotly_chart(fig_wf, use_container_width=True)
