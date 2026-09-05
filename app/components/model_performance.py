"""Causal Uplift Model Performance & Decile Validation Component.

Renders interactive Qini curves, cumulative gain trajectories, AUUC benchmarks,
and 10-decile uplift bar charts with empirical monotonicity verification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.evaluation.decile_analysis import DecileAnalysisResult, compute_decile_analysis
from src.evaluation.qini_metric import QiniCurveResult, compute_qini_curve


def render_model_performance(
    y_test: np.ndarray,
    w_test: np.ndarray,
    model_predictions: dict[str, np.ndarray],
    tau_true: np.ndarray | None = None,
) -> None:
    """Render the model performance, Qini curves, and decile validation tab.

    Parameters
    ----------
    y_test : np.ndarray
        Observed retention outcome (1 = retained, 0 = churned).
    w_test : np.ndarray
        Observed treatment indicator (1 = treated, 0 = control).
    model_predictions : dict[str, np.ndarray]
        Mapping from model name to predicted uplift or targeting score.
    tau_true : np.ndarray | None, default=None
        Optional true latent CATE for oracle upper bound.
    """
    st.header("Causal Uplift Model Performance & Qini Benchmarks")
    st.markdown(
        """
        Because true counterfactuals are unobservable at test time, uplift models are evaluated
        using **cumulative ranking metrics** on held-out A/B trial data. A superior model concentrates
        the highest incremental retention in early population fractions.
        """
    )

    # Compute Qini curves across all available models
    qini_results: dict[str, QiniCurveResult] = {}
    for name, preds in model_predictions.items():
        qini_results[name] = compute_qini_curve(
            y_true=y_test,
            uplift_preds=preds,
            treatment=w_test,
            tau_true=tau_true,
            n_points=60,
        )

    # Model Performance Leaderboard
    st.subheader("Model Benchmark Leaderboard")
    leaderboard_rows = []
    for name, res in qini_results.items():
        leaderboard_rows.append(
            {
                "Model Architecture": name,
                "Normalized Qini (Q_norm)": f"{res.qini_score:.4f}",
                "Area Under Uplift (AUUC)": f"{res.auuc_model:.2f}",
                "Random AUUC Baseline": f"{res.auuc_random:.2f}",
                "AUUC Lift over Random": f"+{(res.auuc_model - res.auuc_random):.2f}",
            }
        )
    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values(
        by="Normalized Qini (Q_norm)", ascending=False
    )
    st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Qini Curve & Cumulative Gain Plots in 2 tabs
    tab_qini, tab_gain = st.tabs(["Cumulative Qini Curves", "Cumulative Gain Curves"])

    with tab_qini:
        st.markdown(
            "The **Qini Curve** plots cumulative incremental responders: "
            "$Q(u) = Y_t(u) - Y_c(u) \\cdot (N_t / N_c)$."
        )
        fig_qini = go.Figure()

        # Color palette for distinct curves
        palette = {
            "T-Learner (Two Models)": "#10B981",
            "X-Learner (Imputation)": "#3B82F6",
            "S-Learner (Single Model)": "#8B5CF6",
            "Propensity Baseline": "#EF4444",
        }

        # Add curves for each model
        for name, res in qini_results.items():
            color = palette.get(name, "#6B7280")
            fig_qini.add_trace(
                go.Scatter(
                    x=res.fractions * 100,
                    y=res.qini_model,
                    mode="lines",
                    name=f"{name} (Q={res.qini_score:.3f})",
                    line={"color": color, "width": 3 if "T-Learner" in name else 2},
                )
            )

        # Reference: Random baseline
        if qini_results:
            sample_res = next(iter(qini_results.values()))
            fig_qini.add_trace(
                go.Scatter(
                    x=sample_res.fractions * 100,
                    y=sample_res.qini_random,
                    mode="lines",
                    name="Uniform Random Targeting",
                    line={"color": "#9CA3AF", "width": 2, "dash": "dash"},
                )
            )
            # Reference: Optimal / Oracle upper bound
            fig_qini.add_trace(
                go.Scatter(
                    x=sample_res.fractions * 100,
                    y=sample_res.qini_optimal,
                    mode="lines",
                    name="Theoretical Optimal Ceiling",
                    line={"color": "#D97706", "width": 2, "dash": "dot"},
                )
            )

        fig_qini.update_layout(
            title="Cumulative Qini Curve Benchmark",
            xaxis_title="Population Fraction Targeted (%)",
            yaxis_title="Cumulative Incremental Responders (Qini)",
            hovermode="x unified",
            legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
            height=480,
            margin={"l": 20, "r": 20, "t": 40, "b": 20},
        )
        st.plotly_chart(fig_qini, use_container_width=True)

    with tab_gain:
        st.markdown(
            "The **Cumulative Gain Curve** displays empirical average uplift scaled by cohort size: "
            "$G(k) = (R_t(k) - R_c(k)) \\cdot k$."
        )
        fig_gain = go.Figure()
        for name, res in qini_results.items():
            color = palette.get(name, "#6B7280")
            fig_gain.add_trace(
                go.Scatter(
                    x=res.fractions * 100,
                    y=res.cum_gain_model,
                    mode="lines",
                    name=name,
                    line={"color": color, "width": 2},
                )
            )
        fig_gain.update_layout(
            title="Cumulative Gain Curve Benchmark",
            xaxis_title="Population Fraction Targeted (%)",
            yaxis_title="Cumulative Incremental Conversions",
            hovermode="x unified",
            height=480,
            margin={"l": 20, "r": 20, "t": 40, "b": 20},
        )
        st.plotly_chart(fig_gain, use_container_width=True)

    st.markdown("---")

    # Decile Analysis Section
    st.subheader("10-Decile Uplift & Monotonicity Verification")
    selected_model = st.selectbox(
        "Select Model for Decile Profile Inspection:",
        options=list(model_predictions.keys()),
        index=0,
    )

    preds_selected = model_predictions[selected_model]
    decile_res: DecileAnalysisResult = compute_decile_analysis(
        y_true=y_test,
        uplift_preds=preds_selected,
        treatment=w_test,
        tau_true=tau_true,
        n_deciles=10,
    )

    col_mono, col_top, col_bottom, col_dog = st.columns(4)
    with col_mono:
        st.metric(
            "Monotonicity (Spearman rho)",
            f"{decile_res.monotonicity_spearman_corr:.3f}",
            delta="p < 0.01" if decile_res.monotonicity_p_value < 0.01 else f"p={decile_res.monotonicity_p_value:.3f}",
            help="Correlation between priority rank and empirical uplift. Near +1.0 indicates perfect monotonic ordering.",
        )
    with col_top:
        st.metric(
            "Decile 1 Uplift (Persuadables)",
            f"+{decile_res.top_decile_uplift * 100:.1f}%",
        )
    with col_bottom:
        st.metric(
            "Decile 10 Uplift",
            f"{decile_res.bottom_decile_uplift * 100:.1f}%",
        )
    with col_dog:
        if decile_res.sleeping_dogs_isolated:
            st.success("Sleeping Dogs Detected & Isolated (Negative Uplift in Bottom Decile)")
        else:
            st.warning("No negative uplift detected in bottom decile.")

    # Plot Decile Empirical Uplift Bar Chart
    d_table = decile_res.decile_table
    fig_decile = go.Figure()

    bar_colors = [
        "#10B981" if u > 0.05 else ("#EF4444" if u < -0.01 else "#F59E0B")
        for u in d_table["empirical_uplift"]
    ]

    fig_decile.add_trace(
        go.Bar(
            x=[f"Decile {d}" for d in d_table["decile"]],
            y=d_table["empirical_uplift"] * 100,
            marker_color=bar_colors,
            name="Empirical Uplift (R_t - R_c)",
            text=[f"{u * 100:+.1f}%" for u in d_table["empirical_uplift"]],
            textposition="auto",
        )
    )

    fig_decile.update_layout(
        title=f"Empirical Retention Uplift Across Deciles: {selected_model}",
        xaxis_title="Predicted Uplift Deciles (Decile 1 = Highest Predicted Uplift)",
        yaxis_title="Empirical Retention Rate Difference (%)",
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        height=400,
    )
    st.plotly_chart(fig_decile, use_container_width=True)

    # Detailed Decile Data Table
    with st.expander("View Full Decile Summary Table"):
        display_df = d_table.copy()
        display_df["rate_treated"] = (display_df["rate_treated"] * 100).round(1).astype(str) + "%"
        display_df["rate_control"] = (display_df["rate_control"] * 100).round(1).astype(str) + "%"
        display_df["empirical_uplift"] = (display_df["empirical_uplift"] * 100).round(1).astype(str) + "%"
        display_df["mean_predicted_uplift"] = (display_df["mean_predicted_uplift"] * 100).round(1).astype(str) + "%"
        st.dataframe(display_df, use_container_width=True, hide_index=True)
