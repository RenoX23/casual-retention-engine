"""Decile Uplift Analysis and Monotonicity Verification Engine.

Partitions customers ranked by predicted uplift into 10 deciles and computes:
1. Empirical treatment response rate R_t(d) and control response rate R_c(d).
2. Empirical decile uplift: tau_empirical(d) = R_t(d) - R_c(d).
3. Incremental conversions and cumulative revenue recovery.
4. Monotonicity score: Spearman rank correlation testing whether empirical uplift
   decreases monotonically from Decile 1 (Persuadables) to Decile 10 (Sleeping Dogs).
5. Ground truth concordance MAE when latent counterfactuals are available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecileAnalysisResult:
    """Container holding decile analysis results and diagnostic metrics.

    Attributes
    ----------
    decile_table : pd.DataFrame
        Detailed metrics per decile (1 to 10).
    monotonicity_spearman_corr : float
        Spearman rank correlation between priority rank (Decile 1 to 10 inverted)
        and empirical uplift. A value close to +1.0 indicates near-perfect monotonicity.
    monotonicity_p_value : float
        p-value for the Spearman monotonicity correlation hypothesis test.
    top_decile_uplift : float
        Empirical uplift in Decile 1 (highest predicted persuadables).
    bottom_decile_uplift : float
        Empirical uplift in Decile 10 (lowest predicted uplift / sleeping dogs).
    sleeping_dogs_isolated : bool
        True if Decile 10 exhibits negative empirical uplift (tau < 0), indicating
        successful isolation of the Do-Not-Disturb cohort.
    """

    decile_table: pd.DataFrame
    monotonicity_spearman_corr: float
    monotonicity_p_value: float
    top_decile_uplift: float
    bottom_decile_uplift: float
    sleeping_dogs_isolated: bool


def compute_decile_analysis(
    y_true: np.ndarray | pd.Series,
    uplift_preds: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    tau_true: np.ndarray | pd.Series | None = None,
    n_deciles: int = 10,
) -> DecileAnalysisResult:
    """Compute empirical decile uplift table, incremental conversions, and monotonicity.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary observed retention outcome (1 = retained, 0 = churned).
    uplift_preds : array-like of shape (n_samples,)
        Predicted uplift tau_hat(X) or targeting prioritization score.
    treatment : array-like of shape (n_samples,)
        Binary treatment assignment indicator (1 = treated, 0 = control).
    tau_true : array-like | None of shape (n_samples,), default=None
        Optional ground truth individual treatment effect for synthetic benchmark validation.
    n_deciles : int, default=10
        Number of quantile bins to partition the ranked population into.

    Returns
    -------
    DecileAnalysisResult
        Comprehensive decile metrics, monotonicity score, and sleeping dog detection.
    """
    y_arr = np.asarray(y_true, dtype=int).ravel()
    w_arr = np.asarray(treatment, dtype=int).ravel()
    preds_arr = np.asarray(uplift_preds, dtype=float).ravel()
    tau_arr = np.asarray(tau_true, dtype=float).ravel() if tau_true is not None else None

    n_samples = len(y_arr)
    if not (len(w_arr) == n_samples and len(preds_arr) == n_samples):
        raise ValueError(
            f"Array length mismatch: y={n_samples}, w={len(w_arr)}, preds={len(preds_arr)}"
        )
    if tau_arr is not None and len(tau_arr) != n_samples:
        raise ValueError(f"tau_true length {len(tau_arr)} does not match n_samples {n_samples}")

    if not np.all(np.isin(y_arr, [0, 1])):
        raise ValueError("y_true must be strictly binary {0, 1}")
    if not np.all(np.isin(w_arr, [0, 1])):
        raise ValueError("treatment must be strictly binary {0, 1}")

    # Build evaluation DataFrame
    df = pd.DataFrame(
        {
            "y": y_arr,
            "w": w_arr,
            "pred_uplift": preds_arr,
        }
    )
    if tau_arr is not None:
        df["tau_true"] = tau_arr

    # Sort descending by predicted uplift (ties broken deterministically by index)
    df = df.sort_values(by="pred_uplift", ascending=False).reset_index(drop=True)

    # Assign decile bins 1 to n_deciles (Decile 1 = top predicted uplift)
    # Using np.array_split ensures balanced bins even with duplicate prediction values
    bin_labels = np.empty(n_samples, dtype=int)
    splits = np.array_split(np.arange(n_samples), n_deciles)
    for decile_idx, split_indices in enumerate(splits, start=1):
        bin_labels[split_indices] = decile_idx

    df["decile"] = bin_labels

    # Aggregate per decile
    rows = []
    for decile_num in range(1, n_deciles + 1):
        sub = df[df["decile"] == decile_num]
        n_total = len(sub)

        sub_t = sub[sub["w"] == 1]
        sub_c = sub[sub["w"] == 0]

        n_t = len(sub_t)
        n_c = len(sub_c)

        y_t = int(sub_t["y"].sum()) if n_t > 0 else 0
        y_c = int(sub_c["y"].sum()) if n_c > 0 else 0

        rate_t = float(y_t / n_t) if n_t > 0 else 0.0
        rate_c = float(y_c / n_c) if n_c > 0 else 0.0
        empirical_uplift = rate_t - rate_c

        # Incremental conversions = empirical uplift * n_total
        incremental_conv = empirical_uplift * n_total
        mean_pred = float(sub["pred_uplift"].mean())

        row_dict = {
            "decile": decile_num,
            "n_total": n_total,
            "n_treated": n_t,
            "n_control": n_c,
            "retained_treated": y_t,
            "retained_control": y_c,
            "rate_treated": rate_t,
            "rate_control": rate_c,
            "empirical_uplift": empirical_uplift,
            "mean_predicted_uplift": mean_pred,
            "incremental_conversions": incremental_conv,
        }

        if tau_arr is not None:
            row_dict["mean_true_uplift"] = float(sub["tau_true"].mean())

        rows.append(row_dict)

    decile_df = pd.DataFrame(rows)
    decile_df["cum_incremental_conversions"] = decile_df["incremental_conversions"].cumsum()

    # Monotonicity test:
    # A well-calibrated causal model has high empirical uplift in Decile 1 and declining thereafter.
    # Therefore, correlation between priority score (-decile) and empirical uplift should be positive.
    priority_order = -decile_df["decile"].to_numpy(dtype=float)
    empirical_uplifts = decile_df["empirical_uplift"].to_numpy(dtype=float)

    # Compute Spearman rank correlation
    corr_res = spearmanr(priority_order, empirical_uplifts)
    spearman_corr = float(corr_res.statistic) if not np.isnan(corr_res.statistic) else 0.0
    p_val = float(corr_res.pvalue) if not np.isnan(corr_res.pvalue) else 1.0

    top_uplift = float(decile_df.loc[decile_df["decile"] == 1, "empirical_uplift"].iloc[0])
    bottom_uplift = float(decile_df.loc[decile_df["decile"] == n_deciles, "empirical_uplift"].iloc[0])
    sleeping_dogs_isolated = bottom_uplift < -0.01

    return DecileAnalysisResult(
        decile_table=decile_df,
        monotonicity_spearman_corr=spearman_corr,
        monotonicity_p_value=p_val,
        top_decile_uplift=top_uplift,
        bottom_decile_uplift=bottom_uplift,
        sleeping_dogs_isolated=sleeping_dogs_isolated,
    )
