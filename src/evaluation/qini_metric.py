"""Qini Metric and Area Under Uplift Curve (AUUC) Evaluation Engine.

Provides mathematically rigorous implementations of:
1. Cumulative Qini Curve: Q(u) = Y_t(u) - Y_c(u) * (N_t / N_c)
2. Cumulative Gain Curve: G(u) = (Y_t(u) / N_t(u) - Y_c(u) / N_c(u)) * k
3. Area Under Uplift Curve (AUUC) via numerical trapezoidal quadrature.
4. Normalized Qini Score (Q_norm):
       Q_norm = (AUUC_model - AUUC_random) / (AUUC_optimal - AUUC_random)
5. Support for both observable RCT data and oracle benchmark evaluation against latent CATE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Compute numerical integral using trapezoidal rule with NumPy 1.x / 2.x compatibility."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))  # type: ignore[attr-defined]


@dataclass(frozen=True)
class QiniCurveResult:
    """Container holding computed Qini and Uplift evaluation trajectories.

    Attributes
    ----------
    fractions : np.ndarray
        Fraction of population targeted, ranging from 0.0 to 1.0.
    qini_model : np.ndarray
        Cumulative incremental gains achieved by model ranking.
    qini_random : np.ndarray
        Expected cumulative incremental gains from uniform random targeting.
    qini_optimal : np.ndarray
        Upper bound cumulative incremental gains (oracle or theoretical best).
    cum_gain_model : np.ndarray
        Cumulative gain curve: incremental conversion volume at each cutoff.
    auuc_model : float
        Area under the model's Qini curve.
    auuc_random : float
        Area under the random baseline curve.
    auuc_optimal : float
        Area under the optimal ceiling curve.
    qini_score : float
        Normalized Qini score in [-1.0, 1.0], where 1.0 represents perfect ranking
        and 0.0 represents parity with random targeting.
    """

    fractions: np.ndarray
    qini_model: np.ndarray
    qini_random: np.ndarray
    qini_optimal: np.ndarray
    cum_gain_model: np.ndarray
    auuc_model: float
    auuc_random: float
    auuc_optimal: float
    qini_score: float


def _compute_raw_qini_trajectory(
    y_sorted: np.ndarray,
    w_sorted: np.ndarray,
    total_nt: int,
    total_nc: int,
    eval_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate cumulative Qini and gain values along designated evaluation indices.

    Parameters
    ----------
    y_sorted : np.ndarray
        Outcomes ordered by targeted score descending.
    w_sorted : np.ndarray
        Treatment indicators ordered by targeted score descending.
    total_nt : int
        Total treated count across entire cohort.
    total_nc : int
        Total control count across entire cohort.
    eval_indices : np.ndarray
        Sample count cutoffs k where metrics are evaluated.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        qini_values, cum_gains along the eval_indices.
    """
    cum_y_treated = np.cumsum(y_sorted * w_sorted)
    cum_y_control = np.cumsum(y_sorted * (1 - w_sorted))
    cum_n_treated = np.cumsum(w_sorted)
    cum_n_control = np.cumsum(1 - w_sorted)

    q_factor = float(total_nt) / float(total_nc) if total_nc > 0 else 1.0

    qini_vals = [0.0]
    gain_vals = [0.0]

    for k in eval_indices:
        idx = k - 1
        y_t = float(cum_y_treated[idx])
        y_c = float(cum_y_control[idx])
        n_t = float(cum_n_treated[idx])
        n_c = float(cum_n_control[idx])

        # Standard Qini: Y_t - Y_c * (N_t / N_c)
        q = y_t - y_c * q_factor
        qini_vals.append(q)

        # Cumulative Gain: (Y_t / N_t - Y_c / N_c) * k
        if n_t > 0 and n_c > 0:
            rate_t = y_t / n_t
            rate_c = y_c / n_c
            gain = (rate_t - rate_c) * float(k)
        else:
            gain = 0.0
        gain_vals.append(gain)

    return np.array(qini_vals, dtype=float), np.array(gain_vals, dtype=float)


def compute_qini_curve(
    y_true: np.ndarray | pd.Series,
    uplift_preds: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    tau_true: np.ndarray | pd.Series | None = None,
    n_points: int = 100,
) -> QiniCurveResult:
    """Compute the Qini curve, cumulative gain, AUUC, and normalized Qini score.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Binary observed retention outcomes (1 = retained, 0 = churned).
    uplift_preds : np.ndarray of shape (n_samples,)
        Predicted uplift / CATE values tau_hat(X), or targeting priority scores.
    treatment : np.ndarray of shape (n_samples,)
        Binary treatment indicator (1 = treated, 0 = control).
    tau_true : np.ndarray | None of shape (n_samples,), default=None
        Optional ground truth individual treatment effect for oracle upper bound.
        If None, the empirical theoretical best ordering is computed from (y, w).
    n_points : int, default=100
        Number of evaluation points along the cumulative population fraction.

    Returns
    -------
    QiniCurveResult
        Structured container with complete trajectories, AUUC integrals, and normalized score.
    """
    y_arr = np.asarray(y_true, dtype=int).ravel()
    w_arr = np.asarray(treatment, dtype=int).ravel()
    preds_arr = np.asarray(uplift_preds, dtype=float).ravel()

    n_samples = len(y_arr)
    if not (len(w_arr) == n_samples and len(preds_arr) == n_samples):
        raise ValueError(
            f"Array length mismatch: y has {n_samples}, w has {len(w_arr)}, preds has {len(preds_arr)}"
        )

    if not np.all(np.isin(y_arr, [0, 1])):
        raise ValueError(f"y_true must be strictly binary {{0, 1}}, found {np.unique(y_arr)}")
    if not np.all(np.isin(w_arr, [0, 1])):
        raise ValueError(f"treatment must be strictly binary {{0, 1}}, found {np.unique(w_arr)}")

    total_nt = int(np.sum(w_arr))
    total_nc = int(n_samples - total_nt)

    if total_nt == 0 or total_nc == 0:
        raise ValueError(
            f"Dataset must contain both treated and control units (found N_t={total_nt}, N_c={total_nc})"
        )

    # 1. Sort by model uplift predictions descending (with stable sort for determinism)
    model_order = np.argsort(-preds_arr, kind="mergesort")
    y_model = y_arr[model_order]
    w_model = w_arr[model_order]

    # Evaluation cutoff points k in [1, N]
    n_eval_points = min(n_points, n_samples)
    eval_indices = np.linspace(1, n_samples, num=n_eval_points, dtype=int)
    # Ensure unique indices
    eval_indices = np.unique(eval_indices)

    fractions = np.concatenate([[0.0], eval_indices / n_samples])

    # 2. Model Qini and Cumulative Gain
    qini_model, cum_gain_model = _compute_raw_qini_trajectory(
        y_model, w_model, total_nt, total_nc, eval_indices
    )

    # 3. Random Baseline: straight line from (0, 0) to (1.0, qini_model[-1])
    qini_end = qini_model[-1]
    qini_random = fractions * qini_end

    # 4. Optimal / Oracle Upper Bound
    if tau_true is not None:
        tau_arr = np.asarray(tau_true, dtype=float).ravel()
        if len(tau_arr) != n_samples:
            raise ValueError(f"tau_true length {len(tau_arr)} does not match n_samples {n_samples}")
        optimal_order = np.argsort(-tau_arr, kind="mergesort")
    else:
        # Theoretical maximum sorting on observable RCT data:
        # 1. Treated responders (W=1, Y=1) [Uplift = +1]
        # 2. Control non-responders (W=0, Y=0) [Uplift = 0, no harm]
        # 3. Treated non-responders (W=1, Y=0) and Control responders (W=0, Y=1)
        # Numerical score: W * Y * 2 + (1 - W) * (1 - Y)
        theoretical_rank_score = (
            w_arr * y_arr * 2.0
            + (1 - w_arr) * (1 - y_arr) * 1.0
            - (1 - w_arr) * y_arr * 2.0
        )
        optimal_order = np.argsort(-theoretical_rank_score, kind="mergesort")

    y_opt = y_arr[optimal_order]
    w_opt = w_arr[optimal_order]
    qini_optimal, _ = _compute_raw_qini_trajectory(
        y_opt, w_opt, total_nt, total_nc, eval_indices
    )

    # 5. Compute Area Under Curves (AUUC) via trapezoidal integration
    auuc_model = _trapezoid(qini_model, fractions)
    auuc_random = _trapezoid(qini_random, fractions)
    auuc_optimal = _trapezoid(qini_optimal, fractions)

    # 6. Normalized Qini Score: Q_norm = (AUUC_model - AUUC_rand) / (AUUC_opt - AUUC_rand)
    optimal_delta = auuc_optimal - auuc_random
    if optimal_delta > 1e-12:
        qini_score = (auuc_model - auuc_random) / optimal_delta
        # Bound score logically to [-1.0, 1.0]
        qini_score = float(np.clip(qini_score, -1.0, 1.0))
    else:
        qini_score = 0.0

    return QiniCurveResult(
        fractions=fractions,
        qini_model=qini_model,
        qini_random=qini_random,
        qini_optimal=qini_optimal,
        cum_gain_model=cum_gain_model,
        auuc_model=float(auuc_model),
        auuc_random=float(auuc_random),
        auuc_optimal=float(auuc_optimal),
        qini_score=float(qini_score),
    )
