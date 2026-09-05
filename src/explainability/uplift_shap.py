"""TreeSHAP Differential Feature Attribution Engine for Causal Uplift Modeling.

Explains heterogeneous treatment effects (why individual accounts exhibit positive
uplift vs. negative uplift / Sleeping Dog risk) by decomposing the T-Learner:
    phi_i(uplift) = phi_i(mu_1) - phi_i(mu_0)
where phi_i(mu_1) is the TreeSHAP attribution on the treatment response surface,
and phi_i(mu_0) is the TreeSHAP attribution on the control response surface.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Any

with contextlib.suppress(ImportError):
    import lightgbm  # noqa: F401

import numpy as np
import pandas as pd
import shap

from src.models.t_learner import TLearner

logger = logging.getLogger(__name__)


def _extract_positive_class_shap(raw_shap: Any) -> np.ndarray:
    """Extract positive class SHAP array handling both list and ndarray return types."""
    if isinstance(raw_shap, list):
        arr = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
        return np.asarray(arr, dtype=float)
    return np.asarray(raw_shap, dtype=float)


def _extract_base_value(expected_val: Any) -> float:
    """Extract scalar baseline expectation handling both float and iterable outputs."""
    if isinstance(expected_val, (list, np.ndarray)):
        arr = np.asarray(expected_val).ravel()
        return float(arr[1] if len(arr) > 1 else arr[0])
    return float(expected_val)


@dataclass(frozen=True)
class UpliftExplanation:
    """Container holding differential TreeSHAP uplift attributions.

    Attributes
    ----------
    shap_values_diff : np.ndarray of shape (n_samples, n_features)
        Differential SHAP values: phi(mu_1) - phi(mu_0).
    shap_values_treated : np.ndarray of shape (n_samples, n_features)
        SHAP values on the treated response model mu_1.
    shap_values_control : np.ndarray of shape (n_samples, n_features)
        SHAP values on the control response model mu_0.
    base_value_diff : float
        Expected value differential: E[mu_1] - E[mu_0].
    feature_names : list[str]
        List of covariate names.
    feature_importance : pd.DataFrame
        Global ranking of features by mean absolute differential uplift impact.
    data : pd.DataFrame
        Input covariate feature matrix.
    """

    shap_values_diff: np.ndarray
    shap_values_treated: np.ndarray
    shap_values_control: np.ndarray
    base_value_diff: float
    feature_names: list[str]
    feature_importance: pd.DataFrame
    data: pd.DataFrame


class UpliftTreeExplainer:
    """Explains causal uplift models using differential TreeSHAP.

    Decomposes the dual-model T-Learner by fitting TreeSHAP explainers to both
    treatment and control tree ensembles, extracting the heterogeneous treatment
    effect attribution for each customer covariate.
    """

    def __init__(
        self,
        t_learner: TLearner,
        feature_names: list[str] | None = None,
    ) -> None:
        """Initialize the uplift explainer with a fitted T-Learner.

        Parameters
        ----------
        t_learner : TLearner
            Fitted TLearner model instance.
        feature_names : list[str] | None, default=None
            Optional list of feature names corresponding to columns in X.
        """
        if not t_learner.is_fitted_:
            raise RuntimeError("TLearner must be fitted before initializing UpliftTreeExplainer.")

        # Utilize the raw uncalibrated tree boosters for exact TreeSHAP computation
        if t_learner.raw_model_0_ is None or t_learner.raw_model_1_ is None:
            raise RuntimeError("TLearner is missing raw fitted tree estimators for TreeSHAP.")

        self.t_learner = t_learner
        self.feature_names = feature_names

        logger.info("Initializing TreeExplainer on raw treated and control LightGBM models...")
        self.explainer_1_ = shap.TreeExplainer(t_learner.raw_model_1_)
        self.explainer_0_ = shap.TreeExplainer(t_learner.raw_model_0_)

        self.base_val_1_ = _extract_base_value(self.explainer_1_.expected_value)
        self.base_val_0_ = _extract_base_value(self.explainer_0_.expected_value)
        self.base_value_diff_ = self.base_val_1_ - self.base_val_0_

    def explain(self, X: pd.DataFrame | np.ndarray) -> UpliftExplanation:
        """Compute differential SHAP values for each sample in X.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray of shape (n_samples, n_features)
            Customer covariates to explain.

        Returns
        -------
        UpliftExplanation
            Container with differential SHAP values, feature importance, and baselines.
        """
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X_df = X.copy()
            X_arr = np.ascontiguousarray(X.to_numpy(), dtype=np.float64)
        else:
            X_arr = np.ascontiguousarray(X, dtype=np.float64)
            feature_names = self.feature_names or [f"feature_{i}" for i in range(X_arr.shape[1])]
            X_df = pd.DataFrame(X_arr, columns=feature_names)

        # Compute raw SHAP attributions on both treatment arms
        raw_shap_1 = self.explainer_1_.shap_values(X_arr)
        raw_shap_0 = self.explainer_0_.shap_values(X_arr)

        shap_1 = _extract_positive_class_shap(raw_shap_1)
        shap_0 = _extract_positive_class_shap(raw_shap_0)

        # Differential SHAP attribution: Delta phi = phi(W=1) - phi(W=0)
        shap_diff = shap_1 - shap_0

        # Compute global feature importance: Mean Absolute Differential SHAP
        mean_abs_impact = np.mean(np.abs(shap_diff), axis=0)
        mean_dir_impact = np.mean(shap_diff, axis=0)

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_uplift_shap": mean_abs_impact,
                "mean_uplift_shap": mean_dir_impact,
            }
        ).sort_values(by="mean_abs_uplift_shap", ascending=False).reset_index(drop=True)

        importance_df["rank"] = np.arange(1, len(importance_df) + 1)

        return UpliftExplanation(
            shap_values_diff=shap_diff,
            shap_values_treated=shap_1,
            shap_values_control=shap_0,
            base_value_diff=self.base_value_diff_,
            feature_names=feature_names,
            feature_importance=importance_df,
            data=X_df,
        )

    def explain_instance(
        self,
        explanation: UpliftExplanation,
        idx: int,
        top_k: int = 8,
    ) -> dict[str, Any]:
        """Generate structured breakdown for a single customer instance.

        Ideal for interactive waterfall plots and UI drill-downs in dashboards.

        Parameters
        ----------
        explanation : UpliftExplanation
            Precomputed explanation container.
        idx : int
            Row index of the customer account.
        top_k : int, default=8
            Number of top driving features to highlight before grouping remainder.

        Returns
        -------
        dict[str, Any]
            Structured waterfall dictionary containing feature values, SHAP contributions,
            and base values.
        """
        n_samples = len(explanation.shap_values_diff)
        if not (0 <= idx < n_samples):
            raise IndexError(f"Index {idx} out of range for explanation with {n_samples} rows.")

        sample_diff = explanation.shap_values_diff[idx]
        sample_vals = explanation.data.iloc[idx].to_dict()

        # Sort features by absolute contribution for this instance
        abs_order = np.argsort(-np.abs(sample_diff))
        sorted_feats = [explanation.feature_names[i] for i in abs_order]
        sorted_shaps = [float(sample_diff[i]) for i in abs_order]
        sorted_vals = [sample_vals[feat] for feat in sorted_feats]

        top_features = sorted_feats[:top_k]
        top_shaps = sorted_shaps[:top_k]
        top_values = sorted_vals[:top_k]

        other_sum = float(sum(sorted_shaps[top_k:])) if len(sorted_shaps) > top_k else 0.0

        return {
            "instance_index": idx,
            "base_value": float(explanation.base_value_diff),
            "total_uplift_shap": float(np.sum(sample_diff) + explanation.base_value_diff),
            "features": top_features,
            "feature_values": top_values,
            "shap_contributions": top_shaps,
            "other_features_contribution": other_sum,
        }

    def explain_archetypes(
        self,
        explanation: UpliftExplanation,
        archetypes: pd.Series | np.ndarray,
    ) -> dict[str, pd.DataFrame]:
        """Aggregate differential SHAP attributions across customer archetypes.

        Compares which features drive positive uplift in Persuadables vs negative
        uplift in Sleeping Dogs.

        Parameters
        ----------
        explanation : UpliftExplanation
            Precomputed explanation container.
        archetypes : pd.Series | np.ndarray of shape (n_samples,)
            Archetype labels (e.g. 'Persuadable', 'Sleeping Dog', 'Sure Thing', 'Lost Cause').

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary mapping each archetype to its feature attribution summary.
        """
        arch_arr = np.asarray(archetypes).ravel()
        if len(arch_arr) != len(explanation.shap_values_diff):
            raise ValueError("Length of archetypes does not match explanation sample count.")

        archetype_summaries = {}
        unique_archetypes = np.unique(arch_arr)

        for arch in unique_archetypes:
            mask = arch_arr == arch
            if not np.any(mask):
                continue

            arch_diff = explanation.shap_values_diff[mask]
            summary_df = pd.DataFrame(
                {
                    "feature": explanation.feature_names,
                    "mean_uplift_shap": np.mean(arch_diff, axis=0),
                    "mean_abs_uplift_shap": np.mean(np.abs(arch_diff), axis=0),
                }
            ).sort_values(by="mean_abs_uplift_shap", ascending=False).reset_index(drop=True)

            archetype_summaries[str(arch)] = summary_df

        return archetype_summaries
