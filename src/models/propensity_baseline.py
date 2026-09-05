"""Propensity Baseline Model (The Traditional Churn Benchmark).

Implements standard supervised classification predicting churn propensity:
    P(Churn | X) = 1 - P(Y=1 | X)
Serves as the benchmark baseline illustrating the economic flaws of targeting
customers solely based on churn risk rather than treatment effect uplift.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin

from src.models.base_learner import BaseUpliftLearner

logger = logging.getLogger(__name__)


class PropensityBaseline(BaseUpliftLearner, ClassifierMixin):
    """Standard supervised churn propensity estimator using LightGBM.

    Trains a gradient boosting classifier on pre-treatment covariates X to
    predict factual retention Y. When scoring for campaign targeting, it ranks
    customers by churn probability P(Churn | X).
    """

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.model_: LGBMClassifier | None = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        w: np.ndarray,
        y: np.ndarray,
    ) -> PropensityBaseline:
        """Fit the supervised retention classifier.

        Note: Traditional churn models ignore treatment assignment w and train
        directly on observable outcome y.
        """
        X_arr, _, y_arr = self._validate_inputs(X, w, y)
        X_clean = np.ascontiguousarray(X_arr, dtype=np.float64)
        y_clean = np.ascontiguousarray(y_arr, dtype=int)

        logger.info("Fitting PropensityBaseline (LGBMClassifier) on %d samples...", len(X_clean))
        self.model_ = LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=-1,
            **self.kwargs,
        )
        self.model_.fit(X_clean, y_clean)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict retention probability array [P(Churn), P(Retained)]."""
        if not self.is_fitted_ or self.model_ is None:
            raise RuntimeError("PropensityBaseline is not fitted yet.")
        X_arr, _, _ = self._validate_inputs(X)
        return self.model_.predict_proba(X_arr)

    def predict_churn_propensity(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict probability of churn: P(Churn | X) = 1 - P(Y=1 | X)."""
        proba = self.predict_proba(X)
        return proba[:, 0]

    def predict_retention_propensity(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict probability of retention: P(Y=1 | X)."""
        proba = self.predict_proba(X)
        return proba[:, 1]

    def predict_uplift(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Rank score for traditional targeting campaigns.

        Traditional churn targeting ranks customers by churn propensity:
        those most likely to cancel are targeted first.
        Returns P(Churn | X) as the targeting prioritization score.
        """
        return self.predict_churn_propensity(X)
