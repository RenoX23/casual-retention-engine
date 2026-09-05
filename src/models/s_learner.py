"""S-Learner (Single Model) Uplift Estimator.

Trains a single supervised model mu(X, W) with treatment indicator W
concatenated as an explicit feature.
Uplift is estimated as:
    tau_hat(X) = mu_hat(X, 1) - mu_hat(X, 0)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.models.base_learner import BaseUpliftLearner

logger = logging.getLogger(__name__)


class SLearner(BaseUpliftLearner):
    """Single-model LightGBM uplift estimator.

    Appends treatment indicator W as a feature to pre-treatment covariates X,
    then predicts counterfactual outcomes by evaluating the model with W=1 and W=0.
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
    ) -> SLearner:
        """Fit single estimator on augmented feature matrix [X, W]."""
        X_arr, w_arr, y_arr = self._validate_inputs(X, w, y)
        assert w_arr is not None and y_arr is not None

        # Augment X with treatment indicator column W
        X_augmented = np.column_stack([X_arr, w_arr.astype(float)])

        logger.info("Fitting SLearner (LGBMClassifier) on %d augmented samples...", len(X_augmented))
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
        self.model_.fit(X_augmented, y_arr)
        self.is_fitted_ = True
        return self

    def predict_potential_outcomes(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate counterfactual potential outcome probabilities (mu_0, mu_1)."""
        if not self.is_fitted_ or self.model_ is None:
            raise RuntimeError("SLearner is not fitted yet.")

        X_arr, _, _ = self._validate_inputs(X)
        n_samples = len(X_arr)

        # Evaluate model with W=0 (control condition)
        X_ctrl = np.column_stack([X_arr, np.zeros(n_samples, dtype=float)])
        mu_0 = self.model_.predict_proba(X_ctrl)[:, 1]

        # Evaluate model with W=1 (treatment condition)
        X_treat = np.column_stack([X_arr, np.ones(n_samples, dtype=float)])
        mu_1 = self.model_.predict_proba(X_treat)[:, 1]

        return mu_0, mu_1

    def predict_uplift(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict Individual Treatment Effect tau_hat(X) = mu_1(X) - mu_0(X)."""
        mu_0, mu_1 = self.predict_potential_outcomes(X)
        return mu_1 - mu_0
