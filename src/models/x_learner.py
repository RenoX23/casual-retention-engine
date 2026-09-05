"""X-Learner (Counterfactual Imputation) Uplift Estimator.

Implements Künzel et al. (2019) X-Learner:
    1. Stage 1: Train mu_1(X) on treated and mu_0(X) on control.
    2. Stage 2: Impute counterfactual effects:
       D_1 = Y_1 - mu_0(X_1) for treated units.
       D_0 = mu_1(X_0) - Y_0 for control units.
    3. Stage 3: Train regressors tau_1(X) on (X_1, D_1) and tau_0(X) on (X_0, D_0).
    4. Stage 4: Combine using propensity score:
       tau_hat(X) = e(X)*tau_0(X) + (1 - e(X))*tau_1(X).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from src.models.base_learner import BaseUpliftLearner

logger = logging.getLogger(__name__)


class XLearner(BaseUpliftLearner):
    """Counterfactual imputation X-Learner using LightGBM.

    Particularly effective in scenarios with asymmetric treatment allocation,
    counterfactually imputing missing potential outcomes before regressing on imputed effects.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = 1,
        propensity_model: bool = True,
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
        self.propensity_model = propensity_model
        self.kwargs = kwargs

        # Stage 1: Response surface classifiers
        self.stage1_mu0_: LGBMClassifier | None = None
        self.stage1_mu1_: LGBMClassifier | None = None

        # Stage 2: Effect regressors
        self.stage2_tau0_: LGBMRegressor | None = None
        self.stage2_tau1_: LGBMRegressor | None = None

        # Propensity model e(X) = P(W=1 | X)
        self.propensity_clf_: LGBMClassifier | None = None
        self.marginal_propensity_: float = 0.5

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        w: np.ndarray,
        y: np.ndarray,
    ) -> XLearner:
        """Fit the four-stage X-Learner meta-algorithm."""
        X_arr, w_arr, y_arr = self._validate_inputs(X, w, y)
        assert w_arr is not None and y_arr is not None

        ctrl_mask = w_arr == 0
        treat_mask = w_arr == 1

        X_0, y_0 = X_arr[ctrl_mask], y_arr[ctrl_mask]
        X_1, y_1 = X_arr[treat_mask], y_arr[treat_mask]

        if len(X_0) == 0 or len(X_1) == 0:
            raise ValueError("Both treatment and control cohorts must have >= 1 sample.")

        logger.info("Fitting X-Learner Stage 1: Response classifiers on %d control, %d treated", len(X_0), len(X_1))

        # 1. Stage 1: Fit base classifiers
        self.stage1_mu0_ = LGBMClassifier(
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
        self.stage1_mu0_.fit(X_0, y_0)

        self.stage1_mu1_ = LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state + 1,
            n_jobs=self.n_jobs,
            verbosity=-1,
            **self.kwargs,
        )
        self.stage1_mu1_.fit(X_1, y_1)

        # 2. Stage 2: Counterfactual imputation
        # D_1 = Y_1 - mu_0(X_1) (imputed counterfactual control outcome subtracted from observed treated)
        # D_0 = mu_1(X_0) - Y_0 (observed control outcome subtracted from imputed counterfactual treated)
        logger.info("Computing X-Learner Stage 2: Counterfactual imputed effects D_1 and D_0...")
        mu0_on_X1 = self.stage1_mu0_.predict_proba(X_1)[:, 1]
        mu1_on_X0 = self.stage1_mu1_.predict_proba(X_0)[:, 1]

        D_1 = y_1.astype(float) - mu0_on_X1
        D_0 = mu1_on_X0 - y_0.astype(float)

        # 3. Stage 3: Train regressors on imputed treatment effects
        logger.info("Fitting X-Learner Stage 3: Effect regressors tau_1(X) and tau_0(X)...")
        self.stage2_tau1_ = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state + 2,
            n_jobs=self.n_jobs,
            verbosity=-1,
            **self.kwargs,
        )
        self.stage2_tau1_.fit(X_1, D_1)

        self.stage2_tau0_ = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state + 3,
            n_jobs=self.n_jobs,
            verbosity=-1,
            **self.kwargs,
        )
        self.stage2_tau0_.fit(X_0, D_0)

        # 4. Stage 4: Propensity score estimation e(X) = P(W=1 | X)
        self.marginal_propensity_ = float(w_arr.mean())
        if self.propensity_model:
            logger.info("Fitting propensity estimator e(X) = P(W=1 | X)...")
            self.propensity_clf_ = LGBMClassifier(
                n_estimators=50,
                learning_rate=self.learning_rate,
                max_depth=3,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=-1,
            )
            self.propensity_clf_.fit(X_arr, w_arr)

        self.is_fitted_ = True
        return self

    def predict_propensity(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict propensity score e(X) = P(W=1 | X)."""
        if not self.is_fitted_:
            raise RuntimeError("XLearner is not fitted yet.")
        X_arr, _, _ = self._validate_inputs(X)
        if self.propensity_clf_ is not None:
            return self.propensity_clf_.predict_proba(X_arr)[:, 1]
        return np.full(len(X_arr), self.marginal_propensity_, dtype=float)

    def predict_uplift(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict uplift: tau_hat(X) = e(X)*tau_0(X) + (1 - e(X))*tau_1(X)."""
        if not self.is_fitted_ or self.stage2_tau0_ is None or self.stage2_tau1_ is None:
            raise RuntimeError("XLearner is not fitted yet.")

        X_arr, _, _ = self._validate_inputs(X)
        tau_0 = self.stage2_tau0_.predict(X_arr)
        tau_1 = self.stage2_tau1_.predict(X_arr)
        e_x = self.predict_propensity(X_arr)

        # Combine: weight tau_0 by e(X) and tau_1 by (1 - e(X))
        tau_combined = e_x * tau_0 + (1.0 - e_x) * tau_1
        return np.clip(tau_combined, -1.0, 1.0)

    def predict_potential_outcomes(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate potential outcome probabilities from Stage 1 models."""
        if not self.is_fitted_ or self.stage1_mu0_ is None or self.stage1_mu1_ is None:
            raise RuntimeError("XLearner is not fitted yet.")

        X_arr, _, _ = self._validate_inputs(X)
        mu_0 = self.stage1_mu0_.predict_proba(X_arr)[:, 1]
        mu_1 = self.stage1_mu1_.predict_proba(X_arr)[:, 1]
        return mu_0, mu_1
