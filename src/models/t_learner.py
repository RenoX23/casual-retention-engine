"""T-Learner (Two Models) Uplift Estimator.

Primary uplift engine training two separate LightGBM models on treatment
and control cohorts respectively:
    mu_1(X) = E[Y | X, W=1]
    mu_0(X) = E[Y | X, W=0]
Uplift is estimated as:
    tau_hat(X) = mu_1_hat(X) - mu_0_hat(X)
Supports probability calibration (Platt scaling / isotonic) to eliminate probability distortion.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.models.base_learner import BaseUpliftLearner

logger = logging.getLogger(__name__)


class TLearner(BaseUpliftLearner):
    """Two-model LightGBM causal uplift estimator with optional probability calibration.

    Forces independent estimation of the treatment response surface and the control
    response surface, avoiding feature masking where treatment effect is regularized
    to zero.
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
        calibrate_probabilities: bool = True,
        calibration_method: str = "sigmoid",
        calibration_cv: int = 3,
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
        self.calibrate_probabilities = calibrate_probabilities
        self.calibration_method = calibration_method
        self.calibration_cv = calibration_cv
        self.kwargs = kwargs

        # Internal model attributes
        self.model_0_: LGBMClassifier | CalibratedClassifierCV | None = None
        self.model_1_: LGBMClassifier | CalibratedClassifierCV | None = None
        self.raw_model_0_: LGBMClassifier | None = None
        self.raw_model_1_: LGBMClassifier | None = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        w: np.ndarray,
        y: np.ndarray,
    ) -> TLearner:
        """Fit dual estimators on separated control (W=0) and treated (W=1) units."""
        X_arr, w_arr, y_arr = self._validate_inputs(X, w, y)
        assert w_arr is not None and y_arr is not None

        ctrl_mask = w_arr == 0
        treat_mask = w_arr == 1

        if ctrl_mask.sum() == 0:
            raise ValueError("No control samples (w=0) found in training data.")
        if treat_mask.sum() == 0:
            raise ValueError("No treated samples (w=1) found in training data.")

        logger.info(
            "Fitting TLearner: Control cohort=%d samples, Treated cohort=%d samples",
            ctrl_mask.sum(),
            treat_mask.sum(),
        )

        # 1. Instantiate base LightGBM models
        self.raw_model_0_ = LGBMClassifier(
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

        self.raw_model_1_ = LGBMClassifier(
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

        # 2. Fit with or without cross-validated probability calibration
        if self.calibrate_probabilities:
            logger.info("Applying %s probability calibration (cv=%d)...", self.calibration_method, self.calibration_cv)
            self.model_0_ = CalibratedClassifierCV(
                estimator=self.raw_model_0_,
                method=self.calibration_method,
                cv=self.calibration_cv,
            )
            self.model_1_ = CalibratedClassifierCV(
                estimator=self.raw_model_1_,
                method=self.calibration_method,
                cv=self.calibration_cv,
            )
            self.model_0_.fit(X_arr[ctrl_mask], y_arr[ctrl_mask])
            self.model_1_.fit(X_arr[treat_mask], y_arr[treat_mask])

            # Also fit raw models on full subsets for TreeSHAP introspection
            self.raw_model_0_.fit(X_arr[ctrl_mask], y_arr[ctrl_mask])
            self.raw_model_1_.fit(X_arr[treat_mask], y_arr[treat_mask])
        else:
            self.raw_model_0_.fit(X_arr[ctrl_mask], y_arr[ctrl_mask])
            self.raw_model_1_.fit(X_arr[treat_mask], y_arr[treat_mask])
            self.model_0_ = self.raw_model_0_
            self.model_1_ = self.raw_model_1_

        self.is_fitted_ = True
        return self

    def predict_potential_outcomes(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate counterfactual potential outcome probabilities (mu_0, mu_1)."""
        if not self.is_fitted_ or self.model_0_ is None or self.model_1_ is None:
            raise RuntimeError("TLearner is not fitted yet.")

        X_arr, _, _ = self._validate_inputs(X)
        mu_0 = self.model_0_.predict_proba(X_arr)[:, 1]
        mu_1 = self.model_1_.predict_proba(X_arr)[:, 1]
        return mu_0, mu_1

    def predict_uplift(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict Individual Treatment Effect tau_hat(X) = mu_1(X) - mu_0(X)."""
        mu_0, mu_1 = self.predict_potential_outcomes(X)
        return mu_1 - mu_0
