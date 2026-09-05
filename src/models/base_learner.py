"""Abstract Base Class for Causal Uplift Estimators."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class BaseUpliftLearner(ABC, BaseEstimator):
    """Abstract base class for all causal uplift meta-learners.

    Enforces standard scikit-learn compatible fit and predict_uplift interfaces
    for estimating the Conditional Average Treatment Effect (CATE):
        tau(X) = E[Y(1) - Y(0) | X]
    """

    def __init__(self) -> None:
        self.is_fitted_: bool = False

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        w: np.ndarray,
        y: np.ndarray,
    ) -> BaseUpliftLearner:
        """Fit the causal estimator on training covariates, treatment, and outcomes.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray of shape (n_samples, n_features)
            Pre-treatment customer covariates.
        w : np.ndarray of shape (n_samples,)
            Binary treatment indicator: 1 = treated, 0 = control.
        y : np.ndarray of shape (n_samples,)
            Binary observable outcome: 1 = retained, 0 = churned.

        Returns
        -------
        BaseUpliftLearner
            Fitted estimator instance.
        """
        pass

    @abstractmethod
    def predict_uplift(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict the conditional average treatment effect (CATE) tau(X).

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray of shape (n_samples, n_features)
            Customer covariates to evaluate.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Estimated uplift tau_hat(X) in range [-1.0, 1.0].
        """
        pass

    def predict_potential_outcomes(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate counterfactual potential outcome probabilities (mu_0, mu_1).

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray of shape (n_samples, n_features)
            Customer covariates to evaluate.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (mu_0, mu_1) where:
            mu_0 = P(Y(0) = 1 | X)
            mu_1 = P(Y(1) = 1 | X)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement explicit potential outcome decomposition."
        )

    def score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Score customer accounts by predicted uplift (alias for predict_uplift)."""
        return self.predict_uplift(X)

    def _validate_inputs(
        self,
        X: pd.DataFrame | np.ndarray,
        w: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Convert and validate array shapes, memory contiguity, and binary constraints."""
        X_raw = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        X_arr = np.ascontiguousarray(X_raw, dtype=np.float64)

        w_arr = None
        if w is not None:
            w_raw = np.asarray(w)
            if not np.all(np.isin(w_raw, [0, 1])):
                raise ValueError(f"Treatment vector w must be strictly binary {{0, 1}}, found {np.unique(w_raw)}")
            w_arr = np.ascontiguousarray(w_raw.astype(int).ravel())
            if len(w_arr) != len(X_arr):
                raise ValueError(f"Length mismatch: X has {len(X_arr)} rows, w has {len(w_arr)}")

        y_arr = None
        if y is not None:
            y_raw = np.asarray(y)
            if not np.all(np.isin(y_raw, [0, 1])):
                raise ValueError(f"Outcome vector y must be strictly binary {{0, 1}}, found {np.unique(y_raw)}")
            y_arr = np.ascontiguousarray(y_raw.astype(int).ravel())
            if len(y_arr) != len(X_arr):
                raise ValueError(f"Length mismatch: X has {len(X_arr)} rows, y has {len(y_arr)}")

        return X_arr, w_arr, y_arr
