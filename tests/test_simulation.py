"""Unit test suite for Financial ROI Optimization and Revenue Recovery Simulator.

Validates:
1. Optimizer initialization and economic parameter validation.
2. Optimal cutoff solver k* identifying marginal profitability inflection point.
3. Strict budget constraint enforcement (total_cost <= campaign_budget).
4. Zero targeting behavior under unprofitable economic regimes.
5. Head-to-head comparative analysis: Causal Uplift vs Churn Propensity.
6. Quantification of avoided marketing waste (Sure Things) and preserved ARR (Sleeping Dogs).
7. pd.Series vs np.ndarray array interoperability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.generate_telemetry import generate_synthetic_telemetry
from src.data_pipeline import split_causal_dataset
from src.models.propensity_baseline import PropensityBaseline
from src.models.t_learner import TLearner
from src.simulation.roi_optimizer import (
    CausalROIOptimizer,
    HeadToHeadComparison,
    TargetingPolicyResult,
)

# -------------------------------------------------------------------------
# 1. Parameter Validation & Boundary Guards
# -------------------------------------------------------------------------

def test_optimizer_invalid_inputs_raises() -> None:
    """Verify that negative intervention cost or budget raises ValueError."""
    with pytest.raises(ValueError, match="intervention_cost cannot be negative"):
        CausalROIOptimizer(intervention_cost=-10.0)

    with pytest.raises(ValueError, match="campaign_budget cannot be negative"):
        CausalROIOptimizer(intervention_cost=50.0, campaign_budget=-500.0)


def test_optimizer_length_mismatch_raises() -> None:
    """Verify that length mismatch between uplift predictions and MRR raises ValueError."""
    optimizer = CausalROIOptimizer(intervention_cost=50.0)
    tau = np.array([0.2, 0.1, 0.3])
    mrr = np.array([100.0, 200.0])
    with pytest.raises(ValueError, match="Length mismatch"):
        optimizer.optimize_policy(tau, mrr)


def test_optimizer_unprofitable_regime_targets_zero() -> None:
    """Verify that if all interventions are unprofitable, optimizer targets 0 accounts."""
    # Intervention cost is $10,000 per account, but max customer ARR is $1,200
    optimizer = CausalROIOptimizer(intervention_cost=10_000.0)
    tau = np.array([0.1, 0.2, 0.05, -0.1])
    mrr = np.array([50.0, 100.0, 40.0, 80.0])

    result: TargetingPolicyResult = optimizer.optimize_policy(tau, mrr)

    assert result.n_targeted == 0
    assert result.fraction_targeted == 0.0
    assert result.total_cost == 0.0
    assert result.gross_revenue_recovered == 0.0
    assert result.net_profit_delta == 0.0
    assert result.roi_percentage == 0.0


# -------------------------------------------------------------------------
# 2. Optimal Cutoff & Budget Enforcement
# -------------------------------------------------------------------------

def test_optimizer_finds_profitable_cutoff() -> None:
    """Verify that optimizer selects accounts with positive marginal ROI."""
    optimizer = CausalROIOptimizer(intervention_cost=50.0, annual_multiplier=12.0)
    # Account 0: tau=0.5, MRR=100 => Saved ARR = 0.5 * 1200 = $600. Cost = $50 => Net = +$550
    # Account 1: tau=0.2, MRR=50  => Saved ARR = 0.2 * 600  = $120. Cost = $50 => Net = +$70
    # Account 2: tau=0.01, MRR=50 => Saved ARR = 0.01 * 600 = $6.   Cost = $50 => Net = -$44 (Unprofitable!)
    # Account 3: tau=-0.2, MRR=100 => Saved ARR = -$240.    Cost = $50 => Net = -$290 (Sleeping Dog!)
    tau = np.array([0.5, 0.2, 0.01, -0.2])
    mrr = np.array([100.0, 50.0, 50.0, 100.0])

    result = optimizer.optimize_policy(tau, mrr)

    # Should target exactly Account 0 and Account 1 (2 accounts)
    assert result.n_targeted == 2
    assert result.total_cost == 100.0
    assert result.gross_revenue_recovered == 720.0  # 600 + 120
    assert result.net_profit_delta == 620.0         # 720 - 100
    assert result.sleeping_dogs_suppressed == 1    # Account 3


def test_optimizer_budget_constraint_clipping() -> None:
    """Verify that campaign_budget limits targeting to affordable accounts."""
    # 5 highly profitable accounts, but budget only affords 2 interventions
    optimizer = CausalROIOptimizer(intervention_cost=100.0, campaign_budget=250.0)
    tau = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    mrr = np.array([200.0, 200.0, 200.0, 200.0, 200.0])

    result = optimizer.optimize_policy(tau, mrr)

    # Budget $250 // $100 = 2 accounts
    assert result.n_targeted == 2
    assert result.total_cost == 200.0
    assert result.total_cost <= 250.0


# -------------------------------------------------------------------------
# 3. Head-to-Head Strategy Comparison
# -------------------------------------------------------------------------

def test_head_to_head_comparison_causal_advantage() -> None:
    """Verify that causal targeting generates superior net profit over propensity targeting."""
    raw_df = generate_synthetic_telemetry(n_samples=1500, random_seed=42)
    dataset = split_causal_dataset(raw_df, test_size=0.30, random_state=42)

    # Train both models
    t_learner = TLearner(n_estimators=30, random_state=42)
    t_learner.fit(dataset.X_train, dataset.w_train, dataset.y_train)

    propensity_model = PropensityBaseline(n_estimators=30, random_state=42)
    propensity_model.fit(dataset.X_train, None, dataset.y_train)

    tau_pred = t_learner.predict_uplift(dataset.X_test)
    p_churn = propensity_model.predict_churn_propensity(dataset.X_test)

    optimizer = CausalROIOptimizer(intervention_cost=60.0, campaign_budget=15000.0)
    comparison: HeadToHeadComparison = optimizer.compare_targeting_strategies(
        uplift_preds=tau_pred,
        churn_propensity=p_churn,
        mrr=dataset.mrr_test,
        archetypes=dataset.latent_test["archetype"],
    )

    # Causal net profit must outperform traditional churn propensity targeting
    assert comparison.causal_policy.net_profit_delta > comparison.propensity_policy.net_profit_delta
    assert comparison.net_profit_advantage > 0.0
    assert comparison.efficiency_multiplier >= 1.0

    # Quantified waste metrics must be non-negative
    assert comparison.sure_things_waste_avoided >= 0.0
    assert comparison.sleeping_dogs_destruction_prevented >= 0.0


def test_optimizer_series_numpy_equivalence() -> None:
    """Verify optimizer produces identical results for pd.Series and np.ndarray."""
    optimizer = CausalROIOptimizer(intervention_cost=40.0)
    tau_arr = np.array([0.4, 0.2, -0.1, 0.3])
    mrr_arr = np.array([120.0, 80.0, 90.0, 150.0])

    res_arr = optimizer.optimize_policy(tau_arr, mrr_arr)
    res_ser = optimizer.optimize_policy(pd.Series(tau_arr), pd.Series(mrr_arr))

    assert res_arr.n_targeted == res_ser.n_targeted
    assert res_arr.total_cost == res_ser.total_cost
    assert res_arr.gross_revenue_recovered == res_ser.gross_revenue_recovered
    assert res_arr.net_profit_delta == res_ser.net_profit_delta
