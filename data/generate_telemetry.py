"""Synthetic B2B SaaS Telemetry and Causal Potential Outcomes Generator.

Simulates 50,000 realistic customer profiles with latent structural equations
modeling baseline retention propensity Y(0), treatment interaction tau_true,
and observable factual outcomes Y across four behavioral archetypes:
Persuadables, Sure Things, Sleeping Dogs, and Lost Causes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import uuid
from pathlib import Path

with contextlib.suppress(ImportError):
    import lightgbm  # noqa: F401

import numpy as np
import pandas as pd
from scipy.special import expit

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_telemetry(
    n_samples: int = 50_000,
    treatment_prob: float = 0.5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic B2B SaaS telemetry data with latent causal ground truth.

    Parameters
    ----------
    n_samples : int, default=50000
        Number of synthetic customer accounts to simulate.
    treatment_prob : float, default=0.5
        Probability of assigning treatment (randomized A/B test condition).
    random_seed : int, default=42
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Complete customer telemetry dataset including observable features,
        treatment assignment W, observable factual outcome Y, and latent
        counterfactual ground truth (Y(0), Y(1), tau_true, archetype).
    """
    rng = np.random.default_rng(random_seed)

    logger.info("Generating %d synthetic SaaS telemetry records...", n_samples)

    # 1. Generate core customer covariates
    # Account age: exponential distribution, clipped 1-60 months
    account_age = np.clip(rng.exponential(scale=18.0, size=n_samples) + 1, 1, 60).astype(int)

    # Plan tier assignment
    plan_tiers = rng.choice(
        ["Starter", "Professional", "Enterprise"],
        p=[0.50, 0.35, 0.15],
        size=n_samples,
    )

    # Monthly Recurring Revenue (MRR): log-normal conditioned on tier
    mrr = np.zeros(n_samples, dtype=float)
    starter_idx = plan_tiers == "Starter"
    pro_idx = plan_tiers == "Professional"
    ent_idx = plan_tiers == "Enterprise"

    mrr[starter_idx] = rng.lognormal(mean=3.7, sigma=0.4, size=starter_idx.sum())  # ~$30 - $80
    mrr[pro_idx] = rng.lognormal(mean=5.3, sigma=0.4, size=pro_idx.sum())          # ~$120 - $350
    mrr[ent_idx] = rng.lognormal(mean=6.9, sigma=0.5, size=ent_idx.sum())          # ~$600 - $2,500
    mrr = np.clip(np.round(mrr, 2), 20.0, 5000.0)

    # Active users ratio (licensed seats actively using product): Beta(3, 2)
    active_users_ratio = np.clip(np.round(rng.beta(a=3.5, b=2.2, size=n_samples), 3), 0.05, 1.0)

    # Login frequency trend over last 30d vs prior 60d: Normal(-0.05, 0.35)
    login_trend_30d = np.clip(np.round(rng.normal(loc=-0.05, scale=0.35, size=n_samples), 3), -0.95, 2.5)

    # Feature usage diversity: Integer 1 to 15
    feature_diversity = np.clip(
        rng.poisson(lam=4.5 + 3.0 * (plan_tiers != "Starter"), size=n_samples),
        1,
        15,
    ).astype(int)

    # Support tickets in trailing 90 days
    support_tickets = np.clip(rng.poisson(lam=2.5, size=n_samples), 0, 25).astype(int)

    # Unresolved Sev-1 (P1) critical tickets: Poisson with low mean, elevated if high tickets
    p1_lambda = np.clip(0.15 + 0.08 * (support_tickets > 4), 0.05, 1.0)
    unresolved_p1 = np.clip(rng.poisson(lam=p1_lambda, size=n_samples), 0, 4).astype(int)

    # CSAT score (1.0 to 5.0): heavily penalised by unresolved P1 tickets
    csat_base = rng.normal(loc=4.1, scale=0.5, size=n_samples)
    csat = csat_base - 0.8 * unresolved_p1 - 0.3 * (support_tickets > 6)
    csat = np.clip(np.round(csat, 1), 1.0, 5.0)

    # Payment failure events (dunning attempts): Poisson(0.2)
    payment_failures = np.clip(rng.poisson(lam=0.22, size=n_samples), 0, 4).astype(int)

    # Contract commitment type: Annual more common for Enterprise
    contract_prob = np.where(plan_tiers == "Enterprise", 0.70, np.where(plan_tiers == "Professional", 0.40, 0.20))
    contract_type = np.where(rng.uniform(size=n_samples) < contract_prob, "Annual", "Monthly")

    # Treatment assignment W: Completely randomized A/B trial
    treatment = rng.binomial(n=1, p=treatment_prob, size=n_samples).astype(int)

    # 2. Latent Structural Equations (Potential Outcomes Framework)
    # Baseline Propensity Score S_0 to retain WITHOUT intervention
    s_0 = (
        0.35
        + 1.70 * login_trend_30d
        + 0.10 * (feature_diversity - 5)
        + 1.60 * (active_users_ratio - 0.50)
        - 1.30 * unresolved_p1
        - 0.90 * payment_failures
        + 0.40 * (csat - 3.5)
        + 0.60 * (contract_type == "Annual")
        + 0.010 * (account_age - 15)
    )

    # True Latent Uplift Function tau_score
    # Persuadable drivers:
    # 1. Moderately dropping usage (trend in [-0.60, -0.10]): outreach re-engages them
    # 2. Unresolved P1 support blocker: proactive CS outreach unblocks them
    # 3. Intermediate seat utilization (0.25 to 0.60): room for onboarding lift
    # Sleeping Dog drivers:
    # 4. High usage trend (> 0.40) or highly satisfied users: unsolicited sales call irritates them
    persuadable_signal = (
        1.50 * ((login_trend_30d >= -0.60) & (login_trend_30d <= -0.10)).astype(float)
        + 1.40 * (unresolved_p1 > 0).astype(float)
        + 0.60 * ((active_users_ratio >= 0.25) & (active_users_ratio <= 0.60)).astype(float)
    )

    sleeping_dog_signal = (
        1.50 * (login_trend_30d > 0.40).astype(float)
        + 0.80 * ((csat >= 4.5) & (unresolved_p1 == 0)).astype(float)
    )

    tau_score = persuadable_signal - sleeping_dog_signal

    # Latent Potential Outcomes
    eps_0 = rng.logistic(loc=0.0, scale=0.65, size=n_samples)
    eps_1 = rng.logistic(loc=0.0, scale=0.65, size=n_samples)

    prob_y0 = expit(s_0 + eps_0)
    prob_y1 = expit(s_0 + tau_score + eps_1)

    y_0 = (prob_y0 >= 0.50).astype(int)
    y_1 = (prob_y1 >= 0.50).astype(int)

    # Individual Treatment Effect (ITE)
    tau_true = y_1 - y_0

    # Categorize Latent Archetypes
    archetypes = np.empty(n_samples, dtype=object)
    archetypes[(y_0 == 0) & (y_1 == 1)] = "Persuadable"
    archetypes[(y_0 == 1) & (y_1 == 0)] = "Sleeping Dog"
    archetypes[(y_0 == 1) & (y_1 == 1)] = "Sure Thing"
    archetypes[(y_0 == 0) & (y_1 == 0)] = "Lost Cause"

    # Factual Observable Outcome: Y = W*Y(1) + (1-W)*Y(0)
    retained_factual = treatment * y_1 + (1 - treatment) * y_0

    # Assemble complete DataFrame
    user_ids = [f"usr_{uuid.UUID(bytes=bytes(rng.bytes(16))).hex[:12]}" for _ in range(n_samples)]

    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "account_age_months": account_age,
            "monthly_recurring_revenue": mrr,
            "plan_tier": plan_tiers,
            "active_users_ratio": active_users_ratio,
            "login_frequency_trend_30d": login_trend_30d,
            "feature_usage_diversity": feature_diversity,
            "support_tickets_90d": support_tickets,
            "unresolved_p1_tickets": unresolved_p1,
            "csat_score": csat,
            "payment_failure_events": payment_failures,
            "contract_type": contract_type,
            "treatment_assigned": treatment,
            "retained": retained_factual,
            # Ground truth latent variables (for research and evaluation benchmarking)
            "y_control_latent": y_0,
            "y_treated_latent": y_1,
            "tau_true": tau_true,
            "archetype": archetypes,
        }
    )

    logger.info("Dataset generated successfully with %d records.", len(df))
    logger.info("Archetype breakdown:\n%s", df["archetype"].value_counts(normalize=True).to_string())
    logger.info(
        "Factual Retention Rate: Overall=%.3f, Treated=%.3f, Control=%.3f",
        df["retained"].mean(),
        df[df["treatment_assigned"] == 1]["retained"].mean(),
        df[df["treatment_assigned"] == 0]["retained"].mean(),
    )

    return df


def validate_schema(df: pd.DataFrame, schema_path: Path) -> bool:
    """Validate DataFrame against JSON schema."""
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)

    required_cols = schema.get("required", [])
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required schema columns: {missing}")

    # Check value bounds
    if not ((df["treatment_assigned"] == 0) | (df["treatment_assigned"] == 1)).all():
        raise ValueError("treatment_assigned contains non-binary values")
    if not ((df["retained"] == 0) | (df["retained"] == 1)).all():
        raise ValueError("retained contains non-binary values")

    logger.info("Schema validation PASSED for all %d records.", len(df))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic B2B SaaS retention telemetry.")
    parser.add_argument("--n-samples", type=int, default=50_000, help="Number of records to generate")
    parser.add_argument("--treatment-prob", type=float, default=0.5, help="Probability of treatment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--save-sample", action="store_true", default=True, help="Save small 1000-row sample")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_telemetry(
        n_samples=args.n_samples,
        treatment_prob=args.treatment_prob,
        random_seed=args.seed,
    )

    schema_file = Path("data/telemetry_schema.json")
    if schema_file.exists():
        validate_schema(df, schema_file)

    output_path = out_dir / "telemetry.csv"
    df.to_csv(output_path, index=False)
    logger.info("Saved full dataset to %s (size: %.2f MB)", output_path, output_path.stat().st_size / (1024 * 1024))

    if args.save_sample:
        sample_path = out_dir / "telemetry_sample.csv"
        df.head(1000).to_csv(sample_path, index=False)
        logger.info("Saved sample dataset to %s", sample_path)


if __name__ == "__main__":
    main()
