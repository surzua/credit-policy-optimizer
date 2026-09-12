"""Synthetic credit portfolio simulator using NumPy and Polars."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from credit_policy_optimizer.data.schema import SimulationResult


@dataclass(frozen=True)
class RiskCalibrationParams:
    """Logistic regression coefficients and calibration terms for Probability of Default (PD)."""

    intercept: float = -3.95
    beta_debt_to_income: float = 1.80
    beta_utilization: float = 1.50
    beta_delinquencies: float = 0.55
    beta_log_income: float = 0.50
    beta_loan_to_income: float = 0.12


class PortfolioSimulator:
    """Deterministic synthetic credit portfolio generator using NumPy and Polars."""

    def __init__(
        self,
        seed: int = 42,
        calibration: RiskCalibrationParams | None = None,
    ) -> None:
        """Initialize the simulator with a deterministic seed and calibration parameters."""
        self.seed = seed
        self.calibration = calibration or RiskCalibrationParams()

    def simulate(self, n_samples: int = 10_000) -> pl.DataFrame:
        """Generate a synthetic portfolio of credit applications with realistic risk metrics.

        Parameters
        ----------
        n_samples : int
            Number of synthetic loan applications to generate.

        Returns
        -------
        pl.DataFrame
            Polars DataFrame containing applicant features, loan parameters, PD, and default flags.
        """
        if n_samples <= 0:
            msg = f"n_samples must be greater than 0, got {n_samples}"
            raise ValueError(msg)

        rng = np.random.default_rng(self.seed)

        # 1. Financial Profile
        # Monthly income: Log-normal distribution (median ~$3,600, min ~$1,000, max ~$30,000)
        raw_income = rng.lognormal(mean=8.2, sigma=0.45, size=n_samples)
        monthly_income = np.clip(np.round(raw_income, 2), 1000.0, 35000.0)

        # Debt-to-income (DTI): Beta distribution scaled to realistic retail bounds [0.05, 0.65]
        debt_to_income = np.clip(
            np.round(0.05 + rng.beta(a=2.0, b=5.0, size=n_samples) * 0.65, 4),
            0.05,
            0.70,
        )

        # Historical delinquencies (30+ days past due in past 24m): Poisson distribution
        historical_delinquencies = rng.poisson(lam=0.35, size=n_samples)

        # Revolving credit utilization: Beta distribution [0.01, 0.99]
        revolving_utilization = np.clip(
            np.round(rng.beta(a=1.8, b=3.2, size=n_samples), 4),
            0.01,
            0.99,
        )

        # 2. Loan Conditions
        # Loan-to-income multiplier (typically 1.5x to 5.5x monthly income, capped [1,500, 50,000])
        loan_multiplier = rng.uniform(low=1.8, high=5.5, size=n_samples)
        raw_loan = np.round(monthly_income * loan_multiplier, -2)
        loan_amount = np.clip(raw_loan, 1500.0, 50000.0)

        # Loan term: categorical terms in months [12, 24, 36, 48, 60]
        term_options = np.array([12, 24, 36, 48, 60])
        term_probs = np.array([0.10, 0.20, 0.40, 0.20, 0.10])
        loan_term_months = rng.choice(term_options, size=n_samples, p=term_probs)

        # Cost of funds: treasury base rate ~ 5.25% with small variance
        cost_of_funds = np.clip(
            np.round(rng.normal(loc=0.0525, scale=0.002, size=n_samples), 4),
            0.045,
            0.065,
        )

        # 3. Probability of Default (PD) & Default Flag
        # Linear risk index (Log-Odds)
        lti_ratio = loan_amount / monthly_income
        log_income_norm = np.log(monthly_income / 1000.0)

        cal = self.calibration
        logit_pd = (
            cal.intercept
            + (cal.beta_debt_to_income * debt_to_income)
            + (cal.beta_utilization * revolving_utilization)
            + (cal.beta_delinquencies * historical_delinquencies)
            - (cal.beta_log_income * log_income_norm)
            + (cal.beta_loan_to_income * lti_ratio)
        )

        # Logistic Sigmoid Transformation
        pd_array = np.round(1.0 / (1.0 + np.exp(-logit_pd)), 5)

        # Active interest rate (tasa activa): cost of funds + risk-based spread + margin
        # Calibrated between ~7.5% and ~28%
        risk_spread = np.round(pd_array * 0.35 + rng.uniform(0.02, 0.05, size=n_samples), 4)
        interest_rate = np.clip(
            np.round(cost_of_funds + risk_spread, 4),
            0.075,
            0.320,
        )

        # Bernoulli trial for ground truth default flag
        uniform_shocks = rng.uniform(low=0.0, high=1.0, size=n_samples)
        default_flag = (uniform_shocks < pd_array).astype(int)

        # Build Polars DataFrame
        data_dict: dict[str, Any] = {
            "application_id": [f"APP_{self.seed}_{i:06d}" for i in range(n_samples)],
            "monthly_income": monthly_income.tolist(),
            "debt_to_income": debt_to_income.tolist(),
            "historical_delinquencies": historical_delinquencies.tolist(),
            "revolving_utilization": revolving_utilization.tolist(),
            "loan_amount": loan_amount.tolist(),
            "loan_term_months": loan_term_months.tolist(),
            "interest_rate": interest_rate.tolist(),
            "cost_of_funds": cost_of_funds.tolist(),
            "pd": pd_array.tolist(),
            "default_flag": default_flag.tolist(),
        }

        schema_overrides = {
            "application_id": pl.String,
            "monthly_income": pl.Float64,
            "debt_to_income": pl.Float64,
            "historical_delinquencies": pl.Int64,
            "revolving_utilization": pl.Float64,
            "loan_amount": pl.Float64,
            "loan_term_months": pl.Int64,
            "interest_rate": pl.Float64,
            "cost_of_funds": pl.Float64,
            "pd": pl.Float64,
            "default_flag": pl.Int64,
        }

        return pl.DataFrame(data_dict, schema=schema_overrides)

    def save_parquet(
        self,
        df: pl.DataFrame,
        output_path: str | Path,
    ) -> Path:
        """Export the generated DataFrame to Parquet format.

        Parameters
        ----------
        df : pl.DataFrame
            Portfolio DataFrame to write.
        output_path : str | Path
            Destination filepath.

        Returns
        -------
        Path
            Path to the saved Parquet file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        return path

    def compute_summary(self, df: pl.DataFrame) -> SimulationResult:
        """Compute aggregate portfolio metrics and validate with SimulationResult schema."""
        total_apps = df.height
        if total_apps == 0:
            msg = "Cannot compute summary on an empty DataFrame"
            raise ValueError(msg)

        default_count = int(df.select(pl.col("default_flag").sum()).item())
        total_exposure = float(df.select(pl.col("loan_amount").sum()).item())
        avg_pd = float(df.select(pl.col("pd").mean()).item())
        avg_loan = float(df.select(pl.col("loan_amount").mean()).item())
        avg_income = float(df.select(pl.col("monthly_income").mean()).item())
        avg_rate = float(df.select(pl.col("interest_rate").mean()).item())
        avg_cost = float(df.select(pl.col("cost_of_funds").mean()).item())

        return SimulationResult(
            total_applications=total_apps,
            total_exposure=round(total_exposure, 2),
            default_count=default_count,
            default_rate=round(default_count / total_apps, 5),
            avg_pd=round(avg_pd, 5),
            avg_loan_amount=round(avg_loan, 2),
            avg_monthly_income=round(avg_income, 2),
            avg_interest_rate=round(avg_rate, 5),
            avg_cost_of_funds=round(avg_cost, 5),
            seed=self.seed,
        )
