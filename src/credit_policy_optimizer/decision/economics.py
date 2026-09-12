"""Financial economics and optimal credit policy decisioning engine.

Models the net expected value (EV) of approving loans:
    EV_i = (1 - PD_i) * Gain(Interest, Amount, Term) - PD_i * Loss(LGD, Amount, FundingCost, Term)

Provides the `CreditPolicyOptimizer` to determine the profit-maximizing probability threshold (p*),
generate approval vs. loss vs. margin trade-off curves, and optimize risk-adjusted credit limits.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class EconomicParameters(BaseModel):
    """Institutional economic and macroeconomic financial assumptions."""

    model_config = ConfigDict(extra="forbid", strict=True)

    default_lgd: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Loss Given Default (LGD) ratio representing net loss rate upon default",
    )
    default_cost_of_funds: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Annual cost of funds / institutional hurdle rate",
    )
    default_interest_rate: float = Field(
        default=0.18,
        gt=0.0,
        le=1.0,
        description="Nominal annual active interest rate charged to borrowers",
    )
    default_loan_term_months: int = Field(
        default=12,
        gt=0,
        description="Default tenure of the credit facility in months",
    )


class PolicyEvaluation(BaseModel):
    """Financial and operational performance metrics for a specific policy threshold."""

    model_config = ConfigDict(extra="forbid", strict=True)

    threshold: float = Field(..., ge=0.0, le=1.0, description="Probability cutoff threshold")
    total_applications: int = Field(..., ge=0, description="Total applications evaluated")
    approved_count: int = Field(..., ge=0, description="Number of approved applications")
    approval_rate: float = Field(..., ge=0.0, le=1.0, description="Approved share of applications")
    total_exposure: float = Field(..., ge=0.0, description="Total loan volume approved in USD")
    gross_margin: float = Field(..., description="Cumulative gross interest margin in USD")
    expected_loss: float = Field(..., ge=0.0, description="Cumulative expected credit loss in USD")
    expected_pnl: float = Field(..., description="Net expected economic profit and loss in USD")
    expected_default_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Average expected PD of approved portfolio"
    )
    return_on_exposure: float = Field(
        ..., description="Net expected return on assets / exposure (PnL / Exposure)"
    )


class OptimizationResult(BaseModel):
    """Result of searching the optimal decision policy threshold."""

    model_config = ConfigDict(extra="forbid", strict=True)

    optimal_threshold: float = Field(
        ..., ge=0.0, le=1.0, description="Optimal probability cutoff p* maximizing objective"
    )
    optimal_policy: PolicyEvaluation = Field(
        ..., description="Financial metrics under optimal policy cutoff p*"
    )
    baseline_policy_05: PolicyEvaluation = Field(
        ..., description="Financial metrics under naive baseline cutoff p=0.5"
    )
    individual_optimal_pnl: float = Field(
        ..., description="Theoretical maximum PnL achievable via applicant-level EV > 0 decisioning"
    )
    incremental_pnl_vs_baseline: float = Field(
        ..., description="Net dollar improvement of optimal policy vs. naive 0.5 cutoff"
    )
    objective_metric: str = Field(..., description="Metric targeted during optimization")


class AmountPolicyResult(BaseModel):
    """Summary of portfolio outcomes under a risk-based credit limit sizing policy."""

    model_config = ConfigDict(extra="forbid", strict=True)

    total_applications: int = Field(..., ge=0)
    approved_count: int = Field(..., ge=0)
    approval_rate: float = Field(..., ge=0.0, le=1.0)
    requested_exposure: float = Field(..., ge=0.0)
    allocated_exposure: float = Field(..., ge=0.0)
    expected_pnl: float = Field(..., description="Cumulative net PnL under sized limits")
    expected_loss: float = Field(..., ge=0.0)
    return_on_exposure: float = Field(...)


def compute_loan_net_gain(
    amount: float | np.ndarray[Any, Any] | pl.Expr,
    interest_rate: float | np.ndarray[Any, Any] | pl.Expr,
    cost_of_funds: float | np.ndarray[Any, Any] | pl.Expr,
    term_months: float | int | np.ndarray[Any, Any] | pl.Expr = 12,
) -> Any:
    """Calculate the net financial revenue generated if the loan is fully performing.

    Gain = Amount * (InterestRate - CostOfFunds) * (TermMonths / 12)
    """
    tenure_years = term_months / 12.0
    net_rate = interest_rate - cost_of_funds
    return amount * net_rate * tenure_years


def compute_loan_net_loss(
    amount: float | np.ndarray[Any, Any] | pl.Expr,
    lgd: float | np.ndarray[Any, Any] | pl.Expr,
    cost_of_funds: float | np.ndarray[Any, Any] | pl.Expr,
    term_months: float | int | np.ndarray[Any, Any] | pl.Expr = 12,
) -> Any:
    """Calculate net financial loss incurred if the loan defaults.

    Loss = Amount * (LGD + CostOfFunds * (TermMonths / 12))
    """
    tenure_years = term_months / 12.0
    loss_factor = lgd + (cost_of_funds * tenure_years)
    return amount * loss_factor


def compute_expected_value(
    pd: float | np.ndarray[Any, Any] | pl.Expr,
    amount: float | np.ndarray[Any, Any] | pl.Expr,
    interest_rate: float | np.ndarray[Any, Any] | pl.Expr,
    cost_of_funds: float | np.ndarray[Any, Any] | pl.Expr,
    lgd: float | np.ndarray[Any, Any] | pl.Expr,
    term_months: float | int | np.ndarray[Any, Any] | pl.Expr = 12,
) -> Any:
    """Compute the net expected value (EV) of approving a credit:

    EV = (1 - PD) * Gain - PD * Loss
    """
    gain = compute_loan_net_gain(amount, interest_rate, cost_of_funds, term_months)
    loss = compute_loan_net_loss(amount, lgd, cost_of_funds, term_months)
    return ((1.0 - pd) * gain) - (pd * loss)


def compute_breakeven_pd(
    interest_rate: float,
    cost_of_funds: float,
    lgd: float,
    term_months: float | int = 12,
) -> float:
    """Compute the exact analytical default probability breakeven threshold p*.

    At p*, EV = 0.
    (1 - p*) * UnitGain = p* * UnitLoss
    p* = UnitGain / (UnitGain + UnitLoss)
    """
    tenure_years = term_months / 12.0
    unit_gain = (interest_rate - cost_of_funds) * tenure_years
    unit_loss = lgd + (cost_of_funds * tenure_years)

    if unit_gain <= 0.0:
        return 0.0
    total_factor = unit_gain + unit_loss
    if total_factor <= 0.0:
        return 0.0
    return float(np.clip(unit_gain / total_factor, 0.0, 1.0))


class CreditPolicyOptimizer:
    """Financial optimization engine for credit portfolio policy decisioning.

    Transforms calibrated default probabilities (PD) and contractual loan terms into
    risk-adjusted economics and computes the optimal decision threshold p* to maximize
    total portfolio P&L.
    """

    def __init__(
        self,
        portfolio: pl.DataFrame | dict[str, Any] | list[dict[str, Any]],
        params: EconomicParameters | None = None,
    ) -> None:
        """Initialize the optimizer with portfolio data and default economic parameters."""
        self.params = params or EconomicParameters()

        if isinstance(portfolio, pl.DataFrame):
            df = portfolio.clone()
        elif isinstance(portfolio, dict):
            df = pl.DataFrame(portfolio)
        else:
            df = pl.DataFrame(portfolio)

        if "pd" not in df.columns:
            msg = "Portfolio must contain a 'pd' (Probability of Default) column."
            raise ValueError(msg)
        if "loan_amount" not in df.columns:
            msg = "Portfolio must contain a 'loan_amount' column."
            raise ValueError(msg)

        # Impute missing economic columns with institutional defaults if not present
        if "interest_rate" not in df.columns:
            df = df.with_columns(pl.lit(self.params.default_interest_rate).alias("interest_rate"))
        if "cost_of_funds" not in df.columns:
            df = df.with_columns(pl.lit(self.params.default_cost_of_funds).alias("cost_of_funds"))
        if "loan_term_months" not in df.columns:
            df = df.with_columns(
                pl.lit(self.params.default_loan_term_months).alias("loan_term_months")
            )
        if "lgd" not in df.columns:
            df = df.with_columns(pl.lit(self.params.default_lgd).alias("lgd"))

        # Cast key columns to Float64 for consistency
        df = df.with_columns(
            [
                pl.col("pd").cast(pl.Float64),
                pl.col("loan_amount").cast(pl.Float64),
                pl.col("interest_rate").cast(pl.Float64),
                pl.col("cost_of_funds").cast(pl.Float64),
                pl.col("loan_term_months").cast(pl.Float64),
                pl.col("lgd").cast(pl.Float64),
            ]
        )

        self._enriched_portfolio = self._enrich_portfolio_economics(df)

    @property
    def portfolio(self) -> pl.DataFrame:
        """Return the enriched portfolio DataFrame."""
        return self._enriched_portfolio

    def _enrich_portfolio_economics(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute loan-level economics: expected gain, loss, EV, and individual breakeven PD."""
        tenure_years = pl.col("loan_term_months") / 12.0
        unit_gain = (pl.col("interest_rate") - pl.col("cost_of_funds")) * tenure_years
        unit_loss = pl.col("lgd") + (pl.col("cost_of_funds") * tenure_years)

        gain_expr = pl.col("loan_amount") * unit_gain
        loss_expr = pl.col("loan_amount") * unit_loss
        ev_expr = ((1.0 - pl.col("pd")) * gain_expr) - (pl.col("pd") * loss_expr)

        breakeven_expr = pl.when(unit_gain <= 0.0).then(0.0).otherwise(
            (unit_gain / (unit_gain + unit_loss)).clip(0.0, 1.0)
        )

        return df.with_columns(
            [
                gain_expr.alias("net_gain_performing"),
                loss_expr.alias("net_loss_default"),
                ev_expr.alias("expected_value"),
                breakeven_expr.alias("breakeven_pd"),
            ]
        )

    def evaluate_policy(self, threshold: float) -> PolicyEvaluation:
        """Evaluate the financial outcome of applying a probability cutoff policy.

        Approves application if PD <= threshold.
        """
        if not (0.0 <= threshold <= 1.0):
            msg = f"Threshold must be between 0.0 and 1.0, got {threshold}"
            raise ValueError(msg)

        total_apps = self._enriched_portfolio.height
        if total_apps == 0:
            return PolicyEvaluation(
                threshold=threshold,
                total_applications=0,
                approved_count=0,
                approval_rate=0.0,
                total_exposure=0.0,
                gross_margin=0.0,
                expected_loss=0.0,
                expected_pnl=0.0,
                expected_default_rate=0.0,
                return_on_exposure=0.0,
            )

        approved = self._enriched_portfolio.filter(pl.col("pd") <= threshold)
        approved_count = approved.height

        if approved_count == 0:
            return PolicyEvaluation(
                threshold=threshold,
                total_applications=total_apps,
                approved_count=0,
                approval_rate=0.0,
                total_exposure=0.0,
                gross_margin=0.0,
                expected_loss=0.0,
                expected_pnl=0.0,
                expected_default_rate=0.0,
                return_on_exposure=0.0,
            )

        exposure = float(approved["loan_amount"].sum())
        expected_gain = float(((1.0 - approved["pd"]) * approved["net_gain_performing"]).sum())
        expected_loss = float((approved["pd"] * approved["net_loss_default"]).sum())
        expected_pnl = float(approved["expected_value"].sum())
        mean_val = approved["pd"].mean()
        avg_pd = float(mean_val) if isinstance(mean_val, (int, float)) else 0.0
        roe = (expected_pnl / exposure) if exposure > 0.0 else 0.0

        return PolicyEvaluation(
            threshold=threshold,
            total_applications=total_apps,
            approved_count=approved_count,
            approval_rate=approved_count / total_apps,
            total_exposure=exposure,
            gross_margin=expected_gain,
            expected_loss=expected_loss,
            expected_pnl=expected_pnl,
            expected_default_rate=avg_pd,
            return_on_exposure=roe,
        )

    def optimize_threshold(
        self,
        grid_size: int = 500,
        metric: Literal["expected_pnl", "return_on_exposure"] = "expected_pnl",
    ) -> OptimizationResult:
        """Find the optimal probability threshold p* maximizing the chosen portfolio objective.

        Evaluates candidate thresholds along the empirical PD distribution and returns a detailed
        comparison against the naive binary classification cutoff (p = 0.5) and the theoretical
        unconstrained individual EV > 0 policy.
        """
        if grid_size < 10:
            msg = f"grid_size must be at least 10, got {grid_size}"
            raise ValueError(msg)

        total_apps = self._enriched_portfolio.height
        if total_apps == 0:
            empty_eval = self.evaluate_policy(0.0)
            return OptimizationResult(
                optimal_threshold=0.0,
                optimal_policy=empty_eval,
                baseline_policy_05=empty_eval,
                individual_optimal_pnl=0.0,
                incremental_pnl_vs_baseline=0.0,
                objective_metric=metric,
            )

        # Extract sorted PDs to form candidate evaluation thresholds
        pds = self._enriched_portfolio["pd"].to_numpy()
        min_pd = float(np.min(pds))
        max_pd = float(np.max(pds))

        # Include boundary points, quantiles, and uniform grid
        grid_uniform = np.linspace(0.0, 1.0, grid_size)
        grid_empirical = np.quantile(pds, np.linspace(0.0, 1.0, min(grid_size, len(pds))))
        candidate_thresholds = np.unique(
            np.clip(
                np.concatenate(([0.0, 0.5, 1.0, min_pd, max_pd], grid_uniform, grid_empirical)),
                0.0,
                1.0,
            )
        )
        candidate_thresholds.sort()

        best_threshold = 0.0
        best_metric_value = -float("inf")
        best_eval: PolicyEvaluation | None = None

        for t in candidate_thresholds:
            pol_eval = self.evaluate_policy(float(t))
            val = (
                pol_eval.expected_pnl
                if metric == "expected_pnl"
                else pol_eval.return_on_exposure
            )
            if val > best_metric_value:
                best_metric_value = val
                best_threshold = float(t)
                best_eval = pol_eval

        # Fallback safeguard
        if best_eval is None:
            best_eval = self.evaluate_policy(best_threshold)

        baseline_05 = self.evaluate_policy(0.5)

        # Theoretical unconstrained individual approval (approve loan i iff EV_i > 0)
        positive_ev_loans = self._enriched_portfolio.filter(pl.col("expected_value") > 0)
        individual_max_pnl = (
            float(positive_ev_loans["expected_value"].sum())
            if positive_ev_loans.height > 0
            else 0.0
        )

        return OptimizationResult(
            optimal_threshold=best_threshold,
            optimal_policy=best_eval,
            baseline_policy_05=baseline_05,
            individual_optimal_pnl=individual_max_pnl,
            incremental_pnl_vs_baseline=best_eval.expected_pnl - baseline_05.expected_pnl,
            objective_metric=metric,
        )

    def compute_tradeoff_curve(self, num_points: int = 100) -> pl.DataFrame:
        """Generate the full trade-off curve across probability thresholds.

        Produces a Polars DataFrame detailing:
        - Threshold (p)
        - Approval Rate (%)
        - Approved Volume / Total Exposure ($)
        - Approved Count
        - Expected Credit Loss ($)
        - Expected Loss Rate (% of exposure)
        - Gross Margin ($)
        - Net Expected PnL / Financial Margin ($)
        - Return on Exposure / ROI (%)
        - Expected Default Rate (% of approved loans)
        """
        if num_points < 2:
            msg = f"num_points must be at least 2, got {num_points}"
            raise ValueError(msg)

        thresholds = np.linspace(0.0, 1.0, num_points)
        rows: list[dict[str, float | int]] = []

        for t in thresholds:
            ev = self.evaluate_policy(float(t))
            loss_rate = (ev.expected_loss / ev.total_exposure) if ev.total_exposure > 0.0 else 0.0
            rows.append(
                {
                    "threshold": round(float(t), 4),
                    "approval_rate": round(ev.approval_rate, 4),
                    "approved_count": ev.approved_count,
                    "total_exposure": round(ev.total_exposure, 2),
                    "expected_loss": round(ev.expected_loss, 2),
                    "expected_loss_rate": round(loss_rate, 4),
                    "gross_margin": round(ev.gross_margin, 2),
                    "net_financial_margin": round(ev.expected_pnl, 2),
                    "return_on_exposure": round(ev.return_on_exposure, 4),
                    "expected_default_rate": round(ev.expected_default_rate, 4),
                }
            )

        return pl.DataFrame(rows)

    def optimize_amount_policy(
        self,
        risk_sensitivity: float = 1.5,
        min_approval_amount: float = 500.0,
    ) -> AmountPolicyResult:
        """Apply a risk-based credit limit sizing rule.

        Instead of binary 0 vs requested amount:
        1. Reject loans with EV <= 0.
        2. For loans with EV > 0, scale the approved amount according to the applicant's risk:
           SizedAmount = RequestedAmount * (1 - (PD / BreakevenPD))^risk_sensitivity
           clamped between min_approval_amount and RequestedAmount.
        """
        if self._enriched_portfolio.height == 0:
            return AmountPolicyResult(
                total_applications=0,
                approved_count=0,
                approval_rate=0.0,
                requested_exposure=0.0,
                allocated_exposure=0.0,
                expected_pnl=0.0,
                expected_loss=0.0,
                return_on_exposure=0.0,
            )

        df = self._enriched_portfolio.clone()

        # Risk-based scaling factor
        safe_breakeven = (
            pl.when(pl.col("breakeven_pd") > 0)
            .then(pl.col("breakeven_pd"))
            .otherwise(1.0)
        )
        scaling_expr = (1.0 - (pl.col("pd") / safe_breakeven)).clip(0.0, 1.0) ** risk_sensitivity

        sized_amount_expr = (pl.col("loan_amount") * scaling_expr)

        # Only approve if EV > 0 and sized amount >= min_approval_amount
        df_sized = df.with_columns(
            pl.when((pl.col("expected_value") > 0) & (sized_amount_expr >= min_approval_amount))
            .then(sized_amount_expr)
            .otherwise(0.0)
            .alias("approved_amount")
        )

        approved = df_sized.filter(pl.col("approved_amount") > 0)
        total_apps = df.height
        approved_count = approved.height
        req_exposure = float(df["loan_amount"].sum())
        alloc_exposure = float(approved["approved_amount"].sum()) if approved_count > 0 else 0.0

        if approved_count == 0 or alloc_exposure == 0.0:
            return AmountPolicyResult(
                total_applications=total_apps,
                approved_count=0,
                approval_rate=0.0,
                requested_exposure=req_exposure,
                allocated_exposure=0.0,
                expected_pnl=0.0,
                expected_loss=0.0,
                return_on_exposure=0.0,
            )

        # Compute economics under sized amount
        tenure_years = approved["loan_term_months"] / 12.0
        unit_gain = (approved["interest_rate"] - approved["cost_of_funds"]) * tenure_years
        unit_loss = approved["lgd"] + (approved["cost_of_funds"] * tenure_years)

        sized_gain = approved["approved_amount"] * unit_gain
        sized_loss = approved["approved_amount"] * unit_loss
        sized_ev = ((1.0 - approved["pd"]) * sized_gain) - (approved["pd"] * sized_loss)
        sized_expected_loss = approved["pd"] * sized_loss

        total_pnl = float(sized_ev.sum())
        total_loss = float(sized_expected_loss.sum())
        roe = total_pnl / alloc_exposure

        return AmountPolicyResult(
            total_applications=total_apps,
            approved_count=approved_count,
            approval_rate=approved_count / total_apps,
            requested_exposure=req_exposure,
            allocated_exposure=alloc_exposure,
            expected_pnl=total_pnl,
            expected_loss=total_loss,
            return_on_exposure=roe,
        )
