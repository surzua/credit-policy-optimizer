"""Unit tests and financial invariant verifications for credit decision economics."""

import numpy as np
import polars as pl
import pytest

from credit_policy_optimizer.data.generator import PortfolioSimulator
from credit_policy_optimizer.decision.economics import (
    CreditPolicyOptimizer,
    compute_breakeven_pd,
    compute_expected_value,
    compute_loan_net_gain,
    compute_loan_net_loss,
)


@pytest.fixture
def sample_portfolio() -> pl.DataFrame:
    """Fixture providing a deterministic portfolio dataset."""
    simulator = PortfolioSimulator(seed=42)
    return simulator.simulate(n_samples=5_000)


def test_formula_breakeven_properties() -> None:
    """Verify analytical properties of expected value and breakeven cutoff."""
    amount = 10_000.0
    interest_rate = 0.20  # 20%
    cost_of_funds = 0.05  # 5%
    lgd = 0.50  # 50%
    term_months = 12

    # Expected gain = 10000 * (0.20 - 0.05) * 1 = 1500
    gain = compute_loan_net_gain(amount, interest_rate, cost_of_funds, term_months)
    assert pytest.approx(gain, rel=1e-5) == 1_500.0

    # Expected loss = 10000 * (0.50 + 0.05 * 1) = 5500
    loss = compute_loan_net_loss(amount, lgd, cost_of_funds, term_months)
    assert pytest.approx(loss, rel=1e-5) == 5_500.0

    # Breakeven p* = 1500 / (1500 + 5500) = 1500 / 7000 = 0.2142857...
    p_star = compute_breakeven_pd(interest_rate, cost_of_funds, lgd, term_months)
    expected_p_star = 1_500.0 / (1_500.0 + 5_500.0)
    assert pytest.approx(p_star, rel=1e-5) == expected_p_star

    # At PD = p*, EV must be exactly 0
    ev_at_breakeven = compute_expected_value(
        p_star, amount, interest_rate, cost_of_funds, lgd, term_months
    )
    assert pytest.approx(ev_at_breakeven, abs=1e-6) == 0.0

    # Below p*, EV must be strictly positive
    ev_safe = compute_expected_value(
        p_star - 0.05, amount, interest_rate, cost_of_funds, lgd, term_months
    )
    assert ev_safe > 0.0

    # Above p*, EV must be strictly negative
    ev_risky = compute_expected_value(
        p_star + 0.05, amount, interest_rate, cost_of_funds, lgd, term_months
    )
    assert ev_risky < 0.0


def test_optimizer_enrichment(sample_portfolio: pl.DataFrame) -> None:
    """Verify that CreditPolicyOptimizer enriches dataset with required economic metrics."""
    optimizer = CreditPolicyOptimizer(sample_portfolio)
    df = optimizer.portfolio

    assert "net_gain_performing" in df.columns
    assert "net_loss_default" in df.columns
    assert "expected_value" in df.columns
    assert "breakeven_pd" in df.columns

    # Verify no nulls or NaNs in economic columns
    for col in ["net_gain_performing", "net_loss_default", "expected_value", "breakeven_pd"]:
        assert df[col].null_count() == 0


def test_financial_invariant_funding_cost_increase(sample_portfolio: pl.DataFrame) -> None:
    """Financial invariant: Higher cost of funds must result in a stricter cutoff (p* decreases)."""
    # 1. Analytical check
    p_star_low_cof = compute_breakeven_pd(interest_rate=0.20, cost_of_funds=0.04, lgd=0.45)
    p_star_high_cof = compute_breakeven_pd(interest_rate=0.20, cost_of_funds=0.12, lgd=0.45)
    assert p_star_high_cof < p_star_low_cof, (
        f"Expected p*(high CoF) < p*(low CoF), got {p_star_high_cof} vs {p_star_low_cof}"
    )

    # 2. Portfolio-level optimization check
    # Create two identical portfolios except for cost_of_funds
    df_low = sample_portfolio.with_columns(pl.lit(0.04).alias("cost_of_funds"))
    df_high = sample_portfolio.with_columns(pl.lit(0.12).alias("cost_of_funds"))

    opt_low = CreditPolicyOptimizer(df_low).optimize_threshold()
    opt_high = CreditPolicyOptimizer(df_high).optimize_threshold()

    # The optimal threshold under high funding costs must be strictly lower or equal
    assert opt_high.optimal_threshold <= opt_low.optimal_threshold, (
        f"High CoF threshold ({opt_high.optimal_threshold}) "
        f"should be <= low CoF ({opt_low.optimal_threshold})"
    )
    # The approval rate under higher funding costs must not be higher
    assert opt_high.optimal_policy.approval_rate <= opt_low.optimal_policy.approval_rate


def test_financial_invariant_interest_rate_increase(sample_portfolio: pl.DataFrame) -> None:
    """Financial invariant: Higher active rate allows absorbing more risk (p* increases)."""
    # 1. Analytical check
    p_star_low_rate = compute_breakeven_pd(interest_rate=0.15, cost_of_funds=0.05, lgd=0.45)
    p_star_high_rate = compute_breakeven_pd(interest_rate=0.25, cost_of_funds=0.05, lgd=0.45)
    assert p_star_high_rate > p_star_low_rate

    # 2. Portfolio-level optimization check
    df_low = sample_portfolio.with_columns(pl.lit(0.15).alias("interest_rate"))
    df_high = sample_portfolio.with_columns(pl.lit(0.25).alias("interest_rate"))

    opt_low = CreditPolicyOptimizer(df_low).optimize_threshold()
    opt_high = CreditPolicyOptimizer(df_high).optimize_threshold()

    assert opt_high.optimal_threshold >= opt_low.optimal_threshold
    assert opt_high.optimal_policy.approval_rate >= opt_low.optimal_policy.approval_rate


def test_financial_invariant_lgd_increase(sample_portfolio: pl.DataFrame) -> None:
    """Financial invariant: Higher LGD forces a stricter policy cutoff (p* decreases)."""
    p_star_low_lgd = compute_breakeven_pd(interest_rate=0.20, cost_of_funds=0.05, lgd=0.30)
    p_star_high_lgd = compute_breakeven_pd(interest_rate=0.20, cost_of_funds=0.05, lgd=0.70)
    assert p_star_high_lgd < p_star_low_lgd

    df_low = sample_portfolio.with_columns(pl.lit(0.30).alias("lgd"))
    df_high = sample_portfolio.with_columns(pl.lit(0.70).alias("lgd"))

    opt_low = CreditPolicyOptimizer(df_low).optimize_threshold()
    opt_high = CreditPolicyOptimizer(df_high).optimize_threshold()

    assert opt_high.optimal_threshold <= opt_low.optimal_threshold


def test_optimal_policy_vs_arbitrary_05_baseline(sample_portfolio: pl.DataFrame) -> None:
    """Demonstrate that optimal threshold p* outperforms naive 0.5 classification cutoff."""
    # 1. On sample portfolio, optimal PnL must be at least as high as baseline
    optimizer_sample = CreditPolicyOptimizer(sample_portfolio)
    result_sample = optimizer_sample.optimize_threshold()
    assert (
        result_sample.optimal_policy.expected_pnl >= result_sample.baseline_policy_05.expected_pnl
    )

    # 2. Portfolio spanning full risk spectrum (PD from 1% to 60%), naive 0.5 causes heavy losses
    n = 1_000
    pds = np.linspace(0.01, 0.60, n)
    amounts = np.full(n, 10_000.0)
    df_spectrum = pl.DataFrame(
        {
            "pd": pds,
            "loan_amount": amounts,
            "interest_rate": np.full(n, 0.18),
            "cost_of_funds": np.full(n, 0.06),
            "lgd": np.full(n, 0.45),
            "loan_term_months": np.full(n, 12.0),
        }
    )

    optimizer_spectrum = CreditPolicyOptimizer(df_spectrum)
    result_spectrum = optimizer_spectrum.optimize_threshold()

    # Optimal threshold is approximately 19%
    assert result_spectrum.optimal_threshold < 0.25
    # Strict superiority over naive 0.5
    assert (
        result_spectrum.optimal_policy.expected_pnl
        > result_spectrum.baseline_policy_05.expected_pnl
    )
    assert result_spectrum.incremental_pnl_vs_baseline > 400_000.0  # Over $400k in extra value

    # Naive p = 0.5 approves high-risk borrowers with PD > 19%, resulting in net negative PnL
    assert result_spectrum.baseline_policy_05.expected_pnl < 0.0
    assert result_spectrum.optimal_policy.expected_pnl > 0.0
    assert (
        result_spectrum.baseline_policy_05.expected_loss
        > result_spectrum.optimal_policy.expected_loss
    )
    assert (
        result_spectrum.optimal_policy.return_on_exposure
        > result_spectrum.baseline_policy_05.return_on_exposure
    )


def test_tradeoff_curve_monotonicity_and_structure(sample_portfolio: pl.DataFrame) -> None:
    """Verify that tradeoff curve metrics follow economic monotonicity laws."""
    optimizer = CreditPolicyOptimizer(sample_portfolio)
    curve = optimizer.compute_tradeoff_curve(num_points=25)

    assert curve.height == 25
    expected_cols = {
        "threshold",
        "approval_rate",
        "approved_count",
        "total_exposure",
        "expected_loss",
        "expected_loss_rate",
        "gross_margin",
        "net_financial_margin",
        "return_on_exposure",
        "expected_default_rate",
    }
    assert expected_cols.issubset(set(curve.columns))

    # Approval rate and total exposure must be monotonically non-decreasing with threshold
    approval_rates = curve["approval_rate"].to_list()
    exposures = curve["total_exposure"].to_list()
    expected_losses = curve["expected_loss"].to_list()

    for i in range(len(approval_rates) - 1):
        assert approval_rates[i + 1] >= approval_rates[i] - 1e-6
        assert exposures[i + 1] >= exposures[i] - 1e-2
        assert expected_losses[i + 1] >= expected_losses[i] - 1e-2

    # At threshold 0.0, approval rate should be minimal/0
    assert approval_rates[0] == 0.0 or approval_rates[0] < 0.05
    # At threshold 1.0, approval rate must be 1.0 (all approved)
    assert approval_rates[-1] == 1.0


def test_amount_policy_sizing(sample_portfolio: pl.DataFrame) -> None:
    """Verify risk-based credit limit sizing."""
    optimizer = CreditPolicyOptimizer(sample_portfolio)
    amount_result = optimizer.optimize_amount_policy(risk_sensitivity=1.2)

    assert amount_result.approved_count > 0
    assert amount_result.allocated_exposure > 0.0
    # Sized exposure must be less than or equal to total requested exposure
    assert amount_result.allocated_exposure <= amount_result.requested_exposure
    # Expected PnL under sized allocation must be positive
    assert amount_result.expected_pnl > 0.0
    assert amount_result.return_on_exposure > 0.0


def test_edge_cases() -> None:
    """Verify edge cases: empty portfolio, zero interest rate, boundary validation."""
    empty_df = pl.DataFrame({"pd": [], "loan_amount": []})
    opt_empty = CreditPolicyOptimizer(empty_df)
    res_empty = opt_empty.optimize_threshold()
    assert res_empty.optimal_policy.approved_count == 0
    assert res_empty.optimal_policy.expected_pnl == 0.0

    # Out of bounds threshold
    optimizer = CreditPolicyOptimizer(
        pl.DataFrame({"pd": [0.05, 0.10], "loan_amount": [1000.0, 2000.0]})
    )
    with pytest.raises(ValueError, match="Threshold must be between"):
        optimizer.evaluate_policy(1.5)
    with pytest.raises(ValueError, match="Threshold must be between"):
        optimizer.evaluate_policy(-0.1)

    # Negative margin (interest rate <= cost of funds)
    p_star_negative_margin = compute_breakeven_pd(interest_rate=0.04, cost_of_funds=0.06, lgd=0.45)
    assert p_star_negative_margin == 0.0


def test_optimize_constrained_policy(sample_portfolio: pl.DataFrame) -> None:
    """Verify that constrained optimization respects default rate and approval caps."""
    optimizer = CreditPolicyOptimizer(sample_portfolio)

    # Tight default rate constraint
    constrained_eval = optimizer.optimize_constrained_policy(
        max_default_rate=0.05,
        min_approval_rate=0.10,
    )
    if constrained_eval.approved_count > 0:
        assert constrained_eval.expected_default_rate <= 0.05
        assert constrained_eval.approval_rate >= 0.10

    # Impossible constraint fallback scenario (max_default_rate = 0.0, min_approval_rate = 0.99)
    impossible_eval = optimizer.optimize_constrained_policy(
        max_default_rate=0.0001,
        min_approval_rate=0.95,
    )
    assert impossible_eval.threshold == 0.0
    assert impossible_eval.approved_count == 0

    # Invalid grid_size error
    with pytest.raises(ValueError, match="grid_size must be at least 10"):
        optimizer.optimize_constrained_policy(grid_size=5)

