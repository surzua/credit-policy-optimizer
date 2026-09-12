"""Unit tests for portfolio simulation and data generator."""

from pathlib import Path

import polars as pl
import pytest

from credit_policy_optimizer.data.generator import PortfolioSimulator
from credit_policy_optimizer.data.schema import SimulatedCreditApplication, SimulationResult


@pytest.fixture
def simulator() -> PortfolioSimulator:
    """Fixture providing a deterministic portfolio simulator instance."""
    return PortfolioSimulator(seed=42)


def test_simulation_dimensions_and_schema(simulator: PortfolioSimulator) -> None:
    """Verify that generated DataFrame has exact expected row and column counts and types."""
    n_samples = 5_000
    df = simulator.simulate(n_samples=n_samples)

    assert df.height == n_samples
    assert df.width == 11

    expected_columns = {
        "application_id",
        "monthly_income",
        "debt_to_income",
        "historical_delinquencies",
        "revolving_utilization",
        "loan_amount",
        "loan_term_months",
        "interest_rate",
        "cost_of_funds",
        "pd",
        "default_flag",
    }
    assert set(df.columns) == expected_columns


def test_absence_of_nulls(simulator: PortfolioSimulator) -> None:
    """Verify that the generated dataset has zero null or NaN values across all columns."""
    df = simulator.simulate(n_samples=2_000)

    for col in df.columns:
        null_count = df[col].null_count()
        assert null_count == 0, f"Column '{col}' contains {null_count} nulls"


def test_default_rate_in_realistic_range(simulator: PortfolioSimulator) -> None:
    """Verify that the default rate stays within plausible retail credit ranges (3% to 8%)."""
    # Test across a statistically representative sample (20,000 observations)
    df = simulator.simulate(n_samples=20_000)

    default_rate = float(df["default_flag"].mean())
    avg_pd = float(df["pd"].mean())

    # Range assertions: expected default rate between 3% and 8%
    assert 0.03 <= default_rate <= 0.08, f"Default rate {default_rate:.4f} is outside 3%-8% bounds"
    assert 0.03 <= avg_pd <= 0.08, f"Average PD {avg_pd:.4f} is outside 3%-8% bounds"


def test_feature_bounds_and_coherence(simulator: PortfolioSimulator) -> None:
    """Verify that financial variables obey domain and business boundaries."""
    df = simulator.simulate(n_samples=1_000)

    # Incomes must be positive and bounded
    assert df["monthly_income"].min() is not None and df["monthly_income"].min() >= 1000.0
    assert df["monthly_income"].max() is not None and df["monthly_income"].max() <= 35000.0

    # DTI between 5% and 70%
    assert df["debt_to_income"].min() is not None and df["debt_to_income"].min() >= 0.05
    assert df["debt_to_income"].max() is not None and df["debt_to_income"].max() <= 0.70

    # Revolving utilization between 0 and 1
    min_util = df["revolving_utilization"].min()
    max_util = df["revolving_utilization"].max()
    assert min_util is not None and min_util >= 0.0
    assert max_util is not None and max_util <= 1.0

    # Loan terms in allowed values
    valid_terms = {12, 24, 36, 48, 60}
    unique_terms = set(df["loan_term_months"].unique().to_list())
    assert unique_terms.issubset(valid_terms)

    # Cost of funds below interest rate on average
    assert float(df["interest_rate"].mean()) > float(df["cost_of_funds"].mean())

    # Default flags strictly binary
    unique_flags = set(df["default_flag"].unique().to_list())
    assert unique_flags.issubset({0, 1})


def test_deterministic_reproducibility() -> None:
    """Verify that running two simulators with the same seed generates identical data."""
    sim1 = PortfolioSimulator(seed=123)
    sim2 = PortfolioSimulator(seed=123)

    df1 = sim1.simulate(n_samples=1_000)
    df2 = sim2.simulate(n_samples=1_000)

    assert df1.equals(df2)


def test_parquet_export_roundtrip(simulator: PortfolioSimulator, tmp_path: Path) -> None:
    """Verify that saving to Parquet produces a valid file matching the in-memory DataFrame."""
    df = simulator.simulate(n_samples=500)
    file_path = tmp_path / "test_portfolio.parquet"

    saved_path = simulator.save_parquet(df, file_path)
    assert saved_path.exists()
    assert saved_path.stat().st_size > 0

    reloaded_df = pl.read_parquet(saved_path)
    assert df.equals(reloaded_df)


def test_pydantic_schema_validation(simulator: PortfolioSimulator) -> None:
    """Verify that generated records conform strictly to the Pydantic v2 schemas."""
    df = simulator.simulate(n_samples=100)
    records = df.to_dicts()

    for rec in records:
        validated = SimulatedCreditApplication.model_validate(rec)
        assert validated.monthly_income > 0
        assert 0.0 <= validated.pd <= 1.0
        assert validated.default_flag in (0, 1)

    # Summary validation
    summary = simulator.compute_summary(df)
    assert isinstance(summary, SimulationResult)
    assert summary.total_applications == 100
    assert summary.default_count == int(df["default_flag"].sum())
    assert summary.total_exposure > 0


def test_invalid_n_samples(simulator: PortfolioSimulator) -> None:
    """Verify that requesting non-positive sample count raises ValueError."""
    with pytest.raises(ValueError, match="n_samples must be greater than 0"):
        simulator.simulate(n_samples=0)
