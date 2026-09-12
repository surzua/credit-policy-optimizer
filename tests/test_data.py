"""Tests for data preparation utilities."""

from credit_policy_optimizer.data import generate_sample_portfolio, prepare_directories


def test_prepare_directories() -> None:
    """Verify that directories are created successfully."""
    prepare_directories()


def test_generate_sample_portfolio() -> None:
    """Verify synthetic portfolio generation output structure and types."""
    df = generate_sample_portfolio(n_samples=25)
    assert len(df) == 25
    assert "customer_id" in df.columns
    assert "monthly_income" in df.columns
    assert "credit_score" in df.columns
    assert df["credit_score"].min() is not None
    assert df["credit_score"].min() >= 300
