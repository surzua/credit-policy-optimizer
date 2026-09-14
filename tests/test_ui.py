"""Tests for the interactive decision dashboard module."""

from __future__ import annotations

import polars as pl

from credit_policy_optimizer.ui.dashboard import (
    generate_scored_portfolio,
    load_cached_model,
)


def test_load_cached_model() -> None:
    """Verify that cached model loads successfully and implements predict_proba."""
    pipeline = load_cached_model()
    assert pipeline is not None
    assert hasattr(pipeline, "predict_proba")


def test_generate_scored_portfolio() -> None:
    """Verify that benchmark portfolio generator returns valid calibrated probabilities."""
    n_samples = 40
    df = generate_scored_portfolio(n_samples=n_samples, seed=123)

    assert isinstance(df, pl.DataFrame)
    assert df.height == n_samples
    assert "calibrated_pd" in df.columns
    assert "loan_amount" in df.columns

    pds = df["calibrated_pd"].to_numpy()
    assert (pds >= 0.0).all()
    assert (pds <= 1.0).all()
