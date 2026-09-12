"""Data pipeline entrypoint for credit-policy-optimizer."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def prepare_directories() -> None:
    """Ensure raw and processed data directories exist."""
    base_dir = Path("data")
    for subdir in ["raw", "processed", "interim"]:
        target = base_dir / subdir
        target.mkdir(parents=True, exist_ok=True)
    logger.info("Data directories initialized successfully.")


def generate_sample_portfolio(n_samples: int = 100) -> pl.DataFrame:
    """Generate synthetic portfolio data for credit risk simulation."""
    rng = np.random.default_rng(seed=42)
    ages: list[int] = rng.integers(18, 70, size=n_samples).tolist()
    incomes: list[float] = np.round(rng.lognormal(mean=10.5, sigma=0.5, size=n_samples), 2).tolist()
    scores: list[int] = rng.integers(300, 850, size=n_samples).tolist()

    df_dict: dict[str, Any] = {
        "customer_id": [f"CUST_{i:04d}" for i in range(n_samples)],
        "age": ages,
        "monthly_income": incomes,
        "credit_score": scores,
    }
    return pl.DataFrame(df_dict)


if __name__ == "__main__":
    logger.info("Executing data preparation pipeline...")
    prepare_directories()
    sample_df = generate_sample_portfolio(10)
    logger.info("Sample portfolio preview:\n%s", sample_df.head())
    logger.info("Data preparation completed.")
