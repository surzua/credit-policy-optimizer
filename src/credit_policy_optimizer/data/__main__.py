"""CLI entrypoint for portfolio dataset generation."""

import logging
from pathlib import Path

from credit_policy_optimizer.data.generator import PortfolioSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Generate deterministic portfolio dataset and export to Parquet."""
    output_path = Path("data/processed/portfolio.parquet")
    logger.info("Initializing PortfolioSimulator (seed=42)...")
    simulator = PortfolioSimulator(seed=42)

    n_samples = 25_000
    logger.info("Simulating %d credit applications...", n_samples)
    df = simulator.simulate(n_samples=n_samples)

    saved_path = simulator.save_parquet(df, output_path)
    logger.info("Portfolio successfully exported to: %s", saved_path)

    summary = simulator.compute_summary(df)
    logger.info(
        "Simulation Summary -> Total: %d, Default Rate: %.2f%%, Avg PD: %.2f%%, Total Volume: $%s",
        summary.total_applications,
        summary.default_rate * 100,
        summary.avg_pd * 100,
        f"{summary.total_exposure:,.2f}",
    )


if __name__ == "__main__":
    main()
