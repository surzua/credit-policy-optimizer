"""Training and probability calibration pipeline CLI entrypoint."""

import logging
from pathlib import Path
from typing import Any

import polars as pl
from sklearn.model_selection import train_test_split

from credit_policy_optimizer.data.generator import PortfolioSimulator
from credit_policy_optimizer.models.calibration import calibrate_pipeline, evaluate_calibration
from credit_policy_optimizer.models.pipeline import LightGBMConfig, build_credit_pipeline
from credit_policy_optimizer.models.serialization import save_pipeline_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_training_pipeline(
    data_path: str | Path | None = None,
    output_model_path: str | Path = "models/credit_pipeline_calibrated.joblib",
    n_samples: int = 25_000,
    random_seed: int = 42,
    calibration_method: str = "isotonic",
) -> tuple[Any, Any, Path]:
    """Execute the complete training, probability calibration, and serialization workflow.

    Parameters
    ----------
    data_path : str | Path | None
        Path to parquet dataset. If None or non-existent, generates synthetic data.
    output_model_path : str | Path
        Target destination for the calibrated pipeline artifact.
    n_samples : int
        Number of samples to simulate if generating on the fly.
    random_seed : int
        Random seed for reproducibility.
    calibration_method : str
        Method for calibration ('isotonic' or 'sigmoid').

    Returns
    -------
    tuple[Any, Any, Path]
        (uncalibrated_pipeline, calibrated_pipeline, saved_path)
    """
    if data_path and Path(data_path).exists():
        logger.info("Loading portfolio data from %s...", data_path)
        df = pl.read_parquet(data_path)
    else:
        logger.info("Simulating %d credit applications with seed %d...", n_samples, random_seed)
        simulator = PortfolioSimulator(seed=random_seed)
        df = simulator.simulate(n_samples=n_samples)

    feature_cols = [
        "monthly_income",
        "debt_to_income",
        "historical_delinquencies",
        "revolving_utilization",
        "loan_amount",
        "loan_term_months",
        "interest_rate",
        "cost_of_funds",
    ]
    target_col = "default_flag"

    # Strict 3-way split: 60% Train, 20% Validation (Calibration), 20% Test (Evaluation)
    logger.info("Partitioning data into Train (60%), Validation (20%), and Test (20%)...")
    targets_full = df[target_col].to_numpy()
    train_df, temp_df = train_test_split(
        df, test_size=0.40, random_state=random_seed, stratify=targets_full
    )

    targets_temp = temp_df[target_col].to_numpy()
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=random_seed, stratify=targets_temp
    )

    X_train = train_df.select(feature_cols)
    y_train = train_df[target_col]

    X_val = val_df.select(feature_cols)
    y_val = val_df[target_col]

    X_test = test_df.select(feature_cols)
    y_test = test_df[target_col]

    logger.info(
        "Split sizes -> Train: %d (%.1f%% defaults), Val: %d (%.1f%% defaults), "
        "Test: %d (%.1f%% defaults)",
        len(X_train),
        float(y_train.mean() or 0.0) * 100,
        len(X_val),
        float(y_val.mean() or 0.0) * 100,
        len(X_test),
        float(y_test.mean() or 0.0) * 100,
    )

    # 1. Train LightGBM Pipeline
    logger.info("Fitting uncalibrated LightGBM pipeline on training set...")
    config = LightGBMConfig(
        objective="binary",
        metric="binary_logloss",
        learning_rate=0.05,
        n_estimators=150,
        random_state=random_seed,
    )
    pipeline = build_credit_pipeline(model_config=config)
    pipeline.fit(X_train, y_train.to_numpy())

    # 2. Probability Calibration on Validation Set
    logger.info(
        "Calibrating probabilities via CalibratedClassifierCV (method='%s') on validation set...",
        calibration_method,
    )
    calibrated_pipeline = calibrate_pipeline(
        fitted_pipeline=pipeline,
        X_val=X_val,
        y_val=y_val,
        method="isotonic" if calibration_method == "isotonic" else "sigmoid",
    )

    # 3. Evaluate Calibration on Independent Test Set
    logger.info("Evaluating performance metrics on independent test set...")
    comparison = evaluate_calibration(
        uncalibrated_model=pipeline,
        calibrated_model=calibrated_pipeline,
        X_eval=X_test,
        y_eval=y_test,
        n_bins=10,
    )
    logger.info("\n%s", comparison.summary())

    # 4. Serialize Calibrated Pipeline Artifact
    metadata = {
        "features": feature_cols,
        "calibration_method": calibration_method,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "test_metrics": {
            "uncalibrated_brier": comparison.uncalibrated.brier_score,
            "calibrated_brier": comparison.calibrated.brier_score,
            "uncalibrated_ece": comparison.uncalibrated.expected_calibration_error,
            "calibrated_ece": comparison.calibrated.expected_calibration_error,
            "uncalibrated_roc_auc": comparison.uncalibrated.roc_auc,
            "calibrated_roc_auc": comparison.calibrated.roc_auc,
            "ece_reduction_pct": comparison.ece_reduction_pct,
            "brier_reduction_pct": comparison.brier_reduction_pct,
        },
    }

    saved_path = save_pipeline_artifact(
        model=calibrated_pipeline,
        output_path=output_model_path,
        metadata=metadata,
    )
    logger.info("Calibrated model successfully serialized to: %s", saved_path)

    return pipeline, calibrated_pipeline, saved_path


def main() -> None:
    """CLI entrypoint."""
    run_training_pipeline()


if __name__ == "__main__":
    main()
