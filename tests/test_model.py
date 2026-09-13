"""Unit and integration tests for credit risk modeling and probability calibration."""

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from credit_policy_optimizer.data.generator import PortfolioSimulator
from credit_policy_optimizer.data.schema import CreditApplication
from credit_policy_optimizer.models.calibration import (
    calibrate_pipeline,
    compute_expected_calibration_error,
    compute_gini_coefficient,
    compute_ks_statistic,
    evaluate_calibration,
)
from credit_policy_optimizer.models.pipeline import (
    LightGBMConfig,
    train_credit_pipeline,
)
from credit_policy_optimizer.models.serialization import (
    load_pipeline,
    load_pipeline_artifact,
    save_pipeline_artifact,
)


@pytest.fixture(scope="module")
def synthetic_portfolio() -> pl.DataFrame:
    """Fixture providing a deterministic synthetic credit portfolio."""
    simulator = PortfolioSimulator(seed=101)
    return simulator.simulate(n_samples=3_000)


@pytest.fixture(scope="module")
def fitted_models(
    synthetic_portfolio: pl.DataFrame,
) -> dict[str, Any]:
    """Train and calibrate credit models for use across the test suite."""
    train_df = synthetic_portfolio[:2_000]
    val_df = synthetic_portfolio[2_000:2_500]
    test_df = synthetic_portfolio[2_500:]

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

    X_train = train_df.select(feature_cols)
    y_train = train_df["default_flag"]

    X_val = val_df.select(feature_cols)
    y_val = val_df["default_flag"]

    X_test = test_df.select(feature_cols)
    y_test = test_df["default_flag"]

    config = LightGBMConfig(
        n_estimators=60,
        max_depth=4,
        learning_rate=0.08,
        random_state=42,
    )
    uncalibrated_pipeline = train_credit_pipeline(
        X_train=X_train,
        y_train=y_train,
        model_config=config,
    )

    isotonic_calibrated = calibrate_pipeline(
        fitted_pipeline=uncalibrated_pipeline,
        X_val=X_val,
        y_val=y_val,
        method="isotonic",
    )

    sigmoid_calibrated = calibrate_pipeline(
        fitted_pipeline=uncalibrated_pipeline,
        X_val=X_val,
        y_val=y_val,
        method="sigmoid",
    )

    return {
        "uncalibrated": uncalibrated_pipeline,
        "isotonic": isotonic_calibrated,
        "sigmoid": sigmoid_calibrated,
        "X_test": X_test,
        "y_test": y_test,
    }


def test_calibrated_probabilities_in_range_0_1(fitted_models: dict[str, Any]) -> None:
    """Verify that calibrated probability of default (PD) is strictly within [0.0, 1.0]."""
    X_test = fitted_models["X_test"]

    for model_key in ("isotonic", "sigmoid"):
        model = fitted_models[model_key]
        probs = model.predict_proba(X_test)

        assert probs.shape == (len(X_test), 2)
        # Class probabilities must sum to 1.0
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

        pd_probs = probs[:, 1]
        assert np.all(pd_probs >= 0.0), f"{model_key} produced probabilities < 0.0"
        assert np.all(pd_probs <= 1.0), f"{model_key} produced probabilities > 1.0"
        assert not np.isnan(pd_probs).any(), f"{model_key} produced NaN probabilities"


def test_pipeline_handles_incomplete_inputs(fitted_models: dict[str, Any]) -> None:
    """Verify that the pipeline gracefully imputes missing features (nulls and absent columns)."""
    calibrated_model = fitted_models["isotonic"]

    # Case 1: DataFrame with explicit nulls/NaNs across numeric and categorical columns
    data_with_nulls = pl.DataFrame(
        [
            {
                "monthly_income": None,
                "debt_to_income": 0.35,
                "historical_delinquencies": None,
                "revolving_utilization": 0.40,
                "loan_amount": 10000.0,
                "loan_term_months": None,
                "interest_rate": 0.12,
                "cost_of_funds": 0.05,
            },
            {
                "monthly_income": 4000.0,
                "debt_to_income": None,
                "historical_delinquencies": 0,
                "revolving_utilization": None,
                "loan_amount": None,
                "loan_term_months": 36,
                "interest_rate": None,
                "cost_of_funds": None,
            },
        ],
        schema_overrides={
            "monthly_income": pl.Float64,
            "debt_to_income": pl.Float64,
            "historical_delinquencies": pl.Float64,
            "revolving_utilization": pl.Float64,
            "loan_amount": pl.Float64,
            "loan_term_months": pl.Int64,
            "interest_rate": pl.Float64,
            "cost_of_funds": pl.Float64,
        },
    )

    preds = calibrated_model.predict_proba(data_with_nulls)
    assert preds.shape == (2, 2)
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)
    assert not np.isnan(preds).any()

    # Case 2: DataFrame missing several required columns completely
    sparse_data = pl.DataFrame(
        [
            {"monthly_income": 5000.0, "loan_amount": 15000.0},
            {"debt_to_income": 0.45},
        ]
    )

    sparse_preds = calibrated_model.predict_proba(sparse_data)
    assert sparse_preds.shape == (2, 2)
    assert np.all(sparse_preds >= 0.0) and np.all(sparse_preds <= 1.0)
    assert not np.isnan(sparse_preds).any()


def test_pipeline_handles_outliers_and_atypical_values(
    fitted_models: dict[str, Any],
) -> None:
    """Verify stability and bounded predictions when exposed to extreme outliers or inputs."""
    calibrated_model = fitted_models["isotonic"]

    atypical_records = pl.DataFrame(
        [
            # Extreme high wealth
            {
                "monthly_income": 10_000_000.0,
                "debt_to_income": 0.001,
                "historical_delinquencies": 0,
                "revolving_utilization": 0.0,
                "loan_amount": 1_000.0,
                "loan_term_months": 12,
                "interest_rate": 0.06,
                "cost_of_funds": 0.05,
            },
            # Extreme subprime / distress: zero income, high DTI, massive delinquencies
            {
                "monthly_income": 0.0,
                "debt_to_income": 10.0,
                "historical_delinquencies": 50,
                "revolving_utilization": 5.0,
                "loan_amount": 1_000_000.0,
                "loan_term_months": 60,
                "interest_rate": 0.60,
                "cost_of_funds": 0.08,
            },
            # Negative income, negative utilization (sensor / input error)
            {
                "monthly_income": -2500.0,
                "debt_to_income": -0.5,
                "historical_delinquencies": -3,
                "revolving_utilization": -0.2,
                "loan_amount": -1000.0,
                "loan_term_months": 36,
                "interest_rate": 0.10,
                "cost_of_funds": 0.05,
            },
            # Unseen categorical loan tenure (e.g. 120 or 999 months)
            {
                "monthly_income": 3500.0,
                "debt_to_income": 0.30,
                "historical_delinquencies": 0,
                "revolving_utilization": 0.25,
                "loan_amount": 5000.0,
                "loan_term_months": 120,
                "interest_rate": 0.11,
                "cost_of_funds": 0.05,
            },
        ]
    )

    preds = calibrated_model.predict_proba(atypical_records)
    assert preds.shape == (4, 2)
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)
    assert not np.isnan(preds).any()

    # Extreme subprime record should have significantly higher default risk than high-wealth record
    prob_wealthy = preds[0, 1]
    prob_distressed = preds[1, 1]
    assert prob_distressed > prob_wealthy


def test_input_types_flexibility(fitted_models: dict[str, Any]) -> None:
    """Verify that the pipeline transparently accepts Polars, dict, list of dicts, and Pydantic."""
    model = fitted_models["isotonic"]

    sample_dict = {
        "monthly_income": 4500.0,
        "debt_to_income": 0.28,
        "historical_delinquencies": 1,
        "revolving_utilization": 0.35,
        "loan_amount": 12000.0,
        "loan_term_months": 36,
        "interest_rate": 0.125,
        "cost_of_funds": 0.0525,
    }

    # 1. Single dict
    pred_dict = model.predict_proba(sample_dict)
    assert pred_dict.shape == (1, 2)

    # 2. List of dicts
    pred_list = model.predict_proba([sample_dict, sample_dict])
    assert pred_list.shape == (2, 2)

    # 3. Polars DataFrame
    pred_polars = model.predict_proba(pl.DataFrame([sample_dict]))
    assert pred_polars.shape == (1, 2)

    # 4. Pydantic Model
    app = CreditApplication(
        application_id="TEST_001",
        monthly_income=4500.0,
        debt_to_income=0.28,
        historical_delinquencies=1,
        revolving_utilization=0.35,
        loan_amount=12000.0,
        loan_term_months=36,
        interest_rate=0.125,
        cost_of_funds=0.0525,
    )
    pred_pydantic = model.predict_proba(app)
    assert pred_pydantic.shape == (1, 2)

    np.testing.assert_allclose(pred_dict[0], pred_polars[0], rtol=1e-5)
    np.testing.assert_allclose(pred_dict[0], pred_pydantic[0], rtol=1e-5)


def test_evaluation_and_calibration_metrics(fitted_models: dict[str, Any]) -> None:
    """Verify that evaluate_calibration computes valid Brier, ECE, and ROC-AUC metrics."""
    uncalibrated = fitted_models["uncalibrated"]
    calibrated = fitted_models["isotonic"]
    X_test = fitted_models["X_test"]
    y_test = fitted_models["y_test"]

    comparison = evaluate_calibration(
        uncalibrated_model=uncalibrated,
        calibrated_model=calibrated,
        X_eval=X_test,
        y_eval=y_test,
        n_bins=10,
    )

    # Assert metric ranges
    assert 0.0 <= comparison.uncalibrated.brier_score <= 1.0
    assert 0.0 <= comparison.calibrated.brier_score <= 1.0
    assert 0.0 <= comparison.uncalibrated.expected_calibration_error <= 1.0
    assert 0.0 <= comparison.calibrated.expected_calibration_error <= 1.0
    assert 0.5 <= comparison.uncalibrated.roc_auc <= 1.0
    assert 0.5 <= comparison.calibrated.roc_auc <= 1.0

    # Formatted summary string checks
    summary_text = comparison.summary()
    assert "Brier Score" in summary_text
    assert "ECE" in summary_text
    assert "ROC-AUC" in summary_text


def test_ece_computation_properties() -> None:
    """Verify ECE calculation edge cases including perfect alignment and error conditions."""
    # Perfect calibration scenario: predictions match binary ground truth perfectly
    y_true_perfect = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob_perfect = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    ece, mce, accs, confs, counts = compute_expected_calibration_error(
        y_true_perfect, y_prob_perfect, n_bins=5
    )
    assert ece == pytest.approx(0.0, abs=1e-6)
    assert mce == pytest.approx(0.0, abs=1e-6)

    # Empty inputs must raise ValueError
    with pytest.raises(ValueError, match="Cannot compute calibration metrics on empty arrays"):
        compute_expected_calibration_error([], [])

    # Length mismatch must raise ValueError
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_expected_calibration_error([0, 1], [0.5])


def test_model_serialization_roundtrip(fitted_models: dict[str, Any], tmp_path: Path) -> None:
    """Verify that saving and loading a calibrated pipeline produces identical inference."""
    calibrated_model = fitted_models["isotonic"]
    X_test = fitted_models["X_test"]
    original_preds = calibrated_model.predict_proba(X_test)

    artifact_path = tmp_path / "models" / "test_pipeline.joblib"
    meta = {"version": "0.1.0", "author": "test_suite"}

    saved_path = save_pipeline_artifact(calibrated_model, artifact_path, metadata=meta)
    assert saved_path.exists()

    # Reload with load_pipeline_artifact
    loaded_model, loaded_meta = load_pipeline_artifact(saved_path)
    assert loaded_meta["version"] == "0.1.0"
    assert "saved_at" in loaded_meta

    reloaded_preds = loaded_model.predict_proba(X_test)
    np.testing.assert_allclose(original_preds, reloaded_preds, rtol=1e-6)

    # Reload directly with load_pipeline
    loaded_direct = load_pipeline(saved_path)
    direct_preds = loaded_direct.predict_proba(X_test)
    np.testing.assert_allclose(original_preds, direct_preds, rtol=1e-6)


def test_invalid_calibration_method(fitted_models: dict[str, Any]) -> None:
    """Verify that unsupported calibration method names raise ValueError."""
    uncalibrated = fitted_models["uncalibrated"]
    X_test = fitted_models["X_test"]
    y_test = fitted_models["y_test"]

    with pytest.raises(ValueError, match="Unsupported calibration method: polynomial"):
        calibrate_pipeline(
            fitted_pipeline=uncalibrated,
            X_val=X_test,
            y_val=y_test,
            method="polynomial",  # type: ignore[arg-type]
        )


def test_gini_and_ks_metrics() -> None:
    """Verify calculation of Gini coefficient and KS statistic."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    gini = compute_gini_coefficient(y_true, y_prob)
    assert gini == pytest.approx(1.0, abs=1e-5)

    ks_stat, ks_thresh = compute_ks_statistic(y_true, y_prob)
    assert ks_stat == pytest.approx(1.0, abs=1e-5)
    assert 0.4 <= ks_thresh <= 0.6

    # Random guessing scenario: AUC ~ 0.5 -> Gini ~ 0.0
    y_prob_random = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    gini_random = compute_gini_coefficient(y_true, y_prob_random)
    assert gini_random == pytest.approx(0.0, abs=1e-5)

