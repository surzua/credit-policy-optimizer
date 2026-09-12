"""Probability calibration and evaluation for credit risk scoring models."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

try:
    from sklearn.frozen import FrozenEstimator

    HAS_FROZEN_ESTIMATOR = True
except ImportError:
    HAS_FROZEN_ESTIMATOR = False


@dataclass(frozen=True)
class ModelCalibrationMetrics:
    """Quantitative calibration and discrimination metrics for a credit risk model."""

    brier_score: float
    expected_calibration_error: float
    maximum_calibration_error: float
    roc_auc: float
    bin_accuracies: list[float]
    bin_confidences: list[float]
    bin_counts: list[int]


@dataclass(frozen=True)
class CalibrationComparison:
    """Comparison of risk model performance before and after probability calibration."""

    uncalibrated: ModelCalibrationMetrics
    calibrated: ModelCalibrationMetrics
    brier_reduction: float
    brier_reduction_pct: float
    ece_reduction: float
    ece_reduction_pct: float
    roc_auc_delta: float

    def summary(self) -> str:
        """Format a human-readable summary of the calibration impact."""
        lines = [
            "=== Probability Calibration Evaluation ===",
            (
                f"Brier Score: {self.uncalibrated.brier_score:.5f} -> "
                f"{self.calibrated.brier_score:.5f} ({self.brier_reduction_pct:+.2f}%)"
            ),
            f"ECE:         {self.uncalibrated.expected_calibration_error:.5f} -> "
            f"{self.calibrated.expected_calibration_error:.5f} ({self.ece_reduction_pct:+.2f}%)",
            f"Max CE:      {self.uncalibrated.maximum_calibration_error:.5f} -> "
            f"{self.calibrated.maximum_calibration_error:.5f}",
            f"ROC-AUC:     {self.uncalibrated.roc_auc:.4f} -> {self.calibrated.roc_auc:.4f} "
            f"({self.roc_auc_delta:+.4f})",
        ]
        return "\n".join(lines)


class CalibratedCreditClassifier(CalibratedClassifierCV):
    """CalibratedClassifierCV subclass with enhanced ergonomics for credit decisioning.

    Accepts single records (dictionary, Pydantic CreditApplication model) or batched
    containers (Polars DataFrame, list of dicts) transparently.
    """

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict calibrated class probabilities."""
        if isinstance(X, dict) or hasattr(X, "model_dump"):
            X = [X]
        result: np.ndarray = np.asarray(super().predict_proba(X))
        return result

    def predict(self, X: Any) -> np.ndarray:
        """Predict calibrated binary class labels."""
        if isinstance(X, dict) or hasattr(X, "model_dump"):
            X = [X]
        result: np.ndarray = np.asarray(super().predict(X))
        return result


def compute_expected_calibration_error(
    y_true: np.ndarray | Sequence[int] | pl.Series,
    y_prob: np.ndarray | Sequence[float] | pl.Series,
    n_bins: int = 10,
) -> tuple[float, float, list[float], list[float], list[int]]:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Partitions predicted probabilities into equal-width bins and measures the weighted
    absolute difference between empirical accuracy (default rate) and mean predicted confidence.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 or 1).
    y_prob : array-like
        Predicted probabilities of class 1.
    n_bins : int, default=10
        Number of equal-width bins between 0.0 and 1.0.

    Returns
    -------
    tuple
        (ece, mce, bin_accuracies, bin_confidences, bin_counts)
    """
    if isinstance(y_true, pl.Series):
        y_true_arr = y_true.to_numpy().astype(np.float64)
    else:
        y_true_arr = np.asarray(y_true, dtype=np.float64)

    if isinstance(y_prob, pl.Series):
        y_prob_arr = y_prob.to_numpy().astype(np.float64)
    else:
        y_prob_arr = np.asarray(y_prob, dtype=np.float64)

    if len(y_true_arr) != len(y_prob_arr):
        msg = f"Length mismatch: y_true ({len(y_true_arr)}) vs y_prob ({len(y_prob_arr)})"
        raise ValueError(msg)

    if len(y_true_arr) == 0:
        msg = "Cannot compute calibration metrics on empty arrays."
        raise ValueError(msg)

    # Enforce probability clipping strictly within [0.0, 1.0]
    y_prob_arr = np.clip(y_prob_arr, 0.0, 1.0)

    # Bin boundaries [0.0, 0.1, 0.2, ..., 1.0]
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin indices: 1 to n_bins (values equal to 1.0 get assigned to last bin)
    bin_assignments = np.digitize(y_prob_arr, bin_edges, right=True)
    # Ensure boundary handling for 0.0
    bin_assignments = np.clip(bin_assignments, 1, n_bins)

    total_samples = len(y_true_arr)
    weighted_ece = 0.0
    max_ce = 0.0

    bin_accuracies: list[float] = []
    bin_confidences: list[float] = []
    bin_counts: list[int] = []

    for bin_idx in range(1, n_bins + 1):
        mask = bin_assignments == bin_idx
        count = int(np.sum(mask))
        bin_counts.append(count)

        if count > 0:
            bin_acc = float(np.mean(y_true_arr[mask]))
            bin_conf = float(np.mean(y_prob_arr[mask]))
            abs_err = abs(bin_acc - bin_conf)

            weighted_ece += (count / total_samples) * abs_err
            max_ce = max(max_ce, abs_err)

            bin_accuracies.append(round(bin_acc, 6))
            bin_confidences.append(round(bin_conf, 6))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)

    return float(weighted_ece), float(max_ce), bin_accuracies, bin_confidences, bin_counts


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> ModelCalibrationMetrics:
    """Calculate Brier Score, ECE, MCE, and ROC-AUC for predicted probabilities."""
    brier = float(brier_score_loss(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    ece, mce, bin_accs, bin_confs, bin_counts = compute_expected_calibration_error(
        y_true, y_prob, n_bins=n_bins
    )

    return ModelCalibrationMetrics(
        brier_score=round(brier, 6),
        expected_calibration_error=round(ece, 6),
        maximum_calibration_error=round(mce, 6),
        roc_auc=round(roc_auc, 6),
        bin_accuracies=bin_accs,
        bin_confidences=bin_confs,
        bin_counts=bin_counts,
    )


def calibrate_pipeline(
    fitted_pipeline: Pipeline | Any,
    X_val: Any,
    y_val: Any,
    method: Literal["isotonic", "sigmoid"] = "isotonic",
    cv: int | None = None,
) -> CalibratedCreditClassifier:
    """Calibrate probabilities of a fitted classifier pipeline using a separate validation set.

    In Scikit-Learn >= 1.6, pre-fitted models are wrapped in `FrozenEstimator` to perform
    calibration over the designated validation set without retraining the base classifier.

    Parameters
    ----------
    fitted_pipeline : Pipeline | Any
        Pre-trained Scikit-Learn Pipeline or estimator.
    X_val : Any
        Validation set features (disjoint from training set).
    y_val : Any
        Validation set binary targets (0 or 1).
    method : {"isotonic", "sigmoid"}, default="isotonic"
        Calibration method. 'isotonic' fits a non-parametric monotonic step function.
        'sigmoid' fits Platt scaling (logistic regression on logits).
    cv : int | None, default=None
        Cross-validation fold count for calibration. If None, dynamically adjusted
        to minority class frequency.

    Returns
    -------
    CalibratedCreditClassifier
        Calibrated model fitted on the validation set (subclass of CalibratedClassifierCV).
    """
    if method not in ("isotonic", "sigmoid"):
        msg = f"Unsupported calibration method: {method}. Must be 'isotonic' or 'sigmoid'."
        raise ValueError(msg)

    # Convert y_val if Polars series or array
    if isinstance(y_val, pl.Series):
        y_val_arr = y_val.to_numpy()
    elif hasattr(y_val, "to_numpy"):
        y_val_arr = y_val.to_numpy()
    else:
        y_val_arr = np.asarray(y_val)

    if HAS_FROZEN_ESTIMATOR:
        if cv is None:
            _, class_counts = np.unique(y_val_arr, return_counts=True)
            min_count = int(np.min(class_counts)) if len(class_counts) > 0 else 5
            cv_splits = max(2, min(5, min_count))
        else:
            cv_splits = cv

        calibrator = CalibratedCreditClassifier(
            estimator=FrozenEstimator(fitted_pipeline),
            method=method,
            cv=cv_splits,
        )
    else:
        # Fallback for scikit-learn < 1.6
        calibrator = CalibratedCreditClassifier(
            estimator=fitted_pipeline,
            method=method,
            cv="prefit",
        )

    calibrator.fit(X_val, y_val_arr)
    return calibrator


def evaluate_calibration(
    uncalibrated_model: Any,
    calibrated_model: Any,
    X_eval: Any,
    y_eval: Any,
    n_bins: int = 10,
) -> CalibrationComparison:
    """Evaluate and compare risk model calibration before and after post-processing.

    Parameters
    ----------
    uncalibrated_model : Any
        Trained uncalibrated pipeline or classifier.
    calibrated_model : Any
        CalibratedClassifierCV or calibrated pipeline.
    X_eval : Any
        Evaluation dataset features (e.g., test set).
    y_eval : Any
        Evaluation dataset ground truth targets.
    n_bins : int, default=10
        Number of calibration bins.

    Returns
    -------
    CalibrationComparison
        Dataclass containing detailed performance metrics before and after calibration.
    """
    if isinstance(y_eval, pl.Series):
        y_true = y_eval.to_numpy()
    elif hasattr(y_eval, "to_numpy"):
        y_true = y_eval.to_numpy()
    else:
        y_true = np.asarray(y_eval)

    # Predict probabilities for positive class (default_flag = 1)
    uncal_probs = uncalibrated_model.predict_proba(X_eval)[:, 1]
    cal_probs = calibrated_model.predict_proba(X_eval)[:, 1]

    metrics_uncal = compute_metrics(y_true, uncal_probs, n_bins=n_bins)
    metrics_cal = compute_metrics(y_true, cal_probs, n_bins=n_bins)

    brier_reduction = metrics_uncal.brier_score - metrics_cal.brier_score
    brier_red_pct = (
        (brier_reduction / metrics_uncal.brier_score) * 100.0
        if metrics_uncal.brier_score > 0
        else 0.0
    )

    ece_reduction = (
        metrics_uncal.expected_calibration_error - metrics_cal.expected_calibration_error
    )
    ece_red_pct = (
        (ece_reduction / metrics_uncal.expected_calibration_error) * 100.0
        if metrics_uncal.expected_calibration_error > 0
        else 0.0
    )

    roc_auc_delta = metrics_cal.roc_auc - metrics_uncal.roc_auc

    return CalibrationComparison(
        uncalibrated=metrics_uncal,
        calibrated=metrics_cal,
        brier_reduction=round(brier_reduction, 6),
        brier_reduction_pct=round(brier_red_pct, 2),
        ece_reduction=round(ece_reduction, 6),
        ece_reduction_pct=round(ece_red_pct, 2),
        roc_auc_delta=round(roc_auc_delta, 6),
    )
