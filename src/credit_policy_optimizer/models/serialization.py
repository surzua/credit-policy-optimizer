"""Serialization and loading utilities for trained and calibrated credit risk pipelines."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib


def save_pipeline_artifact(
    model: Any,
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Serialize a trained/calibrated model pipeline and its metadata to disk.

    Parameters
    ----------
    model : Any
        Trained Scikit-Learn Pipeline or CalibratedClassifierCV.
    output_path : str | Path
        Target destination path (e.g. models/credit_pipeline_calibrated.joblib).
    metadata : dict[str, Any] | None
        Optional dictionary containing training metrics, schema details, or git commits.

    Returns
    -------
    Path
        Resolved Path to the saved artifact.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = metadata.copy() if metadata is not None else {}
    meta.setdefault("saved_at", datetime.now(UTC).isoformat())
    meta.setdefault("model_class", model.__class__.__name__)

    payload = {
        "model": model,
        "metadata": meta,
    }

    joblib.dump(payload, path, compress=3)
    return path


def load_pipeline_artifact(input_path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load a serialized model artifact and return the model along with its metadata.

    Parameters
    ----------
    input_path : str | Path
        Path to the saved .joblib artifact.

    Returns
    -------
    tuple[Any, dict[str, Any]]
        A tuple of (model, metadata_dict).
    """
    path = Path(input_path)
    if not path.exists():
        msg = f"Model artifact not found at: {path.resolve()}"
        raise FileNotFoundError(msg)

    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload.get("metadata", {})

    # Fallback if raw model was saved
    return payload, {}


def load_pipeline(input_path: str | Path) -> Any:
    """Load and return the executable model pipeline directly.

    Parameters
    ----------
    input_path : str | Path
        Path to the saved .joblib artifact.

    Returns
    -------
    Any
        The loaded pipeline or classifier.
    """
    model, _ = load_pipeline_artifact(input_path)
    return model
