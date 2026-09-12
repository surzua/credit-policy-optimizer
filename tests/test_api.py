"""Comprehensive unit and integration tests for the FastAPI credit decisioning API."""

import pytest
from fastapi.testclient import TestClient

from credit_policy_optimizer.api.app import app

client = TestClient(app)


def test_health_check() -> None:
    """Test the /health endpoint returns status 200 and model status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert data["model_loaded"] is True


def test_decision_prime_applicant() -> None:
    """Test decision endpoint for a prime applicant with excellent credit fundamentals."""
    payload = {
        "application_id": "APP-PRIME-001",
        "monthly_income": 8500.0,
        "debt_to_income": 0.20,
        "historical_delinquencies": 0,
        "revolving_utilization": 0.15,
        "loan_amount": 10000.0,
        "loan_term_months": 24,
        "interest_rate": 0.18,
        "cost_of_funds": 0.05,
        "lgd": 0.45,
    }
    response = client.post("/decision", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["application_id"] == "APP-PRIME-001"
    assert data["decision"] == "APROBADO"
    assert 0.0 <= data["calibrated_pd"] <= 0.15
    assert data["expected_value"] > 0
    assert data["breakeven_pd"] > 0
    assert data["net_gain_performing"] > 0
    assert data["net_loss_default"] > 0
    assert len(data["key_factors"]) >= 5

    # Verify presence of favorable indicators
    favorable_factors = [f for f in data["key_factors"] if f["impact"] == "FAVORABLE"]
    assert len(favorable_factors) >= 3


def test_decision_subprime_applicant() -> None:
    """Test decision endpoint for a distressed, high-risk applicant."""
    payload = {
        "application_id": "APP-RISK-999",
        "monthly_income": 1200.0,
        "debt_to_income": 0.85,
        "historical_delinquencies": 4,
        "revolving_utilization": 0.95,
        "loan_amount": 25000.0,
        "loan_term_months": 12,
        "interest_rate": 0.09,
        "cost_of_funds": 0.06,
        "lgd": 0.50,
    }
    response = client.post("/decision", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["application_id"] == "APP-RISK-999"
    assert data["decision"] == "RECHAZADO"
    assert data["expected_value"] < 0
    assert data["calibrated_pd"] > data["breakeven_pd"]

    # Verify presence of unfavorable risk factors
    unfavorable_factors = [f for f in data["key_factors"] if f["impact"] == "DESFAVORABLE"]
    assert len(unfavorable_factors) >= 3


def test_decision_custom_threshold_override() -> None:
    """Test that specifying policy_threshold overrides default breakeven logic."""
    base_payload = {
        "application_id": "APP-OVERRIDE-01",
        "monthly_income": 5000.0,
        "debt_to_income": 0.35,
        "historical_delinquencies": 0,
        "revolving_utilization": 0.40,
        "loan_amount": 12000.0,
        "loan_term_months": 12,
        "interest_rate": 0.18,
        "cost_of_funds": 0.05,
    }

    # Extremely strict threshold -> forces rejection
    strict_payload = {**base_payload, "policy_threshold": 0.00001}
    res_strict = client.post("/decision", json=strict_payload)
    assert res_strict.status_code == 200
    assert res_strict.json()["decision"] == "RECHAZADO"
    assert res_strict.json()["decision_threshold"] == 0.00001

    # Extremely lenient threshold -> forces approval
    lenient_payload = {**base_payload, "policy_threshold": 0.99999}
    res_lenient = client.post("/decision", json=lenient_payload)
    assert res_lenient.status_code == 200
    assert res_lenient.json()["decision"] == "APROBADO"
    assert res_lenient.json()["decision_threshold"] == 0.99999


@pytest.mark.parametrize(
    ("field", "invalid_val"),
    [
        ("monthly_income", -100.0),
        ("monthly_income", 0.0),
        ("debt_to_income", -0.1),
        ("debt_to_income", 2.5),
        ("historical_delinquencies", -1),
        ("revolving_utilization", -0.05),
        ("revolving_utilization", 2.1),
        ("loan_amount", 0.0),
        ("interest_rate", 1.5),
        ("cost_of_funds", -0.01),
        ("policy_threshold", 1.5),
    ],
)
def test_decision_schema_validation_errors(field: str, invalid_val: float | int) -> None:
    """Test that invalid values trigger Pydantic v2 validation errors (422 Unprocessable Entity)."""
    payload = {
        "application_id": "APP-INVALID",
        "monthly_income": 5000.0,
        "debt_to_income": 0.30,
        "historical_delinquencies": 0,
        "revolving_utilization": 0.30,
        "loan_amount": 10000.0,
        "loan_term_months": 12,
        "interest_rate": 0.18,
        "cost_of_funds": 0.05,
        field: invalid_val,
    }
    response = client.post("/decision", json=payload)
    assert response.status_code == 422


def test_decision_extra_fields_forbidden() -> None:
    """Test that extra unmapped fields are rejected by strict extra='forbid'."""
    payload = {
        "application_id": "APP-EXTRA",
        "monthly_income": 5000.0,
        "debt_to_income": 0.30,
        "historical_delinquencies": 0,
        "revolving_utilization": 0.30,
        "loan_amount": 10000.0,
        "loan_term_months": 12,
        "interest_rate": 0.18,
        "cost_of_funds": 0.05,
        "unauthorized_field": "hacker_val",
    }
    response = client.post("/decision", json=payload)
    assert response.status_code == 422


def test_simulate_portfolio_default_grid() -> None:
    """Test portfolio simulation across default threshold grid."""
    applications = [
        # 3 Prime
        {
            "application_id": f"APP-P-{i}",
            "monthly_income": 7000.0 + i * 500,
            "debt_to_income": 0.20 + i * 0.02,
            "historical_delinquencies": 0,
            "revolving_utilization": 0.20,
            "loan_amount": 8000.0,
            "loan_term_months": 12,
            "interest_rate": 0.18,
            "cost_of_funds": 0.05,
        }
        for i in range(3)
    ] + [
        # 3 Subprime
        {
            "application_id": f"APP-S-{i}",
            "monthly_income": 1800.0,
            "debt_to_income": 0.70 + i * 0.05,
            "historical_delinquencies": 2 + i,
            "revolving_utilization": 0.85,
            "loan_amount": 15000.0,
            "loan_term_months": 24,
            "interest_rate": 0.18,
            "cost_of_funds": 0.05,
        }
        for i in range(3)
    ]

    payload = {"applications": applications}
    response = client.post("/simulate-portfolio", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_applications"] == 6
    assert 0.0 <= data["optimal_threshold"] <= 1.0
    assert "max_expected_pnl" in data
    assert "baseline_05_pnl" in data
    assert "incremental_pnl_vs_baseline" in data
    assert len(data["evaluations"]) > 5

    # Check monotonicity of approval rate
    evals = data["evaluations"]
    approval_rates = [e["approval_rate"] for e in evals]
    for i in range(len(approval_rates) - 1):
        assert approval_rates[i] <= approval_rates[i + 1]


def test_simulate_portfolio_custom_thresholds() -> None:
    """Test portfolio simulation using user-specified candidate thresholds."""
    applications = [
        {
            "application_id": f"APP-CUSTOM-{i}",
            "monthly_income": 4000.0 + i * 1000,
            "debt_to_income": 0.30,
            "historical_delinquencies": 0,
            "revolving_utilization": 0.35,
            "loan_amount": 10000.0,
            "loan_term_months": 12,
            "interest_rate": 0.18,
            "cost_of_funds": 0.05,
        }
        for i in range(4)
    ]
    target_thresholds = [0.05, 0.12, 0.25]
    payload = {
        "applications": applications,
        "thresholds": target_thresholds,
    }
    response = client.post("/simulate-portfolio", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert len(data["evaluations"]) == 3
    returned_thresholds = [e["threshold"] for e in data["evaluations"]]
    assert returned_thresholds == target_thresholds


def test_simulate_portfolio_empty_cohort_rejected() -> None:
    """Test that an empty application cohort is rejected."""
    payload = {"applications": []}
    response = client.post("/simulate-portfolio", json=payload)
    assert response.status_code == 422  # pydantic min_length=1


def test_get_model_on_missing_path(tmp_path: pytest.TempPathFactory) -> None:
    """Test that get_model dynamically trains and caches when artifact is absent."""
    from pathlib import Path

    from credit_policy_optimizer.api.app import get_model

    temp_model_path = Path(str(tmp_path)) / "sub" / "auto_trained.joblib"
    assert not temp_model_path.exists()
    model = get_model(temp_model_path)
    assert model is not None
    assert temp_model_path.exists()


def test_app_lifespan() -> None:
    """Test the application startup lifespan handler."""
    import asyncio

    from credit_policy_optimizer.api.app import lifespan

    async def _run() -> None:
        async with lifespan(app):
            pass

    asyncio.run(_run())
