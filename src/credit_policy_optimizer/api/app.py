"""FastAPI delivery and decisioning application for credit policy optimization."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import polars as pl
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from credit_policy_optimizer.decision.economics import (
    CreditPolicyOptimizer,
    EconomicParameters,
    PolicyEvaluation,
    compute_breakeven_pd,
    compute_loan_net_gain,
    compute_loan_net_loss,
)
from credit_policy_optimizer.models.serialization import load_pipeline
from credit_policy_optimizer.models.train import run_training_pipeline

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/credit_pipeline_calibrated.joblib")

# Global singleton container for model caching
_MODEL_PIPELINE: Any = None


def get_model(model_path: Path = DEFAULT_MODEL_PATH) -> Any:
    """Retrieve or lazily initialize the calibrated credit risk pipeline."""
    global _MODEL_PIPELINE
    if model_path == DEFAULT_MODEL_PATH and _MODEL_PIPELINE is not None:
        return _MODEL_PIPELINE

    if model_path.exists():
        logger.info("Loading calibrated model artifact from %s", model_path)
        loaded = load_pipeline(model_path)
        if model_path == DEFAULT_MODEL_PATH:
            _MODEL_PIPELINE = loaded
        return loaded
    else:
        logger.warning(
            "Model artifact not found at %s. Initializing and calibrating pipeline on the fly...",
            model_path,
        )
        _, calibrated_model, _ = run_training_pipeline(
            data_path=None,
            output_model_path=model_path,
            n_samples=1000,
            random_seed=42,
        )
        if model_path == DEFAULT_MODEL_PATH:
            _MODEL_PIPELINE = calibrated_model
        return calibrated_model


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Pre-warm model artifact during application startup."""
    try:
        get_model()
    except Exception as exc:
        logger.error("Failed to load model pipeline during startup: %s", exc)
    yield


app = FastAPI(
    title="Credit Policy Optimizer API",
    version="0.1.0",
    description=(
        "Production-grade decisioning and portfolio simulation API for credit risk "
        "and optimal credit policy evaluation."
    ),
    lifespan=lifespan,
)


# ==============================================================================
# Pydantic v2 Request & Response Schemas
# ==============================================================================


class ApplicantRequest(BaseModel):
    """Schema representing an applicant evaluated for credit policy decisioning."""

    model_config = ConfigDict(extra="forbid", strict=True)

    application_id: str = Field(..., description="Unique credit application identifier")
    monthly_income: float = Field(..., gt=0.0, description="Monthly gross income in USD")
    debt_to_income: float = Field(..., ge=0.0, le=2.0, description="Debt-to-income (DTI) ratio")
    historical_delinquencies: int = Field(
        ..., ge=0, description="Count of historical 30+ day delinquency events"
    )
    revolving_utilization: float = Field(
        ..., ge=0.0, le=2.0, description="Revolving line utilization ratio"
    )
    loan_amount: float = Field(..., gt=0.0, description="Requested principal amount in USD")
    loan_term_months: int = Field(default=12, gt=0, description="Loan repayment tenure in months")
    interest_rate: float = Field(
        default=0.18, gt=0.0, le=1.0, description="Nominal annual active interest rate"
    )
    cost_of_funds: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Annual institutional cost of funds"
    )
    lgd: float = Field(
        default=0.45, ge=0.0, le=1.0, description="Loss Given Default (LGD) expectation"
    )
    policy_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional custom PD policy cutoff. Defaults to financial breakeven PD.",
    )


class DecisionFactor(BaseModel):
    """Interpretable risk and business factor influencing the credit decision."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(..., description="Risk or economic indicator name")
    value: float | str | int = Field(..., description="Reported or derived value")
    impact: Literal["FAVORABLE", "DESFAVORABLE", "NEUTRO"] = Field(
        ..., description="Directional impact of factor on the approval decision"
    )
    description: str = Field(..., description="Human-readable business rationale")


class DecisionResponse(BaseModel):
    """Output schema for the single applicant credit decision endpoint."""

    model_config = ConfigDict(extra="forbid", strict=True)

    application_id: str = Field(..., description="Application identifier")
    decision: Literal["APROBADO", "RECHAZADO"] = Field(
        ..., description="Final policy decision outcome"
    )
    calibrated_pd: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated Probability of Default (PD)"
    )
    expected_value: float = Field(
        ..., description="Net Expected Economic Value (EV) of the loan in USD"
    )
    breakeven_pd: float = Field(
        ..., ge=0.0, le=1.0, description="Financial indifference default probability cutoff p*"
    )
    decision_threshold: float = Field(
        ..., ge=0.0, le=1.0, description="Effective policy cutoff applied"
    )
    net_gain_performing: float = Field(
        ..., description="Contractual revenue if loan fully performs in USD"
    )
    net_loss_default: float = Field(
        ..., description="Economic loss incurred if borrower defaults in USD"
    )
    key_factors: list[DecisionFactor] = Field(
        ..., description="Core risk drivers influencing the decision"
    )


class PortfolioSimulationRequest(BaseModel):
    """Schema for batch portfolio simulation across multiple policy thresholds."""

    model_config = ConfigDict(extra="forbid", strict=True)

    applications: list[ApplicantRequest] = Field(
        ..., min_length=1, description="List of applications to evaluate"
    )
    thresholds: list[float] | None = Field(
        default=None,
        description="Candidate cutoff thresholds. If not provided, a default grid is evaluated.",
    )
    default_lgd: float = Field(default=0.45, ge=0.0, le=1.0, description="Default LGD assumption")
    default_cost_of_funds: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Default cost of funds hurdle"
    )
    default_interest_rate: float = Field(
        default=0.18, gt=0.0, le=1.0, description="Default interest rate"
    )


class ThresholdEvaluation(BaseModel):
    """Financial and portfolio outcomes under a specific policy cutoff."""

    model_config = ConfigDict(extra="forbid", strict=True)

    threshold: float = Field(..., ge=0.0, le=1.0)
    total_applications: int = Field(..., ge=0)
    approved_count: int = Field(..., ge=0)
    approval_rate: float = Field(..., ge=0.0, le=1.0)
    total_exposure: float = Field(..., ge=0.0)
    gross_margin: float = Field(...)
    expected_loss: float = Field(..., ge=0.0)
    expected_pnl: float = Field(...)
    expected_default_rate: float = Field(..., ge=0.0, le=1.0)
    return_on_exposure: float = Field(...)


class PortfolioSimulationResponse(BaseModel):
    """Response payload for portfolio P&L simulation across thresholds."""

    model_config = ConfigDict(extra="forbid", strict=True)

    total_applications: int = Field(..., ge=0, description="Total applications in cohort")
    optimal_threshold: float = Field(
        ..., ge=0.0, le=1.0, description="Policy cutoff that maximizes expected portfolio P&L"
    )
    max_expected_pnl: float = Field(
        ..., description="Maximum expected P&L achieved under optimal threshold"
    )
    baseline_05_pnl: float = Field(
        ..., description="Expected P&L under naive binary classification cutoff p=0.5"
    )
    incremental_pnl_vs_baseline: float = Field(
        ..., description="Dollar gain of optimal policy vs naive 0.5 cutoff"
    )
    evaluations: list[ThresholdEvaluation] = Field(
        ..., description="Detailed economics per simulated threshold"
    )


class HealthResponse(BaseModel):
    """Health check response schema."""

    model_config = ConfigDict(extra="forbid", strict=True)

    status: str = Field(default="ok")
    version: str = Field(default="0.1.0")
    model_loaded: bool = Field(default=True)


# ==============================================================================
# Business Decision Factor Generator
# ==============================================================================


def _extract_decision_factors(
    req: ApplicantRequest,
    pd: float,
    ev: float,
    breakeven_pd: float,
) -> list[DecisionFactor]:
    """Generate interpretable risk drivers for the applicant."""
    factors: list[DecisionFactor] = []

    # 1. Debt-to-Income (DTI)
    dti = req.debt_to_income
    if dti <= 0.30:
        factors.append(
            DecisionFactor(
                name="Debt to Income (DTI)",
                value=round(dti, 4),
                impact="FAVORABLE",
                description=f"DTI saludable ({dti:.1%}), amplia capacidad de servicio de deuda.",
            )
        )
    elif dti > 0.45:
        factors.append(
            DecisionFactor(
                name="Debt to Income (DTI)",
                value=round(dti, 4),
                impact="DESFAVORABLE",
                description=f"DTI elevado ({dti:.1%}), alto compromiso sobre ingresos disponibles.",
            )
        )
    else:
        factors.append(
            DecisionFactor(
                name="Debt to Income (DTI)",
                value=round(dti, 4),
                impact="NEUTRO",
                description=f"DTI moderado ({dti:.1%}) dentro de rangos aceptables.",
            )
        )

    # 2. Historical Delinquencies
    delinq = req.historical_delinquencies
    if delinq == 0:
        factors.append(
            DecisionFactor(
                name="Historial de Morosidades",
                value=delinq,
                impact="FAVORABLE",
                description="Sin morosidades históricas de 30+ días registradas.",
            )
        )
    else:
        factors.append(
            DecisionFactor(
                name="Historial de Morosidades",
                value=delinq,
                impact="DESFAVORABLE",
                description=f"{delinq} eventos de morosidad previa registrados.",
            )
        )

    # 3. Revolving Utilization
    util = req.revolving_utilization
    if util <= 0.30:
        factors.append(
            DecisionFactor(
                name="Uso de Líneas Rotativas",
                value=round(util, 4),
                impact="FAVORABLE",
                description=f"Uso prudente de líneas rotativas ({util:.1%}).",
            )
        )
    elif util > 0.60:
        factors.append(
            DecisionFactor(
                name="Uso de Líneas Rotativas",
                value=round(util, 4),
                impact="DESFAVORABLE",
                description=(
                    f"Uso intensivo de líneas rotativas ({util:.1%}), señal de apalancamiento."
                ),
            )
        )
    else:
        factors.append(
            DecisionFactor(
                name="Uso de Líneas Rotativas",
                value=round(util, 4),
                impact="NEUTRO",
                description=f"Uso moderado de líneas rotativas ({util:.1%}).",
            )
        )

    # 4. Loan to Monthly Income Ratio
    lti = req.loan_amount / req.monthly_income
    if lti <= 3.0:
        factors.append(
            DecisionFactor(
                name="Relación Monto / Ingreso Mensual",
                value=round(lti, 2),
                impact="FAVORABLE",
                description=f"Monto solicitado representa {lti:.1f} meses de ingreso bruto.",
            )
        )
    elif lti > 6.0:
        factors.append(
            DecisionFactor(
                name="Relación Monto / Ingreso Mensual",
                value=round(lti, 2),
                impact="DESFAVORABLE",
                description=f"Monto solicitado elevado ({lti:.1f}x ingreso mensual).",
            )
        )
    else:
        factors.append(
            DecisionFactor(
                name="Relación Monto / Ingreso Mensual",
                value=round(lti, 2),
                impact="NEUTRO",
                description=f"Monto equivalente a {lti:.1f} meses de ingreso bruto.",
            )
        )

    # 5. Financial Net Spread
    net_spread = req.interest_rate - req.cost_of_funds
    if net_spread >= 0.08:
        factors.append(
            DecisionFactor(
                name="Spread Financiero Neto",
                value=round(net_spread, 4),
                impact="FAVORABLE",
                description=f"Spread financiero positivo y atractivo ({net_spread:.1%}).",
            )
        )
    elif net_spread <= 0.0:
        factors.append(
            DecisionFactor(
                name="Spread Financiero Neto",
                value=round(net_spread, 4),
                impact="DESFAVORABLE",
                description=f"Margen financiero nulo o negativo ({net_spread:.1%}).",
            )
        )
    else:
        factors.append(
            DecisionFactor(
                name="Spread Financiero Neto",
                value=round(net_spread, 4),
                impact="NEUTRO",
                description=f"Spread financiero estándar ({net_spread:.1%}).",
            )
        )

    # 6. Economic Value (EV) vs Breakeven
    if ev > 0:
        factors.append(
            DecisionFactor(
                name="Valor Esperado (EV)",
                value=round(ev, 2),
                impact="FAVORABLE",
                description=(
                    f"Retorno esperado positivo (+${ev:,.2f} USD). "
                    f"PD calibrada ({pd:.2%}) inferior a breakeven ({breakeven_pd:.2%})."
                ),
            )
        )
    else:
        factors.append(
            DecisionFactor(
                name="Valor Esperado (EV)",
                value=round(ev, 2),
                impact="DESFAVORABLE",
                description=(
                    f"Riesgo de pérdida esperada (-${abs(ev):,.2f} USD). "
                    f"PD calibrada ({pd:.2%}) supera breakeven ({breakeven_pd:.2%})."
                ),
            )
        )

    return factors


# ==============================================================================
# API Endpoints
# ==============================================================================


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check(model: Annotated[Any, Depends(get_model)]) -> HealthResponse:
    """Check API operational health and model readiness."""
    return HealthResponse(status="ok", version="0.1.0", model_loaded=model is not None)


@app.post("/decision", response_model=DecisionResponse, tags=["Decisioning"])
def evaluate_decision(
    request: ApplicantRequest,
    model: Annotated[Any, Depends(get_model)],
) -> DecisionResponse:
    """Evaluate an individual credit applicant and return optimal policy decision.

    Calculates:
    - Calibrated Probability of Default (PD) via trained pipeline
    - Net Expected Monetary Value (EV) = (1 - PD) * Gain - PD * Loss
    - Financial Breakeven PD p*
    - Binary decision outcome: `APROBADO` or `RECHAZADO`
    - Key interpretability decision factors and risk drivers
    """
    # 1. Align features for model inference
    input_data = {
        "monthly_income": request.monthly_income,
        "debt_to_income": request.debt_to_income,
        "historical_delinquencies": request.historical_delinquencies,
        "revolving_utilization": request.revolving_utilization,
        "loan_amount": request.loan_amount,
        "loan_term_months": request.loan_term_months,
        "interest_rate": request.interest_rate,
        "cost_of_funds": request.cost_of_funds,
    }
    df = pl.DataFrame([input_data])

    try:
        proba = model.predict_proba(df)
        calibrated_pd = float(proba[0, 1])
    except Exception as exc:
        logger.error("Inference failed for application %s: %s", request.application_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing risk prediction: {exc}",
        ) from exc

    # 2. Financial Economics Computations
    net_gain = float(
        compute_loan_net_gain(
            amount=request.loan_amount,
            interest_rate=request.interest_rate,
            cost_of_funds=request.cost_of_funds,
            term_months=request.loan_term_months,
        )
    )
    net_loss = float(
        compute_loan_net_loss(
            amount=request.loan_amount,
            lgd=request.lgd,
            cost_of_funds=request.cost_of_funds,
            term_months=request.loan_term_months,
        )
    )
    ev = ((1.0 - calibrated_pd) * net_gain) - (calibrated_pd * net_loss)

    breakeven_pd = compute_breakeven_pd(
        interest_rate=request.interest_rate,
        cost_of_funds=request.cost_of_funds,
        lgd=request.lgd,
        term_months=request.loan_term_months,
    )

    # 3. Decision Evaluation
    effective_threshold = (
        request.policy_threshold if request.policy_threshold is not None else breakeven_pd
    )

    if request.policy_threshold is not None:
        # Caller explicitly specified cutoff
        is_approved = calibrated_pd <= request.policy_threshold
    else:
        # Default policy: approve if economically accretive (EV > 0, equiv. PD <= breakeven_pd)
        is_approved = ev > 0 and calibrated_pd <= breakeven_pd

    decision_status: Literal["APROBADO", "RECHAZADO"] = "APROBADO" if is_approved else "RECHAZADO"

    # 4. Generate Interpretable Factors
    factors = _extract_decision_factors(
        req=request,
        pd=calibrated_pd,
        ev=ev,
        breakeven_pd=breakeven_pd,
    )

    return DecisionResponse(
        application_id=request.application_id,
        decision=decision_status,
        calibrated_pd=round(calibrated_pd, 6),
        expected_value=round(ev, 2),
        breakeven_pd=round(breakeven_pd, 6),
        decision_threshold=round(effective_threshold, 6),
        net_gain_performing=round(net_gain, 2),
        net_loss_default=round(net_loss, 2),
        key_factors=factors,
    )


@app.post("/simulate-portfolio", response_model=PortfolioSimulationResponse, tags=["Simulation"])
def simulate_portfolio(
    request: PortfolioSimulationRequest,
    model: Annotated[Any, Depends(get_model)],
) -> PortfolioSimulationResponse:
    """Simulate the financial and P&L impact across multiple policy thresholds.

    Evaluates:
    - Expected P&L curve across candidate thresholds
    - Approval rate, default rate, and return on exposure (ROE)
    - Optimal cutoff threshold p* maximizing portfolio profitability
    - Comparison vs naive baseline cutoff (p = 0.5)
    """
    total_apps = len(request.applications)
    if total_apps == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Applications cohort cannot be empty.",
        )

    # 1. Batch build features DataFrame for model inference
    app_rows = [
        {
            "monthly_income": app.monthly_income,
            "debt_to_income": app.debt_to_income,
            "historical_delinquencies": app.historical_delinquencies,
            "revolving_utilization": app.revolving_utilization,
            "loan_amount": app.loan_amount,
            "loan_term_months": app.loan_term_months,
            "interest_rate": app.interest_rate,
            "cost_of_funds": app.cost_of_funds,
            "lgd": app.lgd,
        }
        for app in request.applications
    ]
    df_features = pl.DataFrame(app_rows)

    try:
        probabilities = model.predict_proba(df_features)[:, 1]
    except Exception as exc:
        logger.error("Batch inference failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing batch inference: {exc}",
        ) from exc

    # 2. Enrich DataFrame with predicted PD
    df_portfolio = df_features.with_columns(pl.Series("pd", probabilities))

    # 3. Initialize CreditPolicyOptimizer
    economic_params = EconomicParameters(
        default_lgd=request.default_lgd,
        default_cost_of_funds=request.default_cost_of_funds,
        default_interest_rate=request.default_interest_rate,
    )
    optimizer = CreditPolicyOptimizer(portfolio=df_portfolio, params=economic_params)

    # 4. Determine thresholds to evaluate
    if request.thresholds is not None and len(request.thresholds) > 0:
        eval_thresholds = sorted(list(set(request.thresholds)))
    else:
        eval_thresholds = [
            0.01,
            0.02,
            0.05,
            0.08,
            0.10,
            0.12,
            0.15,
            0.20,
            0.25,
            0.30,
            0.40,
            0.50,
            1.0,
        ]

    # 5. Evaluate policy across candidate thresholds
    evaluations: list[ThresholdEvaluation] = []
    for t in eval_thresholds:
        ev_res: PolicyEvaluation = optimizer.evaluate_policy(t)
        evaluations.append(
            ThresholdEvaluation(
                threshold=round(ev_res.threshold, 4),
                total_applications=ev_res.total_applications,
                approved_count=ev_res.approved_count,
                approval_rate=round(ev_res.approval_rate, 4),
                total_exposure=round(ev_res.total_exposure, 2),
                gross_margin=round(ev_res.gross_margin, 2),
                expected_loss=round(ev_res.expected_loss, 2),
                expected_pnl=round(ev_res.expected_pnl, 2),
                expected_default_rate=round(ev_res.expected_default_rate, 4),
                return_on_exposure=round(ev_res.return_on_exposure, 4),
            )
        )

    # 6. Global optimization search
    opt_result = optimizer.optimize_threshold(metric="expected_pnl")

    return PortfolioSimulationResponse(
        total_applications=total_apps,
        optimal_threshold=round(opt_result.optimal_threshold, 4),
        max_expected_pnl=round(opt_result.optimal_policy.expected_pnl, 2),
        baseline_05_pnl=round(opt_result.baseline_policy_05.expected_pnl, 2),
        incremental_pnl_vs_baseline=round(opt_result.incremental_pnl_vs_baseline, 2),
        evaluations=evaluations,
    )
