"""Pydantic v2 validation schemas for credit applications and simulation outputs."""

from pydantic import BaseModel, ConfigDict, Field


class CreditApplication(BaseModel):
    """Schema representing an individual credit application."""

    model_config = ConfigDict(extra="forbid", strict=True)

    application_id: str = Field(..., description="Unique application identifier")
    monthly_income: float = Field(..., gt=0, description="Monthly gross income in USD")
    debt_to_income: float = Field(..., ge=0.0, le=2.0, description="Debt-to-income (DTI) ratio")
    historical_delinquencies: int = Field(
        ..., ge=0, description="Count of historical 30+ day delinquency events"
    )
    revolving_utilization: float = Field(
        ..., ge=0.0, le=2.0, description="Revolving line utilization ratio"
    )
    loan_amount: float = Field(..., gt=0, description="Requested principal amount in USD")
    loan_term_months: int = Field(..., gt=0, description="Loan repayment tenure in months")
    interest_rate: float = Field(
        ..., gt=0.0, le=1.0, description="Nominal annual active interest rate"
    )
    cost_of_funds: float = Field(
        ..., gt=0.0, le=1.0, description="Annual cost of funds / funding hurdle rate"
    )


class SimulatedCreditApplication(CreditApplication):
    """Schema representing an application enriched with simulation outcomes."""

    pd: float = Field(..., ge=0.0, le=1.0, description="Calibrated Probability of Default (PD)")
    default_flag: int = Field(
        ..., ge=0, le=1, description="Binary simulated default outcome (1=default, 0=performing)"
    )


class SimulationResult(BaseModel):
    """Aggregate metrics and summary of a portfolio simulation run."""

    model_config = ConfigDict(extra="forbid", strict=True)

    total_applications: int = Field(..., ge=0, description="Total number of simulated applications")
    total_exposure: float = Field(..., ge=0.0, description="Total portfolio loan volume in USD")
    default_count: int = Field(..., ge=0, description="Total count of simulated defaults")
    default_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Portfolio-level realized default rate"
    )
    avg_pd: float = Field(..., ge=0.0, le=1.0, description="Average probability of default")
    avg_loan_amount: float = Field(..., ge=0.0, description="Average loan amount in USD")
    avg_monthly_income: float = Field(..., ge=0.0, description="Average monthly income in USD")
    avg_interest_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Average nominal active interest rate"
    )
    avg_cost_of_funds: float = Field(..., ge=0.0, le=1.0, description="Average cost of funds")
    seed: int | None = Field(default=None, description="Random seed used for reproducibility")
