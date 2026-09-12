"""FastAPI application entrypoint for credit-policy-optimizer."""

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Credit Policy Optimizer API",
    version="0.1.0",
    description="API for credit policy simulation and risk decisioning",
)


class HealthResponse(BaseModel):
    status: str
    version: str


@app.get("/health", response_model=HealthResponse)
def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}
