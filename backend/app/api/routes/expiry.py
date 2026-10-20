from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from backend.app.schemas.expiry import (
    ExpiryRiskRequest,
    ExpiryRiskResponse,
)
from backend.app.services.expiry_service import expiry_risk_service

router = APIRouter(prefix="/api/expiry", tags=["Expiry Risk"])


@router.post("/predict", response_model=ExpiryRiskResponse)
def predict_expiry_risk(request: ExpiryRiskRequest):
    try:
        result = expiry_risk_service.predict(request)
        return ExpiryRiskResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Expiry-risk prediction failed.",
        ) from exc
