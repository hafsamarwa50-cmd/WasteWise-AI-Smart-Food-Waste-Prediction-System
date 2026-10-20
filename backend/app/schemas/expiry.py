from pydantic import BaseModel, Field, ConfigDict


class ExpiryRiskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    category: str = Field(min_length=1, max_length=50)
    initial_quantity: int = Field(gt=0, le=1_000_000)
    days_to_expiry: int = Field(ge=0, le=365)
    avg_daily_sales_7d: float = Field(ge=0, le=1_000_000)
    unit_price: float = Field(ge=0, le=1_000_000)
    unit_cost: float = Field(ge=0, le=1_000_000)
    discount_pct: float = Field(ge=0, le=100)


class ExpiryRiskResponse(BaseModel):
    risk_label: str
    risk_probability: float
    model_version: str
    sale_blocked: bool
    message: str
