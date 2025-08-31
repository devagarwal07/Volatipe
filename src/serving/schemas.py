from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional

class PredictRequest(BaseModel):
    horizon: int = Field(1, ge=1, le=30)
    symbol: str = Field("^NSEI")

class PredictResponse(BaseModel):
    symbol: str
    horizon: int
    forecast: float
    components: dict

class RegimeResponse(BaseModel):
    regime: str
    confidence: float

class StatusResponse(BaseModel):
    models: list
    timestamp: str

class UpdateDataResponse(BaseModel):
    status: str
    records: int
