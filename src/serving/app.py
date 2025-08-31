from __future__ import annotations
from fastapi import FastAPI
from .routes import router
from .live import live_router
try:
    from .mlflow_routes import mlflow_router
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

app = FastAPI(title="MSAE-India Volatility API")
app.include_router(router)
app.include_router(live_router)

# Conditionally include MLflow routes if available
if HAS_MLFLOW:
    app.include_router(mlflow_router)

@app.get('/')
def root():
    endpoints = [
        "/health", 
        "/predict", 
        "/predict/har", 
        "/predict/combined", 
        "/model/status"
    ]
    if HAS_MLFLOW:
        endpoints.extend(["/mlflow/status", "/mlflow/runs", "/mlflow/runs/latest"])
    return {"message": "Volatility API", "endpoints": endpoints}