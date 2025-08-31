from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from .schemas import PredictRequest, PredictResponse, RegimeResponse, StatusResponse, UpdateDataResponse
from datetime import datetime
from ..models.predict import load_models, predict_garch, predict_har
from ..utils.logging import get_logger
from ..data.live_data import LiveDataFetcher

logger = get_logger(__name__)
router = APIRouter()

MODELS_CACHE = load_models()
DATA_FETCHER = LiveDataFetcher([
    "RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS", 
    "ICICIBANK.NS", "ITC.NS", "SBIN.NS"
])


def get_models():
    if not MODELS_CACHE:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return MODELS_CACHE

@router.get('/health')
def health():
    return {"status": "ok"}

@router.get('/model/status', response_model=StatusResponse)
def model_status():
    return StatusResponse(models=list(MODELS_CACHE.keys()), timestamp=datetime.utcnow().isoformat())

@router.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest, models=Depends(get_models)):
    try:
        garch_out = predict_garch(models, horizon=req.horizon)
        if 'ensemble' not in garch_out:
            logger.warning(f"No ensemble key in GARCH output: {garch_out.keys()}")
            raise ValueError("Invalid forecast format")
            
        forecast_value = float(garch_out['ensemble'])
        return PredictResponse(
            symbol=req.symbol, 
            horizon=req.horizon, 
            forecast=forecast_value,
            components=garch_out
        )
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/predict/har', response_model=PredictResponse)
def predict_har_endpoint(req: PredictRequest, models=Depends(get_models)):
    """Get predictions from HAR-RV model"""
    try:
        if 'har' not in models:
            raise HTTPException(status_code=404, detail="HAR-RV model not loaded")
            
        har_out = predict_har(models, horizon=req.horizon, symbol=req.symbol)
        if 'ensemble' not in har_out:
            logger.warning(f"No ensemble key in HAR output: {har_out.keys()}")
            raise ValueError("Invalid HAR forecast format")
            
        forecast_value = float(har_out['ensemble'])
        return PredictResponse(
            symbol=req.symbol, 
            horizon=req.horizon, 
            forecast=forecast_value,
            components=har_out
        )
    except Exception as e:
        logger.exception(f"HAR prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def update_model_data(symbol: str):
    """Background task to update model data with latest market data"""
    try:
        df = await DATA_FETCHER.get_ohlcv_data(f"{symbol}.NS")
        if df is not None and not df.empty:
            # Update model's historical data here
            if 'har' in MODELS_CACHE:
                MODELS_CACHE['har'].update_data(df)
    except Exception as e:
        logger.error(f"Failed to update model data: {e}")

@router.post('/predict/combined', response_model=PredictResponse)
async def predict_combined(req: PredictRequest, background_tasks: BackgroundTasks, models=Depends(get_models)):
    """Get combined predictions from GARCH and HAR models with equal weighting"""
    # Queue background update of model data
    background_tasks.add_task(update_model_data, req.symbol)
    try:
        results = {}
        forecast_values = []
        
        # Try GARCH prediction
        try:
            if 'garch' in models:
                garch_out = predict_garch(models, horizon=req.horizon)
                if 'ensemble' in garch_out:
                    forecast_values.append(float(garch_out['ensemble']))
                    results['garch'] = garch_out
        except Exception as e:
            logger.warning(f"GARCH prediction failed in combined: {e}")
            
        # Try HAR prediction
        try:
            if 'har' in models:
                har_out = predict_har(models, horizon=req.horizon, symbol=req.symbol)
                if 'ensemble' in har_out:
                    forecast_values.append(float(har_out['ensemble']))
                    results['har'] = har_out
        except Exception as e:
            logger.warning(f"HAR prediction failed in combined: {e}")
            
        if not forecast_values:
            raise HTTPException(status_code=500, detail="No predictions available")
            
        # Simple average of available forecasts
        combined_forecast = sum(forecast_values) / len(forecast_values)
        results['combined_forecast'] = combined_forecast
        results['model_count'] = len(forecast_values)
            
        return PredictResponse(
            symbol=req.symbol, 
            horizon=req.horizon, 
            forecast=combined_forecast,
            components=results
        )
    except Exception as e:
        logger.exception(f"Combined prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/regime', response_model=RegimeResponse)
def regime():
    # Placeholder: return normal regime with dummy confidence
    return RegimeResponse(regime='normal', confidence=0.5)

@router.post('/update/data', response_model=UpdateDataResponse)
def update_data():
    # Placeholder for async background task trigger
    return UpdateDataResponse(status='queued', records=0)
