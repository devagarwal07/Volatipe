from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
import mlflow
from mlflow.tracking.client import MlflowClient
from .schemas import StatusResponse

# Create router for MLflow metrics
mlflow_router = APIRouter(prefix="/mlflow")

# Schema for MLflow run info
class RunInfo(BaseModel):
    run_id: str
    experiment_id: str
    status: str
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    artifact_uri: Optional[str] = None
    
class RunMetric(BaseModel):
    key: str
    value: float
    timestamp: Optional[int] = None
    step: Optional[int] = None
    
class RunParameter(BaseModel):
    key: str
    value: str
    
class RunData(BaseModel):
    params: Dict[str, str] = {}
    metrics: Dict[str, float] = {}

class RunResponse(BaseModel):
    info: RunInfo
    data: RunData

class RunListResponse(BaseModel):
    runs: List[RunResponse]
    total: int = 0
    
# Create MLflow client
try:
    client = MlflowClient()
    mlflow.set_experiment('volatility_india')
except Exception as e:
    client = None
    print(f"Error initializing MLflow client: {e}")

def get_mlflow_client():
    if not client:
        raise HTTPException(status_code=503, detail="MLflow client not initialized")
    return client

@mlflow_router.get("/status", response_model=StatusResponse)
def mlflow_status(client=Depends(get_mlflow_client)):
    """Get MLflow connection status"""
    try:
        # Test connection by getting list of experiments
        experiments = client.search_experiments()
        expt_names = [exp.name for exp in experiments]
        return StatusResponse(
            models=expt_names,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MLflow error: {str(e)}")

@mlflow_router.get("/runs/latest", response_model=RunResponse)
def latest_run(client=Depends(get_mlflow_client)):
    """Get latest run data from MLflow"""
    try:
        # Get the experiment by name
        experiment = client.get_experiment_by_name("volatility_india")
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment 'volatility_india' not found")
            
        # Get latest run
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1
        )
        
        if not runs:
            raise HTTPException(status_code=404, detail="No runs found")
            
        run = runs[0]
        
        # Convert run info
        run_info = RunInfo(
            run_id=run.info.run_id,
            experiment_id=run.info.experiment_id,
            status=run.info.status,
            start_time=run.info.start_time,
            end_time=run.info.end_time,
            artifact_uri=run.info.artifact_uri
        )
        
        # Extract params and metrics
        run_data = RunData(
            params={k: v for k, v in run.data.params.items()},
            metrics={k: v for k, v in run.data.metrics.items()}
        )
        
        return RunResponse(
            info=run_info,
            data=run_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MLflow error: {str(e)}")

@mlflow_router.get("/runs", response_model=RunListResponse)
def list_runs(max_results: int = 10, client=Depends(get_mlflow_client)):
    """List runs from MLflow"""
    try:
        # Get the experiment by name
        experiment = client.get_experiment_by_name("volatility_india")
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment 'volatility_india' not found")
            
        # Get runs
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=max_results
        )
        
        results = []
        for run in runs:
            # Convert run info
            run_info = RunInfo(
                run_id=run.info.run_id,
                experiment_id=run.info.experiment_id,
                status=run.info.status,
                start_time=run.info.start_time,
                end_time=run.info.end_time,
                artifact_uri=run.info.artifact_uri
            )
            
            # Extract params and metrics
            run_data = RunData(
                params={k: v for k, v in run.data.params.items()},
                metrics={k: v for k, v in run.data.metrics.items()}
            )
            
            results.append(
                RunResponse(
                    info=run_info,
                    data=run_data
                )
            )
        
        return RunListResponse(
            runs=results,
            total=len(results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MLflow error: {str(e)}")
