from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import json

from .database import Database
from .model_monitor import ModelMonitor
from .config_manager import ConfigManager
from .logging_config import logger

# Initialize components
app = FastAPI(title="Customer Segmentation API")
db = Database()
monitor = ModelMonitor()
config_manager = ConfigManager()
api_config = config_manager.get_config('api')

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key security
api_key_header = APIKeyHeader(name=api_config.get('api_key_header', 'X-API-Key'))

async def get_api_key(api_key: str = Header(...)):
    """Validate API key."""
    # In a real application, validate against a database of API keys
    if api_key != "your-secret-api-key":  # Replace with proper validation
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Customer Segmentation API"}

@app.get("/models", dependencies=[Depends(get_api_key)])
async def get_models():
    """Get available models."""
    try:
        models = []
        models_dir = Path("models")
        
        if (models_dir / "kmeans.joblib").exists():
            models.append("kmeans")
        if (models_dir / "dbscan.joblib").exists():
            models.append("dbscan")
            
        return {"models": models}
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/{model_type}", dependencies=[Depends(get_api_key)])
async def predict(model_type: str, data: List[dict]):
    """Make predictions using the specified model."""
    try:
        # Load models
        models_dir = Path("models")
        model = joblib.load(models_dir / f"{model_type.lower()}.joblib")
        scaler = joblib.load(models_dir / "scaler.joblib")
        pca = joblib.load(models_dir / "pca.joblib")
        features = joblib.load(models_dir / "features.joblib")
        
        # Prepare data
        df = pd.DataFrame(data)
        X = df[features]
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)
        
        # Make predictions
        predictions = model.predict(X_pca)
        
        # Format response
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "customer_id": df.index[i],
                "segment": int(pred),
                "features": X.iloc[i].to_dict()
            })
        
        return {"predictions": results}
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/status/{model_type}", dependencies=[Depends(get_api_key)])
async def get_model_status(model_type: str):
    """Get model status and performance metrics."""
    try:
        latest_model = db.get_latest_model_version(model_type.lower())
        if not latest_model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        metrics = json.loads(latest_model[3])
        return {
            "model_type": model_type,
            "version": latest_model[1],
            "created_at": latest_model[4],
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/monitor/drift", dependencies=[Depends(get_api_key)])
async def check_drift(model_type: str, data: List[dict]):
    """Check for data drift."""
    try:
        # Load reference data
        reference_data = pd.read_csv("data/marketing_campaign.csv")
        current_data = pd.DataFrame(data)
        
        # Check model health
        results = monitor.check_model_health(model_type, current_data, reference_data)
        
        return {
            "model_type": model_type,
            "drift_detected": results["significant_drift"],
            "drift_metrics": results["drift_metrics"],
            "performance_metrics": results["performance_metrics"]
        }
    except Exception as e:
        logger.error(f"Error checking drift: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/segments/{model_type}", dependencies=[Depends(get_api_key)])
async def get_segments(model_type: str, limit: Optional[int] = 100):
    """Get customer segments for the specified model."""
    try:
        latest_model = db.get_latest_model_version(model_type.lower())
        if not latest_model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Query segments from database
        segments = db.get_segments(latest_model[0], limit)
        return {"segments": segments}
    except Exception as e:
        logger.error(f"Error getting segments: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audit/logs", dependencies=[Depends(get_api_key)])
async def get_audit_logs(limit: Optional[int] = 100):
    """Get audit logs."""
    try:
        logs = db.get_audit_logs(limit)
        return {"logs": logs}
    except Exception as e:
        logger.error(f"Error getting audit logs: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 8000),
        debug=api_config.get('debug', False)
    ) 