#!/usr/bin/env python3
"""
FastAPI Model Serving Application
Serves the trained wine quality prediction model via REST API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Wine Quality Prediction API",
    description="ML model serving for wine quality prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and metadata
model = None
scaler = None
model_info = None
feature_names = None

class WineFeatures(BaseModel):
    """Input schema for wine features."""
    alcohol: float = Field(..., ge=0, le=20, description="Alcohol content (%)")
    malic_acid: float = Field(..., ge=0, le=10, description="Malic acid (g/L)")
    ash: float = Field(..., ge=0, le=5, description="Ash content (g/L)")
    alcalinity_of_ash: float = Field(..., ge=0, le=50, description="Alcalinity of ash")
    magnesium: float = Field(..., ge=0, le=200, description="Magnesium (mg/L)")
    total_phenols: float = Field(..., ge=0, le=5, description="Total phenols")
    flavanoids: float = Field(..., ge=0, le=5, description="Flavanoids")
    nonflavanoid_phenols: float = Field(..., ge=0, le=1, description="Nonflavanoid phenols")
    proanthocyanins: float = Field(..., ge=0, le=5, description="Proanthocyanins")
    color_intensity: float = Field(..., ge=0, le=20, description="Color intensity")
    hue: float = Field(..., ge=0, le=2, description="Hue")
    od280_od315: float = Field(..., ge=0, le=5, description="OD280/OD315 of diluted wines")
    proline: float = Field(..., ge=0, le=2000, description="Proline (mg/L)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "alcohol": 13.2,
                "malic_acid": 1.78,
                "ash": 2.14,
                "alcalinity_of_ash": 11.2,
                "magnesium": 100,
                "total_phenols": 2.65,
                "flavanoids": 2.76,
                "nonflavanoid_phenols": 0.26,
                "proanthocyanins": 1.28,
                "color_intensity": 4.38,
                "hue": 1.05,
                "od280_od315": 3.4,
                "proline": 1050
            }
        }
    
    class Config:
        json_schema_extra = {
            "example": {
                "alcohol": 13.2,
                "malic_acid": 1.78,
                "ash": 2.14,
                "alcalinity_of_ash": 11.2,
                "magnesium": 100,
                "total_phenols": 2.65,
                "flavanoids": 2.76,
                "nonflavanoid_phenols": 0.26,
                "proanthocyanins": 1.28,
                "color_intensity": 4.38,
                "hue": 1.05,
                "od280_od315_of_diluted_wines": 3.4,
                "proline": 1050
            }
        }

class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    quality_class: int
    confidence: float
    quality_label: str
    probabilities: Dict[str, float]
    timestamp: str

class ModelInfo(BaseModel):
    """Response schema for model information."""
    model_type: str
    metrics: Dict[str, float]
    training_date: str
    feature_names: List[str]
    version: str = "1.0.0"

def load_model():
    """Load the trained model and preprocessing artifacts."""
    global model, scaler, model_info, feature_names
    
    try:
        # Load model info
        with open('models/artifacts/model_info.json', 'r') as f:
            model_info = json.load(f)
        
        # Load model based on type
        if 'xgboost' in model_info['model_type'].lower():
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model('models/artifacts/best_model.xgb')
        else:
            model = joblib.load('models/artifacts/best_model.pkl')
        
        # Load scaler
        scaler = joblib.load('data/processed/scaler.pkl')
        
        # Get feature names
        with open('data/processed/metadata.json', 'r') as f:
            metadata = json.load(f)
            feature_names = metadata['feature_names']
        
        logger.info(f"✅ Model loaded successfully: {model_info['model_type']}")
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise

def prepare_features(wine_data: WineFeatures) -> pd.DataFrame:
    """Prepare input features for prediction."""
    # Convert to dataframe
    data_dict = wine_data.dict()
    # Fix the feature name with forward slash
    data_dict['od280/od315_of_diluted_wines'] = data_dict.pop('od280_od315')
    input_df = pd.DataFrame([data_dict])
    
    # Add engineered features (must match training)
    input_df['alcohol_squared'] = input_df['alcohol'] ** 2
    input_df['phenols_flavanoids_ratio'] = input_df['total_phenols'] / (input_df['flavanoids'] + 1e-6)
    input_df['color_hue_interaction'] = input_df['color_intensity'] * input_df['hue']
    input_df['proline_log'] = np.log1p(input_df['proline'])
    input_df['magnesium_scaled'] = (input_df['magnesium'] - 100) / 30  # Approximate scaling
    
    # Ensure all features are present and in correct order
    return input_df[feature_names]

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Wine Quality Prediction API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model."""
    if model_info is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_type=model_info['model_type'],
        metrics=model_info['metrics'],
        training_date=model_info['training_date'],
        feature_names=feature_names
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(wine_features: WineFeatures):
    """Make a prediction for wine quality."""
    try:
        # Prepare features
        input_features = prepare_features(wine_features)
        
        # Scale features
        input_scaled = scaler.transform(input_features)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_scaled)[0]
            prob_dict = {f"class_{i}": float(p) for i, p in enumerate(probabilities)}
            confidence = float(max(probabilities))
        else:
            prob_dict = {f"class_{prediction}": 1.0}
            confidence = 1.0
        
        # Map to quality label
        quality_labels = {0: "Low Quality", 1: "Medium Quality", 2: "High Quality"}
        quality_label = quality_labels.get(int(prediction), "Unknown")
        
        return PredictionResponse(
            quality_class=int(prediction),
            confidence=confidence,
            quality_label=quality_label,
            probabilities=prob_dict,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(wine_samples: List[WineFeatures]):
    """Make predictions for multiple wine samples."""
    if len(wine_samples) > 100:
        raise HTTPException(status_code=400, detail="Batch size limited to 100 samples")
    
    predictions = []
    for sample in wine_samples:
        pred = await predict(sample)
        predictions.append(pred)
    
    return predictions

if __name__ == "__main__":
    # Run the API
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )