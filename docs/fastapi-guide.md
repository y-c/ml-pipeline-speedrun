# FastAPI Guide for ML Model Serving

This guide explains how FastAPI serves our trained ML model and makes it accessible via REST API.

## Table of Contents
- [What is FastAPI?](#what-is-fastapi)
- [Why FastAPI for ML?](#why-fastapi-for-ml)
- [How It Works in Our Pipeline](#how-it-works-in-our-pipeline)
- [API Architecture](#api-architecture)
- [Understanding the Code](#understanding-the-code)
- [API Endpoints](#api-endpoints)
- [Testing the API](#testing-the-api)
- [Deployment Flow](#deployment-flow)

## What is FastAPI?

FastAPI is a modern Python web framework for building APIs. Think of it as the "serving layer" that:
- Takes HTTP requests (wine features)
- Runs them through our trained model
- Returns predictions as JSON responses

## Why FastAPI for ML?

**Traditional ML Workflow:**
```
Train Model → Save as .pkl file → ??? → How do others use it?
```

**With FastAPI:**
```
Train Model → Save as .pkl → FastAPI loads it → REST API → Anyone can use it!
```

Benefits:
- **Automatic validation**: Ensures input data is correct format
- **Auto-documentation**: Swagger UI generated automatically
- **Fast**: Built on modern Python async features
- **Type hints**: Catches errors before they happen
- **Easy testing**: Simple to test with curl or any HTTP client

## How It Works in Our Pipeline

### The Complete Flow:
```
1. Data Prep (data_prep.py)
   ↓ Creates scaler.pkl, metadata.json
   
2. Training (train.py)
   ↓ Creates best_model.pkl, model_info.json
   
3. Serving (serve.py) ← FastAPI starts here
   ↓ Loads model + scaler
   
4. API Ready at http://localhost:8000
   ↓ Accepts POST requests with wine features
   
5. Returns predictions as JSON
```

## API Architecture

### Key Components in `serve.py`:

```python
# 1. App Initialization
app = FastAPI(
    title="Wine Quality Prediction API",
    description="ML model serving for wine quality prediction",
    version="1.0.0"
)

# 2. Data Models (Pydantic)
class WineFeatures(BaseModel):
    """Defines expected input structure"""
    alcohol: float
    malic_acid: float
    # ... all features

class PredictionResponse(BaseModel):
    """Defines response structure"""
    quality_class: int
    confidence: float
    quality_label: str

# 3. Model Loading
def load_model():
    """Loads trained model and scaler on startup"""
    model = joblib.load('models/artifacts/best_model.pkl')
    scaler = joblib.load('data/processed/scaler.pkl')

# 4. Endpoints
@app.post("/predict")
async def predict(wine_features: WineFeatures):
    """Main prediction endpoint"""
```

## Understanding the Code

### 1. Input Validation with Pydantic
```python
class WineFeatures(BaseModel):
    alcohol: float = Field(..., ge=0, le=20, description="Alcohol content (%)")
    # Validates: must be float, between 0-20
```

If you send invalid data:
```json
{"alcohol": "high"}  // ❌ Not a number
{"alcohol": -5}      // ❌ Below minimum
{"alcohol": 13.2}    // ✅ Valid
```

### 2. Feature Engineering Must Match Training
```python
def prepare_features(wine_data: WineFeatures) -> pd.DataFrame:
    # Recreate EXACT same features as training
    input_df['alcohol_squared'] = input_df['alcohol'] ** 2
    input_df['phenols_flavanoids_ratio'] = ...
    # Must match data_prep.py exactly!
```

### 3. Model Prediction Flow
```python
@app.post("/predict")
async def predict(wine_features: WineFeatures):
    # 1. Convert input to DataFrame
    input_features = prepare_features(wine_features)
    
    # 2. Scale using saved scaler
    input_scaled = scaler.transform(input_features)
    
    # 3. Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # 4. Get confidence scores
    probabilities = model.predict_proba(input_scaled)[0]
    
    # 5. Return formatted response
    return PredictionResponse(...)
```

### 4. Startup Events
```python
@app.on_event("startup")
async def startup_event():
    load_model()  # Load model once when server starts
```

## API Endpoints

### 1. Root Endpoint
```bash
GET http://localhost:8000/
```
Returns:
```json
{
  "message": "Wine Quality Prediction API",
  "docs": "/docs",
  "health": "/health"
}
```

### 2. Health Check
```bash
GET http://localhost:8000/health
```
Returns:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-08-09T14:09:49.732526"
}
```

### 3. Model Information
```bash
GET http://localhost:8000/model/info
```
Returns:
```json
{
  "model_type": "Random Forest",
  "metrics": {
    "accuracy": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0
  },
  "training_date": "2025-08-09T13:40:37.530620",
  "feature_names": [...],
  "version": "1.0.0"
}
```

### 4. Single Prediction
```bash
POST http://localhost:8000/predict
Content-Type: application/json

{
  "alcohol": 13.2,
  "malic_acid": 1.78,
  ...
}
```
Returns:
```json
{
  "quality_class": 0,
  "confidence": 0.9746,
  "quality_label": "Low Quality",
  "probabilities": {
    "class_0": 0.9746,
    "class_1": 0.0194,
    "class_2": 0.0060
  },
  "timestamp": "2025-08-09T14:15:07.303135"
}
```

### 5. Batch Prediction
```bash
POST http://localhost:8000/predict/batch
Content-Type: application/json

[
  {"alcohol": 13.2, ...},
  {"alcohol": 12.8, ...}
]
```

### 6. Auto-Documentation
```bash
GET http://localhost:8000/docs
```
Opens Swagger UI with interactive API testing!

## Testing the API

### 1. Using curl
```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"alcohol": 13.2, ...}'
```

### 2. Using Python
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"alcohol": 13.2, ...}
)
print(response.json())
```

### 3. Using Swagger UI
1. Go to http://localhost:8000/docs
2. Click on any endpoint
3. Click "Try it out"
4. Edit the request body
5. Click "Execute"

## Deployment Flow

### Local Development:
```bash
python3 src/serve.py
# Runs on http://localhost:8000
```

### Docker Container:
```dockerfile
# Dockerfile packages everything
COPY models/artifacts/ ./models/artifacts/
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0"]
```

### Production Flow:
```
1. Build Docker image with model inside
2. Push to container registry
3. Deploy to cloud (AWS, GCP, Azure)
4. API available globally
```

## Common Issues and Solutions

### 1. Field Name Issues
```python
# Problem: "od280/od315_of_diluted_wines" has forward slash
# Solution: Renamed to "od280_od315" for JSON compatibility
```

### 2. Feature Mismatch
```python
# Problem: API features don't match training features
# Solution: prepare_features() must exactly recreate training features
```

### 3. Model Not Loading
```python
# Problem: Can't find model file
# Solution: Check paths, ensure model artifacts are committed/copied
```

## Why This Architecture?

### Separation of Concerns:
- **Training** (train.py): Focus on ML, experimentation
- **Serving** (serve.py): Focus on reliability, validation, API design
- **Docker**: Focus on deployment, dependencies

### Benefits:
1. **Scalability**: Can run multiple API instances
2. **Language Agnostic**: Any language can call REST API
3. **Monitoring**: Easy to add logging, metrics
4. **Versioning**: Can serve multiple model versions
5. **Testing**: Easy to test endpoints independently

## Integration with Frontend

A frontend application could use our API like this:

```javascript
// React example
const predictWineQuality = async (wineData) => {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(wineData)
  });
  return response.json();
};
```

## Best Practices

### 1. Input Validation
Always validate inputs thoroughly:
```python
alcohol: float = Field(..., ge=0, le=20)  # Range validation
```

### 2. Error Handling
```python
try:
    prediction = model.predict(input_scaled)
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

### 3. Logging
```python
logger.info(f"Prediction made: {prediction}")
```

### 4. Version Your API
```python
app = FastAPI(version="1.0.0")
# Consider /v1/predict for versioned endpoints
```

## Next Steps

1. **Add Authentication**: Protect endpoints with API keys
2. **Add Monitoring**: Track prediction latency, errors
3. **Add Model Versioning**: Serve multiple models
4. **Add Caching**: Cache repeated predictions
5. **Add Rate Limiting**: Prevent abuse

## Summary

FastAPI transforms our saved ML model into a production-ready web service that:
- Anyone can use (not just Python users)
- Validates inputs automatically
- Documents itself
- Handles errors gracefully
- Scales horizontally
- Integrates with any frontend

It's the bridge between "I trained a model" and "My model is serving predictions to users worldwide"!

---

*This guide explains FastAPI's role in our ML pipeline. For API reference, check http://localhost:8000/docs when running.*