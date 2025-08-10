# ML Pipeline Speedrun: From Data to Deployment

**Date**: Saturday, August 09, 2025  
**Time Budget**: 2 hours  
**Goal**: Learn the complete ML pipeline by building a wine quality predictor

## 🎯 Learning Objectives Achieved
- ✅ End-to-end ML workflow implementation
- ✅ MLOps tools experience (MLflow, Docker, FastAPI)
- ✅ Built and deployed ML service
- ✅ Learned ML-specific versioning with MLflow

## 🏗️ Architecture Overview
```
Data → Preprocessing → Training → Versioning → Deployment → Serving
 ↓         ↓              ↓           ↓            ↓           ↓
CSV    Feature Eng    sklearn     MLflow      Docker      FastAPI
       pandas         XGBoost      DVC                    Monitoring
```

## 📁 Project Structure
```
ml-pipeline-speedrun/
├── data/
│   ├── raw/              # Original data
│   └── processed/        # Cleaned, feature-engineered
├── figures/              # EDA visualizations
├── notebooks/
│   └── exploration.ipynb # Quick EDA
├── src/
│   ├── data_prep.py      # Data pipeline ✅
│   ├── train.py          # Model training ✅
│   └── serve.py          # FastAPI app ✅
├── models/
│   └── artifacts/        # Saved models
├── mlruns/               # MLflow experiments (gitignored)
├── Dockerfile
├── requirements.txt      ✅
└── README.md
```

## ⏱️ Timeline & Tasks

### ✅ 0:00-0:20 | Data Preparation
- [x] Load wine quality dataset from sklearn
- [x] Exploratory analysis (distributions, correlations)
- [x] Handle missing values, outliers
- [x] Feature engineering (6 new features created)
- [x] Create reproducible train/test split
- [x] Save processed data with metadata

### ✅ 0:20-0:40 | Model Training
- [x] Baseline model (Logistic Regression)
- [x] Advanced model (Random Forest with GridSearchCV)
- [x] XGBoost model with early stopping
- [x] Cross-validation
- [x] Log all experiments with MLflow
- [x] Compare model performances

### ✅ 0:40-1:00 | Model Versioning & API
- [x] Set up MLflow tracking server
- [x] Log models, parameters, metrics
- [x] Track feature importance
- [x] Select and save best model
- [x] Create FastAPI application
- [x] Add prediction endpoints

### ✅ 1:00-1:30 | Deployment & Serving
- [x] FastAPI with health checks
- [x] Model metadata endpoint
- [x] Single and batch prediction endpoints
- [x] Test API with curl commands
- [x] Write Dockerfile
- [x] Build container image
- [x] Test containerized app locally

### ✅ 1:30-2:00 | Final Steps
- [x] Run production server in Docker
- [ ] Create simple web UI for predictions (future improvement)
- [x] Test with sample requests
- [x] Document API with Swagger
- [x] Push everything to GitHub

## 🛠️ Tech Stack
- **Data**: pandas, numpy, scikit-learn
- **Training**: XGBoost, scikit-learn
- **Tracking**: MLflow ~~, DVC~~ (skipped for time)
- **API**: FastAPI, pydantic
- **Deployment**: Docker, uvicorn
- **Monitoring**: Basic logging ~~, prometheus~~ (optional)

## 📊 Results & Metrics
- **Dataset**: 178 samples, 13 original features + 6 engineered features
- **Best Model**: Random Forest (100% accuracy on test set)
- **API Performance**: <100ms response time ✅
- **Endpoints**: `/health`, `/model/info`, `/predict`, `/predict/batch`
- **Docker Image**: ~1.5GB (Python slim + dependencies)
- **Container Health**: Automated health checks every 30s

## 🚀 Quick Start Commands

### Setup
```bash
# Clone and setup
git clone https://github.com/y-c/ml-pipeline-speedrun.git
cd ml-pipeline-speedrun

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip3 install -r requirements.txt
```

### Run the Pipeline
```bash
# 1. Data preparation
python3 src/data_prep.py

# 2. Model training
python3 src/train.py

# 3. View experiments (separate terminal)
python3 -m mlflow ui
# Open http://127.0.0.1:5000 to see experiments

# 4. Serve API
python3 src/serve.py
# Open http://localhost:8000/docs for API documentation
```

### Make Predictions
```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Docker Commands
```bash
# Start Docker Desktop (if not running)
open -a Docker  # macOS
# Wait for Docker to fully start (whale icon in menu bar)

# Build the Docker image
docker build -t wine-predictor .

# Run the container
docker run -p 8000:8000 wine-predictor

# Run in detached mode
docker run -d -p 8000:8000 --name wine-api wine-predictor

# Check logs
docker logs wine-api

# Stop the container
docker stop wine-api
```

## 📝 Key Learnings
1. **Data versioning vs code versioning**: Data changes affect model performance, need tracking
2. **MLflow benefits**: Automatic experiment tracking, model comparison, artifact storage
3. **ML deployment challenges**: Feature engineering consistency, model versioning, API design
4. **Reproducibility**: Random seeds, data splits, dependency management are crucial

## 🔍 Interesting Findings
- Using sklearn wine dataset (different from UCI wine quality dataset)
- Feature engineering improved model performance significantly
- XGBoost and Random Forest outperformed Logistic Regression
- MLflow UI makes experiment comparison much easier
- FastAPI auto-generates interactive API documentation

## 📌 Notes & Gotchas
- Use `python3 -m mlflow ui` if mlflow command not found
- Start Docker Desktop first: `open -a Docker` (macOS)
- Sklearn wine dataset has different feature names than UCI dataset
- JSON serialization issues with numpy types (fixed in code)
- Remember to match feature engineering in serving code
- API field name `od280_od315` (simplified from `od280/od315_of_diluted_wines` for JSON compatibility)
- API expects all 13 original features for prediction
- Model achieves 100% accuracy on test set (might indicate overfitting on small dataset)

## 🔄 Future Improvements
- [ ] Add A/B testing capability
- [ ] Implement model retraining pipeline
- [ ] Add data drift detection
- [ ] Scale with Kubernetes
- [ ] Add GPU support
- [ ] Implement proper monitoring/alerting
- [ ] Add DVC for data versioning
- [ ] Create web UI for easier predictions
- [ ] Add model explainability (SHAP/LIME)

## 📚 Resources
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Docker for Data Scientists](https://docker-curriculum.com/)
- **[📖 MLflow Guide for this Project](docs/mlflow-guide.md)** - Detailed explanation of MLflow usage
- **[🚀 FastAPI Guide for this Project](docs/fastapi-guide.md)** - How FastAPI serves our ML model

---
*Remember: The goal isn't perfection in 2 hours - it's understanding the complete pipeline. Each component can be a deep rabbit hole for future exploration!*

**Status**: ✅ 100% COMPLETE! Full ML pipeline from data to Docker deployment achieved in 2 hours!