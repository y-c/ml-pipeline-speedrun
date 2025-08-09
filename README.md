# ML Pipeline Speedrun: From Data to Deployment

**Date**: Saturday, August 09, 2025  
**Time Budget**: 2 hours  
**Goal**: Learn the complete ML pipeline by building a wine quality predictor

## 🎯 Learning Objectives
- Understand end-to-end ML workflow
- Experience MLOps tools and practices
- Build a deployable ML service
- Learn ML-specific version control

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
├── notebooks/
│   └── exploration.ipynb # Quick EDA
├── src/
│   ├── data_prep.py      # Data pipeline
│   ├── train.py          # Model training
│   └── serve.py          # FastAPI app
├── models/
│   └── artifacts/        # Saved models
├── mlruns/               # MLflow experiments
├── Dockerfile
├── requirements.txt
└── README.md
```

## ⏱️ Timeline & Tasks

### 0:00-0:20 | Data Preparation
- [ ] Load wine quality dataset from UCI/sklearn
- [ ] Exploratory analysis (distributions, correlations)
- [ ] Handle missing values, outliers
- [ ] Feature engineering (polynomial features, scaling)
- [ ] Create reproducible train/test split
- [ ] Save processed data with versioning

### 0:20-0:40 | Model Training
- [ ] Baseline model (Logistic Regression)
- [ ] Advanced model (Random Forest/XGBoost)
- [ ] Hyperparameter tuning (quick GridSearch)
- [ ] Cross-validation
- [ ] Log all experiments with MLflow
- [ ] Compare model performances

### 0:40-1:00 | Model Versioning
- [ ] Set up MLflow tracking server
- [ ] Log models, parameters, metrics
- [ ] Create model registry
- [ ] Initialize DVC for data versioning
- [ ] Tag best model for production
- [ ] Document model lineage

### 1:00-1:30 | Deployment
- [ ] Create FastAPI application
- [ ] Add prediction endpoint
- [ ] Add model metadata endpoint
- [ ] Write Dockerfile
- [ ] Build container image
- [ ] Test containerized app locally

### 1:30-2:00 | Serving & Monitoring
- [ ] Run production server
- [ ] Create simple web UI for predictions
- [ ] Add basic monitoring (latency, requests)
- [ ] Test with sample requests
- [ ] Document API with Swagger
- [ ] Push everything to GitHub

## 🛠️ Tech Stack
- **Data**: pandas, numpy, scikit-learn
- **Training**: XGBoost, scikit-learn
- **Tracking**: MLflow, DVC
- **API**: FastAPI, pydantic
- **Deployment**: Docker, uvicorn
- **Monitoring**: Basic logging, prometheus (optional)

## 📊 Success Metrics
- ✅ Complete pipeline runs end-to-end
- ✅ Model achieves >85% accuracy
- ✅ API responds <100ms
- ✅ All code committed to GitHub
- ✅ Can reproduce results from scratch

## 🚀 Quick Start Commands
```bash
# Clone and setup
git clone <your-repo>
cd ml-pipeline-speedrun
pip install -r requirements.txt

# Run pipeline
python src/data_prep.py
python src/train.py
mlflow ui  # View experiments

# Deploy
docker build -t wine-predictor .
docker run -p 8000:8000 wine-predictor

# Test
curl -X POST http://localhost:8000/predict -d '{...}'
```

## 📝 Key Learnings to Document
1. Why is data versioning different from code versioning?
2. How does MLflow differ from traditional logging?
3. What makes ML deployment unique?
4. How to ensure reproducibility?

## 🔄 Future Improvements
- Add A/B testing capability
- Implement model retraining pipeline
- Add data drift detection
- Scale with Kubernetes
- Add GPU support

## 📚 Resources
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Docker for Data Scientists](https://docker-curriculum.com/)
