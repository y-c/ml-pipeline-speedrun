# MLflow Guide for ML Pipeline Project

This guide explains how MLflow is integrated into our wine quality prediction pipeline and how to use it effectively.

## Table of Contents
- [What is MLflow?](#what-is-mlflow)
- [How MLflow Works in Our Code](#how-mlflow-works-in-our-code)
- [Running the MLflow UI](#running-the-mlflow-ui)
- [Understanding the UI](#understanding-the-ui)
- [Directory Structure](#directory-structure)
- [Common Tasks](#common-tasks)
- [Best Practices](#best-practices)

## What is MLflow?

MLflow is like "Git for Machine Learning" - it automatically tracks:
- **Parameters**: Hyperparameters used for each model
- **Metrics**: Model performance (accuracy, precision, etc.)
- **Models**: The actual trained model artifacts
- **Code**: Which code version produced which model
- **Environment**: Dependencies and versions

## How MLflow Works in Our Code

### 1. Setup (in `train.py`)
```python
import mlflow
import mlflow.sklearn

# Configure where to store experiments
mlflow.set_tracking_uri("mlruns")  # Local directory
mlflow.set_experiment("wine-quality-prediction")  # Experiment name
```

### 2. Tracking a Model Run
Each model training is wrapped in an MLflow run:

```python
def train_random_forest(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="random_forest"):
        # 1. Log hyperparameters
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_params(grid_search.best_params_)
        
        # 2. Train model
        model = grid_search.best_estimator_
        
        # 3. Log metrics
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        
        # 4. Log the model itself
        mlflow.sklearn.log_model(model, "model")
        
        # 5. Log additional artifacts
        mlflow.log_text(feature_importance.to_string(), "feature_importance.txt")
```

### 3. What Gets Logged

For each model run, MLflow stores:
- **Parameters**: All hyperparameters (n_estimators, max_depth, etc.)
- **Metrics**: Performance metrics (accuracy, precision, recall, f1)
- **Artifacts**: Model file, feature importance, confusion matrix
- **Metadata**: Start time, duration, user, source code version

## Running the MLflow UI

```bash
# Start the UI server
python3 -m mlflow ui

# Output:
# [INFO] Listening at: http://127.0.0.1:5000

# Open in browser
open http://127.0.0.1:5000  # macOS
# or just click the link
```

## Understanding the UI

### Main Page - Experiments List
Shows all experiments. Click on "wine-quality-prediction" to see all runs.

### Runs Table
```
┌──────────────┬─────────────────┬──────────┬───────────┬─────────┐
│ Run Name     │ Created         │ Duration │ Accuracy  │ F1      │
├──────────────┼─────────────────┼──────────┼───────────┼─────────┤
│ random_forest│ 2025-08-09 13:40│ 15s      │ 1.000000 │ 1.0000  │
│ xgboost      │ 2025-08-09 13:40│ 12s      │ 0.972222 │ 0.9722  │
│ logistic_reg │ 2025-08-09 13:40│ 8s       │ 0.944444 │ 0.9444  │
└──────────────┴─────────────────┴──────────┴───────────┴─────────┘
```

### Run Details Page
Click any run to see:
- **Parameters**: All logged hyperparameters
- **Metrics**: All logged performance metrics
- **Artifacts**: Downloadable files (model, plots, text files)
- **Metadata**: Run ID, start time, duration, etc.

### Comparing Runs
1. Select multiple runs (checkbox)
2. Click "Compare"
3. View side-by-side:
   - Parameter differences
   - Metric comparisons
   - Parallel coordinates plot
   - Scatter plots

## Directory Structure

MLflow creates this structure:
```
mlruns/
├── 0/                          # Experiment ID (0 = Default)
│   └── meta.yaml              # Experiment metadata
├── 1/                          # Our experiment ID
│   └── meta.yaml              # Experiment name and details
├── <run_id_1>/                # Each run has unique ID
│   ├── artifacts/
│   │   ├── model/             # Saved model
│   │   │   ├── model.pkl
│   │   │   ├── MLmodel        # Model metadata
│   │   │   └── requirements.txt
│   │   ├── feature_importance.txt
│   │   └── confusion_matrix.txt
│   ├── metrics/               # One file per metric
│   │   ├── accuracy
│   │   ├── precision
│   │   └── f1
│   ├── params/                # One file per parameter
│   │   ├── model_type
│   │   └── n_estimators
│   ├── tags/                  # MLflow tags
│   └── meta.yaml             # Run metadata
```

## Common Tasks

### 1. Find the Best Model
- Sort by accuracy column in UI
- Or use MLflow API:
```python
from mlflow.tracking import MlflowClient
client = MlflowClient()
runs = client.search_runs(experiment_ids=["1"], order_by=["metrics.accuracy DESC"])
best_run = runs[0]
```

### 2. Load a Previous Model
```python
import mlflow
# Using run ID from UI
model = mlflow.sklearn.load_model("runs:/a3f4d5e6b7c8d9e0/model")
```

### 3. Compare Hyperparameters
- Select runs in UI → Compare
- Look at "Parameters" section
- See which hyperparameters led to best performance

### 4. Export Results
- Screenshot the comparison view
- Or use MLflow API to export to CSV:
```python
import mlflow
df = mlflow.search_runs(experiment_ids=["1"])
df.to_csv("experiment_results.csv")
```

## Best Practices

### 1. Consistent Naming
```python
with mlflow.start_run(run_name="model_type_version"):
    # Use descriptive run names
```

### 2. Log Everything Important
```python
# Not just final metrics
mlflow.log_metric("train_accuracy", train_acc)
mlflow.log_metric("val_accuracy", val_acc)
mlflow.log_metric("test_accuracy", test_acc)
```

### 3. Use Tags for Organization
```python
mlflow.set_tag("version", "v1.2")
mlflow.set_tag("dataset", "wine_quality_red")
```

### 4. Track Data Versions
```python
mlflow.log_param("data_version", "2025-08-09")
mlflow.log_param("n_samples", len(X_train))
```

## Troubleshooting

### MLflow UI Won't Start
```bash
# Check if port 5000 is in use
lsof -i :5000

# Use different port
mlflow ui --port 5001
```

### Can't Find mlflow Command
```bash
# Use Python module syntax
python3 -m mlflow ui
```

### Experiments Not Showing
- Check you're in the right directory
- MLflow looks for `mlruns/` in current directory
- Run from project root

## Why MLflow Matters

**Without MLflow:**
- "Which hyperparameters did I use for that good model last week?"
- "Did changing max_depth actually improve performance?"
- Multiple model files with unclear naming: `model_final_v2_really_final.pkl`

**With MLflow:**
- Every experiment is tracked automatically
- Easy to reproduce any previous result
- Clear history of what was tried and what worked
- Professional ML workflow

## Further Learning

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking Guide](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html) (for production)

---

*This guide is specific to our wine quality prediction project. The concepts apply to any ML project using MLflow.*