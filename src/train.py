#!/usr/bin/env python3
"""
Model Training Pipeline with MLflow Tracking
Trains multiple models, tracks experiments, and saves the best model.
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Set MLflow tracking URI
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("wine-quality-prediction")

def load_processed_data():
    """Load the processed data from the data preparation step."""
    print("📂 Loading processed data...")
    
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    # Load metadata
    with open('data/processed/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"✅ Loaded train set: {X_train.shape}")
    print(f"✅ Loaded test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, metadata

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    print(f"\n📊 {model_name} Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return metrics, y_pred

def train_baseline_model(X_train, X_test, y_train, y_test):
    """Train a baseline logistic regression model."""
    print("\n🎯 Training Baseline Model (Logistic Regression)...")
    
    with mlflow.start_run(run_name="logistic_regression"):
        # Model definition
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            multi_class='multinomial'
        )
        
        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("multi_class", "multinomial")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics, y_pred = evaluate_model(model, X_test, y_test, "Logistic Regression")
        
        # Log metrics
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        
    return model, metrics

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train a Random Forest model with hyperparameter tuning."""
    print("\n🌳 Training Random Forest Model...")
    
    with mlflow.start_run(run_name="random_forest"):
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
        
        # Base model
        rf = RandomForestClassifier(random_state=42)
        
        # Grid search
        print("🔍 Running hyperparameter search...")
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, 
            scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Best model
        model = grid_search.best_estimator_
        print(f"✅ Best parameters: {grid_search.best_params_}")
        
        # Log parameters
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("cv_folds", 3)
        
        # Evaluate
        metrics, y_pred = evaluate_model(model, X_test, y_test, "Random Forest")
        
        # Log metrics
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
        mlflow.log_metric("cv_best_score", grid_search.best_score_)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Top 5 Important Features:")
        print(feature_importance.head())
        
        # Log model and artifacts
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_text(feature_importance.to_string(), "feature_importance.txt")
        
    return model, metrics

def train_xgboost(X_train, X_test, y_train, y_test):
    """Train an XGBoost model."""
    print("\n🚀 Training XGBoost Model...")
    
    with mlflow.start_run(run_name="xgboost"):
        # Parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': len(np.unique(y_train)),
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42
        }
        
        # Log parameters
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_params(params)
        
        # Train model
        model = xgb.XGBClassifier(**params)
        
        # Train with evaluation set
        eval_set = [(X_test, y_test)]
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Evaluate
        metrics, y_pred = evaluate_model(model, X_test, y_test, "XGBoost")
        
        # Log metrics
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
        
        # Log model
        mlflow.xgboost.log_model(model, "model")
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
        mlflow.log_metric("cv_std_accuracy", cv_scores.std())
        
        print(f"📈 Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
    return model, metrics

def select_best_model(models_metrics):
    """Select the best model based on metrics."""
    best_model_name = max(models_metrics, key=lambda x: models_metrics[x]['metrics']['accuracy'])
    best_metrics = models_metrics[best_model_name]['metrics']
    best_model = models_metrics[best_model_name]['model']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
    
    return best_model, best_model_name, best_metrics

def save_production_model(model, model_name, metrics):
    """Save the best model for production."""
    print(f"\n💾 Saving {model_name} as production model...")
    
    # Create models directory
    os.makedirs('models/artifacts', exist_ok=True)
    
    # Save model
    if 'xgboost' in model_name.lower():
        model.save_model('models/artifacts/best_model.xgb')
    else:
        joblib.dump(model, 'models/artifacts/best_model.pkl')
    
    # Save model metadata
    model_info = {
        'model_type': model_name,
        'metrics': metrics,
        'feature_names': model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else None,
        'training_date': pd.Timestamp.now().isoformat()
    }
    
    with open('models/artifacts/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("✅ Production model saved!")

def main():
    """Run the complete training pipeline."""
    print("🚀 Starting Model Training Pipeline")
    print("=" * 50)
    
    # Load data
    X_train, X_test, y_train, y_test, metadata = load_processed_data()
    
    # Initialize MLflow
    print("\n📊 Starting MLflow tracking server...")
    print("   View experiments at: http://localhost:5000")
    print("   Run 'mlflow ui' in another terminal to see the UI")
    
    # Train models
    models_metrics = {}
    
    # 1. Baseline model
    lr_model, lr_metrics = train_baseline_model(X_train, X_test, y_train, y_test)
    models_metrics['Logistic Regression'] = {'model': lr_model, 'metrics': lr_metrics}
    
    # 2. Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)
    models_metrics['Random Forest'] = {'model': rf_model, 'metrics': rf_metrics}
    
    # 3. XGBoost
    xgb_model, xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test)
    models_metrics['XGBoost'] = {'model': xgb_model, 'metrics': xgb_metrics}
    
    # Select best model
    best_model, best_model_name, best_metrics = select_best_model(models_metrics)
    
    # Save for production
    save_production_model(best_model, best_model_name, best_metrics)
    
    print("\n✨ Model training complete!")
    print("📁 Check 'mlruns/' directory for experiment tracking")
    print("🏆 Best model saved to 'models/artifacts/'")
    print("\n💡 Run 'mlflow ui' to view experiments in the browser")
    
    return best_model

if __name__ == "__main__":
    main()