#!/usr/bin/env python3
"""
Data Preparation Pipeline for Wine Quality Prediction
This script handles data loading, exploration, cleaning, and feature engineering.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import json

# Create directories if they don't exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('figures', exist_ok=True)

def load_wine_data(use_sklearn=True):
    """Load wine quality dataset from UCI repository or sklearn."""
    print("📊 Loading wine quality dataset...")
    
    if use_sklearn:
        # For speed, we'll use sklearn's wine dataset (similar structure)
        wine_data = load_wine()
        df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
        df['quality'] = wine_data.target
        print("ℹ️ Using sklearn wine dataset (for speed). Features are different from UCI dataset.")
    else:
        # Load actual wine quality dataset from UCI
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
            df = pd.read_csv(url, sep=';')
            print("ℹ️ Loaded UCI wine quality dataset")
        except Exception as e:
            print(f"⚠️ Failed to load from UCI: {e}")
            print("Falling back to sklearn dataset...")
            return load_wine_data(use_sklearn=True)
    
    # Save raw data
    df.to_csv('data/raw/wine_quality.csv', index=False)
    print(f"✅ Loaded {len(df)} samples with {len(df.columns)} features")
    
    return df

def explore_data(df):
    """Perform exploratory data analysis."""
    print("\n🔍 Exploratory Data Analysis...")
    
    # Basic statistics
    print("\nDataset shape:", df.shape)
    print("\nMissing values:", df.isnull().sum().sum())
    print("\nTarget distribution:")
    print(df['quality'].value_counts().sort_index())
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Target distribution
    df['quality'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Wine Quality Distribution')
    axes[0, 0].set_xlabel('Quality Score')
    
    # Correlation heatmap (top features)
    correlation = df.corr()['quality'].abs().sort_values(ascending=False)[1:11]
    correlation.plot(kind='barh', ax=axes[0, 1])
    axes[0, 1].set_title('Top 10 Features Correlated with Quality')
    
    # Feature distributions
    df['alcohol'].hist(bins=30, ax=axes[1, 0])
    axes[1, 0].set_title('Alcohol Content Distribution')
    
    # Box plot for outliers
    df.boxplot(column='alcohol', by='quality', ax=axes[1, 1])
    axes[1, 1].set_title('Alcohol Content by Quality')
    
    plt.tight_layout()
    plt.savefig('figures/eda_summary.png')
    plt.close()
    
    return {
        'n_samples': len(df),
        'n_features': len(df.columns) - 1,
        'target_distribution': df['quality'].value_counts().to_dict(),
        'missing_values': df.isnull().sum().sum()
    }

def engineer_features(df):
    """Create new features and handle preprocessing."""
    print("\n🔧 Engineering features...")
    
    # Create a copy
    df_processed = df.copy()
    
    # Print available features for debugging
    print(f"Available features: {df.columns.tolist()}")
    
    # 1. Create wine quality bins (low, medium, high)
    df_processed['quality_category'] = pd.cut(df['quality'], 
                                               bins=[0, 1, 2, 3], 
                                               labels=['low', 'medium', 'high'])
    
    # 2. Create interaction features based on available columns
    if 'alcohol' in df.columns:
        df_processed['alcohol_squared'] = df['alcohol'] ** 2
        
    if 'total_phenols' in df.columns and 'flavanoids' in df.columns:
        df_processed['phenols_flavanoids_ratio'] = df['total_phenols'] / (df['flavanoids'] + 1e-6)
    
    if 'color_intensity' in df.columns and 'hue' in df.columns:
        df_processed['color_hue_interaction'] = df['color_intensity'] * df['hue']
    
    # 3. Create normalized features for highly varying columns
    if 'proline' in df.columns:
        df_processed['proline_log'] = np.log1p(df['proline'])
    
    if 'magnesium' in df.columns:
        df_processed['magnesium_scaled'] = (df['magnesium'] - df['magnesium'].mean()) / df['magnesium'].std()
    
    print(f"✅ Created {len(df_processed.columns) - len(df.columns)} new features")
    
    return df_processed

def prepare_train_test_split(df):
    """Split data and apply scaling."""
    print("\n🔄 Splitting data into train/test sets...")
    
    # Separate features and target
    X = df.drop(['quality', 'quality_category'], axis=1, errors='ignore')
    y = df['quality']
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    print(f"✅ Train set: {X_train_scaled.shape}")
    print(f"✅ Test set: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_processed_data(X_train, X_test, y_train, y_test, scaler, metadata):
    """Save processed data and artifacts."""
    print("\n💾 Saving processed data...")
    
    # Save datasets
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    # Save scaler
    joblib.dump(scaler, 'data/processed/scaler.pkl')
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        else:
            return obj
    
    # Save metadata
    metadata['processing_date'] = datetime.now().isoformat()
    metadata['feature_names'] = X_train.columns.tolist()
    metadata = convert_to_serializable(metadata)
    
    with open('data/processed/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ All data saved successfully!")

def main():
    """Run the complete data preparation pipeline."""
    print("🚀 Starting Data Preparation Pipeline")
    print("=" * 50)
    
    # Load data
    df = load_wine_data()
    
    # Explore data
    metadata = explore_data(df)
    
    # Engineer features
    df_processed = engineer_features(df)
    
    # Prepare train/test split
    X_train, X_test, y_train, y_test, scaler = prepare_train_test_split(df_processed)
    
    # Save everything
    save_processed_data(X_train, X_test, y_train, y_test, scaler, metadata)
    
    print("\n✨ Data preparation complete!")
    print(f"📁 Check the 'data/processed/' directory for outputs")
    print(f"📊 Check the 'figures/' directory for EDA visualizations")
    
    # Return for potential use in notebooks
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    main()