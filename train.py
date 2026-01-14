#!/usr/bin/env python3
"""
AWS SageMaker ML Training Pipeline
End-to-end ML model training with MLflow tracking and CI/CD integration
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import mlflow
import mlflow.sklearn
import boto3
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--min-samples-split', type=int, default=2)
    return parser.parse_args()

def load_data(train_path, test_path):
    """Load training and test data from S3"""
    print("Loading data...")
    train_data = pd.read_csv(os.path.join(train_path, 'train.csv'))
    test_data = pd.read_csv(os.path.join(test_path, 'test.csv'))
    
    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    
    print(f"Train data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train, n_estimators, max_depth, min_samples_split):
    """Train Random Forest model with MLflow tracking"""
    print("Training model...")
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print("Model training completed")
        return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and log metrics"""
    print("Evaluating model...")
    
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Log metrics to MLflow
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    
    print(f"Model Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")
    
    return {"rmse": rmse, "mae": mae, "r2": r2}

def save_model(model, model_dir):
    """Save trained model for SageMaker deployment"""
    print(f"Saving model to {model_dir}")
    path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, path)
    print("Model saved successfully")

def main():
    args = parse_args()
    
    # Load data
    X_train, y_train, X_test, y_test = load_data(args.train, args.test)
    
    # Train model
    model = train_model(
        X_train, y_train,
        args.n_estimators,
        args.max_depth,
        args.min_samples_split
    )
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, args.model_dir)
    
    # Save metrics for CI/CD pipeline
    metrics_path = os.path.join(args.model_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    print("Training pipeline completed successfully")
    return 0

if __name__ == '__main__':
    main()
