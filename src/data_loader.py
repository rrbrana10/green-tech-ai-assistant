"""
data_loader.py — Loads dataset and handles preprocessing/scaling.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.utils import (
    DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMNS, 
    FEATURE_SCALER_PATH, TARGET_SCALER_PATH, ensure_dirs,
    FEATURE_NAMES, TARGET_NAMES
)

def load_data():
    """Loads dataset from CSV."""
    df = pd.read_csv(DATA_PATH)
    
    # Optional: ensure we map columns to meaningful names if needed inside dfs
    # But usually, it's easier to keep them as arrays for Keras and use names later
    return df

def prepare_data(df, test_size=0.2, random_state=42):
    """
    Splits into X and y, applies train_test_split (stratified by building height),
    and scales the data.
    """
    ensure_dirs()
    
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMNS].values
    
    # We can stratify by "Overall Height" (X5) to ensure equal representation
    # X5 is at index 4
    heights = df['X5'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=heights
    )
    
    # Scale Features
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    # Scale Targets (MinMaxScaler is good to keep targets within [0,1] or similar ranges, 
    # though with linear output in NN it's not strictly required, it helps convergence)
    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train)
    y_test_scaled = target_scaler.transform(y_test)
    
    # Save scalers for later use by the genetic algorithm and explainer
    joblib.dump(feature_scaler, FEATURE_SCALER_PATH)
    joblib.dump(target_scaler, TARGET_SCALER_PATH)
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scaler

def load_scalers():
    """Loads pre-trained scalers."""
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    return feature_scaler, target_scaler
