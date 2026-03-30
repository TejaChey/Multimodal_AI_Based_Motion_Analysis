#!/usr/bin/env python3
"""Prepare data for machine learning"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle

_HERE      = Path(__file__).resolve().parent.parent  # Phase_1/
DATA_PATH  = _HERE / "results" / "multimodal_features" / "multimodal_features.csv"
OUTPUT_DIR = _HERE / "results" / "ml_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def prepare_data():
    # Load
    df = pd.read_csv(DATA_PATH)
    
    print(" Loaded data:", df.shape)
    
    # Create labels (we'll do simple binary: fast vs slow)
    # For now, split by median of a key metric
    median_hip = df['hip_angle_mean'].median()
    
    # Lower hip angle = more explosive = better (elite)
    df['label'] = (df['hip_angle_mean'] < median_hip).astype(int)
    # 1 = fast/elite, 0 = slow/beginner
    
    print(f"\n Label distribution:")
    print(df['label'].value_counts())
    
    # Separate features and labels
    feature_cols = [c for c in df.columns if c not in ['subject', 'trial', 'label', 'frames_detected', 'file']]
    
    X = df[feature_cols].values
    y = df['label'].values
    subjects = df['subject'].values
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"   {feature_cols}")
    
    # Split by subject (not by sample!)
    unique_subjects = np.unique(subjects)
    train_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=0.25, random_state=42
    )
    
    train_mask = np.isin(subjects, train_subjects)
    test_mask = np.isin(subjects, test_subjects)
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"\n Split:")
    print(f"   Train: {len(X_train)} samples ({len(train_subjects)} subjects)")
    print(f"   Test:  {len(X_test)} samples ({len(test_subjects)} subjects)")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save
    data = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_cols,
        'scaler': scaler
    }
    
    output_path = OUTPUT_DIR / "ml_data.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n Saved: {output_path}")
    
    return data

if __name__ == "__main__":
    data = prepare_data()