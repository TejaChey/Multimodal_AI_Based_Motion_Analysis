#!/usr/bin/env python3
"""Compare multimodal vs unimodal approaches"""

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

_HERE     = Path(__file__).resolve().parent.parent  # Phase_1/
DATA_PATH = _HERE / "results" / "ml_data" / "ml_data.pkl"
OUTPUT_DIR = _HERE / "results"

def compare_modalities():
    # Load data
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']
    
    # Identify video vs IMU features
    video_features = [i for i, f in enumerate(feature_names) 
                     if 'angle' in f or 'asymmetry' in f or 'velocity' in f]
    imu_features = [i for i, f in enumerate(feature_names) 
                   if 'accel' in f or 'gyro' in f or 'stride' in f or 'steps' in f or 'smoothness' in f]
    
    print("="*60)
    print("MODALITY COMPARISON")
    print("="*60)
    
    print(f"\n Feature breakdown:")
    print(f"   Video features: {len(video_features)}")
    print(f"   IMU features: {len(imu_features)}")
    print(f"   Total features: {len(feature_names)}")
    
    results = {}
    
    # 1. Video-only
    print("\n VIDEO-ONLY MODEL:")
    X_train_video = X_train[:, video_features]
    X_test_video = X_test[:, video_features]
    
    rf_video = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_video.fit(X_train_video, y_train)
    y_pred_video = rf_video.predict(X_test_video)
    acc_video = accuracy_score(y_test, y_pred_video)
    
    print(f"   Accuracy: {acc_video:.2%}")
    results['Video-only'] = acc_video
    
    # 2. IMU-only
    print("\nIMU-ONLY MODEL:")
    X_train_imu = X_train[:, imu_features]
    X_test_imu = X_test[:, imu_features]
    
    rf_imu = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_imu.fit(X_train_imu, y_train)
    y_pred_imu = rf_imu.predict(X_test_imu)
    acc_imu = accuracy_score(y_test, y_pred_imu)
    
    print(f"   Accuracy: {acc_imu:.2%}")
    results['IMU-only'] = acc_imu
    
    # 3. Multimodal (all features)
    print("\n MULTIMODAL (Video + IMU):")
    rf_multi = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_multi.fit(X_train, y_train)
    y_pred_multi = rf_multi.predict(X_test)
    acc_multi = accuracy_score(y_test, y_pred_multi)
    
    print(f"   Accuracy: {acc_multi:.2%}")
    results['Multimodal'] = acc_multi
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    df_results = pd.DataFrame([results]).T
    df_results.columns = ['Accuracy']
    df_results['Improvement'] = df_results['Accuracy'] - df_results['Accuracy'].iloc[:-1].max()
    
    print(df_results)
    
    best = max(results, key=results.get)
    improvement = results['Multimodal'] - max(results['Video-only'], results['IMU-only'])
    
    print(f"\n Best approach: {best} ({results[best]:.2%})")
    print(f" Multimodal improvement: +{improvement:.1%}")
    
    # Save results
    output_path = OUTPUT_DIR / "modality_comparison.csv"
    df_results.to_csv(output_path)
    print(f"\nSaved: {output_path}")
    
    return results

if __name__ == "__main__":
    compare_modalities()