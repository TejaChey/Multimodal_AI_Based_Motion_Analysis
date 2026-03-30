#!/usr/bin/env python3
"""Compare modalities with multiple classifiers"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pathlib import Path

_HERE     = Path(__file__).resolve().parent.parent  # Phase_1/
DATA_PATH = _HERE / "results" / "ml_data" / "ml_data_3class.pkl"

def compare_all():
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']
    
    # Identify features
    video_features = [i for i, f in enumerate(feature_names) 
                     if 'angle' in f or 'asymmetry' in f or 'velocity' in f]
    imu_features = [i for i, f in enumerate(feature_names) 
                   if 'accel' in f or 'gyro' in f or 'stride' in f or 'steps' in f or 'smoothness' in f]
    
    print("="*70)
    print("COMPREHENSIVE MODALITY COMPARISON")
    print("="*70)
    
    print(f"\n Dataset:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test:  {len(X_test)} samples")
    print(f"   Classes: {len(np.unique(y_train))} (Beginner/Intermediate/Elite)")
    
    # Classifiers to try
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=42),
        'SVM': SVC(kernel='rbf', C=1, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    
    for clf_name, clf in classifiers.items():
        print(f"\n{'='*70}")
        print(f" {clf_name}")
        print(f"{'='*70}")
        
        # Video-only
        X_train_video = X_train[:, video_features]
        X_test_video = X_test[:, video_features]
        clf_video = clf.__class__(**clf.get_params())
        clf_video.fit(X_train_video, y_train)
        acc_video = accuracy_score(y_test, clf_video.predict(X_test_video))
        
        # IMU-only
        X_train_imu = X_train[:, imu_features]
        X_test_imu = X_test[:, imu_features]
        clf_imu = clf.__class__(**clf.get_params())
        clf_imu.fit(X_train_imu, y_train)
        acc_imu = accuracy_score(y_test, clf_imu.predict(X_test_imu))
        
        # Multimodal
        clf_multi = clf.__class__(**clf.get_params())
        clf_multi.fit(X_train, y_train)
        acc_multi = accuracy_score(y_test, clf_multi.predict(X_test))
        
        print(f"  Video-only:    {acc_video:.1%}")
        print(f"  IMU-only:      {acc_imu:.1%}")
        print(f"  Multimodal:    {acc_multi:.1%}")
        
        improvement = acc_multi - max(acc_video, acc_imu)
        print(f"  Improvement:   {improvement:+.1%}")
        
        results[clf_name] = {
            'Video': acc_video,
            'IMU': acc_imu,
            'Multimodal': acc_multi,
            'Improvement': improvement
        }
    
    # Best results
    print(f"\n{'='*70}")
    print("BEST RESULTS")
    print(f"{'='*70}")
    
    best_multi = max(results.items(), key=lambda x: x[1]['Multimodal'])
    best_improvement = max(results.items(), key=lambda x: x[1]['Improvement'])
    
    print(f"\n Best Multimodal Accuracy: {best_multi[0]}")
    print(f"   {best_multi[1]['Multimodal']:.1%} accuracy")
    
    print(f"\n Best Fusion Improvement: {best_improvement[0]}")
    print(f"   {best_improvement[1]['Improvement']:+.1%} over single modality")
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Classifier':<20} {'Video':<10} {'IMU':<10} {'Multi':<10} {'Gain':<10}")
    print("-"*70)
    
    for clf_name, scores in results.items():
        print(f"{clf_name:<20} {scores['Video']:<10.1%} {scores['IMU']:<10.1%} "
              f"{scores['Multimodal']:<10.1%} {scores['Improvement']:+<10.1%}")
    
    return results

if __name__ == "__main__":
    results = compare_all()