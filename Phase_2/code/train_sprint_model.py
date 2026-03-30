#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ── CONFIG ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent

INPUT_CSV = PROJECT_ROOT / "Phase_2" / "results" / "multimodal_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "Phase_2" / "results" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def train_and_evaluate():
    print(f"{'='*50}\n  Phase 2: Multimodal Sprint Classifier\n{'='*50}")
    
    # 1. Load Data
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing fused features at {INPUT_CSV}")
        
    df = pd.read_csv(INPUT_CSV)
    athletes = df['athlete'].values
    features_df = df.drop(columns=['athlete'])
    feature_names = features_df.columns.tolist()
    
    X_raw = features_df.values
    
    # 2. Scale Features
    print("  Scaling 23 features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # 3. K-Means Clustering (Auto-Labeling)
    print(" Running K-Means to discover Elite/Intermediate/Beginner clusters...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # We heuristically assign labels based on Stride Frequency (higher = better)
    stride_col_idx = feature_names.index('stride_freq_hz')
    
    cluster_means = []
    for c in range(3):
        mean_stride = X_raw[clusters == c, stride_col_idx].mean()
        cluster_means.append((c, mean_stride))
        
    # Sort clusters by stride rate (descending)
    cluster_means.sort(key=lambda x: x[1], reverse=True)
    
    # Map cluster ID to label string and integer class
    label_map = {
        cluster_means[0][0]: ("Elite", 2),
        cluster_means[1][0]: ("Intermediate", 1),
        cluster_means[2][0]: ("Beginner", 0)
    }
    
    y = np.zeros(len(clusters), dtype=int)
    y_names = []
    print("\n Cluster Assignments:")
    for i in range(len(clusters)):
        c_id = clusters[i]
        label_name, class_idx = label_map[c_id]
        y[i] = class_idx
        y_names.append(label_name)
        print(f"   {athletes[i]:<15} -> {label_name}")
        
    df['Skill_Level'] = y_names
    df.to_csv(OUTPUT_DIR / "labeled_athletes.csv", index=False)
    
    # 4. Leave-One-Out Cross-Validation (LOOCV)
    print(f"\n Running Leave-One-Out Cross-Validation on {len(X_scaled)} samples...")
    loo = LeaveOneOut()
    
    y_true = []
    y_pred_rf = []
    rf_feature_importances = np.zeros(X_scaled.shape[1])
    
    for train_index, test_index in loo.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Predict
        pred = rf.predict(X_test)[0]
        
        y_true.append(y_test[0])
        y_pred_rf.append(pred)
        
        # Accumulate feature importances
        rf_feature_importances += rf.feature_importances_
        
    accuracy = accuracy_score(y_true, y_pred_rf)
    print(f"\n Random Forest LOOCV Accuracy: {accuracy:.2%}")
    
    # 5. Final Model Retraining (on all data for saving)
    rf_final = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_final.fit(X_scaled, y)
    
    # 6. Evaluation & Plots
    target_names = ['Beginner', 'Intermediate', 'Elite']
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_rf, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Random Forest - LOOCV Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f" Saved Confusion Matrix: {cm_path}")
    
    # Feature Importances (Average over folds)
    avg_importances = rf_feature_importances / len(X_scaled)
    indices = np.argsort(avg_importances)[::-1][:10] # Top 10
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(10), avg_importances[indices])
    plt.xticks(range(10), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title('Top 10 Biomechanical Sprint Features (Multimodal)')
    plt.xlabel('Features')
    plt.ylabel('Relative Importance')
    plt.tight_layout()
    fi_path = OUTPUT_DIR / "feature_importance.png"
    plt.savefig(fi_path, dpi=150, bbox_inches='tight')
    print(f" Saved Feature Importances: {fi_path}")
    
    # 7. Save Model
    model_path = OUTPUT_DIR / "sprint_model_rf.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': rf_final,
            'scaler': scaler,
            'feature_names': feature_names,
            'label_map': {v[1]: v[0] for k, v in label_map.items()} # {2: 'Elite', ...}
        }, f)
    print(f" Saved Final Model: {model_path}")

if __name__ == "__main__":
    train_and_evaluate()
