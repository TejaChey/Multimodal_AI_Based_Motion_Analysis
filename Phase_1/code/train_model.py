#!/usr/bin/env python3
"""Train classification model"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

_HERE      = Path(__file__).resolve().parent.parent  # Phase_1/
DATA_PATH  = _HERE / "results" / "ml_data" / "ml_data.pkl"
OUTPUT_DIR = _HERE / "results" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def train_and_evaluate():
    # Load data
    print(" Loading data...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"   Train: {X_train.shape}")
    print(f"   Test:  {X_test.shape}")
    
    # Train Random Forest
    print("\n Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    
    print(f"   Accuracy: {acc_rf:.2%}")
    
    # Train SVM
    print("\n Training SVM...")
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train, y_train)
    
    y_pred_svm = svm.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    
    print(f"   Accuracy: {acc_svm:.2%}")
    
    # Pick best model
    if acc_rf >= acc_svm:
        best_model = rf
        best_name = "Random Forest"
        best_acc = acc_rf
        y_pred = y_pred_rf
    else:
        best_model = svm
        best_name = "SVM"
        best_acc = acc_svm
        y_pred = y_pred_svm
    
    print(f"\nBest model: {best_name} ({best_acc:.2%})")
    
    # Detailed evaluation
    print(f"\n Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Slow', 'Fast']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Slow', 'Fast'], 
                yticklabels=['Slow', 'Fast'])
    plt.title(f'{best_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"\n Saved confusion matrix: {cm_path}")
    
    # Feature importance (if Random Forest)
    if best_name == "Random Forest":
        importances = rf.feature_importances_
        feature_names = data['feature_names']
        
        indices = np.argsort(importances)[::-1][:10]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.title('Top 10 Feature Importances')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        fi_path = OUTPUT_DIR / "feature_importance.png"
        plt.savefig(fi_path, dpi=150, bbox_inches='tight')
        print(f" Saved feature importance: {fi_path}")
    
    # Save model
    model_path = OUTPUT_DIR / "best_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({'model': best_model, 'scaler': data['scaler'], 
                     'feature_names': data['feature_names']}, f)
    
    print(f" Saved model: {model_path}")
    
    plt.show()
    
    return best_model, best_acc

if __name__ == "__main__":
    model, acc = train_and_evaluate()