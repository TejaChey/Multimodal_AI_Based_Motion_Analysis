"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       PHASE 3 — DEEP LEARNING ABLATION STUDY (ROTATIONS)                     ║
║       Multimodal AI-Based Motion Analysis                                    ║
║                                                                              ║
║  Goal: Test model robustness via 3 Data Rotations:                           ║
║        A) Synthetic Only (Tested on 100% Original Data)                      ║
║        B) Mixed Data (Tested on 20% Original Data)                           ║
║        C) Original Only (Tested on 20% Original Data)                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# MLOps Setup
import yaml
import mlflow
import mlflow.keras

# Force deterministic behavior
np.random.seed(42)
tf.random.set_seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION (MLOPS)
# ─────────────────────────────────────────────────────────────────────────────
_HERE            = Path(__file__).resolve().parent
PROJECT_ROOT     = _HERE.parent.parent

# Load Hyperparameters from params.yaml
with open(PROJECT_ROOT / "params.yaml", "r") as f:
    params = yaml.safe_load(f)

INPUT_CSV        = PROJECT_ROOT / params['data']['input_csv_path']
OUTPUT_DIR       = PROJECT_ROOT / "Phase_3" / "results" / "model"
PLOTS_DIR        = PROJECT_ROOT / "Phase_3" / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH  = OUTPUT_DIR / "best_sprint_model.keras"
SCALER_SAVE_PATH = OUTPUT_DIR / "fitted_scaler.joblib"

CLASS_NAMES      = {0: "Beginner", 1: "Intermediate", 2: "Elite"}
EXCLUDE_COLS     = params['data']['exclude_cols']

# Hyperparameters mapped dynamically
BATCH_SIZE = params['training']['batch_size']
EPOCHS     = params['training']['epochs']
LR         = params['training']['learning_rate']
L2_REG     = params['training']['l2_regularization']

mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
mlflow.set_experiment(params['mlflow']['experiment_name'])


# ─────────────────────────────────────────────────────────────────────────────
# 2. NEURAL NETWORK ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
def build_mlp_model(input_dim: int) -> Sequential:
    """Builds a fresh, untrained regularized feed-forward neural network."""
    model = Sequential([
        Input(shape=(input_dim,)),
        
        Dense(128, activation='relu', kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(64, activation='relu', kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu', kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(3, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',  
        metrics=['accuracy']
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3. ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    features = [c for c in df.columns if c not in EXCLUDE_COLS]
    
    real_df = df[df["aug_method"] == "original"]
    synth_df = df[df["aug_method"] != "original"]

    # Manually split 3 original athletes for Test Set (for Rotations B & C)
    test_indices = []
    for c in [0, 1, 2]:
        try:
            target_row = real_df[real_df['label'] == c].sample(n=1, random_state=42)
            test_indices.append(target_row.index[0])
        except ValueError:
            pass # Failsafe
            
    real_test_df = real_df.loc[test_indices]
    real_train_df = real_df.drop(test_indices)

    results = {}
    best_overall_acc = 0.0

    print("\n[{}]".format("="*60))
    print(" COMMENCING ABLATION STUDY: 3 DATA ROTATIONS")
    print("[{}]".format("="*60))

    rotations = ["A_Synthetic_Only", "B_Mixed", "C_Original_Only"]

    for rot in rotations:
        print(f"\n>>>> Executing Rotation: {rot} <<<<")
        
        # --- 3.1 Data Splitting Logic ---
        if rot == "A_Synthetic_Only":
            # Train/Val = 80/20 of Synthetic. Test = 100% Real (9 items)
            X_raw = synth_df[features].values
            y = synth_df["label"].values
            X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_raw, y, test_size=0.2, random_state=42)
            
            X_test_raw = real_df[features].values
            y_test = real_df["label"].values
            
        elif rot == "B_Mixed":
            # Train/Val = Mixed. Test = 3 Real items.
            mixed_train_df = pd.concat([synth_df, real_train_df])
            X_raw = mixed_train_df[features].values
            y = mixed_train_df["label"].values
            X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_raw, y, test_size=0.2, random_state=42)
            
            X_test_raw = real_test_df[features].values
            y_test = real_test_df["label"].values

        elif rot == "C_Original_Only":
            # Train = Remaining Real (6 items). Test = 3 Real items.
            # No Val set exists logically due to N=6, so we mirror train to val just to allow Keras compilation.
            # We disable early stopping via the tiny dataset.
            X_train_raw = real_train_df[features].values
            y_train = real_train_df["label"].values
            X_val_raw, y_val = X_train_raw, y_train
            
            X_test_raw = real_test_df[features].values
            y_test = real_test_df["label"].values

        # --- 3.2 Scaling ---
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val   = scaler.transform(X_val_raw)
        X_test  = scaler.transform(X_test_raw)

        # --- 3.3 Modeling ---
        model = build_mlp_model(len(features))
        
        callbacks = []
        if rot != "C_Original_Only":
            callbacks.append(EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0))
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5, verbose=0))
        
        # MLOps: MLflow Tracking Hook
        with mlflow.start_run(run_name=f"rotation_{rot}"):
            # Train
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                      epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0)
            
            # Evaluate
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"   => Test Accuracy : {test_acc * 100:.2f}%")
            print(f"   => Test Loss     : {test_loss:.4f}")
            
            # Log Hyperparameters
            mlflow.log_param("rotation", rot)
            mlflow.log_param("epochs", EPOCHS)
            mlflow.log_param("batch_size", BATCH_SIZE)
            mlflow.log_param("learning_rate", LR)
            
            # Log Metrics
            mlflow.log_metric("test_accuracy", test_acc * 100)
            mlflow.log_metric("test_loss", test_loss)
            
            results[rot] = {"acc": test_acc * 100, "loss": test_loss}

            # Saving best weights and SCALER for real-time web deployment
            if test_acc >= best_overall_acc and rot == "A_Synthetic_Only":
                best_overall_acc = test_acc
                model.save(str(MODEL_SAVE_PATH))
                joblib.dump(scaler, str(SCALER_SAVE_PATH))
                
                # Register the production model to MLflow
                mlflow.keras.log_model(model, artifact_path="model", registered_model_name="Sprint_MLP_Model")


    # ─────────────────────────────────────────────────────────────────────────────
    # 4. PLOTTING RESULTS
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[==================================================]")
    print(" GENERATING ABLATION COMPARISON CHART")
    print("[==================================================]")
    
    bars_acc = [results[r]["acc"] for r in rotations]
    names = ["A:\nSynthetic Only\n(Test N=9)", "B:\nMixed\n(Test N=3)", "C:\nOriginal Only\n(Test N=3)"]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=names, y=bars_acc, palette="coolwarm")
    plt.axhline(66.0, color='red', linestyle='--', label="Random Forest Baseline (66%)")
    plt.title("Ablation Study: Deep Learning Accuracy Across Data Rotations")
    plt.ylabel("Test Accuracy (%)")
    plt.ylim(0, 110)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    for i, acc in enumerate(bars_acc):
        plt.text(i, acc + 2, f"{acc:.1f}%", ha='center', fontweight='bold')

    plt.tight_layout()
    chart_path = PLOTS_DIR / "09_ablation_study_results.png"
    plt.savefig(chart_path, dpi=150)
    print(f" ✓ Saved: {chart_path}")
    print(f" ✓ Web Deployment Scaler Saved: {SCALER_SAVE_PATH}")
