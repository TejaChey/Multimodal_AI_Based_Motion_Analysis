#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, roc_curve, auc
import scipy.signal

# ── CONFIG ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent

INPUT_CSV = PROJECT_ROOT / "Phase_2" / "results" / "multimodal_features.csv"
IMU_RAW_DIR = PROJECT_ROOT / "Phase_2" / "data" / "sprint_raw"
OUTPUT_DIR = PROJECT_ROOT / "Phase_2" / "results" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define feature subsets
VIDEO_FEATURES = [
    'duration_s', 'total_frames', 'knee_angle_mean', 'hip_angle_mean',
    'knee_asymmetry', 'hip_asymmetry', 'vert_oscillation_norm',
    'total_steps_detected', 'stride_freq_hz', 'stride_len_m',
    'gct_ratio', 'flight_ratio'
]

IMU_FEATURES = [
    'peak_accel_mag', 'mean_accel_mag', 'std_accel_mag', 'stride_rate_accel_hz',
    'movement_smoothness_accel', 'peak_gyro_mag', 'mean_gyro_mag', 'std_gyro_mag',
    'stride_rate_gyro_hz', 'movement_smoothness_gyro', 'steps_per_min_gyro'
]

def map_labels(df):
    X_raw = df.drop(columns=['athlete']).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Map based on stride frequency (higher = Elite)
    stride_idx = df.columns.get_loc('stride_freq_hz') - 1
    means = [(c, X_raw[clusters == c, stride_idx].mean()) for c in range(3)]
    means.sort(key=lambda x: x[1], reverse=True)
    
    label_map = {
        means[0][0]: ("Elite", 2),
        means[1][0]: ("Intermediate", 1),
        means[2][0]: ("Beginner", 0)
    }
    y = np.array([label_map[c][1] for c in clusters])
    return y, label_map

def plot_modality_comparison(df, y):
    print("\n Running Modality Comparison...")
    
    results = {}
    subsets = {
        'Video-Only': VIDEO_FEATURES,
        'IMU-Only': IMU_FEATURES,
        'Multimodal': VIDEO_FEATURES + IMU_FEATURES
    }
    
    loo = LeaveOneOut()
    
    for name, features in subsets.items():
        X = df[features].values
        X_scaled = StandardScaler().fit_transform(X)
        
        y_pred = []
        for train_idx, test_idx in loo.split(X_scaled):
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_scaled[train_idx], y[train_idx])
            y_pred.append(model.predict(X_scaled[test_idx])[0])
            
        results[name] = accuracy_score(y, y_pred)
        print(f"   {name} Accuracy: {results[name]:.2%}")
        
    plt.figure(figsize=(8, 5))
    names = list(results.keys())
    accs = [results[n]*100 for n in names]
    
    bars = plt.bar(names, accs, color=['#3498db', '#e74c3c', '#2ecc71'])
    plt.title('Performance Comparison: Single vs. Multimodal')
    plt.ylabel('LOOCV Accuracy (%)')
    plt.ylim(0, 100)
    
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 2, f'{h:.1f}%', ha='center', fontweight='bold')
        
    out_path = OUTPUT_DIR / 'modality_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f" Saved: {out_path}")

def plot_roc_curve(df, y, label_map):
    print("\n Ploting ROC Curve...")
    
    X = df[VIDEO_FEATURES + IMU_FEATURES].values
    X_scaled = StandardScaler().fit_transform(X)
    
    # Binarize labels for One-vs-Rest ROC
    y_bin = label_binarize(y, classes=[0, 1, 2])
    n_classes = 3
    
    # Use cross_val_predict with LOOCV to get valid prediction probabilities
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    y_score = cross_val_predict(model, X_scaled, y, cv=LeaveOneOut(), method='predict_proba')
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    class_names = {v[1]: v[0] for k, v in label_map.items()} # {0: 'Beginner', 1: 'Intermediate', 2: 'Elite'}
    colors = {0: '#e74c3c', 1: '#f1c40f', 2: '#2ecc71'}
    
    plt.figure(figsize=(8, 6))
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
                 
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve per Class (Random Forest - Multimodal)')
    plt.legend(loc="lower right")
    
    out_path = OUTPUT_DIR / 'roc_curve_multimodal.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f" Saved: {out_path}")

def plot_imu_gait_events():
    print("\n Plotting IMU Gait Events...")
    # Attempt to load Chandra1 for visualization (one of the Elite athletes)
    athlete = "Chandra1"
    imu_file = IMU_RAW_DIR / athlete / "WatchAccelerometer.csv"
    
    if not imu_file.exists():
        print(f" Could not find raw IMU for {athlete}. Skipping gait plot.")
        return
        
    df_imu = pd.read_csv(imu_file)
    
    # Calculate magnitude
    acc_x = df_imu['x'].values
    acc_y = df_imu['y'].values
    acc_z = df_imu['z'].values
    t = df_imu['seconds_elapsed'].values
    t = t - t[0] # Start at 0
    mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    # Smooth signal
    mag_smooth = scipy.signal.savgol_filter(mag, window_length=15, polyorder=3)
    
    # Detect intense sprinting window (peaks)
    peaks, _ = scipy.signal.find_peaks(mag_smooth, distance=10, prominence=5, height=20)
    
    if len(peaks) > 0:
        win_start = max(0, peaks[0] - 20)
        win_end = min(len(t)-1, peaks[-1] + 20)
    else:
        win_start, win_end = 0, len(t)-1
        
    plt.figure(figsize=(12, 5))
    
    # Plot only the active sprint window
    t_win = t[win_start:win_end]
    mag_win = mag_smooth[win_start:win_end]
    
    plt.plot(t_win, mag_win, color='#2c3e50', alpha=0.9, label='Filtered Acceleration Magnitude')
    
    # Re-detect peaks in the isolated window to highlight them
    win_peaks, _ = scipy.signal.find_peaks(mag_win, distance=10, prominence=4, height=30)
    
    plt.plot(t_win[win_peaks], mag_win[win_peaks], "ro", markersize=8, label='Detected Gait Event (Foot Strike)')
    
    plt.title(f'IMU Time-Series with Annotated Gait Events ({athlete})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration (m/s²)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    out_path = OUTPUT_DIR / 'imu_gait_events.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f" Saved: {out_path}")

if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    y, label_map = map_labels(df)
    
    plot_modality_comparison(df, y)
    plot_roc_curve(df, y, label_map)
    plot_imu_gait_events()
    print("\nAdvanced visualizations completed successfully!")
