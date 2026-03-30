#!/usr/bin/env python3
"""Explore the merged features"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

_HERE      = Path(__file__).resolve().parent.parent  # Phase_1/
DATA_PATH  = _HERE / "results" / "multimodal_features" / "multimodal_features.csv"
OUTPUT_DIR = _HERE / "results" / "exploration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def explore():
    # Load data
    df = pd.read_csv(DATA_PATH)
    
    print("="*60)
    print("DATA EXPLORATION")
    print("="*60)
    
    print(f"\nShape: {df.shape}")
    print(f"   Samples: {len(df)}")
    print(f"   Features: {len(df.columns)}")
    
    print(f"\n Columns:")
    print(df.columns.tolist())
    
    print(f"\n First 5 rows:")
    print(df.head())
    
    print(f"\nStatistics:")
    print(df.describe())
    
    print(f"\n Missing values:")
    print(df.isnull().sum())
    
    print(f"\n Subjects: {df['subject'].nunique()}")
    print(f"   Subject IDs: {sorted(df['subject'].unique())}")
    
    print(f"\n Samples per subject:")
    print(df['subject'].value_counts().sort_index())
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Hip angle distribution
    axes[0, 0].hist(df['hip_angle_mean'], bins=15, edgecolor='black')
    axes[0, 0].set_title('Hip Angle Distribution')
    axes[0, 0].set_xlabel('Degrees')
    axes[0, 0].set_ylabel('Count')
    
    # Plot 2: Knee angle distribution
    axes[0, 1].hist(df['knee_angle_mean'], bins=15, edgecolor='black')
    axes[0, 1].set_title('Knee Angle Distribution')
    axes[0, 1].set_xlabel('Degrees')
    axes[0, 1].set_ylabel('Count')
    
    # Plot 3: Correlation heatmap (select numeric columns)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [c for c in numeric_cols if c not in ['subject', 'trial']]
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, ax=axes[1, 0], cmap='coolwarm', center=0, 
                square=True, cbar_kws={'shrink': 0.8})
    axes[1, 0].set_title('Feature Correlation')
    
    # Plot 4: Subject variability
    df.boxplot(column='hip_angle_mean', by='subject', ax=axes[1, 1])
    axes[1, 1].set_title('Hip Angle by Subject')
    axes[1, 1].set_xlabel('Subject')
    axes[1, 1].set_ylabel('Hip Angle (degrees)')
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "data_exploration.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n Saved visualizations: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    explore()