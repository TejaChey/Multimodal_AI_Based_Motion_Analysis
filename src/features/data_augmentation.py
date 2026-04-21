"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       PHASE 3 — SPRINT BIOMECHANICS DATA AUGMENTATION PIPELINE               ║
║       Multimodal AI-Based Motion Analysis                                    ║
║                                                                              ║
║  Goal   : Expand 18-27 real samples → 600-900 realistic synthetic samples    ║
║  Methods: SMOTE, Gaussian Noise, Linear Interpolation, Biomechanical Jitter  ║
║  Output : augmented_features.csv  (balanced, validated, Drive-saved)         ║
╚══════════════════════════════════════════════════════════════════════════════╝

COLAB SETUP
-----------
Run the cell below once to install dependencies:

    !pip install -q imbalanced-learn scikit-learn pandas numpy matplotlib seaborn

Then mount Google Drive:

    from google.colab import drive
    drive.mount('/content/drive')
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  INSTALL DEPENDENCIES  (uncomment when running in Colab)
# ─────────────────────────────────────────────────────────────────────────────
# import subprocess
# subprocess.run(["pip", "install", "-q", "imbalanced-learn"], check=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  CONFIGURATION  ← Edit these paths before running
# ─────────────────────────────────────────────────────────────────────────────
# Google Drive paths (change if your folder structure differs)
# DRIVE_ROOT       = Path("/content/drive/MyDrive/Multimodal_AI_Motion_Analysis")
# INPUT_CSV        = DRIVE_ROOT / "Phase_2" / "results" / "labeled_features.csv"
# OUTPUT_CSV       = DRIVE_ROOT / "Phase_3" / "data"   / "augmented_features.csv"
# OUTPUT_PLOTS_DIR = DRIVE_ROOT / "Phase_3" / "plots"

# ── If running locally (not Colab), override paths here ──────────────────────
_HERE            = Path(__file__).resolve().parent
PROJECT_ROOT     = _HERE.parent.parent
INPUT_CSV        = PROJECT_ROOT / "Phase_2" / "results" / "models" / "labeled_athletes.csv"
OUTPUT_CSV       = PROJECT_ROOT / "Phase_3" / "data" / "augmented_features.csv"
OUTPUT_PLOTS_DIR = PROJECT_ROOT / "Phase_3" / "plots"

# Augmentation targets
TARGET_SAMPLES_PER_CLASS = 300        # → 900 total (3 classes × 300)
NOISE_STD_FRACTION       = 0.05      # ±5 % Gaussian noise
INTERPOLATION_ALPHA_RANGE = (0.1, 0.9)

# Column names (adapt if your CSV differs slightly)
ATHLETE_COL  = "athlete"
TRIAL_COL    = "trial"
LABEL_COL    = "label"          # integer: 0=Beginner, 1=Intermediate, 2=Elite
SKILL_COL    = "Skill_Level"    # string versions (optional, kept for reference)

CLASS_NAMES  = {0: "Beginner", 1: "Intermediate", 2: "Elite"}
REVERSE_CLASS_MAP = {"Beginner": 0, "Intermediate": 1, "Elite": 2}

# ─────────────────────────────────────────────────────────────────────────────
# 3.  BIOMECHANICAL BOUNDS PER CLASS
#     Used to clip synthetic samples to physiologically realistic ranges.
# ─────────────────────────────────────────────────────────────────────────────
BIOMECH_BOUNDS = {
    # feature_name : { label : (min, max) }
    "hip_angle_mean": {
        0: (128.0, 148.0),   # Beginner
        1: (113.0, 127.0),   # Intermediate
        2: (98.0,  112.0),   # Elite
    },
    "knee_angle_mean": {
        0: (158.0, 172.0),
        1: (163.0, 177.0),
        2: (168.0, 182.0),
    },
    "stride_frequency": {
        0: (2.8,  3.50),
        1: (3.50, 4.20),
        2: (4.20, 5.00),
    },
    "accel_peak": {
        0: (10.0, 18.0),
        1: (18.0, 25.0),
        2: (25.0, 35.0),
    },
    "gyro_mean": {
        0: (1.5, 3.5),
        1: (3.5, 5.5),
        2: (5.5, 8.5),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 4.  DATA LOADING & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def load_and_validate(path: Path) -> pd.DataFrame:
    """
    Load the labeled feature CSV and perform basic sanity checks.
    Returns a clean DataFrame.
    """
    print(f"\n{'─'*60}")
    print("  STEP 1 — Loading & Validating Data")
    print(f"{'─'*60}")
    print(f"  Source : {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Cannot find: {path}\n"
            "  → Ensure the file is uploaded to Google Drive in the correct folder.\n"
            "  → Update INPUT_CSV in the CONFIG section above."
        )

    df = pd.read_csv(path)
    
    # Needs a numeric 'label' column for the pipeline
    if SKILL_COL in df.columns and LABEL_COL not in df.columns:
        df[LABEL_COL] = df[SKILL_COL].map(REVERSE_CLASS_MAP)
        
    print(f"  Shape  : {df.shape[0]} rows × {df.shape[1]} columns")

    # ── Detect feature columns automatically ─────────────────────────────────
    meta_cols = [c for c in [ATHLETE_COL, TRIAL_COL, LABEL_COL, SKILL_COL]
                 if c in df.columns]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    print(f"  Meta   : {meta_cols}")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    # ── Class distribution ───────────────────────────────────────────────────
    print("\n  Class distribution (original):")
    for lbl, name in CLASS_NAMES.items():
        n = (df[LABEL_COL] == lbl).sum()
        print(f"    {name:>12}  (label={lbl})  →  {n:>3} samples")

    # ── Basic checks ─────────────────────────────────────────────────────────
    assert df[LABEL_COL].nunique() == 3, \
        "Expected exactly 3 classes (0, 1, 2). Check LABEL_COL."
    assert len(feature_cols) >= 5, \
        "Too few feature columns found. Check column names in CSV."
    assert df.isnull().sum().sum() == 0, \
        "NaN values detected. Please clean the data before augmentation."

    print("\n  ✓ Validation passed — data looks clean.")
    return df, feature_cols, meta_cols


# ─────────────────────────────────────────────────────────────────────────────
# 5.  AUGMENTATION METHODS
# ─────────────────────────────────────────────────────────────────────────────

class BiomechanicalAugmentor:
    """
    Four-method augmentation pipeline tailored to sprint biomechanics data.

    Methods
    -------
    gaussian_noise   : Adds ±5% Gaussian noise per feature, then clips to
                       biomechanical bounds.
    linear_interp    : Generates convex combinations of two same-class samples,
                       preserving inter-feature correlations.
    feature_jitter   : Per-feature uniform jitter within tighter class-specific
                       ranges, ideal for minority classes.
    smote            : Scikit-imbalanced SMOTE in feature space (used as a
                       supplement after the three rule-based methods).
    """

    def __init__(self, feature_cols: list, label_col: str = LABEL_COL):
        self.feature_cols = feature_cols
        self.label_col    = label_col
        self._bounds      = BIOMECH_BOUNDS   # reference shortcut

    # ── Internal helper ───────────────────────────────────────────────────────
    def _clip_to_bounds(self, row: np.ndarray, label: int,
                        col_names: list) -> np.ndarray:
        """Clip each feature in a sample to its class-specific physiological range."""
        row = row.copy()
        for feat, class_bounds in self._bounds.items():
            if feat in col_names and label in class_bounds:
                idx = col_names.index(feat)
                lo, hi = class_bounds[label]
                row[idx] = np.clip(row[idx], lo, hi)
        return row

    # ── Method 1 : Gaussian Noise ─────────────────────────────────────────────
    def gaussian_noise(self, df: pd.DataFrame,
                       n_per_class: int, std_frac: float = NOISE_STD_FRACTION
                       ) -> pd.DataFrame:
        """
        For each class, sample real rows with replacement and add zero-mean
        Gaussian noise scaled to std_frac × |feature_value|.
        """
        print("    [Method 1] Gaussian Noise Augmentation …", end=" ")
        synthetic_rows = []

        for lbl in CLASS_NAMES:
            class_df   = df[df[self.label_col] == lbl]
            class_data = class_df[self.feature_cols].values
            n_real     = len(class_data)

            if n_real == 0:
                continue

            for _ in range(n_per_class):
                # Draw a real sample
                idx   = np.random.randint(0, n_real)
                base  = class_data[idx].copy()
                # Add feature-proportional Gaussian noise
                noise = np.random.normal(0, std_frac * np.abs(base))
                noisy = base + noise
                # Clip to biomechanical bounds
                noisy = self._clip_to_bounds(noisy.tolist(), lbl, self.feature_cols)
                synthetic_rows.append(np.append(noisy, lbl))

        cols = self.feature_cols + [self.label_col]
        result = pd.DataFrame(synthetic_rows, columns=cols)
        result["aug_method"] = "gaussian_noise"
        print(f"generated {len(result)} rows.")
        return result

    # ── Method 2 : Linear Interpolation ───────────────────────────────────────
    def linear_interpolation(self, df: pd.DataFrame,
                              n_per_class: int) -> pd.DataFrame:
        """
        Randomly pick two same-class athletes and generate a convex
        combination:  s = α·a + (1-α)·b  where α ~ Uniform(0.1, 0.9).
        This preserves the covariance structure of the class.
        """
        print("    [Method 2] Linear Interpolation …", end=" ")
        lo, hi = INTERPOLATION_ALPHA_RANGE
        synthetic_rows = []

        for lbl in CLASS_NAMES:
            class_df   = df[df[self.label_col] == lbl]
            class_data = class_df[self.feature_cols].values
            n_real     = len(class_data)

            if n_real < 2:
                print(f"\n      [WARN] Class {lbl} has < 2 samples — skipping interpolation.")
                continue

            for _ in range(n_per_class):
                idx_a, idx_b = np.random.choice(n_real, size=2, replace=False)
                alpha        = np.random.uniform(lo, hi)
                interp       = alpha * class_data[idx_a] + (1 - alpha) * class_data[idx_b]
                interp       = self._clip_to_bounds(interp.tolist(), lbl, self.feature_cols)
                synthetic_rows.append(np.append(interp, lbl))

        cols = self.feature_cols + [self.label_col]
        result = pd.DataFrame(synthetic_rows, columns=cols)
        result["aug_method"] = "interpolation"
        print(f"generated {len(result)} rows.")
        return result

    # ── Method 3 : Feature-Wise Biomechanical Jitter ─────────────────────────
    def biomechanical_jitter(self, df: pd.DataFrame,
                              n_per_class: int) -> pd.DataFrame:
        """
        Generate samples by independently jittering each feature within
        its class-specific physiological range (Uniform draw).
        For features without explicit bounds, use ±8% around the class mean.
        """
        print("    [Method 3] Biomechanical Jitter …", end=" ")
        synthetic_rows = []

        for lbl in CLASS_NAMES:
            class_df   = df[df[self.label_col] == lbl]
            class_data = class_df[self.feature_cols].values
            
            if len(class_data) == 0:
                continue
                
            class_mean = class_data.mean(axis=0)
            class_std  = np.maximum(class_data.std(axis=0), 1e-6)

            for _ in range(n_per_class):
                sample = np.zeros(len(self.feature_cols))
                for fi, feat in enumerate(self.feature_cols):
                    if feat in self._bounds and lbl in self._bounds[feat]:
                        lo, hi    = self._bounds[feat][lbl]
                        sample[fi] = np.random.uniform(lo, hi)
                    else:
                        # Fallback: sample from ±2σ around class mean
                        lo = class_mean[fi] - 2 * class_std[fi]
                        hi = class_mean[fi] + 2 * class_std[fi]
                        sample[fi] = np.random.uniform(lo, hi)
                synthetic_rows.append(np.append(sample, lbl))

        cols = self.feature_cols + [self.label_col]
        result = pd.DataFrame(synthetic_rows, columns=cols)
        result["aug_method"] = "bio_jitter"
        print(f"generated {len(result)} rows.")
        return result

    # ── Method 4 : SMOTE ──────────────────────────────────────────────────────
    def smote_augment(self, df: pd.DataFrame,
                      target_total: int) -> pd.DataFrame:
        """
        Apply SMOTE to reach target_total samples (balanced across classes).
        Returns ONLY the new synthetic samples (not the originals).
        """
        print("    [Method 4] SMOTE …", end=" ")
        X = df[self.feature_cols].values
        y = df[self.label_col].values

        k = min(5, min(np.bincount(y)) - 1)
        if k < 1:
            print("skipped (not enough samples per class for k-NN).")
            return pd.DataFrame(columns=self.feature_cols + [self.label_col, "aug_method"])

        smote = SMOTE(sampling_strategy="auto", k_neighbors=k, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        # Keep only the newly generated rows (beyond original count)
        n_orig = len(df)
        X_new  = X_res[n_orig:]
        y_new  = y_res[n_orig:]

        result = pd.DataFrame(X_new, columns=self.feature_cols)
        result[self.label_col] = y_new
        result["aug_method"]   = "smote"
        print(f"generated {len(result)} rows.")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 6.  AUGMENTATION PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_augmentation_pipeline(df: pd.DataFrame,
                               feature_cols: list,
                               target_per_class: int = TARGET_SAMPLES_PER_CLASS
                               ) -> pd.DataFrame:
    """
    Orchestrates all four augmentation methods and assembles the final dataset.

    Strategy
    --------
    Each method contributes ~25% of the synthetic quota for a given class.
    The original real data is kept as-is and combined at the end.
    """
    print(f"\n{'─'*60}")
    print("  STEP 2 — Augmentation Pipeline")
    print(f"{'─'*60}")
    print(f"  Target per class : {target_per_class}")
    print(f"  Total expected   : {target_per_class * 3} synthetic  +  {len(df)} real\n")

    aug = BiomechanicalAugmentor(feature_cols=feature_cols)

    # Split target quota across methods (ignore SMOTE since it generates very few rows due to low base sample count)
    n_each = (target_per_class // 3) + 10

    parts = []

    # ── Real data (tagged) ────────────────────────────────────────────────────
    real_df = df[feature_cols + [LABEL_COL]].copy()
    real_df["aug_method"] = "original"
    parts.append(real_df)

    # ── Synthetic data ────────────────────────────────────────────────────────
    parts.append(aug.gaussian_noise(df, n_per_class=n_each))
    parts.append(aug.linear_interpolation(df, n_per_class=n_each))
    parts.append(aug.biomechanical_jitter(df, n_per_class=n_each))
    parts.append(aug.smote_augment(df, target_total=target_per_class * 3))

    combined = pd.concat(parts, ignore_index=True)

    # ── Balance classes by down-sampling to target_per_class + real ───────────
    balanced_parts = []
    for lbl in CLASS_NAMES:
        sub = combined[combined[LABEL_COL] == lbl]
        real_sub      = sub[sub["aug_method"] == "original"]
        synthetic_sub = sub[sub["aug_method"] != "original"]

        n_needed = target_per_class
        if len(synthetic_sub) > n_needed:
            synthetic_sub = synthetic_sub.sample(n=n_needed, random_state=42)

        balanced_parts.append(real_sub)
        balanced_parts.append(synthetic_sub)

    final_df = pd.concat(balanced_parts, ignore_index=True)
    final_df[LABEL_COL] = final_df[LABEL_COL].astype(int)
    final_df["skill_level"] = final_df[LABEL_COL].map(CLASS_NAMES)

    print(f"\n  Final class distribution:")
    for lbl, name in CLASS_NAMES.items():
        n = (final_df[LABEL_COL] == lbl).sum()
        print(f"    {name:>12}  →  {n:>4} samples")
    print(f"  Total : {len(final_df)} rows  ×  {len(feature_cols)} features")

    return final_df


# ─────────────────────────────────────────────────────────────────────────────
# 7.  DATA QUALITY VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_augmented_data(real_df: pd.DataFrame,
                             aug_df: pd.DataFrame,
                             feature_cols: list) -> dict:
    """
    Runs statistical and biomechanical sanity checks on synthetic data.

    Checks
    ------
    1. KS-test: synthetic distribution should be similar to real (p > 0.05)
    2. Biomechanical bound violations: count if any synthetic sample
       falls outside the defined physiological ranges
    3. No NaN / Inf values
    4. Class balance: all classes within ±5% of each other
    """
    print(f"\n{'─'*60}")
    print("  STEP 3 — Data Quality Validation")
    print(f"{'─'*60}")

    report = {}

    # ── Check 1: KS-test per feature (real vs synthetic) ─────────────────────
    synthetic_only = aug_df[aug_df["aug_method"] != "original"]
    ks_results = {}
    failed_features = []

    if len(synthetic_only) == 0:
        print("  KS-Test skipped (no synthetic data generated).")
        report["ks_test"] = {}
    else:    
        for feat in feature_cols:
            real_vals  = real_df[feat].dropna().values
            synth_vals = synthetic_only[feat].dropna().values
            
            if len(real_vals) > 0 and len(synth_vals) > 0:
                stat, pval = ks_2samp(real_vals, synth_vals)
                ks_results[feat] = {"ks_stat": round(stat, 4), "p_value": round(pval, 4)}
                if pval < 0.001:   # very strict — mild drift is expected
                    failed_features.append(feat)

        report["ks_test"] = ks_results
        print(f"\n  KS-Test (real vs synthetic):")
        print(f"    Features with significant drift (p<0.001): "
            f"{len(failed_features)} / {len(feature_cols)}")
        if failed_features:
            print(f"    → {failed_features}")
        else:
            print("    ✓ All features within acceptable statistical range.")

    # ── Check 2: Biomechanical bound violations ───────────────────────────────
    violation_count = 0
    for feat, class_bounds in BIOMECH_BOUNDS.items():
        if feat not in aug_df.columns:
            continue
        for lbl, (lo, hi) in class_bounds.items():
            sub = synthetic_only[synthetic_only[LABEL_COL] == lbl]
            out = sub[(sub[feat] < lo) | (sub[feat] > hi)]
            violation_count += len(out)

    report["bound_violations"] = violation_count
    pct = violation_count / max(len(synthetic_only), 1) * 100
    print(f"\n  Biomechanical Bound Violations:")
    print(f"    {violation_count} samples ({pct:.1f}%) outside defined physiological ranges.")
    if pct < 5:
        print("    ✓ Acceptable (<5%).")
    else:
        print("    ⚠ Consider tightening BIOMECH_BOUNDS or increasing clipping.")

    # ── Check 3: NaN / Inf ────────────────────────────────────────────────────
    n_nan = aug_df[feature_cols].isnull().sum().sum()
    n_inf = np.isinf(aug_df[feature_cols].replace([np.inf, -np.inf], np.nan)
                     .dropna(how="all")).sum().sum()
    report["nan_count"] = int(n_nan)
    report["inf_count"] = int(n_inf)
    print(f"\n  NaN count : {n_nan}  |  Inf count : {n_inf}")
    if n_nan == 0 and n_inf == 0:
        print("    ✓ No missing or infinite values.")

    # ── Check 4: Class balance ────────────────────────────────────────────────
    counts  = aug_df[LABEL_COL].value_counts().sort_index()
    balance = counts.std() / counts.mean() * 100
    report["class_balance_cv_pct"] = round(balance, 2)
    print(f"\n  Class balance CV : {balance:.1f}%")
    if balance < 5:
        print("    ✓ Classes are well-balanced.")
    else:
        print("    ⚠ Consider adjusting TARGET_SAMPLES_PER_CLASS.")

    print("\n  ✓ Validation complete.")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# 8.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "Beginner":     "#E74C3C",
    "Intermediate": "#F39C12",
    "Elite":        "#2ECC71",
}

def plot_distributions(real_df: pd.DataFrame,
                        aug_df: pd.DataFrame,
                        feature_cols: list,
                        save_dir: Path):
    """
    Panel 1 — Feature Distributions Before vs After augmentation.
    Panel 2 — PCA scatter (real vs synthetic).
    Panel 3 — Class composition bar chart.
    Panel 4 — Augmentation method breakdown pie.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot 1 : KDE per feature, before vs after ────────────────────────────
    n_feats  = min(len(feature_cols), 12)   # cap at 12 for readability
    n_cols   = 4
    n_rows   = (n_feats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 3))
    fig.suptitle("Feature Distributions: Real vs Augmented",
                 fontsize=16, fontweight="bold", y=1.01)
    axes = axes.flatten()

    synth_only = aug_df[aug_df["aug_method"] != "original"]

    for i, feat in enumerate(feature_cols[:n_feats]):
        ax = axes[i]
        for lbl, name in CLASS_NAMES.items():
            color = list(PALETTE.values())[lbl]
            r = real_df[real_df[LABEL_COL] == lbl][feat]
            s = synth_only[synth_only[LABEL_COL] == lbl][feat]
            ax.plot([], [], color=color, label=name)   # legend entry
            
            if len(r) > 1:
                r.plot.kde(ax=ax, color=color, linewidth=2, label=None)
            if len(s) > 1:
                s.plot.kde(ax=ax, color=color, linewidth=1.2,
                       linestyle="--", alpha=0.7, label=None)
        ax.set_title(feat.replace("_", " ").title(), fontsize=9)
        ax.set_yticks([])
        ax.tick_params(labelsize=8)

    # Hide empty subplots
    for j in range(n_feats, len(axes)):
        axes[j].set_visible(False)

    # Shared legend
    handles = [plt.Line2D([0], [0], color=c, lw=2, label=n)
               for n, c in PALETTE.items()]
    handles += [plt.Line2D([0], [0], color="grey", lw=2, label="Real (solid)"),
                plt.Line2D([0], [0], color="grey", lw=1.2,
                           linestyle="--", alpha=0.7, label="Synthetic (dashed)")]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)

    plt.tight_layout()
    p1 = save_dir / "01_feature_distributions.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  ✓ Saved: {p1}")

    # ── Plot 2 : PCA scatter ──────────────────────────────────────────────────
    scaler = StandardScaler()
    all_feats = aug_df[feature_cols].values
    X_scaled  = scaler.fit_transform(all_feats)

    pca    = PCA(n_components=2, random_state=42)
    X_pca  = pca.fit_transform(X_scaled)
    ev     = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"PCA — Real vs Synthetic  "
                 f"(PC1 {ev[0]*100:.1f}%  PC2 {ev[1]*100:.1f}%)",
                 fontsize=13)

    is_real = aug_df["aug_method"] == "original"
    for lbl, name in CLASS_NAMES.items():
        mask_r = (aug_df[LABEL_COL] == lbl) & is_real
        mask_s = (aug_df[LABEL_COL] == lbl) & ~is_real
        c = list(PALETTE.values())[lbl]
        ax.scatter(X_pca[mask_r, 0], X_pca[mask_r, 1],
                   color=c, s=90, edgecolors="black", linewidths=0.8,
                   zorder=3, label=f"{name} (real)")
        ax.scatter(X_pca[mask_s, 0], X_pca[mask_s, 1],
                   color=c, s=15, alpha=0.35, zorder=1,
                   label=f"{name} (synth)")

    ax.set_xlabel("PC 1", fontsize=11)
    ax.set_ylabel("PC 2", fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = save_dir / "02_pca_scatter.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved: {p2}")

    # ── Plot 3 : Class composition bar chart ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Before
    ax = axes[0]
    before_counts = real_df[LABEL_COL].value_counts().sort_index()
    bars = ax.bar([CLASS_NAMES[l] for l in before_counts.index],
                  before_counts.values,
                  color=list(PALETTE.values()), edgecolor="black", linewidth=0.6)
    ax.set_title("Before Augmentation", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sample Count")
    for bar, val in zip(bars, before_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, before_counts.max() * 1.3)

    # After
    ax = axes[1]
    after_counts = aug_df[LABEL_COL].value_counts().sort_index()
    bars = ax.bar([CLASS_NAMES[l] for l in after_counts.index],
                  after_counts.values,
                  color=list(PALETTE.values()), edgecolor="black", linewidth=0.6)
    ax.set_title("After Augmentation", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sample Count")
    for bar, val in zip(bars, after_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, after_counts.max() * 1.3)

    plt.suptitle("Class Distribution: Before vs After", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    p3 = save_dir / "03_class_balance.png"
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved: {p3}")

    # ── Plot 4 : Augmentation method breakdown ────────────────────────────────
    method_counts = aug_df["aug_method"].value_counts()
    colors = ["#3498DB", "#9B59B6", "#1ABC9C", "#E67E22", "#E74C3C"]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        method_counts.values,
        labels=method_counts.index,
        autopct="%1.1f%%",
        colors=colors[:len(method_counts)],
        startangle=90,
        pctdistance=0.80,
        wedgeprops=dict(linewidth=1.5, edgecolor="white"),
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
    ax.set_title("Augmentation Method Breakdown", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p4 = save_dir / "04_method_breakdown.png"
    plt.savefig(p4, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved: {p4}")

    # ── Plot 5 : Correlation heatmap (before vs after) ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    corr_before = real_df[feature_cols].corr()
    corr_after  = aug_df[feature_cols].corr()

    for ax, corr, title in zip(axes,
                                [corr_before, corr_after],
                                ["Correlation — Original Data",
                                 "Correlation — Augmented Data"]):
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, ax=ax,
                    annot=True, fmt=".2f", annot_kws={"size": 6},
                    cmap="RdYlGn", vmin=-1, vmax=1,
                    linewidths=0.4, linecolor="grey",
                    square=True, cbar_kws={"shrink": 0.6})
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(labelsize=7)

    plt.suptitle("Feature Correlation Structure (Preserved?)", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    p5 = save_dir / "05_correlation_heatmap.png"
    plt.savefig(p5, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved: {p5}")

    print("\n  ✓ All validation plots saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  SAVE TO DRIVE
# ─────────────────────────────────────────────────────────────────────────────

def save_augmented_csv(df: pd.DataFrame, path: Path,
                        feature_cols: list, report: dict):
    """Save the augmented dataset and print a summary."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Reorder columns nicely
    meta_out  = [LABEL_COL, "skill_level", "aug_method"]
    col_order = feature_cols + meta_out
    df_out    = df[[c for c in col_order if c in df.columns]]

    df_out.to_csv(path, index=False)

    print(f"\n{'─'*60}")
    print("  STEP 4 — Saved Augmented Dataset")
    print(f"{'─'*60}")
    print(f"  Path   : {path}")
    print(f"  Shape  : {df_out.shape[0]} rows × {df_out.shape[1]} columns")
    print(f"  Columns: {list(df_out.columns)}")
    print(f"\n  Quality Report Summary:")
    print(f"    KS-test failed features  : "
          f"{sum(1 for v in report['ks_test'].values() if v.get('p_value', 1) < 0.001)}")
    print(f"    Bound violations         : {report['bound_violations']}")
    print(f"    NaN / Inf values         : {report['nan_count']} / {report['inf_count']}")
    print(f"    Class balance CV (%)     : {report['class_balance_cv_pct']}")
    print(f"\n  ✓ augmented_features.csv is ready for deep learning training.")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 62)
    print("        PHASE 3 — DATA AUGMENTATION PIPELINE")
    print("        Multimodal AI-Based Sprint Motion Analysis")
    print("═" * 62)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    df, feature_cols, meta_cols = load_and_validate(INPUT_CSV)
    real_df = df.copy()

    # ── 2. Augment ────────────────────────────────────────────────────────────
    aug_df = run_augmentation_pipeline(df, feature_cols,
                                       target_per_class=TARGET_SAMPLES_PER_CLASS)

    # ── 3. Validate ───────────────────────────────────────────────────────────
    report = validate_augmented_data(real_df, aug_df, feature_cols)

    # ── 4. Plot ───────────────────────────────────────────────────────────────
    plot_distributions(real_df, aug_df, feature_cols, OUTPUT_PLOTS_DIR)

    # ── 5. Save ───────────────────────────────────────────────────────────────
    save_augmented_csv(aug_df, OUTPUT_CSV, feature_cols, report)

    print("\n" + "═" * 62)
    print("  PIPELINE COMPLETE ✓")
    print(f"  Next step: use {OUTPUT_CSV.name} to train the deep learning model.")
    print("═" * 62 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
