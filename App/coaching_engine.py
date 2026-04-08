import pandas as pd
from pathlib import Path

# ── 1. INITIALIZE ELITE BASELINES ──────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
DATA_CSV = PROJECT_ROOT / "Phase_3" / "data" / "augmented_features.csv"

# Pre-calculate Elite baselines on server startup to save runtime latency
try:
    _elite_df = pd.read_csv(DATA_CSV)
    _elite_df = _elite_df[_elite_df['label'] == 2] # 2 = Elite
    ELITE_BASELINES = _elite_df.mean(numeric_only=True).to_dict()
except Exception as e:
    # Failsafe fallback metrics if CSV goes missing
    ELITE_BASELINES = {
        'stride_freq_hz': 4.3,
        'knee_asymmetry': 3.5,
        'flight_ratio': 0.61,
        'vert_oscillation_norm': 0.045,
        'movement_smoothness_accel': -12.0
    }

# ── 2. INFERENCE ENGINE ──────────────────────────────────────────────────────────
def generate_coaching_feedback(user_metrics: dict) -> list[str]:
    """
    Acts as an Expert System Rule-Based AI. Checks the user's specific 23 extracted
    biomechanical features and compares them against the generated Elite class
    averages. Outputs coaching text.
    """
    tips = []
    
    # 1. Stride Frequency (Higher is better)
    elite_sf = ELITE_BASELINES.get('stride_freq_hz', 4.3)
    user_sf  = user_metrics.get('stride_freq_hz', 0)
    if user_sf < elite_sf - 0.2:
        diff = round(elite_sf - user_sf, 2)
        tips.append(f"🏃💨 **Stride Frequency is too low.** (You: {user_sf} Hz | Elite: {elite_sf:.1f} Hz). Try to manually increase your leg turnover rate cadence by {diff} Hz.")
        
    # 2. Asymmetry (Lower is better)
    elite_knee_asym = ELITE_BASELINES.get('knee_asymmetry', 3.5)
    user_knee_asym  = user_metrics.get('knee_asymmetry', 0)
    if user_knee_asym > elite_knee_asym + 3.0:
        tips.append(f"⚖️ **High Knee Asymmetry Detected!** (You: {user_knee_asym:.1f}° diff | Elite: <{elite_knee_asym:.1f}°). You are heavily favoring one leg, which leaks power and causes injury.")

    # 3. Flight Ratio (Higher is better)
    elite_flight = ELITE_BASELINES.get('flight_ratio', 0.6)
    user_flight  = user_metrics.get('flight_ratio', 0)
    if user_flight < elite_flight - 0.05:
        tips.append(f"✈️ **Low Flight Ratio.** (You spent {user_flight*100:.1f}% of the stride in the air | Elite: {elite_flight*100:.1f}%). Focus on explosive, punchy foot strikes away from the ground.")
        
    # 4. Vertical Oscillation (Lower is better)
    elite_osc = ELITE_BASELINES.get('vert_oscillation_norm', 0.045)
    user_osc  = user_metrics.get('vert_oscillation_norm', 0)
    if user_osc > elite_osc + 0.02:
        tips.append(f"🦘 **Too Much Vertical Bouncing.** You are bouncing up and down too much instead of projecting forward. Focus on driving your hips strictly horizontal.")
        
    # 5. Smoothness (Closer to 0 is better, but negative)
    elite_smooth = ELITE_BASELINES.get('movement_smoothness_accel', -12.0)
    user_smooth  = user_metrics.get('movement_smoothness_accel', -20.0)
    if user_smooth < elite_smooth - 5.0:
        tips.append(f"📉 **Jerky Acceleration Profile.** Your acceleration curve is highly jagged compared to Elite models. Ensure your upper body and arm swings are locked and smooth.")

    # 6. Peak Acceleration (Higher is better)
    elite_accel = ELITE_BASELINES.get('peak_accel_mag', 28.0)
    user_accel  = user_metrics.get('peak_accel_mag', 0)
    if user_accel < elite_accel - 4.0:
        tips.append(f"🚀 **Low Peak Force Generation.** (You: {user_accel:.1f} m/s² | Elite: {elite_accel:.1f} m/s²). You lack the raw power output. Incorporate heavy sled pushes and plyometrics.")

    # Fallback if perfect
    if len(tips) == 0:
        tips.append("🌟 **Perfect Biomechanics!** Your mathematical kinematics align completely with our Elite athlete baselines. Maintain this exact form.")
        
    # Return max 3 tips to avoid overwhelming the user interface
    return tips[:3]
