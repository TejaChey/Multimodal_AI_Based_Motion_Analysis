import streamlit as st
import tempfile
import time
from pathlib import Path

from inference import run_inference
from coaching_engine import generate_coaching_feedback

# ── 1. DASHBOARD CONFIGURATION ───────────────────────────────────────────────
st.set_page_config(page_title="Multimodal Sprint Analytics", layout="wide")

# Custom Premium Dark Theme CSS
st.markdown("""
<style>
    /* Global Typography and Beautiful Background Gradient */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    .stApp {
        background: radial-gradient(ellipse at top, #11142b, #000000 80%) !important;
        background-attachment: fixed !important;
        color: #f5f5f7;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Typography with subtle metallic gradient for headers */
    h1, h2, h3, h4 {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-weight: 600 !important;
        letter-spacing: -0.015em;
        background: linear-gradient(180deg, #ffffff 0%, #a0a0a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Streamlit Metric Boxes overrides: Premium Glassmorphism */
    div[data-testid="metric-container"] {
        background: rgba(30, 30, 36, 0.4) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 18px !important;
        padding: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        transition: transform 0.25s ease-out, background 0.25s ease-out;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        background: rgba(40, 40, 46, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }
    
    /* Fix text coloring inside metrics (so gradient doesn't override values) */
    div[data-testid="metric-container"] label {
        color: #a0a0a5 !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* Custom Coaching Note Panels */
    .coaching-tip {
        background: rgba(30, 30, 36, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 22px;
        border-radius: 18px;
        margin-top: 15px;
        margin-bottom: 8px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid #0A84FF;
        font-size: 16px;
        line-height: 1.6;
        color: #f5f5f7;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    /* Button overrides to look like Premium Apple Control */
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #0A84FF, #0055B3) !important;
        color: #ffffff !important;
        border-radius: 980px !important;
        font-weight: 500 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 4px 15px rgba(10, 132, 255, 0.3) !important;
        transition: transform 0.2s ease, filter 0.2s ease;
    }
    
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        transform: scale(1.02);
        filter: brightness(1.1);
    }
</style>
""", unsafe_allow_html=True)

# ── 2. UI LAYOUT ─────────────────────────────────────────────────────────────
st.title("Multimodal AI Sprint Analytics")
st.markdown("Upload your raw **15-meter Fly Zone Video** and synchronized **IMU Sensor** data below to instantly classify athletic skill level and generate personalized biomechanical coaching.")

st.markdown("---")
# Upload Zone
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Primary Modality (Vision)")
    video_file = st.file_uploader("Upload Athlete Video (.mp4)", type=["mp4", "mov", "avi"])

with col2:
    st.markdown("### Secondary Modality (Sensors)")
    st.caption("Galaxy Watch exports 2 files — upload both below:")
    accel_file = st.file_uploader("Accelerometer CSV", type=["csv"], key="accel")
    gyro_file  = st.file_uploader("Gyroscope CSV", type=["csv"], key="gyro")

# ── 3. EXECUTION LOGIC ───────────────────────────────────────────────────────
if video_file and accel_file and gyro_file:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Giant Execution Button
    if st.button("INITIALIZE AI INFERENCE", type="primary", use_container_width=True):
        
        # Loader
        with st.status("Initializing Multimodal Analysis...", expanded=True) as status:
            
            st.write("Loading temp files into memory...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tv:
                tv.write(video_file.read())
                temp_video_path = Path(tv.name)
                
            with tempfile.NamedTemporaryFile(delete=False, suffix="_accel.csv") as ta:
                ta.write(accel_file.read())
                temp_accel_path = Path(ta.name)

            with tempfile.NamedTemporaryFile(delete=False, suffix="_gyro.csv") as tg:
                tg.write(gyro_file.read())
                temp_gyro_path = Path(tg.name)
            
            st.write("Extracting MediaPipe Pose Landmarks & IMU Features...")
            # Run the actual deep learning bridged python script
            label, metrics, error = run_inference(temp_video_path, temp_accel_path, temp_gyro_path)
            
            if error:
                status.update(label="Critical System Error", state="error", expanded=True)
                st.error(error)
            else:
                st.write("Feeding Kinematics to Deep Learning Classifier...")
                status.update(label="Analysis Pipeline Complete", state="complete", expanded=False)
                
            # Cleanup
            temp_video_path.unlink(missing_ok=True)
            temp_accel_path.unlink(missing_ok=True)
            temp_gyro_path.unlink(missing_ok=True)

        # ── 4. RESULTS DASHBOARD (Rendered outside the collapsed status box) ──
        if not error and label and metrics:
            st.markdown("---")
            
            # Skill Badge Banner
            st.markdown(f"## Predicted Athlete Skill Class: <span style='color:#000000; background:#ffffff; padding:5px 15px; border-radius:12px;'>{label.upper()}</span>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            dash_col1, dash_col2 = st.columns([1, 2])
            
            # Left Column: Raw Metrics
            with dash_col1:
                st.markdown("### Extracted Features")
                st.metric("Stride Freq (Hz)", f"{metrics.get('stride_freq_hz', 0)}")
                st.metric("Peak Accel (m/s²)", f"{metrics.get('peak_accel_mag', 0)}")
                st.metric("Flight Ratio (%)", f"{metrics.get('flight_ratio', 0)*100:.1f}")
                st.metric("Knee Angle Mean (°)", f"{metrics.get('knee_angle_mean', 0)}")
                st.metric("Accel Smoothness", f"{metrics.get('movement_smoothness_accel', 0)}")

            # Right Column: The Coaching Engine Feed
            with dash_col2:
                st.markdown("### Expert Coaching Engine Notes")
                st.write("*Analyzing structural deviations against Elite mathematical profiles...*")
                
                tips = generate_coaching_feedback(metrics)
                for tip in tips:
                    st.markdown(f"<div class='coaching-tip'>{tip}</div>", unsafe_allow_html=True)
