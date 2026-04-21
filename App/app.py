import streamlit as st
import tempfile
import time
from pathlib import Path

from inference import run_inference
from coaching_engine import generate_coaching_feedback

# ── 1. DASHBOARD CONFIGURATION ───────────────────────────────────────────────
st.set_page_config(page_title="Multimodal Sprint Analytics", layout="wide")

# Custom Apple-like Dark Mode CSS Injector
st.markdown("""
<style>
    /* Global Typography and Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    .stApp {
        background-color: #000000;
        color: #f5f5f7;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Typography */
    h1, h2, h3, h4 {
        color: #f5f5f7 !important;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-weight: 600 !important;
        letter-spacing: -0.015em;
    }
    
    /* Streamlit Metric Boxes overrides to look like iOS Widgets */
    div[data-testid="metric-container"] {
        background: #1c1c1e;
        border-radius: 18px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Custom Coaching Note Panels */
    .coaching-tip {
        background: #1c1c1e;
        padding: 20px;
        border-radius: 18px;
        margin-top: 15px;
        margin-bottom: 5px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 16px;
        line-height: 1.5;
        color: #f5f5f7;
    }
    
    /* Button overrides to look like Apple Pill Buttons */
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 980px !important;
        font-weight: 500 !important;
        border: none !important;
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
