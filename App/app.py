import streamlit as st
import tempfile
import time
from pathlib import Path

from inference import run_inference
from coaching_engine import generate_coaching_feedback

# ── 1. DASHBOARD CONFIGURATION ───────────────────────────────────────────────
st.set_page_config(page_title="AI Sprint Analysis", layout="wide", page_icon="⚡")

# Custom Dark Mode & Glassmorphism CSS Injector
st.markdown("""
<style>
    /* Main Background Override */
    .stApp {
        background-color: #0b0c10;
        color: #c5c6c7;
    }
    
    /* Neon Typography */
    h1, h2, h3, h4 {
        color: #66fcf1 !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    
    /* Streamlit Metric Boxes overrides to look more tactical */
    div[data-testid="metric-container"] {
        background: rgba(31, 40, 51, 0.45);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid rgba(102, 252, 241, 0.3);
        box-shadow: 0 4px 6px rgba(0,0,0,0.4);
    }

    /* Custom Coaching Note Panels */
    .coaching-tip {
        background: rgba(69, 162, 158, 0.15);
        padding: 20px;
        border-radius: 12px;
        margin-top: 15px;
        margin-bottom: 5px;
        border-left: 5px solid #45a29e;
        font-size: 16px;
        line-height: 1.5;
        color: #e0e0e0;
        box-shadow: inset 0px 0px 10px rgba(69,162,158, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ── 2. UI LAYOUT ─────────────────────────────────────────────────────────────
st.title("⚡ Multimodal AI Sprint Analytics")
st.markdown("Upload your raw **15-meter Fly Zone Video** and synchronized **IMU Sensor** data below to instantly classify athletic skill level and generate personalized biomechanical coaching.")

st.markdown("---")
# Upload Zone
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎥 Primary Modality (Vision)")
    video_file = st.file_uploader("Upload Athlete Video (.mp4)", type=["mp4", "mov", "avi"])

with col2:
    st.markdown("### ⌚ Secondary Modality (Sensors)")
    st.caption("Galaxy Watch exports 2 files — upload both below:")
    accel_file = st.file_uploader("📈 Accelerometer CSV", type=["csv"], key="accel")
    gyro_file  = st.file_uploader("🔄 Gyroscope CSV", type=["csv"], key="gyro")

# ── 3. EXECUTION LOGIC ───────────────────────────────────────────────────────
if video_file and accel_file and gyro_file:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Giant Execution Button
    if st.button("INITIALIZE AI INFERENCE 🚀", type="primary", use_container_width=True):
        
        # Loader
        with st.status("Initializing Multimodal Analysis...", expanded=True) as status:
            
            st.write("1️⃣  Loading temp files into memory...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tv:
                tv.write(video_file.read())
                temp_video_path = Path(tv.name)
                
            with tempfile.NamedTemporaryFile(delete=False, suffix="_accel.csv") as ta:
                ta.write(accel_file.read())
                temp_accel_path = Path(ta.name)

            with tempfile.NamedTemporaryFile(delete=False, suffix="_gyro.csv") as tg:
                tg.write(gyro_file.read())
                temp_gyro_path = Path(tg.name)
            
            st.write("2️⃣  Extracting MediaPipe Pose Landmarks & IMU Features...")
            # Run the actual deep learning bridged python script
            label, metrics, error = run_inference(temp_video_path, temp_accel_path, temp_gyro_path)
            
            if error:
                status.update(label="Critical System Error", state="error", expanded=True)
                st.error(error)
            else:
                st.write("3️⃣  Feeding Kinematics to Deep Learning Classifier...")
                status.update(label="Analysis Pipeline Complete!", state="complete", expanded=False)
                
            # Cleanup
            temp_video_path.unlink(missing_ok=True)
            temp_accel_path.unlink(missing_ok=True)
            temp_gyro_path.unlink(missing_ok=True)

        # ── 4. RESULTS DASHBOARD (Rendered outside the collapsed status box) ──
        if not error and label and metrics:
            st.markdown("---")
            
            # Skill Badge Banner
            st.markdown(f"## 🏅 Predicted Athlete Skill Class: <span style='color:white; background:#45a29e; padding:5px 15px; border-radius:5px;'>{label.upper()}</span>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            dash_col1, dash_col2 = st.columns([1, 2])
            
            # Left Column: Raw Metrics
            with dash_col1:
                st.markdown("### 📊 Extracted Features")
                st.metric("Stride Freq (Hz)", f"{metrics.get('stride_freq_hz', 0)}")
                st.metric("Peak Accel (m/s²)", f"{metrics.get('peak_accel_mag', 0)}")
                st.metric("Flight Ratio (%)", f"{metrics.get('flight_ratio', 0)*100:.1f}")
                st.metric("Knee Angle Mean (°)", f"{metrics.get('knee_angle_mean', 0)}")
                st.metric("Accel Smoothness", f"{metrics.get('movement_smoothness_accel', 0)}")

            # Right Column: The Coaching Engine Feed
            with dash_col2:
                st.markdown("### 📋 Expert Coaching Engine Notes")
                st.write("*Analyzing structural deviations against Elite mathematical profiles...*")
                
                tips = generate_coaching_feedback(metrics)
                for tip in tips:
                    st.markdown(f"<div class='coaching-tip'>{tip}</div>", unsafe_allow_html=True)
                    
                if label == "Elite":
                    st.balloons()
