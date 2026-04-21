import streamlit as st
import tempfile
import time
from pathlib import Path

from inference import run_inference
from coaching_engine import generate_coaching_feedback

# ── 1. DASHBOARD CONFIGURATION ───────────────────────────────────────────────
st.set_page_config(page_title="Multimodal Sprint Analytics", layout="wide")

# Custom Black & White Theme CSS
st.markdown("""
<style>
    /* Global Typography and White Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    .stApp {
        background: #ffffff !important; /* Pure white background */
        color: #000000 !important; /* Solid black text */
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Typography Overrides */
    h1, h2, h3, h4, p, span {
        color: #000000 !important;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    h1, h2, h3, h4 {
        font-weight: 700 !important;
        letter-spacing: -0.015em;
    }
    
    /* Streamlit Metric Boxes overrides: Crisp B&W styling */
    div[data-testid="metric-container"] {
        background: #ffffff !important;
        border-radius: 8px !important;
        padding: 20px !important;
        border: 2px solid #000000 !important;
        box-shadow: 4px 4px 0px #000000 !important; /* Hard black shadow for contrast */
        transition: transform 0.2s ease-out;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
    }
    
    /* Text coloring inside metrics */
    div[data-testid="metric-container"] label {
        color: #444444 !important; /* Slightly lighter gray for subheaders */
        font-weight: 600 !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #000000 !important;
    }

    /* Custom Coaching Note Panels */
    .coaching-tip {
        background: #ffffff;
        padding: 22px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 8px;
        border: 1px solid #000000;
        border-left: 6px solid #000000; /* Bold black accent line */
        font-size: 16px;
        line-height: 1.6;
        color: #000000;
    }
    
    /* Button overrides: Black block with white text */
    div[data-testid="stButton"] > button[kind="primary"] {
        background: #000000 !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        border: 2px solid #000000 !important;
        transition: all 0.2s ease;
    }
    
    div[data-testid="stButton"] > button[kind="primary"] * {
        color: #ffffff !important; /* Force internal text to white */
    }
    
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: #ffffff !important;
        color: #000000 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover * {
        color: #000000 !important; /* Force internal text to black on hover */
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
            
            # Skill Badge Banner (Inverted to black background with white text for the new theme)
            st.markdown(f"## Predicted Athlete Skill Class: <span style='color:#ffffff; background:#000000; padding:5px 15px; border-radius:8px;'>{label.upper()}</span>", unsafe_allow_html=True)
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