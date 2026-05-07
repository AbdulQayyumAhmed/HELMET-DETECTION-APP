# app.py

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

# Page configuration
st.set_page_config(
    page_title="Guardian AI Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Compact SaaS Dashboard Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    :root {
        --primary: #6366f1;
        --bg: #0f172a;
        --card-bg: #1e293b;
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
    }

    .stApp {
        background-color: var(--bg);
        color: var(--text-main);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Hide Streamlit components */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Remove default padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--card-bg);
    }

    /* Compact Header */
    .hero-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }

    .hero-text {
        background: linear-gradient(to right, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
    }

    /* Force Main Cards Side-by-Side */
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: stretch !important;
    }

    [data-testid="column"] {
        width: 50% !important;
        min-width: 50% !important;
        flex: 1 1 auto !important;
    }

    /* Compact Cards */
    .dashboard-card {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 0.5rem;
        height: 100%;
        display: flex;
        flex-direction: column;
    }

    /* Compact Stat Boxes - Force 4 Columns in a Row */
    .stat-container {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        width: 100% !important;
        gap: 10px !important;
        margin-bottom: 15px !important;
    }

    .stat-box {
        flex: 1;
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.01));
        padding: 10px 15px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        display: flex;
        flex-direction: row; /* Side-by-side */
        align-items: center;
        justify-content: center;
        gap: 10px;
        min-width: 180px;
        transition: all 0.2s ease;
        cursor: default;
    }

    .stat-box:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: var(--primary);
    }

    .stat-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: white;
    }

    .stat-label {
        color: var(--text-muted);
        text-transform: uppercase;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        white-space: nowrap;
    }

    /* Status Badge */
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 700;
    }

    .badge-success { background: rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.2); }
    .badge-danger { background: rgba(239, 68, 68, 0.1); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.2); }

    /* Fix image height to keep on one screen */
    img {
        max-height: 400px;
        object-fit: contain;
    }

</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h2 style='color: white; font-weight: 800; font-size: 1.2rem;'>GUARDIAN AI</h2>", unsafe_allow_html=True)
    st.markdown("<div class='status-badge badge-success'>Live System</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Model: YOLOv8 Engine")
    st.caption("Scan: Automatic Mode")

# Default thresholds
conf_threshold = 0.5
iou_threshold = 0.45

# --- Model Loading ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- Main Layout ---
# Top Header Line
st.markdown("""
<div class='hero-container'>
    <h1 class='hero-text'>Neural Intelligence</h1>
    <div style='color: #94a3b8; font-size: 0.8rem;'>Real-time Safety Dashboard v2.5</div>
</div>
""", unsafe_allow_html=True)

# Initial results logic to show metrics at the TOP
uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    start = time.time()
    results = model.predict(source=img, conf=conf_threshold, iou=iou_threshold)
    proc_time = (time.time() - start) * 1000
    
    # Process Metrics FIRST for top display
    boxes = results[0].boxes
    total = len(boxes)
    names = model.names
    helmets = 0
    violations = 0
    for box in boxes:
        label = names[int(box.cls[0])].lower()
        if 'helmet' in label and 'no' not in label: helmets += 1
        elif 'head' in label or 'no' in label: violations += 1

    # DASHBOARD ON TOP - Combined into a single markdown call to ensure flexbox works
    st.markdown(f"""
    <div class='stat-container'>
        <div class='stat-box'><div class='stat-val'>{total}</div><div class='stat-label'>Total Detected</div></div>
        <div class='stat-box'><div class='stat-val' style='color: #10b981;'>{helmets}</div><div class='stat-label'>PPE Safe</div></div>
        <div class='stat-box'><div class='stat-val' style='color: #ef4444;'>{violations}</div><div class='stat-label'>Safety Alerts</div></div>
        <div class='stat-box'><div class='stat-val' style='color: #6366f1;'>{proc_time:.0f}ms</div><div class='stat-label'>Inference</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Main View Columns
    c1, c2 = st.columns([1, 1], gap="medium")
    
    with c1:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.image(img, caption="Original Stream", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        annotated = results[0].plot()
        st.image(annotated, caption="Neural Analysis", use_container_width=True)
        
        if violations > 0:
            st.markdown("<div class='status-badge badge-danger' style='text-align:center;'>🚨 VIOLATION DETECTED: ACTION REQUIRED</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='status-badge badge-success' style='text-align:center;'>✅ ALL PERSONNEL COMPLIANT</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Upload an image to activate the neural safety dashboard.")
    st.image("https://images.unsplash.com/photo-1590486803833-ffc6f11f8fd8?q=80&w=1000&auto=format&fit=crop", use_container_width=True) # Placeholder for UI feel

