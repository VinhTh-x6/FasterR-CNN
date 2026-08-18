import streamlit as st
import torch
import numpy as np
import cv2
import tempfile
import os
import time
from load_model import load_model
from test_image import predict_image
from test_video import process_video

# Fixed technical configuration — not shown to the end user
CHECKPOINT_PATH = "trained_models/best.pt"

st.set_page_config(
    page_title="Detect • Faster R-CNN",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================================
# DESIGN SYSTEM — "Viewfinder"
# 4 corner brackets around the image/video mimic the bounding box the
# model draws. A thin blueprint-style grid background evokes a
# technical/computer-vision tool feel. Space Grotesk for headings,
# JetBrains Mono for numeric readouts.
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
        --bg: #0a0e13;
        --surface: #131a22;
        --surface-2: #1a222c;
        --line: rgba(148,163,184,0.13);
        --text: #e6edf3;
        --muted: #7d8998;
        --cyan: #5eead4;
        --amber: #fbbf24;
        --rose: #fb7185;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp {
        background-color: var(--bg);
        background-image:
            linear-gradient(rgba(148,163,184,0.045) 1px, transparent 1px),
            linear-gradient(90deg, rgba(148,163,184,0.045) 1px, transparent 1px),
            radial-gradient(circle at 50% 0%, rgba(94,234,212,0.08), transparent 55%);
        background-size: 42px 42px, 42px 42px, 100% 100%;
    }
    .block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1100px; }

    /* ---------- corner-bracket frame (signature element) ---------- */
    div[data-testid="stImage"] > img, div[data-testid="stVideo"] video {
        background-image:
            linear-gradient(to right, var(--cyan) 2px, transparent 2px) top left / 18px 2px no-repeat,
            linear-gradient(to bottom, var(--cyan) 2px, transparent 2px) top left / 2px 18px no-repeat,
            linear-gradient(to left, var(--cyan) 2px, transparent 2px) top right / 18px 2px no-repeat,
            linear-gradient(to bottom, var(--cyan) 2px, transparent 2px) top right / 2px 18px no-repeat,
            linear-gradient(to right, var(--cyan) 2px, transparent 2px) bottom left / 18px 2px no-repeat,
            linear-gradient(to top, var(--cyan) 2px, transparent 2px) bottom left / 2px 18px no-repeat,
            linear-gradient(to left, var(--cyan) 2px, transparent 2px) bottom right / 18px 2px no-repeat,
            linear-gradient(to top, var(--cyan) 2px, transparent 2px) bottom right / 2px 18px no-repeat;
        padding: 10px;
        border-radius: 2px;
    }
    div[data-testid="stImage"], div[data-testid="stVideo"] { background: transparent; }

    /* ---------- hero ---------- */
    .hero { text-align: center; padding: 3rem 1.5rem 2.2rem; }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase;
        color: var(--cyan);
        background: rgba(94,234,212,0.08);
        border: 1px solid rgba(94,234,212,0.3);
        border-radius: 999px;
        padding: 0.3rem 0.9rem;
        margin-bottom: 1.4rem;
    }
    .hero-badge .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--cyan);
        box-shadow: 0 0 8px var(--cyan); }
    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text);
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.15;
        margin: 0;
    }
    .hero h1 span { color: var(--cyan); }
    .hero p {
        color: var(--muted);
        font-size: 1.05rem;
        max-width: 40ch;
        margin: 1rem auto 0;
    }

    /* ---------- how it works ---------- */
    .steps { display: grid; grid-template-columns: 1fr auto 1fr auto 1fr; align-items: stretch; gap: 0.7rem; margin: 2.2rem 0 2.6rem; }
    .step {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 1.2rem 1.2rem 1.3rem;
    }
    .step .num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--cyan);
        margin-bottom: 0.5rem;
    }
    .step .title { font-weight: 600; color: var(--text); font-size: 0.98rem; margin-bottom: 0.3rem; }
    .step .desc { color: var(--muted); font-size: 0.85rem; line-height: 1.45; }

    /* ---------- sidebar ---------- */
    section[data-testid="stSidebar"] { background: #070a0e; border-right: 1px solid var(--line); }
    section[data-testid="stSidebar"] label { color: var(--muted) !important; }
    .sb-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase;
        color: var(--cyan);
        border-bottom: 1px solid var(--line);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* ---------- section labels ---------- */
    .sec-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase;
        color: var(--muted);
        margin: 0 0 0.6rem 0;
    }

    /* ---------- metrics ---------- */
    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-left: 2px solid var(--cyan);
        border-radius: 8px;
        padding: 0.85rem 1rem;
    }
    div[data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    div[data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'Space Grotesk', sans-serif !important; }

    /* ---------- HUD readout list ---------- */
    .hud-row {
        display: flex; justify-content: space-between; align-items: baseline;
        padding: 0.75rem 1rem; margin-bottom: 0.4rem;
        background: var(--surface);
        border: 1px solid var(--line);
        border-left: 3px solid var(--tier-color, var(--cyan));
        border-radius: 6px;
    }
    .hud-name { font-weight: 600; color: var(--text); font-size: 1rem; letter-spacing: -0.01em; }
    .hud-name::before { content: "▸ "; color: var(--tier-color, var(--cyan)); }
    .hud-score { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.05rem; letter-spacing: 0.02em; color: var(--tier-color, var(--cyan)); }

    /* ---------- tabs ---------- */
    .stTabs { margin-top: 0.4rem; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; border-bottom: 1px solid var(--line); justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem; letter-spacing: 0.05em;
        color: var(--muted);
        border-radius: 0;
        padding: 0.6rem 1.4rem;
    }
    .stTabs [aria-selected="true"] {
        color: var(--cyan) !important;
        border-bottom: 2px solid var(--cyan) !important;
        background: transparent !important;
    }

    /* ---------- buttons ---------- */
    div.stButton > button {
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600; font-size: 0.82rem; letter-spacing: 0.04em; text-transform: uppercase;
        padding: 0.65rem 1.2rem;
        border: 1px solid var(--cyan);
        color: var(--cyan);
        background: transparent;
        transition: all 0.15s ease;
    }
    div.stButton > button:hover { background: rgba(94,234,212,0.1); }
    div.stButton > button[kind="primary"] {
        background: var(--cyan); color: #04120f; border: none;
        box-shadow: 0 4px 20px rgba(94,234,212,0.25);
    }
    div.stDownloadButton > button {
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        border: 1px solid var(--line);
        background: var(--surface);
        color: var(--text);
    }

    /* ---------- uploader ---------- */
    div[data-testid="stFileUploaderDropzone"] {
        background: var(--surface);
        border: 1px dashed var(--line);
        border-radius: 10px;
    }

    h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--text) !important; font-weight: 600 !important; }

    /* ---------- accessibility ---------- */
    button:focus-visible, input:focus-visible, [role="slider"]:focus-visible,
    div[data-testid="stFileUploaderDropzone"]:focus-within {
        outline: 2px solid var(--cyan) !important; outline-offset: 2px;
    }

    /* ---------- responsive ---------- */
    @media (max-width: 640px) {
        .block-container { padding-left: 0.9rem; padding-right: 0.9rem; }
        .hero { padding: 2rem 0.4rem 1.4rem; }
        .hero h1 { font-size: 1.7rem; }
        .hero p { font-size: 0.9rem; }
        .steps { grid-template-columns: 1fr; }
        .flow-arrow { display: none; }
        .stTabs [data-baseweb="tab"] { padding: 0.55rem 0.9rem; font-size: 0.76rem; }
        div.stButton > button, div.stDownloadButton > button { width: 100%; padding: 0.75rem 1rem; }
        .hud-name { font-size: 0.95rem; }
        .hud-score { font-size: 0.98rem; }
        .hud-row { padding: 0.65rem 0.85rem; }
    }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--surface-2); border-radius: 4px; }

    /* ---------- animations ---------- */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes popIn {
        from { opacity: 0; transform: scale(0.97); }
        to   { opacity: 1; transform: scale(1); }
    }
    @media (prefers-reduced-motion: reduce) {
        .hero-badge, .hero h1, .hero p, .step, .hud-row,
        div[data-testid="stImage"], div[data-testid="stVideo"], .panel { animation: none !important; }
    }

    .hero-badge { animation: fadeUp 0.5s ease both; }
    .hero h1    { animation: fadeUp 0.5s ease 0.08s both; }
    .hero p     { animation: fadeUp 0.5s ease 0.16s both; }

    .step { animation: fadeUp 0.5s ease both; transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease; }
    .step:nth-child(1) { animation-delay: 0.20s; }
    .step:nth-child(2) { animation-delay: 0.28s; }
    .step:nth-child(3) { animation-delay: 0.36s; }
    .step:hover {
        transform: translateY(-3px);
        border-color: rgba(94,234,212,0.4);
        box-shadow: 0 10px 24px rgba(0,0,0,0.35);
    }

    .panel { animation: fadeUp 0.45s ease both; }

    div[data-testid="stImage"], div[data-testid="stVideo"] { animation: popIn 0.4s ease both; }

    .hud-row { animation: fadeUp 0.35s ease both; transition: transform 0.15s ease, border-color 0.15s ease; }
    .hud-row:hover { transform: translateX(3px); border-color: rgba(148,163,184,0.35); }
    .hud-row:nth-child(1) { animation-delay: 0.02s; } .hud-row:nth-child(2) { animation-delay: 0.06s; }
    .hud-row:nth-child(3) { animation-delay: 0.10s; } .hud-row:nth-child(4) { animation-delay: 0.14s; }
    .hud-row:nth-child(5) { animation-delay: 0.18s; } .hud-row:nth-child(6) { animation-delay: 0.22s; }

    div.stButton > button, div.stDownloadButton > button { transition: all 0.15s ease; }
    div.stButton > button:active { transform: scale(0.97); }
    div.stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 26px rgba(94,234,212,0.4);
        transform: translateY(-1px);
    }

    div[data-testid="stMetric"] { transition: border-color 0.2s ease, transform 0.2s ease; }
    div[data-testid="stMetric"]:hover { border-color: rgba(94,234,212,0.35); transform: translateY(-2px); }

    div[data-testid="stProgress"] > div > div > div { background: var(--cyan) !important; }

    .flow-arrow {
        display: flex; align-items: center; justify-content: center;
        color: var(--muted); font-family: 'JetBrains Mono', monospace; font-size: 1rem;
    }

    /* panel: wraps result content using st.container(border=True) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface) !important;
        border: 1px solid var(--line) !important;
        border-radius: 14px !important;
        padding: 0.4rem 0.2rem !important;
        animation: fadeUp 0.4s ease both;
    }
</style>
""", unsafe_allow_html=True)

def tier(score):
    if score >= 0.7:
        return "#5eead4"
    if score >= 0.4:
        return "#fbbf24"
    return "#fb7185"

def render_detections(detections):
    if not detections:
        st.info("No objects detected at the current confidence threshold.")
        return
    for d in sorted(detections, key=lambda x: x["score"], reverse=True):
        color = tier(d["score"])
        st.markdown(
            f"""<div class="hud-row" style="--tier-color:{color}">
                    <span class="hud-name">{d['category']}</span>
                    <span class="hud-score">{d['score']*100:.1f}%</span>
                </div>""",
            unsafe_allow_html=True,
        )

# ---------------- Hero ----------------
st.markdown("""
<div class="hero">
    <div class="hero-badge"><span class="dot"></span> Faster R-CNN · Real-time inference</div>
    <h1>Object detection<br>in <span>one click</span></h1>
    <p>Upload an image or video — the system automatically draws boxes around and labels each object.</p>
</div>

<div class="steps">
    <div class="step">
        <div class="num">01</div>
        <div class="title">Upload a file</div>
        <div class="desc">Drag and drop an image or video into the field below.</div>
    </div>
    <div class="flow-arrow">→</div>
    <div class="step">
        <div class="num">02</div>
        <div class="title">Model processes it</div>
        <div class="desc">Faster R-CNN locates and labels each object.</div>
    </div>
    <div class="flow-arrow">→</div>
    <div class="step">
        <div class="num">03</div>
        <div class="title">Get your results</div>
        <div class="desc">View the confidence for each label and download the result file.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- Sidebar: keep only options relevant to the user experience ----------------
with st.sidebar:
    st.markdown('<p class="sb-title">Settings</p>', unsafe_allow_html=True)
    conf_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, 0.3, 0.05,
        help="Increase the threshold if you notice too many incorrect labels."
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.markdown(
        f'<p class="sec-label">Runtime — <span style="color:#5eead4">{device.type.upper()}</span></p>',
        unsafe_allow_html=True,
    )

if not os.path.exists(CHECKPOINT_PATH):
    st.error("Model not found. Please check the system configuration.")
    st.stop()

@st.cache_resource
def get_model(path, device_str):
    return load_model(path, torch.device(device_str))

with st.spinner("Loading model..."):
    model = get_model(CHECKPOINT_PATH, str(device))

tab_img, tab_video = st.tabs(["IMAGE", "VIDEO"])

# ---------------- IMAGE TAB ----------------
with tab_img:
    uploaded_image = st.file_uploader("Drag and drop or select an image", type=["jpg", "jpeg", "png", "bmp"], key="img")

    if uploaded_image is None:
        st.info("No image has been uploaded yet.")
    else:
        file_bytes = np.frombuffer(uploaded_image.read(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Running prediction..."):
            t0 = time.time()
            result_bgr, detections = predict_image(model, device, image_bgr, conf_threshold)
            elapsed = time.time() - t0

        m1, m2, m3 = st.columns(3)
        m1.metric("OBJECTS", len(detections))
        m2.metric("LATENCY", f"{elapsed:.2f}s")
        m3.metric("THRESHOLD", f"{conf_threshold:.2f}")

        st.write("")
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<p class="sec-label">Input</p>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.markdown('<p class="sec-label">Detection output</p>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.write("")
            st.markdown('<p class="sec-label">Readout</p>', unsafe_allow_html=True)
            render_detections(detections)

        st.write("")
        _, buf = cv2.imencode(".jpg", result_bgr)
        st.download_button("Download result", data=buf.tobytes(),
                            file_name="prediction.jpg", mime="image/jpeg", use_container_width=True)

# ---------------- VIDEO TAB ----------------
with tab_video:
    uploaded_video = st.file_uploader("Drag and drop or select a video", type=["mp4", "avi", "mov", "mkv"], key="vid")

    if uploaded_video is None:
        st.info("No video has been uploaded yet.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded_video.read())
            input_path = tmp_in.name

        output_path = os.path.join(tempfile.gettempdir(), "result.mp4")

        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<p class="sec-label">Input</p>', unsafe_allow_html=True)
                st.video(input_path)

            st.write("")
            run = st.button("Run detection", use_container_width=True, type="primary")

            if run:
                progress_bar = st.progress(0.0, text="Processing video...")

                def update_progress(p):
                    progress_bar.progress(min(p, 1.0), text=f"Processing video... {int(p * 100)}%")

                t0 = time.time()
                process_video(model, device, input_path, output_path, conf_threshold, update_progress)
                elapsed = time.time() - t0
                progress_bar.empty()

                st.success(f"Done in {elapsed:.1f}s")

                with col2:
                    st.markdown('<p class="sec-label">Detection output</p>', unsafe_allow_html=True)
                    st.video(output_path)

        if run:
            st.write("")
            with open(output_path, "rb") as f:
                st.download_button("Download result", data=f.read(),
                                    file_name="result.mp4", mime="video/mp4", use_container_width=True)

        os.remove(input_path)