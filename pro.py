"""
QR Code + Face Recognition Attendance System  v3
=================================================
Requirements: pip install streamlit opencv-python mediapipe face_recognition qrcode pandas numpy pillow
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import os
import time
import qrcode
from datetime import datetime, date
from PIL import Image
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AttendX Pro",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CUSTOM CSS  – Dark luxury theme
# ─────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    /* ── Root tokens ── */
    :root {
        --bg:        #0a0d14;
        --surface:   #111520;
        --surface2:  #161b2e;
        --border:    #1e2540;
        --accent:    #4f8ef7;
        --accent2:   #7c3aed;
        --green:     #22c55e;
        --red:       #ef4444;
        --amber:     #f59e0b;
        --text:      #e2e8f0;
        --muted:     #64748b;
        --radius:    14px;
        --shadow:    0 8px 32px rgba(0,0,0,.5);
    }

    /* ── Base reset ── */
    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }

    /* ── Remove default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden !important; }
    .block-container { padding: 2rem 2.5rem !important; max-width: 1400px !important; }

    /* ── Typography ── */
    h1, h2, h3, .main-title {
        font-family: 'Syne', sans-serif !important;
        letter-spacing: -0.02em;
    }
    h1 { font-size: 2.2rem !important; font-weight: 800 !important;
         background: linear-gradient(135deg, #4f8ef7, #7c3aed);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    h2 { font-size: 1.3rem !important; font-weight: 700 !important; color: var(--text) !important; }
    h3 { font-size: 1rem !important; font-weight: 600 !important; color: var(--muted) !important; text-transform: uppercase; letter-spacing:.08em; }

    /* ── Cards ── */
    .att-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.5rem;
        box-shadow: var(--shadow);
        transition: border-color .25s, transform .2s;
    }
    .att-card:hover { border-color: var(--accent); transform: translateY(-2px); }

    /* ── Floating nav buttons in sidebar ── */
    [data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        border: 1px solid transparent !important;
        color: var(--muted) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        padding: 0.6rem 1rem !important;
        border-radius: 10px !important;
        transition: all 0.22s cubic-bezier(.4,0,.2,1) !important;
        width: 100% !important;
        letter-spacing: 0.01em !important;
        box-shadow: none !important;
        margin: 2px 0 !important;
        position: relative;
        overflow: hidden;
    }
    [data-testid="stSidebar"] .stButton > button::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(90deg, rgba(79,142,247,.12), rgba(124,58,237,.08));
        opacity: 0;
        transition: opacity .22s;
        border-radius: 10px;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        color: var(--text) !important;
        border-color: var(--accent) !important;
        transform: translateX(5px) !important;
        box-shadow: 0 4px 20px rgba(79,142,247,.18) !important;
        background: rgba(79,142,247,.07) !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover::before { opacity: 1; }

    /* ── Main action buttons ── */
    .main-content .stButton > button, [data-testid="stMainBlockContainer"] .stButton > button {
        background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 1.4rem !important;
        transition: all .2s ease !important;
        box-shadow: 0 4px 15px rgba(79,142,247,.3) !important;
        letter-spacing: 0.02em !important;
    }
    [data-testid="stMainBlockContainer"] .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(79,142,247,.45) !important;
        filter: brightness(1.1) !important;
    }
    [data-testid="stMainBlockContainer"] .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Inputs ── */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stTextArea textarea {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
        transition: border-color .2s !important;
    }
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(79,142,247,.15) !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 1.1rem 1.2rem !important;
        transition: border-color .2s, transform .2s !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: var(--accent) !important;
        transform: translateY(-2px) !important;
    }
    [data-testid="stMetricLabel"] > div {
        color: var(--muted) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: .07em !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    [data-testid="stMetricValue"] > div {
        color: var(--text) !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        border: 1px solid var(--border) !important;
        gap: 2px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--muted) !important;
        border-radius: 9px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        padding: 0.4rem 0.9rem !important;
        transition: all .2s !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
        color: white !important;
        font-weight: 600 !important;
        box-shadow: 0 3px 12px rgba(79,142,247,.35) !important;
    }
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        color: var(--text) !important;
        background: rgba(79,142,247,.08) !important;
    }

    /* ── Alerts / info boxes ── */
    .stAlert { border-radius: var(--radius) !important; border-left-width: 4px !important; }
    [data-testid="stInfo"]    { background: rgba(79,142,247,.1) !important; border-color: var(--accent) !important; }
    [data-testid="stSuccess"] { background: rgba(34,197,94,.1) !important; border-color: var(--green) !important; }
    [data-testid="stError"]   { background: rgba(239,68,68,.1) !important; border-color: var(--red) !important; }
    [data-testid="stWarning"] { background: rgba(245,158,11,.1) !important; border-color: var(--amber) !important; }

    /* ── Dataframes ── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        overflow: hidden !important;
    }

    /* ── Progress bar ── */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
        border-radius: 99px !important;
    }
    .stProgress > div > div {
        background: var(--surface2) !important;
        border-radius: 99px !important;
    }

    /* ── Checkbox ── */
    .stCheckbox span { color: var(--text) !important; font-family: 'DM Sans',sans-serif !important; }

    /* ── Divider ── */
    hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }

    /* ── Step indicator ── */
    .step-bar {
        display: flex; gap: 0; margin-bottom: 1.5rem;
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 12px; padding: 0.7rem 1.2rem;
        align-items: center; justify-content: space-between;
    }
    .step-item {
        display: flex; align-items: center; gap: 0.5rem;
        font-family: 'DM Sans', sans-serif; font-size: 0.85rem;
        font-weight: 500; color: var(--muted);
    }
    .step-item.active { color: var(--accent); font-weight: 600; }
    .step-item.done   { color: var(--green); }
    .step-dot {
        width: 28px; height: 28px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.75rem; font-weight: 700;
        background: var(--surface2); border: 2px solid var(--border);
        font-family: 'Syne', sans-serif;
    }
    .step-dot.active { background: var(--accent); border-color: var(--accent); color: white; }
    .step-dot.done   { background: var(--green);  border-color: var(--green);  color: white; }
    .step-connector  { flex: 1; height: 1px; background: var(--border); margin: 0 0.6rem; }

    /* ── Logo / brand ── */
    .brand-logo {
        font-family: 'Syne', sans-serif;
        font-size: 1.4rem; font-weight: 800;
        background: linear-gradient(135deg, #4f8ef7, #7c3aed);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    .brand-sub {
        font-size: 0.72rem; color: var(--muted);
        text-transform: uppercase; letter-spacing: .1em;
        font-family: 'DM Sans', sans-serif; margin-top: -4px;
    }

    /* ── Badge ── */
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 99px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: .04em;
        font-family: 'DM Sans', sans-serif;
    }
    .badge-green  { background: rgba(34,197,94,.15);  color: #22c55e; border:1px solid rgba(34,197,94,.3); }
    .badge-blue   { background: rgba(79,142,247,.15); color: #4f8ef7; border:1px solid rgba(79,142,247,.3);}
    .badge-amber  { background: rgba(245,158,11,.15); color: #f59e0b; border:1px solid rgba(245,158,11,.3);}
    .badge-red    { background: rgba(239,68,68,.15);  color: #ef4444; border:1px solid rgba(239,68,68,.3); }
    .badge-purple { background: rgba(124,58,237,.15); color: #a78bfa; border:1px solid rgba(124,58,237,.3);}

    /* ── Form labels ── */
    .stTextInput label, .stSelectbox label, .stTextArea label, .stRadio label {
        color: var(--muted) !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: .07em !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
        transition: border-color .2s !important;
    }
    .streamlit-expanderHeader:hover { border-color: var(--accent) !important; }
    .streamlit-expanderContent {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
    }

    /* ── Download button ── */
    [data-testid="stDownloadButton"] button {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        color: var(--accent) !important;
        font-family: 'DM Sans', sans-serif !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all .2s !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        border-color: var(--accent) !important;
        background: rgba(79,142,247,.1) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent); }

    /* Sidebar nav section headers */
    .nav-section-label {
        font-size: 0.68rem; text-transform: uppercase; letter-spacing: .12em;
        color: var(--muted); font-family: 'DM Sans', sans-serif;
        padding: 0.8rem 0.4rem 0.3rem; font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
USER_DATA_FILE    = "user_data.json"
ATTENDANCE_FILE   = "attendance_log.csv"
HISTORY_FILE      = "login_logout_history.csv"
FACE_MATCH_THRESH = 0.55
AUTO_REDIRECT_SEC = 3

ROLE_OPTIONS = [
    "Select Role",
    "Software Engineer", "Senior Engineer", "Lead Engineer", "Engineering Manager",
    "Product Manager", "Project Manager",
    "Data Analyst", "Data Scientist", "ML Engineer",
    "UI/UX Designer", "Graphic Designer",
    "HR Manager", "HR Executive", "Recruiter",
    "Finance Manager", "Accountant",
    "Sales Executive", "Sales Manager", "Business Development",
    "Marketing Manager", "Content Writer",
    "DevOps Engineer", "System Administrator", "Network Engineer",
    "QA Engineer", "Security Analyst",
    "Operations Manager", "Team Lead",
    "CEO", "CTO", "CFO", "COO", "Director", "VP",
    "Intern", "Contractor", "Consultant",
    "Other",
]

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def init_session_state():
    defaults = {
        "page":            "qr_scanner",
        "scanned_user":    None,
        "face_verified":   False,
        "confidence":      0.0,
        "attendance_done": False,
        "error_msg":       "",
        "success_msg":     "",
        "redirect_time":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# ─────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────
def load_user_data() -> dict:
    if not os.path.exists(USER_DATA_FILE):
        return {}
    with open(USER_DATA_FILE, "r") as f:
        raw = json.load(f)
    users = {}
    for uid, info in raw.items():
        info = info.copy()
        if info.get("face_encoding"):
            info["face_encoding"] = np.array(info["face_encoding"])
        users[uid] = info
    return users

def save_user_data(users: dict):
    serialisable = {}
    for uid, info in users.items():
        rec = info.copy()
        if isinstance(rec.get("face_encoding"), np.ndarray):
            rec["face_encoding"] = rec["face_encoding"].tolist()
        serialisable[uid] = rec
    with open(USER_DATA_FILE, "w") as f:
        json.dump(serialisable, f, indent=2)

def load_attendance() -> pd.DataFrame:
    cols = ["user_id", "name", "date", "login_time", "logout_time"]
    if not os.path.exists(ATTENDANCE_FILE):
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(ATTENDANCE_FILE, dtype=str)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df

def save_attendance(df: pd.DataFrame):
    df.to_csv(ATTENDANCE_FILE, index=False)

def load_history() -> pd.DataFrame:
    cols = ["event_id","user_id","name","role","department",
            "date","time","event_type","confidence_pct","duration_since_login"]
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(HISTORY_FILE, dtype=str)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df

def save_history(df: pd.DataFrame):
    df.to_csv(HISTORY_FILE, index=False)

def append_history_event(user_id: str, name: str, role: str, department: str,
                          event_type: str, confidence: float):
    df       = load_history()
    now      = datetime.now()
    today    = now.strftime("%Y-%m-%d")
    now_str  = now.strftime("%I:%M:%S %p")
    event_id = f"{user_id}_{now.strftime('%Y%m%d%H%M%S')}"
    duration_str = ""
    if event_type == "LOGOUT":
        user_logins = df[
            (df["user_id"] == user_id) &
            (df["date"]    == today)   &
            (df["event_type"] == "LOGIN")
        ]
        if not user_logins.empty:
            try:
                login_dt  = datetime.strptime(f"{today} {user_logins.iloc[-1]['time']}", "%Y-%m-%d %I:%M:%S %p")
                delta     = now - login_dt
                total_sec = int(delta.total_seconds())
                hours, rem = divmod(total_sec, 3600)
                mins, secs = divmod(rem, 60)
                if hours > 0:   duration_str = f"{hours}h {mins}m {secs}s"
                elif mins > 0:  duration_str = f"{mins}m {secs}s"
                else:           duration_str = f"{secs}s"
            except Exception:
                duration_str = ""
    new_row = pd.DataFrame([{
        "event_id":             event_id,
        "user_id":              user_id,
        "name":                 name,
        "role":                 role,
        "department":           department,
        "date":                 today,
        "time":                 now_str,
        "event_type":           event_type,
        "confidence_pct":       str(round(confidence, 1)),
        "duration_since_login": duration_str,
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    save_history(df)

def mark_attendance(user_id: str, name: str, role: str, department: str, confidence: float) -> str:
    today   = date.today().strftime("%Y-%m-%d")
    now_str = datetime.now().strftime("%I:%M:%S %p")
    df      = load_attendance()
    mask     = (df["user_id"] == user_id) & (df["date"] == today)
    existing = df[mask]
    if existing.empty:
        df = pd.concat([df, pd.DataFrame([{
            "user_id": user_id, "name": name,
            "date": today, "login_time": now_str, "logout_time": "",
        }])], ignore_index=True)
        save_attendance(df)
        append_history_event(user_id, name, role, department, "LOGIN", confidence)
        return "login"
    else:
        idx = existing.index[-1]
        if df.at[idx, "logout_time"] == "" or pd.isna(df.at[idx, "logout_time"]):
            df.at[idx, "logout_time"] = now_str
            save_attendance(df)
            append_history_event(user_id, name, role, department, "LOGOUT", confidence)
            return "logout"
        else:
            df = pd.concat([df, pd.DataFrame([{
                "user_id": user_id, "name": name,
                "date": today, "login_time": now_str, "logout_time": "",
            }])], ignore_index=True)
            save_attendance(df)
            append_history_event(user_id, name, role, department, "LOGIN", confidence)
            return "login"

def today_summary() -> pd.DataFrame:
    today = date.today().strftime("%Y-%m-%d")
    df    = load_attendance()
    return df[df["date"] == today].reset_index(drop=True)

def get_user_history(user_id: str = None, filter_date: str = None) -> pd.DataFrame:
    df = load_history()
    if user_id:     df = df[df["user_id"] == user_id]
    if filter_date: df = df[df["date"] == filter_date]
    return df.sort_values(["date","time"], ascending=[False,False]).reset_index(drop=True)

# ─────────────────────────────────────────────
# QR HELPERS
# ─────────────────────────────────────────────
def generate_qr(user_id, name, department):
    data = f"{user_id}|{name}|{department}"
    qr   = qrcode.QRCode(version=1, box_size=8, border=4)
    qr.add_data(data)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white").convert("RGB")

def decode_qr_from_frame(frame):
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(frame)
    annotated = frame.copy()
    if points is not None:
        pts = points[0].astype(int)
        cv2.polylines(annotated, [pts], True, (79,142,247), 3)
    return (data if data else None), annotated

def parse_qr_data(raw):
    parts = raw.strip().split("|")
    if len(parts) != 3:
        return None
    return {"user_id": parts[0].strip(), "name": parts[1].strip(), "department": parts[2].strip()}

# ─────────────────────────────────────────────
# FACE HELPERS
# ─────────────────────────────────────────────
def get_face_encoding(frame_rgb):
    try:
        import face_recognition
        locs = face_recognition.face_locations(frame_rgb, model="hog")
        encs = face_recognition.face_encodings(frame_rgb, locs)
        return encs[0] if encs else None
    except ImportError:
        st.error("face_recognition not installed.")
        return None

def compare_faces(stored, live):
    try:
        import face_recognition
        dist  = face_recognition.face_distance([stored], live)[0]
        conf  = (1.0 - float(dist)) * 100.0
        return float(dist) < FACE_MATCH_THRESH, round(conf, 1)
    except ImportError:
        return False, 0.0

def draw_face_boxes(frame):
    try:
        import mediapipe as mp
        ann   = frame.copy()
        h, w  = frame.shape[:2]
        with mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as det:
            res = det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.detections:
                for d in res.detections:
                    bb = d.location_data.relative_bounding_box
                    x1, y1 = max(0,int(bb.xmin*w)), max(0,int(bb.ymin*h))
                    x2, y2 = min(w,int((bb.xmin+bb.width)*w)), min(h,int((bb.ymin+bb.height)*h))
                    cv2.rectangle(ann, (x1,y1), (x2,y2), (79,142,247), 2)
        return ann
    except ImportError:
        return frame

# ─────────────────────────────────────────────
# NAVIGATION
# ─────────────────────────────────────────────
def go_to(page: str):
    st.session_state.page            = page
    st.session_state.error_msg       = ""
    st.session_state.success_msg     = ""
    st.session_state.attendance_done = False
    if page == "qr_scanner":
        st.session_state.scanned_user  = None
        st.session_state.face_verified = False
        st.session_state.confidence    = 0.0

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Brand
        st.markdown("""
        <div style="padding: 0.5rem 0.2rem 1rem">
            <div class="brand-logo">AttendX Pro</div>
            <div class="brand-sub">Smart Attendance System</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="nav-section-label">Check-In Flow</div>', unsafe_allow_html=True)
        if st.button("⬡  QR Scanner",            use_container_width=True): go_to("qr_scanner");      st.rerun()
        if st.button("⬡  Face Recognition",       use_container_width=True): go_to("face_recognition"); st.rerun()
        st.markdown('<div class="nav-section-label">Reports</div>', unsafe_allow_html=True)
        if st.button("⬡  Login / Logout History", use_container_width=True): go_to("history");         st.rerun()
        st.markdown('<div class="nav-section-label">Management</div>', unsafe_allow_html=True)
        if st.button("⬡  Admin Panel",            use_container_width=True): go_to("admin");            st.rerun()

        st.markdown("---")

        # Today's snapshot
        st.markdown('<div class="nav-section-label">Today\'s Snapshot</div>', unsafe_allow_html=True)
        df = today_summary()
        users = load_user_data()
        cols = st.columns(2)
        cols[0].metric("Present", len(df))
        cols[1].metric("Registered", len(users))

        if not df.empty:
            for _, row in df.iterrows():
                login  = row.get("login_time","–")
                logout = row.get("logout_time","–") or "–"
                status = "🟢" if (logout == "–") else "🔵"
                st.markdown(f"""
                <div style="background:var(--surface2);border:1px solid var(--border);
                     border-radius:9px;padding:.5rem .75rem;margin:3px 0;font-size:.8rem;">
                    <span style="font-weight:600;color:var(--text)">{status} {row['name']}</span><br>
                    <span style="color:var(--muted)">In: {login} · Out: {logout}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:var(--muted);font-size:.8rem;padding:.3rem">No entries yet today.</p>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"""
        <div style="font-size:.75rem;color:var(--muted);font-family:'DM Sans',sans-serif;line-height:1.9">
            🕐 {datetime.now().strftime('%I:%M %p')}<br>
            📆 {date.today().strftime('%a, %b %d %Y')}
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# STEP INDICATOR
# ─────────────────────────────────────────────
def render_steps(current: int):
    steps = [("01","QR Scan"), ("02","Face Verify"), ("03","Attendance")]
    html  = '<div class="step-bar">'
    for i, (num, label) in enumerate(steps, 1):
        cls = "done" if i < current else ("active" if i == current else "")
        dot_content = "✓" if i < current else num
        html += f'''
        <div class="step-item {cls}">
            <div class="step-dot {cls}">{dot_content}</div>
            <span>{label}</span>
        </div>'''
        if i < len(steps):
            html += '<div class="step-connector"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: QR SCANNER
# ─────────────────────────────────────────────
def page_qr_scanner():
    render_steps(1)
    st.markdown("## 🔷 QR Code Scanner")

    if st.session_state.error_msg:
        st.error(st.session_state.error_msg)

    col1, col2 = st.columns([3, 1], gap="large")

    with col1:
        run   = st.checkbox("▶  Start Camera Feed", value=True, key="qr_cam_run")
        fph   = st.empty()
        sph   = st.empty()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot access camera. Check permissions and try again.")
            return

        detected  = None
        t0        = time.time()
        TIMEOUT   = 60

        while run and (time.time() - t0) < TIMEOUT:
            ret, frame = cap.read()
            if not ret: st.error("❌ Camera read error."); break
            qr_data, annotated = decode_qr_from_frame(frame)
            fph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            if qr_data:
                parsed = parse_qr_data(qr_data)
                if parsed:
                    users = load_user_data()
                    if parsed["user_id"] in users:
                        sph.success(f"✅ Detected — **{parsed['name']}** ({parsed['user_id']})")
                        detected = parsed
                        time.sleep(0.8)
                        break
                    else:
                        sph.error(f"❌ Unknown user ID: {parsed['user_id']}")
                else:
                    sph.warning("⚠️ Invalid QR format")
            else:
                sph.info(f"🔎 Scanning for QR code… {int(time.time()-t0)}s / {TIMEOUT}s")
            time.sleep(0.04)

        cap.release()

        if detected:
            st.session_state.scanned_user = detected
            st.session_state.error_msg    = ""
            go_to("face_recognition")
            st.rerun()
        elif not run:
            st.warning("Camera paused.")
        else:
            st.error("⏱ Scan timed out. Please try again.")

    with col2:
        st.markdown("""
        <div class="att-card">
            <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;margin-bottom:.8rem">How to Scan</div>
            <ol style="color:var(--muted);font-size:.85rem;line-height:2;padding-left:1.1rem">
                <li>Start camera feed</li>
                <li>Hold QR card steady</li>
                <li>Ensure good lighting</li>
                <li>System auto-advances</li>
            </ol>
            <div style="margin-top:1rem;padding-top:1rem;border-top:1px solid var(--border);
                 font-size:.78rem;color:var(--muted)">
                <b style="color:var(--text)">QR Format</b><br>
                <code style="font-size:.75rem;color:var(--accent)">UID|Name|Department</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        users = load_user_data()
        st.markdown('<div style="font-family:\'DM Sans\',sans-serif;font-size:.78rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem">Registered Users</div>', unsafe_allow_html=True)
        if users:
            for uid, info in users.items():
                role   = info.get("role","–")
                has_face = info.get("face_encoding") is not None
                badge  = '<span class="badge badge-green">Face ✓</span>' if has_face else '<span class="badge badge-amber">No Face</span>'
                st.markdown(f"""
                <div style="background:var(--surface);border:1px solid var(--border);border-radius:9px;
                     padding:.5rem .8rem;margin:3px 0;font-size:.8rem;transition:border-color .2s">
                    <span style="font-weight:600;color:var(--text)">{uid}</span>
                    <span style="color:var(--muted)"> · {info.get('name','?')}</span><br>
                    <span style="color:var(--muted);font-size:.73rem">{role}</span> {badge}
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:var(--muted);font-size:.82rem">No users registered yet.</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: FACE RECOGNITION
# ─────────────────────────────────────────────
def page_face_recognition():
    user = st.session_state.scanned_user
    if not user:
        st.error("No QR data found.")
        if st.button("← Back to QR Scanner"): go_to("qr_scanner"); st.rerun()
        return

    render_steps(2)
    users      = load_user_data()
    user_info  = users.get(user["user_id"], {})
    role       = user_info.get("role", "–")
    stored_enc = user_info.get("face_encoding")

    st.markdown(f"## 👤 Face Verification")
    st.info(f"Verifying **{user['name']}** · {role} · {user['department']}")

    if stored_enc is None:
        st.error("❌ No face encoding registered for this user. Contact admin.")
        if st.button("← Back"): go_to("qr_scanner"); st.rerun()
        return

    col1, col2 = st.columns([3, 1], gap="large")
    with col1:
        run      = st.checkbox("▶  Start Camera Feed", value=True, key="face_cam_run")
        frame_ph = st.empty()
        status_ph= st.empty()
        progress = st.progress(0)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot access camera.")
            return

        match_found = False
        confidence  = 0.0
        t0          = time.time()
        TIMEOUT     = 30

        while run and (time.time() - t0) < TIMEOUT:
            ret, frame = cap.read()
            if not ret: break
            frame_ph.image(cv2.cvtColor(draw_face_boxes(frame), cv2.COLOR_BGR2RGB),
                           channels="RGB", use_column_width=True)
            progress.progress(min(int((time.time()-t0)/TIMEOUT*100), 99))
            live_enc = get_face_encoding(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if live_enc is not None:
                matched, confidence = compare_faces(stored_enc, live_enc)
                if matched:
                    status_ph.success(f"✅ Verified! Confidence: {confidence}%")
                    match_found = True
                    time.sleep(0.8)
                    break
                else:
                    status_ph.warning(f"🔄 Scanning… {confidence}% (need ≥{round((1-FACE_MATCH_THRESH)*100)}%)")
            else:
                status_ph.info("🔎 Looking for face…")
            time.sleep(0.04)

        cap.release()
        progress.progress(100)

        if match_found:
            st.session_state.face_verified = True
            st.session_state.confidence    = confidence
            go_to("attendance")
            st.rerun()
        elif not run:
            st.warning("Camera paused.")
        else:
            st.error("❌ Verification failed. Returning to scanner…")
            time.sleep(2)
            go_to("qr_scanner")
            st.session_state.error_msg = "Face mismatch. Please try again."
            st.rerun()

    with col2:
        st.markdown(f"""
        <div class="att-card">
            <div style="font-family:'Syne',sans-serif;font-weight:700;margin-bottom:.8rem">Tips</div>
            <ul style="color:var(--muted);font-size:.85rem;line-height:2;padding-left:1.1rem">
                <li>Face camera directly</li>
                <li>Good lighting required</li>
                <li>Remove glasses if needed</li>
                <li>Only one face in frame</li>
                <li>Stay still during scan</li>
            </ul>
            <div style="margin-top:1rem;padding-top:1rem;border-top:1px solid var(--border)">
                <span class="badge badge-blue">Threshold: ≥{round((1-FACE_MATCH_THRESH)*100)}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Cancel & Back", use_container_width=True):
            go_to("qr_scanner"); st.rerun()

# ─────────────────────────────────────────────
# PAGE: ATTENDANCE
# ─────────────────────────────────────────────
def page_attendance():
    user = st.session_state.scanned_user
    if not user or not st.session_state.face_verified:
        st.error("Verification incomplete.")
        if st.button("← Back"): go_to("qr_scanner"); st.rerun()
        return

    render_steps(3)

    if not st.session_state.attendance_done:
        users      = load_user_data()
        user_info  = users.get(user["user_id"], {})
        role       = user_info.get("role", "–")
        department = user_info.get("department", user.get("department",""))
        action     = mark_attendance(user["user_id"], user["name"], role, department, st.session_state.confidence)
        st.session_state.attendance_done   = True
        st.session_state.attendance_action = action
        st.session_state.redirect_time     = time.time()
        st.session_state.att_role          = role

    action = st.session_state.get("attendance_action","login")
    role   = st.session_state.get("att_role","–")
    now    = datetime.now()

    if action == "login":
        st.markdown("## ✅ Login Recorded")
        st.success(f"🎉 Welcome, **{user['name']}**! Your login has been marked.")
    else:
        st.markdown("## ✅ Logout Recorded")
        st.success(f"👋 Goodbye, **{user['name']}**! See you next time.")

    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("👤 Name",       user["name"])
    c2.metric("🆔 Employee ID", user["user_id"])
    c3.metric("💼 Role",        role)
    c4, c5, c6 = st.columns(3)
    c4.metric("🏢 Department",  user["department"])
    c5.metric("🕐 Time",        now.strftime("%I:%M:%S %p"))
    c6.metric("🔐 Confidence",  f"{st.session_state.confidence}%")

    st.markdown("---")
    st.markdown("### 🕐 Recent Activity")

    history_df = get_user_history(user_id=user["user_id"])
    if not history_df.empty:
        show = history_df[["date","time","event_type","confidence_pct","duration_since_login"]].head(8).copy()
        show.columns = ["Date","Time","Event","Confidence %","Session Duration"]

        def style_event(val):
            if val == "LOGIN":  return "background-color:#0d2e1a;color:#22c55e;font-weight:700;text-align:center;"
            if val == "LOGOUT": return "background-color:#0d1e35;color:#4f8ef7;font-weight:700;text-align:center;"
            return ""

        st.dataframe(show.style.applymap(style_event, subset=["Event"]),
                     use_container_width=True, hide_index=True)
        if st.button("📖 Full History"):
            go_to("history"); st.rerun()
    else:
        st.info("No history yet.")

    st.markdown("---")
    elapsed   = time.time() - st.session_state.redirect_time
    remaining = max(0, AUTO_REDIRECT_SEC - int(elapsed))
    st.info(f"⏳ Returning to QR scanner in **{remaining}s**…")

    if remaining == 0:
        go_to("qr_scanner"); st.rerun()
    else:
        time.sleep(1); st.rerun()

    if st.button("← Back to Scanner", use_container_width=True):
        go_to("qr_scanner"); st.rerun()

# ─────────────────────────────────────────────
# PAGE: HISTORY
# ─────────────────────────────────────────────
def page_history():
    st.markdown("## 🕐 Login / Logout History")
    st.markdown('<p style="color:var(--muted);font-size:.9rem;margin-top:-.5rem">Complete audit trail of every verified check-in and check-out.</p>', unsafe_allow_html=True)

    df    = load_history()
    users = load_user_data()

    if df.empty:
        st.info("No history recorded yet.")
        return

    st.markdown("---")
    st.markdown("### 🔍 Filters")
    f1, f2, f3, f4 = st.columns(4)

    with f1:
        user_opts = ["All Users"] + sorted(df["user_id"].unique().tolist())
        sel_user  = st.selectbox("Employee", user_opts,
                                  format_func=lambda x: x if x=="All Users"
                                  else f"{x} · {users.get(x,{}).get('name',x)}")
    with f2:
        date_opts = ["All Dates"] + sorted(df["date"].unique().tolist(), reverse=True)
        sel_date  = st.selectbox("Date", date_opts)
    with f3:
        sel_event = st.selectbox("Event Type", ["All","LOGIN","LOGOUT"])
    with f4:
        role_opts  = ["All Roles"] + sorted(df["role"].dropna().unique().tolist()) if "role" in df.columns else ["All Roles"]
        sel_role   = st.selectbox("Role", role_opts)

    filtered = df.copy()
    if sel_user  != "All Users":  filtered = filtered[filtered["user_id"]    == sel_user]
    if sel_date  != "All Dates":  filtered = filtered[filtered["date"]        == sel_date]
    if sel_event != "All":        filtered = filtered[filtered["event_type"] == sel_event]
    if sel_role  != "All Roles" and "role" in filtered.columns:
        filtered = filtered[filtered["role"] == sel_role]
    filtered = filtered.sort_values(["date","time"], ascending=[False,False]).reset_index(drop=True)

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🟢 Logins",      len(filtered[filtered["event_type"]=="LOGIN"]))
    m2.metric("🔵 Logouts",     len(filtered[filtered["event_type"]=="LOGOUT"]))
    m3.metric("👥 Unique Users", filtered["user_id"].nunique())
    m4.metric("📅 Days",         filtered["date"].nunique())

    st.markdown("---")
    st.markdown(f"### 📋 Event Log <span style='font-size:.85rem;color:var(--muted);font-weight:400'>({len(filtered)} records)</span>", unsafe_allow_html=True)

    if filtered.empty:
        st.warning("No records match the selected filters.")
    else:
        show_cols = ["date","time","user_id","name","role","department","event_type","confidence_pct","duration_since_login"]
        avail     = [c for c in show_cols if c in filtered.columns]
        display   = filtered[avail].copy()
        display.columns = [c.replace("_"," ").title() for c in avail]

        def style_ev(val):
            if val == "LOGIN":  return "background:#0d2e1a;color:#22c55e;font-weight:700;text-align:center;"
            if val == "LOGOUT": return "background:#0d1e35;color:#4f8ef7;font-weight:700;text-align:center;"
            return ""

        ev_col = "Event Type" if "Event Type" in display.columns else "Eventtype"
        try:
            st.dataframe(display.style.applymap(style_ev, subset=[ev_col]),
                         use_container_width=True, hide_index=True, height=450)
        except Exception:
            st.dataframe(display, use_container_width=True, hide_index=True, height=450)

        buf = io.BytesIO()
        filtered.to_csv(buf, index=False); buf.seek(0)
        st.download_button("⬇ Download CSV", data=buf,
                           file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv", use_container_width=True)

    st.markdown("---")
    st.markdown("### 👤 Per-User Daily Timeline")
    if users:
        uid_tl = st.selectbox("Select Employee", list(users.keys()),
                               format_func=lambda x: f"{x} · {users[x]['name']} · {users[x].get('role','–')}",
                               key="tl_uid")
        uhist = df[df["user_id"]==uid_tl].sort_values(["date","time"]).copy()
        if uhist.empty:
            st.info("No history for this employee yet.")
        else:
            for d in sorted(uhist["date"].unique(), reverse=True):
                dev = uhist[uhist["date"]==d]
                lc  = len(dev[dev["event_type"]=="LOGIN"])
                lo  = len(dev[dev["event_type"]=="LOGOUT"])
                expanded = (d == date.today().strftime("%Y-%m-%d"))
                with st.expander(f"📅 {d}  ·  🟢 {lc} login(s)  ·  🔵 {lo} logout(s)", expanded=expanded):
                    for _, row in dev.iterrows():
                        icon = "🟢" if row["event_type"]=="LOGIN" else "🔵"
                        dur  = f"  ·  session: **{row['duration_since_login']}**" if row.get("duration_since_login") else ""
                        role_tag = f" · {row['role']}" if row.get("role") else ""
                        st.markdown(f"{icon} **{row['event_type']}** — {row['time']}{role_tag}  ·  conf: {row['confidence_pct']}%{dur}")

# ─────────────────────────────────────────────
# PAGE: ADMIN
# ─────────────────────────────────────────────
def page_admin():
    st.markdown("## ⚙️ Admin Panel")
    st.markdown('<p style="color:var(--muted);font-size:.9rem;margin-top:-.5rem">Manage employees, enrol biometrics, generate credentials, and view reports.</p>', unsafe_allow_html=True)

    tabs = st.tabs(["➕ Register Employee", "📸 Enrol Face", "🔑 Generate QR",
                    "📋 Attendance Log", "🕐 History Log", "👥 Employee Directory"])

    # ── Tab 0: Register ──────────────────────
    with tabs[0]:
        st.markdown("### Register New Employee")
        with st.form("register_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                uid  = st.text_input("Employee ID", placeholder="e.g. EMP001")
                name = st.text_input("Full Name",    placeholder="e.g. John Doe")
            with c2:
                dept = st.text_input("Department",   placeholder="e.g. Engineering")
                role = st.selectbox("Job Role", ROLE_OPTIONS)

            submitted = st.form_submit_button("✚  Register Employee", use_container_width=True)

        if submitted:
            if not uid or not name or not dept:
                st.error("Employee ID, Name and Department are required.")
            elif role == "Select Role":
                st.error("Please select a job role.")
            else:
                users = load_user_data()
                if uid in users:
                    users[uid].update({"name":name,"department":dept,"role":role,
                                        "qr_data":f"{uid}|{name}|{dept}"})
                    st.warning(f"Employee **{uid}** already existed — details updated.")
                else:
                    users[uid] = {"user_id":uid,"name":name,"department":dept,"role":role,
                                  "face_encoding":None,"qr_data":f"{uid}|{name}|{dept}"}
                    st.success(f"✅ **{name}** registered as *{role}* in {dept}!")
                save_user_data(users)

    # ── Tab 1: Enrol Face ────────────────────
    with tabs[1]:
        st.markdown("### Enrol Face Encoding")
        users = load_user_data()
        if not users:
            st.warning("No employees registered. Register first.")
        else:
            uid_sel = st.selectbox("Select Employee", list(users.keys()),
                                   format_func=lambda x: f"{x} · {users[x]['name']} · {users[x].get('role','–')}")
            info    = users[uid_sel]
            has_face = info.get("face_encoding") is not None
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("Name",       info.get("name",""))
            fc2.metric("Role",       info.get("role","–"))
            fc3.metric("Face Status", "Enrolled ✅" if has_face else "Pending ⚠️")

            run_cam = st.checkbox("▶  Open Camera for Face Capture", key="enrol_cam")
            fph = st.empty(); sph = st.empty()

            if run_cam:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Cannot access camera.")
                else:
                    captured_enc = None
                    t0 = time.time()
                    while time.time()-t0 < 15:
                        ret, frame = cap.read()
                        if not ret: break
                        fph.image(cv2.cvtColor(draw_face_boxes(frame), cv2.COLOR_BGR2RGB),
                                  channels="RGB", use_column_width=True)
                        enc = get_face_encoding(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        if enc is not None:
                            sph.success("✅ Face detected! Saving…")
                            captured_enc = enc
                            time.sleep(1); break
                        else:
                            sph.info(f"🔎 Looking for face… ({int(time.time()-t0)}s)")
                        time.sleep(0.04)
                    cap.release()
                    if captured_enc is not None:
                        users[uid_sel]["face_encoding"] = captured_enc
                        save_user_data(users)
                        st.success(f"✅ Face enrolled for **{users[uid_sel]['name']}**!")
                    else:
                        st.error("No face detected within 15 seconds.")

    # ── Tab 2: Generate QR ───────────────────
    with tabs[2]:
        st.markdown("### Generate Employee QR Code")
        users = load_user_data()
        if not users:
            st.warning("No employees registered.")
        else:
            uid_qr = st.selectbox("Select Employee", list(users.keys()),
                                  format_func=lambda x: f"{x} · {users[x]['name']} · {users[x].get('role','–')}",
                                  key="qr_sel")
            u = users[uid_qr]
            qc1, qc2 = st.columns([1,2])
            with qc1:
                if st.button("⬡  Generate QR Code", use_container_width=True):
                    img = generate_qr(u["user_id"], u["name"], u["department"])
                    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
                    st.image(img, width=220)
                    st.download_button("⬇ Download QR", data=buf,
                                       file_name=f"QR_{uid_qr}.png", mime="image/png")
            with qc2:
                st.markdown(f"""
                <div class="att-card" style="margin-top:.5rem">
                    <div style="font-family:'Syne',sans-serif;font-weight:700;margin-bottom:.8rem">Employee Info</div>
                    <table style="width:100%;font-size:.85rem;color:var(--muted);border-collapse:collapse">
                        <tr><td style="padding:.3rem 0;color:var(--text);font-weight:600">ID</td><td>{u['user_id']}</td></tr>
                        <tr><td style="padding:.3rem 0;color:var(--text);font-weight:600">Name</td><td>{u['name']}</td></tr>
                        <tr><td style="padding:.3rem 0;color:var(--text);font-weight:600">Role</td><td>{u.get('role','–')}</td></tr>
                        <tr><td style="padding:.3rem 0;color:var(--text);font-weight:600">Department</td><td>{u['department']}</td></tr>
                        <tr><td style="padding:.3rem 0;color:var(--text);font-weight:600">Face</td><td>{'✅ Enrolled' if u.get('face_encoding') is not None else '❌ Not enrolled'}</td></tr>
                    </table>
                    <div style="margin-top:1rem;padding-top:1rem;border-top:1px solid var(--border);
                         font-size:.78rem;color:var(--muted)">QR encodes:
                        <code style="color:var(--accent)">{u['user_id']}|{u['name']}|{u['department']}</code>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Tab 3: Attendance Log ────────────────
    with tabs[3]:
        st.markdown("### Attendance Log")
        df = load_attendance()
        if df.empty:
            st.info("No attendance records yet.")
        else:
            today = date.today().strftime("%Y-%m-%d")
            show  = st.radio("View", ["Today","All Time"], horizontal=True)
            view  = df[df["date"]==today] if show=="Today" else df
            st.dataframe(view.sort_values("date", ascending=False), use_container_width=True)
            buf = io.BytesIO(); view.to_csv(buf, index=False); buf.seek(0)
            st.download_button("⬇ Export CSV", data=buf,
                               file_name="attendance_export.csv", mime="text/csv")

    # ── Tab 4: History Log ───────────────────
    with tabs[4]:
        st.markdown("### Login / Logout History Log")
        df = load_history()
        if df.empty:
            st.info("No history recorded yet.")
        else:
            st.dataframe(df.sort_values(["date","time"], ascending=[False,False]),
                         use_container_width=True)
            buf = io.BytesIO(); df.to_csv(buf, index=False); buf.seek(0)
            st.download_button("⬇ Export Full History", data=buf,
                               file_name="history_export.csv", mime="text/csv")
            st.markdown("---")
            if st.button("🗑  Clear All History", type="primary"):
                if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
                st.success("History cleared."); st.rerun()

    # ── Tab 5: Employee Directory ────────────
    with tabs[5]:
        st.markdown("### Employee Directory")
        users = load_user_data()
        if not users:
            st.info("No employees registered.")
        else:
            rows = []
            for uid, info in users.items():
                rows.append({
                    "Employee ID":   uid,
                    "Name":          info.get("name",""),
                    "Role":          info.get("role","–"),
                    "Department":    info.get("department",""),
                    "Face Enrolled": "✅ Yes" if info.get("face_encoding") is not None else "❌ No",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("### 🗑 Remove Employee")
            del_uid = st.selectbox("Select employee to remove", list(users.keys()),
                                   format_func=lambda x: f"{x} · {users[x]['name']} · {users[x].get('role','–')}",
                                   key="del_sel")
            ec1, ec2 = st.columns([1,3])
            with ec1:
                if st.button("Remove Employee", type="primary"):
                    del users[del_uid]
                    save_user_data(users)
                    st.success("Employee removed."); st.rerun()
            with ec2:
                u = users.get(del_uid, {})
                st.markdown(f'<p style="color:var(--muted);font-size:.82rem;padding-top:.4rem">Will remove: <b style="color:var(--red)">{u.get("name","?")} ({del_uid})</b> · {u.get("role","–")}</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────
def main():
    render_sidebar()
    page = st.session_state.page
    if   page == "qr_scanner":      page_qr_scanner()
    elif page == "face_recognition": page_face_recognition()
    elif page == "attendance":       page_attendance()
    elif page == "history":          page_history()
    elif page == "admin":            page_admin()

if __name__ == "__main__":
    main()