import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import sys
import os
from pathlib import Path
from collections import deque
from datetime import datetime

# ── 팀 모듈 import ──────────────────────────────────────────────
# Dashboard/ 폴더 → 프로젝트 루트로 경로 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import torch
    from feature_extract.feature_extractor import FeatureExtractor
    from model import DrowsyLSTM
    TEAM_MODULES_AVAILABLE = True
except Exception as e:
    print(f"[경고] 팀 모듈 로드 실패: {e}")
    TEAM_MODULES_AVAILABLE = False

# BiLSTM은 선택적 — 모델 담당이 model.py에 추가하기 전까지는 없을 수 있음.
# 없어도 LSTM 모드는 정상 동작하도록 분리해서 import.
try:
    from model import DrowsyBiLSTM
    BILSTM_AVAILABLE = True
except Exception:
    BILSTM_AVAILABLE = False

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="실시간 졸음 감지 모니터링 시스템",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
# 색상 팔레트:
#   BLUE   #2563EB  → 헤더, 섹션 레이블, 버튼, 현황 패널 좌선
#   VIOLET #7C3AED  → 로그 패널 좌선, 스크롤바
#   SLATE  #1E293B  → 텍스트, 탑바 배경
#   BG     #FFFFFF  → 페이지 배경
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background: #FFFFFF !important;
    color: #1E293B;
    font-family: 'DM Sans', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp > div:first-child { padding: 0 !important; }

/* ── Top bar ── */
.topbar {
    background: #1E293B;
    padding: 12px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 3px solid #2563EB;
}
.topbar-left { display: flex; align-items: baseline; gap: 14px; }
.topbar-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: 3px;
}
.topbar-badge {
    background: #2563EB;
    color: #fff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 2px;
    padding: 2px 8px;
    border-radius: 2px;
}
.topbar-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #94A3B8;
}

/* ── Section label ── */
.section-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 2px;
    color: #2563EB;
    text-transform: uppercase;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.section-label::before {
    content: "";
    display: inline-block;
    width: 3px; height: 12px;
    background: #2563EB;
    border-radius: 2px;
}

/* ── Log panel ── */
.log-panel {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-left: 3px solid #7C3AED;
    border-radius: 6px;
    height: 454px;
    overflow-y: auto;
    padding: 10px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
}
.log-panel::-webkit-scrollbar { width: 4px; }
.log-panel::-webkit-scrollbar-track { background: #F1F5F9; }
.log-panel::-webkit-scrollbar-thumb { background: #7C3AED; border-radius: 4px; }

.log-entry {
    display: flex; align-items: center; gap: 8px;
    padding: 4px 0;
    border-bottom: 1px solid #F1F5F9;
    animation: fadeSlide 0.25s ease;
}
@keyframes fadeSlide {
    from { opacity: 0; transform: translateX(-6px); }
    to   { opacity: 1; transform: translateX(0); }
}
.log-time  { color: #94A3B8; font-size: 9px; white-space: nowrap; min-width: 54px; }
.log-score { font-size: 13px; font-weight: 700; min-width: 36px; }
.log-stage { font-size: 9px; padding: 2px 6px; border-radius: 3px; font-weight: 600; letter-spacing: 1px; }

/* ── Alert panel ── */
.alert-panel {
    border-radius: 8px;
    padding: 22px 16px;
    text-align: center;
    min-height: 160px;
    max-height: 160px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 6px;
    transition: all 0.3s ease;
    border: 2px solid transparent;
}
.stage-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 56px; font-weight: 800; line-height: 1;
}
.stage-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px; letter-spacing: 2px; font-weight: 700;
}

/* ── Activity panel ── */
.activity-panel {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-left: 3px solid #7C3AED;
    border-radius: 6px;
    padding: 14px; min-height: 286px;
}
.activity-row  { margin-bottom: 10px; }
.activity-time { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #94A3B8; margin-bottom: 2px; }
.activity-text { font-size: 12px; color: #334155; line-height: 1.5; font-weight: 500; }

/* ── Gauge ── */
.gauge-wrap {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 10px 14px; margin-top: 8px;
}
.gauge-track {
    height: 30px; background: #E2E8F0;
    border-radius: 4px; position: relative; overflow: visible;
}
.gauge-segments { display: flex; height: 100%; overflow: hidden; border-radius: 4px; }
.gauge-seg { border-right: 2px solid #fff; }
.gauge-fill {
    position: absolute; top: 0; left: 0; height: 100%;
    border-radius: 4px;
    transition: width 0.4s cubic-bezier(0.4,0,0.2,1);
    background: linear-gradient(90deg, #22C55E, #EAB308, #F97316, #EF4444, #DC2626);
    opacity: 0.85;
}
.gauge-siren {
    position: absolute; top: 50%; transform: translate(-50%,-50%);
    font-size: 22px;
    transition: left 0.4s cubic-bezier(0.4,0,0.2,1);
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.25));
    z-index: 10;
}
.gauge-ticks {
    display: flex; justify-content: space-between; margin-top: 6px;
    font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #94A3B8;
}

/* ── Camera standby ── */
.cam-standby {
    background: #F8FAFC; border: 2px dashed #CBD5E1; border-radius: 8px;
    height: 454px; display: flex; align-items: center; justify-content: center;
    flex-direction: column; gap: 12px;
}

/* ── Buttons ── */
div[data-testid="stButton"] button {
    background: #2563EB !important;
    border: none !important;
    color: #ffffff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    border-radius: 6px !important;

    padding: 6px 12px !important;
    min-height: 38px !important;
    margin-top: 0px !important;
    line-height: 1 !important;

    transition: all 0.15s !important;
    width: 100%;
    box-shadow: 0 1px 3px rgba(37,99,235,0.3) !important;
}
div[data-testid="stButton"] button:hover {
    background: #1D4ED8 !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.4) !important;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] {
    margin: 0 !important;
    padding: 0 !important;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button {
    margin: 0 !important;
}

/* ── REC indicator ── */
.rec-indicator {
    display: inline-flex; align-items: center; gap: 5px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #EF4444; font-weight: 700;
}
.rec-dot {
    width: 7px; height: 7px; background: #EF4444;
    border-radius: 50%; animation: blink 1s step-start infinite;
}
@keyframes blink { 50% { opacity: 0; } }

/* ── Alert animations ── */
@keyframes sirenPulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(220,38,38,0.35); }
    50%      { box-shadow: 0 0 0 18px rgba(220,38,38,0); }
}
.alert-danger { animation: sirenPulse 0.7s ease-out infinite; }

@keyframes warnPulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(249,115,22,0.3); }
    50%      { box-shadow: 0 0 0 12px rgba(249,115,22,0); }
}
.alert-warn { animation: warnPulse 1.2s ease-out infinite; }

.log-index {
    color: #64748B;
    font-size: 10px;
    font-weight: 700;
    min-width: 30px;
}

</style>
""", unsafe_allow_html=True)

# ── MediaPipe (시각화용 랜드마크 표시에만 사용) ─────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

def euclidean(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def calc_ear(lm, eye_idx):
    pts = [lm[i] for i in eye_idx]
    A = euclidean(pts[1], pts[5])
    B = euclidean(pts[2], pts[4])
    C = euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C else 0

def calc_perclos(ear_buf):
    if not ear_buf: return 0.0
    return sum(1 for e in ear_buf if e < 0.21) / len(ear_buf)

def perclos_to_score(perclos):
    """LSTM 모델 미사용 시 폴백용 — 단순 EAR 규칙."""
    return min(100.0, perclos * 250)


# ── LSTM 추론 관련 ──────────────────────────────────────────────────────────────
WINDOW_SIZE   = 30   # train.py / dataset.py와 동일
LSTM_CLASSES  = 3    # NORMAL / WARNING / DANGER

# LSTM 3클래스 → 0~100 점수 매핑 가중치
# softmax 확률에 가중치를 곱해 부드러운 점수 산출
CLASS_SCORE_WEIGHTS = np.array([0.0, 50.0, 100.0], dtype=np.float32)


@st.cache_resource
def load_lstm_model(model_type: str = "lstm"):
    """
    학습된 LSTM/BiLSTM 가중치를 로드.

    Args:
        model_type: "lstm" 또는 "bilstm"
    Returns:
        (model, device) 또는 (None, None) — 로드 실패 시
    """
    if not TEAM_MODULES_AVAILABLE:
        return None, None

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # BiLSTM 요청인데 모델 클래스가 없으면(모델 담당 미반영) LSTM으로 자동 폴백
    if model_type == "bilstm" and BILSTM_AVAILABLE:
        weight_path = ROOT / "models" / "best_bilstm_model.pth"
        model = DrowsyBiLSTM(input_dim=4, hidden_dim=64, num_layers=2, num_classes=3)
    else:
        weight_path = ROOT / "models" / "best_lstm_model.pth"
        model = DrowsyLSTM(input_dim=4, hidden_dim=64, num_layers=2, num_classes=3)

    if not weight_path.exists():
        print(f"[경고] 가중치 없음: {weight_path}")
        return None, None

    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device).eval()
        return model, device
    except Exception as e:
        print(f"[경고] 모델 로드 실패: {e}")
        return None, None


@st.cache_resource
def load_feature_extractor():
    """장원석 팀원의 통합 추출기 (MediaPipe + EAR + Head Pose 한 번에)."""
    if not TEAM_MODULES_AVAILABLE:
        return None
    return FeatureExtractor()


def predict_with_lstm(model, device, frame_window: deque) -> tuple:
    """
    30프레임 시퀀스를 LSTM에 넣어 졸음 점수와 클래스 예측.

    Returns:
        (score 0~100, pred_class 0/1/2, probs np.array)
    """
    arr = np.array(list(frame_window), dtype=np.float32)   # (30, 4)
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device) # (1, 30, 4)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (3,)
        pred   = int(np.argmax(probs))

    # softmax 확률 가중합으로 0~100 점수 계산
    score = float(np.dot(probs, CLASS_SCORE_WEIGHTS))
    return score, pred, probs

# ── Stage config ──────────────────────────────────────────────────────────────────
STAGE_CFG = [
    {"label": "정상", "color": "#16A34A", "bg": "#DCFCE7", "border": "#86EFAC", "text": "#14532D"},
    {"label": "주의", "color": "#CA8A04", "bg": "#FEF9C3", "border": "#FDE047", "text": "#713F12"},
    {"label": "경고", "color": "#EA580C", "bg": "#FFEDD5", "border": "#FDBA74", "text": "#7C2D12"},
    {"label": "위험", "color": "#DC2626", "bg": "#FEE2E2", "border": "#FCA5A5", "text": "#7F1D1D"},
    {"label": "긴급", "color": "#9F1239", "bg": "#FFE4E6", "border": "#FDA4AF", "text": "#4C0519"},
]

def score_to_stage(score, thresholds):
    t = thresholds
    if score < t[1]: return 0
    if score < t[2]: return 1
    if score < t[3]: return 2
    if score < t[4]: return 3
    return 4

# ── Session state ─────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "running": False,
        "score": 0.0,
        "ear_buffer": deque(maxlen=150),
        "feature_window": deque(maxlen=WINDOW_SIZE),  # LSTM 입력용 30프레임 버퍼
        "log_entries": [],
        "thresholds": [0, 20, 40, 60, 80, 100],
        "last_log_time": 0,
        "activity_messages": [],
        "alert_stage": 0,
        "stage_start_time": None,
        "auto_called": False,
        "cap": None,
        # 기본은 LSTM (어디서든 보장). BiLSTM은 모델 담당 반영 후 선택 가능.
        "model_type": "bilstm" if BILSTM_AVAILABLE else "lstm",
        "last_pred_class": 0,         # 마지막 LSTM 예측 클래스
        "last_probs": None,           # 마지막 LSTM 확률 분포
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# 모델/추출기 로드 (캐시되어 한 번만 실행됨)
lstm_model, lstm_device = load_lstm_model(st.session_state["model_type"])
feature_extractor       = load_feature_extractor()
LSTM_READY = lstm_model is not None and feature_extractor is not None

# ── Helpers ───────────────────────────────────────────────────────────────────────
def adapt_thresholds(log_entries):
    if len(log_entries) < 20:
        add_activity("⚠ 로그 데이터 부족 (최소 20개 필요)")
        return
    scores = [e["score"] for e in log_entries]
    median = float(np.median(scores))
    shift  = max(0.0, median - 40.0)
    step   = (100.0 - shift) / 5
    new_t  = [round(shift + step * i) for i in range(6)]
    new_t[0] = 0; new_t[-1] = 100
    st.session_state["thresholds"] = new_t
    st.session_state["score"] = 0.0
    add_activity(f"⚙ 임계값 변경: {new_t[0]}→{new_t[1]}→{new_t[2]}→{new_t[3]}→{new_t[4]}→{new_t[5]}")

def reset_thresholds():
    st.session_state["thresholds"] = [0, 20, 40, 60, 80, 100]
    st.session_state["score"] = 0.0
    st.session_state["log_entries"] = []
    st.session_state["auto_called"] = False
    add_activity("🔄 임계값 초기화 완료")

def add_activity(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state["activity_messages"].append({"time": ts, "msg": msg})
    st.session_state["activity_messages"] = st.session_state["activity_messages"][-10:]

# ── Render helpers ────────────────────────────────────────────────────────────────
def normalize_score(score, thresholds):
    t = thresholds

    for i in range(5):
        low = t[i]
        high = t[i + 1]

        if low <= score <= high:
            ratio = (score - low) / (high - low)
            return (i * 20) + (ratio * 20)

    return 100

def render_gauge(score, thresholds):
    pct = normalize_score(score, thresholds)
    t   = thresholds
    seg_widths = [t[i+1] - t[i] for i in range(5)]
    seg_colors = [
        "rgba(34,197,94,0.18)",
        "rgba(234,179,8,0.18)",
        "rgba(249,115,22,0.18)",
        "rgba(239,68,68,0.18)",
        "rgba(159,18,57,0.18)",
    ]
    segs_html = "".join(
        f'<div class="gauge-seg" style="flex:{w};background:{c};"></div>'
        for w, c in zip(seg_widths, seg_colors)
    )
    tick_html = "".join(f'<span>{v}</span>' for v in t)
    return f"""
    <div class="gauge-wrap">
        <div class="section-label">졸음 위험 점수</div>
        <div class="gauge-track">
            <div class="gauge-segments">{segs_html}</div>
            <div class="gauge-fill" style="width:{pct}%;"></div>
            <div class="gauge-siren" style="left:{pct}%;">🚨</div>
        </div>
        <div class="gauge-ticks">{tick_html}</div>
    </div>
    """

def render_log(log_entries, thresholds):
    if not log_entries:
        return ('<div class="log-panel" style="min-height:527px;">'
                '<span style="color:#CBD5E1;font-size:11px;">'
                '로그 없음 — 모니터링을 시작하세요</span></div>')
    rows = ""
    for idx, e in enumerate(reversed(log_entries[-80:]), start=1):
        stage = score_to_stage(e["score"], thresholds)
        cfg   = STAGE_CFG[stage]
        rows += (
            f'<div class="log-entry">'
            f'<span class="log-index">#{len(log_entries)-idx+1}</span>'
            f'<span class="log-time">{e["time"]}</span>'
            f'<span class="log-score" style="color:{cfg["color"]};">{e["score"]:.1f}</span>'
            f'<span class="log-stage" style="background:{cfg["bg"]};color:{cfg["text"]};'
            f'border:1px solid {cfg["border"]};">{cfg["label"]}</span>'
            f'</div>'
        )
    return f'<div class="log-panel" style="min-height:527px;">{rows}</div>'

def render_alert(stage):
    cfg   = STAGE_CFG[stage]
    pulse = "alert-danger" if stage == 4 else ("alert-warn" if stage >= 2 else "")
    icon  = "🚨" if stage >= 3 else ("⚠️" if stage >= 2 else "🔔")

    return (
        f'<div class="alert-panel {pulse}" '
        f'style="background:{cfg["bg"]};border-color:{cfg["border"]};position:relative;">'

        f'<div style="position:absolute;top:10px;right:12px;font-size:24px;">{icon}</div>'

        f'<div class="stage-number" style="color:{cfg["color"]};">{stage}</div>'
        f'<div class="stage-label"  style="color:{cfg["text"]};">{cfg["label"]}</div>'
        f'</div>'
    )

def render_activity(messages):
    if not messages:
        body = '<span style="color:#CBD5E1;font-size:11px;">시스템 대기 중...</span>'
    else:
        body = "".join(
            f'<div class="activity-row">'
            f'<div class="activity-time">{m["time"]}</div>'
            f'<div class="activity-text">{m["msg"]}</div>'
            f'</div>'
            for m in reversed(messages[-6:])
        )
    return (
        '<div class="section-label">실시간 현황</div>'
        f'<div class="activity-panel" style="min-height:370px;">{body}</div>'
    )

# ── Top bar ───────────────────────────────────────────────────────────────────────
now_str    = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
st.markdown(
    f'<div class="topbar">'
    f'<div class="topbar-left">'
    f'<span class="topbar-title">🚨 실시간 졸음 감지 모니터링 시스템</span>'
    f'<span class="topbar-badge">DROWSY-DETECTION</span>'
    f'</div>'
    f'<div class="topbar-time">{now_str}</div>'
    f'</div>',
    unsafe_allow_html=True,
)
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# ── 3-column layout ───────────────────────────────────────────────────────────────
col_left, col_mid, col_right = st.columns([1, 2.2, 1])

# ═══════════════════════════════════════
# LEFT — Log + bottom controls
# ═══════════════════════════════════════
with col_left:
    st.markdown(
        '''
        <div style="
            display:flex;
            align-items:center;
        ">
            <div class="section-label" style="margin-bottom:0;">
                측정 로그
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    
    st.markdown("<div style='height:27px;'></div>", unsafe_allow_html=True)
    log_placeholder = st.empty()
    log_placeholder.markdown(
        render_log(
            st.session_state["log_entries"],
            st.session_state["thresholds"]
        ),
        unsafe_allow_html=True,
    )
    
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    left_btn, right_btn = st.columns(2)

    # 변경
    with left_btn:

        if st.button(
            "⚙ 변경",
            key="adapt_btn",
            use_container_width=True
        ):
            adapt_thresholds(
                st.session_state["log_entries"]
            )
            st.rerun()

    # 초기화
    with right_btn:

        if st.button(
            "🔄 초기화",
            key="reset_btn",
            use_container_width=True
        ):
            reset_thresholds()
            st.rerun()
    
# ═══════════════════════════════════════
# MIDDLE — Video frame + Gauge
# ═══════════════════════════════════════
with col_mid:
    header_left, header_right = st.columns([5.5, 1.5])

    with header_left:
        rec_html = ""
        if st.session_state["running"]:
            rec_html = '''
            <span style="
                margin-left:10px;
                display:flex;
                align-items:center;
            ">
                <span class="rec-indicator">
                    <span class="rec-dot"></span>REC
                </span>
            </span>
            '''
        st.markdown(
            f'''
            <div style="
                display:flex;
                align-items:center;
            ">
                <div class="section-label" style="margin-bottom:0;">
                    실시간 모니터링
                </div>

                {rec_html}
            </div>
            ''',
            unsafe_allow_html=True,
        )
    with header_right:
        model_col, real_btn_col = st.columns([1.5, 3])

        with model_col:
            # LSTM ↔ BiLSTM 모델 선택
            # BiLSTM은 모델 담당이 model.py에 반영한 경우에만 옵션으로 노출
            model_options = ["lstm"]
            if BILSTM_AVAILABLE:
                model_options = ["bilstm", "lstm"]

            cur = st.session_state["model_type"]
            cur_index = model_options.index(cur) if cur in model_options else 0

            model_choice = st.selectbox(
                "모델",
                options=model_options,
                index=cur_index,
                key="model_select",
                label_visibility="collapsed",
            )
            if model_choice != st.session_state["model_type"]:
                st.session_state["model_type"] = model_choice
                st.session_state["feature_window"] = deque(maxlen=WINDOW_SIZE)
                add_activity(f"🔄 모델 전환 → {model_choice.upper()}")
                st.rerun()

        with real_btn_col:
            is_webcam_running = st.session_state["running"]

            btn_label = "■ 중지" if is_webcam_running else "▶ 모니터링 시작"

            if st.button(btn_label, key="toggle"):
                if not is_webcam_running:
                    st.session_state["running"]        = True
                    st.session_state["ear_buffer"]     = deque(maxlen=150)
                    st.session_state["feature_window"] = deque(maxlen=WINDOW_SIZE)
                    st.session_state["auto_called"]    = False
                    add_activity("웹캠 모니터링 시작")

                else:
                    st.session_state["running"] = False

                    cap_obj = st.session_state.get("cap")

                    if cap_obj and cap_obj.isOpened():
                        cap_obj.release()
                        st.session_state["cap"] = None

                    add_activity("모니터링 중지")

                st.rerun()
    frame_placeholder = st.empty()
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    gauge_placeholder = st.empty()
    gauge_placeholder.markdown(
        render_gauge(st.session_state["score"], st.session_state["thresholds"]),
        unsafe_allow_html=True,
    )

    if not st.session_state["running"]:
        frame_placeholder.markdown(
            '<div class="cam-standby">'
            '<div style="font-size:52px;">📷</div>'
            '<div style="font-size:13px;color:#94A3B8;font-weight:600;letter-spacing:2px;">STANDBY</div>'
            '<div style="font-size:11px;color:#CBD5E1;">실시간 졸음 감지 모니터링을 실행하세요!</div>'
            "</div>",
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════
# RIGHT — Alert + Activity
# ═══════════════════════════════════════
with col_right:
    st.markdown('<div class="section-label">위험 경보</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:19px;'></div>", unsafe_allow_html=True)
    alert_placeholder = st.empty()
    alert_placeholder.markdown(
        render_alert(st.session_state["alert_stage"]),
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    activity_placeholder = st.empty()
    activity_placeholder.markdown(
        render_activity(st.session_state["activity_messages"]),
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════
# SHARED UI UPDATE FUNCTION
# ═══════════════════════════════════════════════════════════════
LOG_INTERVAL = 1

def update_ui(score, thresholds, prev_score, frame=None):
    stage      = score_to_stage(score, thresholds)
    prev_stage = score_to_stage(prev_score, thresholds)
    st.session_state["alert_stage"] = stage

    if frame is not None:
        frame_placeholder.image(frame, channels="BGR", use_container_width=True)

    # gauge
    gauge_placeholder.markdown(render_gauge(score, thresholds), unsafe_allow_html=True)
    
    #alert panel
    alert_placeholder.markdown(render_alert(stage), unsafe_allow_html=True)
    
    # stage change messages
    if stage != prev_stage:
        msgs = {
            0: "✅ 정상 상태입니다.",
            1: "📢 주의 단계 진입 — 상태를 확인하세요.",
            2: "⚠️ 경고 단계! — 졸음이 감지되었습니다.",
            3: "🔴 위험 단계! — 즉시 휴식을 취하세요!",
            4: "🚨 긴급 단계! (경보 울림) — 즉시 정차하세요!",
        }
        add_activity(msgs.get(stage, ""))
        if stage >= 3:
            st.session_state["stage_start_time"] = time.time()

    # auto 119 call
    if stage == 4 and not st.session_state["auto_called"]:
        t_start = st.session_state.get("stage_start_time")
        if t_start and time.time() - t_start > 5:
            add_activity("📞 긴급 단계 5초 지속! — 119 자동 신고합니다.")
            st.session_state["auto_called"] = True
    if stage < 3:
        st.session_state["auto_called"]    = False
        st.session_state["stage_start_time"] = None

    activity_placeholder.markdown(
        render_activity(st.session_state["activity_messages"]),
        unsafe_allow_html=True,
    )

    # periodic log
    now = time.time()
    if now - st.session_state["last_log_time"] >= LOG_INTERVAL:
        st.session_state["log_entries"].append({
            "time":  datetime.now().strftime("%H:%M:%S"),
            "score": round(score, 1),
            "stage": stage,
        })
        st.session_state["last_log_time"] = now
        log_placeholder.markdown(
            render_log(st.session_state["log_entries"], thresholds),
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════
# WEBCAM LOOP
# ═══════════════════════════════════════════════════════════════
#
# 통합된 파이프라인:
#   웹캠 프레임
#     ├─ feature_extractor.extract_from_frame()  ← 장원석 모듈
#     │     → (ear, pitch, yaw, roll)
#     │
#     ├─ 30프레임 슬라이딩 윈도우 누적
#     │
#     └─ LSTM/BiLSTM 추론  ← 강영한 모듈
#           → 클래스 확률 → 0~100 점수
#
# 모델 로드 실패 시: 기존 EAR-PERCLOS 규칙 폴백
# ═══════════════════════════════════════════════════════════════
if st.session_state["running"]:
    cap_obj = st.session_state.get("cap")
    if cap_obj is None or not cap_obj.isOpened():
        cap_obj = cv2.VideoCapture(0)
        cap_obj.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap_obj.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap_obj.set(cv2.CAP_PROP_FPS, 30)
        st.session_state["cap"] = cap_obj

    # 시각화용 MediaPipe (눈 랜드마크 표시만 담당)
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # 모드 안내
    if LSTM_READY:
        add_activity(f"🧠 LSTM 모델 활성화 ({st.session_state['model_type'].upper()})")
    else:
        add_activity("⚠ LSTM 모델 미사용 — EAR 규칙 폴백 모드")

    frame_count = 0
    while st.session_state["running"]:
        ret, frame = cap_obj.read()
        if not ret:
            add_activity("⚠ 카메라 읽기 실패")
            break

        frame = cv2.flip(frame, 1)

        # ─────────────────────────────────────────────────────
        # 1) 특징 추출 — 팀원 모듈 사용
        # ─────────────────────────────────────────────────────
        ear_val  = None
        features = None

        if LSTM_READY:
            features = feature_extractor.extract_from_frame(frame)
            if features is not None:
                ear_val = features["ear"]
                # LSTM 입력 윈도우에 누적
                st.session_state["feature_window"].append([
                    features["ear"],
                    features["pitch"],
                    features["yaw"],
                    features["roll"],
                ])
                st.session_state["ear_buffer"].append(features["ear"])

        # ─────────────────────────────────────────────────────
        # 2) 눈 랜드마크 시각화 (별도 MediaPipe 호출)
        #    추출기와 중복 호출이지만 UI 표시용으로만 사용
        # ─────────────────────────────────────────────────────
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            if ear_val is None:
                # 폴백 경로: 추출기를 못 쓰면 직접 EAR 계산
                ear_l   = calc_ear(lm, LEFT_EYE)
                ear_r   = calc_ear(lm, RIGHT_EYE)
                ear_val = (ear_l + ear_r) / 2.0
                st.session_state["ear_buffer"].append(ear_val)

            h, w = frame.shape[:2]
            for idx in LEFT_EYE + RIGHT_EYE:
                cx = int(lm[idx].x * w)
                cy = int(lm[idx].y * h)
                cv2.circle(frame, (cx, cy), 2, (37, 99, 235), -1)

        # ─────────────────────────────────────────────────────
        # 3) 점수 계산: LSTM 우선, 실패 시 EAR 규칙 폴백
        # ─────────────────────────────────────────────────────
        perclos    = calc_perclos(st.session_state["ear_buffer"])
        prev_score = st.session_state["score"]

        used_lstm = False
        pred_class = st.session_state["last_pred_class"]
        probs      = st.session_state["last_probs"]

        if LSTM_READY and len(st.session_state["feature_window"]) == WINDOW_SIZE:
            raw_score, pred_class, probs = predict_with_lstm(
                lstm_model, lstm_device, st.session_state["feature_window"]
            )
            used_lstm = True
            st.session_state["last_pred_class"] = pred_class
            st.session_state["last_probs"]      = probs
        else:
            # LSTM 미사용 또는 30프레임 미달 → EAR-PERCLOS 규칙
            raw_score = perclos_to_score(perclos)

        # 지수 평활화 (UI 흔들림 방지)
        st.session_state["score"] = 0.85 * prev_score + 0.15 * raw_score
        score      = st.session_state["score"]
        thresholds = st.session_state["thresholds"]

        # ─────────────────────────────────────────────────────
        # 4) 영상 오버레이
        # ─────────────────────────────────────────────────────
        stage = score_to_stage(score, thresholds)
        cfg   = STAGE_CFG[stage]
        color_rgb = tuple(int(cfg["color"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]),
                      color_bgr, 2 if stage < 3 else 4)

        ear_txt = f"EAR: {ear_val:.3f}" if ear_val else "EAR: --"
        cv2.putText(frame, ear_txt,                          (12, 28),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 1, cv2.LINE_AA)
        cv2.putText(frame, f"PERCLOS: {perclos*100:.1f}%",   (12, 52),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 1, cv2.LINE_AA)
        cv2.putText(frame, f"SCORE: {score:.1f}",            (12, 78),  cv2.FONT_HERSHEY_SIMPLEX, 0.70, color_bgr, 2, cv2.LINE_AA)
        cv2.putText(frame, f"STAGE: {stage}",                (12, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color_bgr, 2, cv2.LINE_AA)

        # 모델 추론 정보
        if used_lstm and probs is not None:
            mode_txt = f"{st.session_state['model_type'].upper()} | N={probs[0]:.2f} W={probs[1]:.2f} D={probs[2]:.2f}"
        elif LSTM_READY:
            need = WINDOW_SIZE - len(st.session_state["feature_window"])
            mode_txt = f"LSTM warming up ({need} frames left)"
        else:
            mode_txt = "EAR fallback (LSTM unavailable)"
        cv2.putText(frame, mode_txt, (12, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        if stage >= 3 and frame_count % 15 < 8:
            cv2.rectangle(frame, (3, 3), (frame.shape[1]-3, frame.shape[0]-3), (220, 38, 38), 3)

        update_ui(score, thresholds, prev_score, frame=frame)
        frame_count += 1
        time.sleep(1 / 30)

    face_mesh.close()
