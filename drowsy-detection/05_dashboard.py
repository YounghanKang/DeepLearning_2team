"""
05_dashboard.py
===============
졸음 운전 탐지 Streamlit 대시보드.

04_demo.py와 동일한 파이프라인을 Streamlit UI로 시각화합니다.
좌측: 실시간 웹캠 + 오버레이
우측: 상태 표시, EAR/PERCLOS 그래프, 통계

실행:
    streamlit run 05_dashboard.py

요구사항:
    pip install streamlit plotly pandas
    + 기존 패키지 (mediapipe, opencv-python, numpy)

본 파일은 PIPELINE.md의 인터페이스 명세에 기반합니다.
"""

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────
# 모듈 import
# ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
for sub in ["eye_extract", "ear", "head_pose"]:
    sys.path.insert(0, str(ROOT / sub))

from eye_extractor import EyeExtractor          # type: ignore
from ear import EARCalculator                   # type: ignore
from head_pose import HeadPoseEstimator         # type: ignore


# ─────────────────────────────────────────────────────────
# 공통 상수
# ─────────────────────────────────────────────────────────
FPS = 30
PERCLOS_WINDOW_SECONDS = 60
PERCLOS_WARNING = 0.10
PERCLOS_DANGER  = 0.30

HEAD_DOWN_PITCH = 20.0
HEAD_POSE_MIN_CONFIDENCE = 0.5

GRAPH_WINDOW = 180   # 그래프에 표시할 최근 프레임 수 (약 6초)


# ─────────────────────────────────────────────────────────
# PERCLOS 임시 구현 (04_demo.py와 동일)
# ─────────────────────────────────────────────────────────
class SimplePerclos:
    def __init__(self, fps=FPS, window_seconds=PERCLOS_WINDOW_SECONDS):
        self.window_size = fps * window_seconds
        self.history = deque(maxlen=self.window_size)

    def update(self, eye_closed: bool) -> float:
        self.history.append(1 if eye_closed else 0)
        return self.get_perclos()

    def get_perclos(self) -> float:
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    def reset(self):
        self.history.clear()


# ─────────────────────────────────────────────────────────
# 신호 융합 함수 (04_demo.py와 동일)
# ─────────────────────────────────────────────────────────
CNN_AVAILABLE = False


def fuse_eye(ear_closed, cnn_closed, cnn_available):
    return cnn_closed if cnn_available else ear_closed


def fuse_head(head_result):
    if head_result is None or head_result.confidence < HEAD_POSE_MIN_CONFIDENCE:
        return False
    return head_result.pitch >= HEAD_DOWN_PITCH


def decide_state(p_eye, p_head):
    combined = max(p_eye, p_head)
    if combined >= PERCLOS_DANGER:  return 'DANGER'
    if combined >= PERCLOS_WARNING: return 'WARNING'
    return 'NORMAL'


# ─────────────────────────────────────────────────────────
# Streamlit 세션 상태 초기화
# ─────────────────────────────────────────────────────────
def init_session_state():
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'modules' not in st.session_state:
        st.session_state.modules = None
    if 'history' not in st.session_state:
        st.session_state.history = {
            'ear': deque(maxlen=GRAPH_WINDOW),
            'pitch': deque(maxlen=GRAPH_WINDOW),
            'perclos_eye': deque(maxlen=GRAPH_WINDOW),
            'perclos_head': deque(maxlen=GRAPH_WINDOW),
            'frame_idx': deque(maxlen=GRAPH_WINDOW),
        }
    if 'stats' not in st.session_state:
        st.session_state.stats = {
            'total_frames': 0,
            'danger_frames': 0,
            'warning_frames': 0,
            'alert_count': 0,
            'last_state': 'NORMAL',
        }


def get_modules():
    if st.session_state.modules is None:
        st.session_state.modules = {
            'eye_ext':   EyeExtractor(),
            'head_est':  HeadPoseEstimator(),
            'ear_calc':  EARCalculator(smoothing_window=5, threshold_ratio=0.7),
            'perclos_e': SimplePerclos(),
            'perclos_h': SimplePerclos(),
            'cap':       cv2.VideoCapture(0),
            'frame_idx': 0,
        }
    return st.session_state.modules


def cleanup_modules():
    if st.session_state.modules is not None:
        st.session_state.modules['eye_ext'].close()
        st.session_state.modules['head_est'].close()
        st.session_state.modules['cap'].release()
        st.session_state.modules = None


# ─────────────────────────────────────────────────────────
# UI 컴포넌트
# ─────────────────────────────────────────────────────────
STATE_STYLES = {
    'NORMAL':  {'color': '#10B981', 'emoji': '🟢', 'msg': '정상'},
    'WARNING': {'color': '#F59E0B', 'emoji': '🟡', 'msg': '주의 (졸음 의심)'},
    'DANGER':  {'color': '#EF4444', 'emoji': '🔴', 'msg': '위험 (졸음 추정)'},
}


def render_state_panel(state, p_eye, p_head):
    style = STATE_STYLES[state]
    st.markdown(
        f"""
        <div style="
            background:{style['color']};
            padding:24px;
            border-radius:12px;
            text-align:center;
            color:white;
            font-weight:700;
        ">
            <div style="font-size:64px;line-height:1;">{style['emoji']}</div>
            <div style="font-size:28px;margin-top:8px;">{state}</div>
            <div style="font-size:14px;opacity:.9;margin-top:4px;">{style['msg']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_ear_chart(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=list(history['ear']),
        mode='lines',
        line=dict(color='#3B82F6', width=2),
        name='EAR',
    ))
    fig.update_layout(
        title='EAR (실시간)',
        height=180,
        margin=dict(l=20, r=10, t=30, b=20),
        yaxis=dict(range=[0, 0.45]),
        showlegend=False,
    )
    return fig


def build_perclos_gauges(p_eye, p_head):
    """좌우로 두 PERCLOS 게이지."""
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=p_eye * 100,
        title={'text': "PERCLOS (Eye)"},
        domain={'x': [0, 0.48], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#3B82F6"},
            'steps': [
                {'range': [0,  PERCLOS_WARNING*100], 'color': "#D1FAE5"},
                {'range': [PERCLOS_WARNING*100, PERCLOS_DANGER*100], 'color': "#FEF3C7"},
                {'range': [PERCLOS_DANGER*100,  100], 'color': "#FEE2E2"},
            ],
        },
    ))

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=p_head * 100,
        title={'text': "PERCLOS (Head)"},
        domain={'x': [0.52, 1.0], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#8B5CF6"},
            'steps': [
                {'range': [0,  PERCLOS_WARNING*100], 'color': "#D1FAE5"},
                {'range': [PERCLOS_WARNING*100, PERCLOS_DANGER*100], 'color': "#FEF3C7"},
                {'range': [PERCLOS_DANGER*100,  100], 'color': "#FEE2E2"},
            ],
        },
    ))

    fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def build_pitch_chart(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=list(history['pitch']),
        mode='lines',
        line=dict(color='#8B5CF6', width=2),
        name='Pitch',
    ))
    fig.add_hline(y=HEAD_DOWN_PITCH, line_dash='dash', line_color='red',
                  annotation_text="head down threshold")
    fig.update_layout(
        title='Pitch (고개 상하 각도)',
        height=180,
        margin=dict(l=20, r=10, t=30, b=20),
        yaxis=dict(range=[-45, 45]),
        showlegend=False,
    )
    return fig


def draw_overlays_on_frame(frame, eye_data, state):
    """OpenCV BGR 프레임에 눈 다각형 + 상태 배지 그리기."""
    color = {
        'NORMAL':  (0, 200, 0),
        'WARNING': (0, 220, 220),
        'DANGER':  (0, 0, 230),
    }[state]

    for pts in [eye_data.left_eye_landmarks, eye_data.right_eye_landmarks]:
        pts_i = pts.astype(np.int32)
        cv2.polylines(frame, [pts_i], True, color, 2)

    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 40), color, -1)
    cv2.putText(frame, f"STATE: {state}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


# ─────────────────────────────────────────────────────────
# 메인 처리 1 스텝
# ─────────────────────────────────────────────────────────
def process_one_frame(m):
    ok, frame = m['cap'].read()
    if not ok:
        return None

    frame = cv2.flip(frame, 1)

    eye_data  = m['eye_ext'].extract(frame)
    head_data = m['head_est'].estimate(frame)

    if eye_data is None:
        return {
            'frame': frame,
            'face_detected': False,
        }

    ear_result = m['ear_calc'].process(
        eye_data.left_eye_landmarks,
        eye_data.right_eye_landmarks,
    )

    cnn_closed = False  # placeholder
    is_closed   = fuse_eye(ear_result.is_closed, cnn_closed, CNN_AVAILABLE)
    is_head_dn  = fuse_head(head_data)

    p_eye  = m['perclos_e'].update(is_closed)
    p_head = m['perclos_h'].update(is_head_dn)
    state  = decide_state(p_eye, p_head)

    frame = draw_overlays_on_frame(frame, eye_data, state)

    m['frame_idx'] += 1

    return {
        'frame':         frame,
        'face_detected': True,
        'ear':           ear_result.ear_smoothed,
        'pitch':         head_data.pitch if head_data else 0.0,
        'p_eye':         p_eye,
        'p_head':        p_head,
        'state':         state,
        'frame_idx':     m['frame_idx'],
    }


def update_history_and_stats(result):
    h = st.session_state.history
    s = st.session_state.stats

    h['ear'].append(result['ear'])
    h['pitch'].append(result['pitch'])
    h['perclos_eye'].append(result['p_eye'] * 100)
    h['perclos_head'].append(result['p_head'] * 100)
    h['frame_idx'].append(result['frame_idx'])

    s['total_frames'] += 1
    if result['state'] == 'DANGER':
        s['danger_frames'] += 1
        if s['last_state'] != 'DANGER':
            s['alert_count'] += 1
    elif result['state'] == 'WARNING':
        s['warning_frames'] += 1
    s['last_state'] = result['state']


# ─────────────────────────────────────────────────────────
# Streamlit 페이지
# ─────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="졸음 운전 탐지",
        page_icon="🚗",
        layout="wide",
    )

    init_session_state()

    st.title("🚗 졸음 운전 탐지 대시보드")
    st.caption("PIPELINE.md 명세 기반 통합 데모 · 본 화면은 자동 갱신됩니다.")

    # 컨트롤
    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        if st.button("▶ 시작" if not st.session_state.running else "⏸ 정지",
                     use_container_width=True):
            st.session_state.running = not st.session_state.running
            if not st.session_state.running:
                cleanup_modules()
            st.rerun()
    with c2:
        if st.button("🔄 리셋", use_container_width=True):
            for v in st.session_state.history.values():
                v.clear()
            st.session_state.stats = {
                'total_frames': 0, 'danger_frames': 0, 'warning_frames': 0,
                'alert_count': 0, 'last_state': 'NORMAL',
            }
            if st.session_state.modules:
                st.session_state.modules['perclos_e'].reset()
                st.session_state.modules['perclos_h'].reset()
            st.rerun()

    st.markdown("---")

    # 레이아웃: 좌(영상) / 우(상태·그래프)
    left, right = st.columns([3, 2])

    with left:
        video_slot = st.empty()
    with right:
        state_slot   = st.empty()
        gauge_slot   = st.empty()
        ear_slot     = st.empty()
        pitch_slot   = st.empty()

    stats_slot = st.empty()

    # 정지 상태 안내
    if not st.session_state.running:
        with video_slot:
            st.info("▶ 시작 버튼을 눌러 웹캠을 켜세요.")
        with state_slot:
            render_state_panel('NORMAL', 0.0, 0.0)
        return

    # 실행 루프 (Streamlit의 rerun 모델 안에서 짧은 루프로 갱신)
    m = get_modules()
    loop_start = time.time()
    LOOP_DURATION = 0.1  # 100ms마다 rerun

    while st.session_state.running and (time.time() - loop_start) < LOOP_DURATION:
        result = process_one_frame(m)
        if result is None:
            st.error("웹캠 읽기 실패")
            cleanup_modules()
            st.session_state.running = False
            break

        # 영상
        rgb = cv2.cvtColor(result['frame'], cv2.COLOR_BGR2RGB)
        with video_slot:
            st.image(rgb, channels="RGB", use_container_width=True)

        if result['face_detected']:
            update_history_and_stats(result)

            with state_slot:
                render_state_panel(result['state'], result['p_eye'], result['p_head'])

            with gauge_slot:
                st.plotly_chart(
                    build_perclos_gauges(result['p_eye'], result['p_head']),
                    use_container_width=True,
                )

            with ear_slot:
                st.plotly_chart(
                    build_ear_chart(st.session_state.history),
                    use_container_width=True,
                )

            with pitch_slot:
                st.plotly_chart(
                    build_pitch_chart(st.session_state.history),
                    use_container_width=True,
                )
        else:
            with state_slot:
                st.warning("얼굴 미검출 — 카메라 정면을 응시하세요.")

        # 통계
        s = st.session_state.stats
        with stats_slot.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("총 프레임", s['total_frames'])
            c2.metric("주의 누적", s['warning_frames'])
            c3.metric("위험 누적", s['danger_frames'])
            c4.metric("경보 발생", s['alert_count'])

    # 다음 루프로 자동 진입
    if st.session_state.running:
        time.sleep(0.01)
        st.rerun()


if __name__ == "__main__":
    main()
