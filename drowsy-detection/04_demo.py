"""
04_demo.py
==========
졸음 운전 탐지 실시간 데모 — 통합 파이프라인 프로토타입.

현재 사용 가능한 모듈만으로 전체 흐름을 구현합니다.
CNN과 외부 PERCLOS 모듈이 완성되면 표시된 위치에 끼워넣기만 하면 됩니다.

실행:
    python 04_demo.py

조작:
    c : EAR 개인 baseline 측정 (5초 정면 응시)
    r : PERCLOS 누적 리셋
    q : 종료

요구사항:
    - Python 3.11
    - mediapipe 0.10.9, opencv-python, numpy

본 파일은 PIPELINE.md의 인터페이스 명세에 기반합니다.
모듈 위치가 바뀌면 import 경로만 수정하면 됩니다.
"""

import sys
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────
# 모듈 import (각 모듈 폴더가 분리되어 있어 sys.path 추가)
# ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
for sub in ["eye_extract", "ear", "head_pose"]:
    sys.path.insert(0, str(ROOT / sub))

from eye_extractor import EyeExtractor          # type: ignore
from ear import EARCalculator                   # type: ignore
from head_pose import HeadPoseEstimator         # type: ignore


# ─────────────────────────────────────────────────────────
# 공통 상수 (추후 src/config.py로 이전 예정)
# ─────────────────────────────────────────────────────────
FPS = 30
PERCLOS_WINDOW_SECONDS = 60
PERCLOS_WARNING = 0.10
PERCLOS_DANGER  = 0.30

HEAD_DOWN_PITCH = 20.0
HEAD_POSE_MIN_CONFIDENCE = 0.5

CALIBRATION_SECONDS = 5
FACE_MISS_RESET_SECONDS = 5  # 얼굴 미검출이 길어지면 PERCLOS 리셋


# ─────────────────────────────────────────────────────────
# 임시 PERCLOS (ML 담당이 perclos.py 만들면 교체)
# ─────────────────────────────────────────────────────────
class SimplePerclos:
    """
    PIPELINE.md의 PerclosCalculator 인터페이스와 호환되는 최소 구현.
    ML 담당이 완성한 perclos.py가 들어오면 그대로 교체 가능.
    """
    def __init__(self, fps: int = FPS, window_seconds: int = PERCLOS_WINDOW_SECONDS):
        self.window_size = fps * window_seconds
        self.history = deque(maxlen=self.window_size)

    def update(self, eye_closed: bool) -> float:
        self.history.append(1 if eye_closed else 0)
        return self.get_perclos()

    def get_perclos(self) -> float:
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    def get_state(self) -> str:
        p = self.get_perclos()
        if p >= PERCLOS_DANGER:  return 'DANGER'
        if p >= PERCLOS_WARNING: return 'WARNING'
        return 'NORMAL'

    def reset(self):
        self.history.clear()


# ─────────────────────────────────────────────────────────
# CNN 자리표시자 — ML 담당 완성 후 교체
# ─────────────────────────────────────────────────────────
def cnn_predict_placeholder(eye_image: np.ndarray) -> float:
    """
    CNN 모델 완성 전 자리표시자.
    실제 CNN이 들어오면:
        - 모델 로드 (models/eye_cnn.pth)
        - 정규화 후 forward
        - softmax → closed 확률 반환
    """
    return 0.5  # 의미 없는 더미 값


CNN_AVAILABLE = False  # CNN 학습 완료 시 True로 전환


# ─────────────────────────────────────────────────────────
# 신호 융합 로직
# ─────────────────────────────────────────────────────────
def fuse_eye(ear_closed: bool, cnn_closed: bool, cnn_available: bool) -> bool:
    """EAR과 CNN 결과 융합. PIPELINE.md 5-1절."""
    if cnn_available:
        return cnn_closed
    return ear_closed


def fuse_head(head_result) -> bool:
    """Head pose → 고개 떨굼 여부. PIPELINE.md 5-2절."""
    if head_result is None or head_result.confidence < HEAD_POSE_MIN_CONFIDENCE:
        return False
    return head_result.pitch >= HEAD_DOWN_PITCH


def decide_state(p_eye: float, p_head: float) -> str:
    """두 PERCLOS 종합 → 최종 상태. PIPELINE.md 5-3절."""
    combined = max(p_eye, p_head)
    if combined >= PERCLOS_DANGER:  return 'DANGER'
    if combined >= PERCLOS_WARNING: return 'WARNING'
    return 'NORMAL'


# ─────────────────────────────────────────────────────────
# UI 헬퍼
# ─────────────────────────────────────────────────────────
STATE_COLORS = {
    'NORMAL':  (0, 200, 0),     # 초록
    'WARNING': (0, 220, 220),   # 노랑
    'DANGER':  (0, 0, 230),     # 빨강
}


def draw_eye_polygon(frame, landmarks, color):
    pts = landmarks.astype(np.int32)
    cv2.polylines(frame, [pts], True, color, 1)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 2, color, -1)


def draw_overlay(frame, ear_result, head_result, p_eye, p_head, state, calib):
    h, w = frame.shape[:2]
    color = STATE_COLORS[state]

    # 상단 상태 박스
    cv2.rectangle(frame, (0, 0), (w, 50), color, -1)
    cv2.putText(frame, f"STATE: {state}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # 하단 정보 박스
    cv2.rectangle(frame, (0, h - 140), (w, h), (0, 0, 0), -1)

    line1 = f"EAR: {ear_result.ear_smoothed:.3f}  (th={ear_result.threshold:.3f})  closed={ear_result.is_closed}"
    line2 = (f"PITCH: {head_result.pitch:+.1f}  YAW: {head_result.yaw:+.1f}  "
             f"ROLL: {head_result.roll:+.1f}  conf={head_result.confidence:.2f}"
             if head_result else "PITCH/YAW/ROLL: head pose 미검출")
    line3 = f"PERCLOS  eye: {p_eye*100:5.1f}%   head: {p_head*100:5.1f}%"
    line4 = f"CNN: {'ON' if CNN_AVAILABLE else 'OFF (EAR fallback)'}    [c]calib  [r]reset  [q]quit"

    cv2.putText(frame, line1, (15, h - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, line2, (15, h - 85),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, line3, (15, h - 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, line4, (15, h - 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)

    # calibration 카운트다운
    if calib['active']:
        remaining = max(0, CALIBRATION_SECONDS - (time.time() - calib['start']))
        cv2.putText(frame, f"CALIBRATING {remaining:.1f}s",
                    (w - 360, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)


# ─────────────────────────────────────────────────────────
# 메인 루프
# ─────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 웹캠 열기 실패")
        return

    print("=" * 60)
    print("Drowsy Detection Demo (prototype)")
    print("=" * 60)
    print(f"CNN: {'ENABLED' if CNN_AVAILABLE else 'DISABLED (EAR fallback)'}")
    print("c: calibration   r: reset PERCLOS   q: quit")
    print("=" * 60)

    # 모듈 초기화
    eye_ext   = EyeExtractor()
    head_est  = HeadPoseEstimator()
    ear_calc  = EARCalculator(smoothing_window=5, threshold_ratio=0.7)
    perclos_e = SimplePerclos()
    perclos_h = SimplePerclos()

    calib = {'active': False, 'start': 0}
    last_face_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            # 1) 얼굴/눈/머리 추출 — 현재 MediaPipe 2회 호출 (병목, PIPELINE.md 7-2)
            eye_data  = eye_ext.extract(frame)
            head_data = head_est.estimate(frame)

            now = time.time()

            # 2) 얼굴 미검출 처리
            if eye_data is None:
                if now - last_face_time > FACE_MISS_RESET_SECONDS:
                    perclos_e.reset()
                    perclos_h.reset()
                cv2.putText(frame, "FACE NOT DETECTED", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow("Drowsy Demo", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            last_face_time = now

            # 3) EAR
            ear_result = ear_calc.process(
                eye_data.left_eye_landmarks,
                eye_data.right_eye_landmarks,
            )

            # 4) calibration 자동 종료
            if calib['active'] and now - calib['start'] >= CALIBRATION_SECONDS:
                ear_calc.finish_calibration()
                calib['active'] = False

            # 5) CNN (자리표시자)
            cnn_closed_prob = cnn_predict_placeholder(eye_data.left_eye_image)
            cnn_closed = cnn_closed_prob > 0.5

            # 6) 신호 융합
            is_closed   = fuse_eye(ear_result.is_closed, cnn_closed, CNN_AVAILABLE)
            is_head_dn  = fuse_head(head_data)

            # 7) PERCLOS
            p_eye  = perclos_e.update(is_closed)
            p_head = perclos_h.update(is_head_dn)
            state  = decide_state(p_eye, p_head)

            # 8) UI 렌더링
            line_color = STATE_COLORS[state] if is_closed else (0, 200, 0)
            draw_eye_polygon(frame, eye_data.left_eye_landmarks, line_color)
            draw_eye_polygon(frame, eye_data.right_eye_landmarks, line_color)
            draw_overlay(frame, ear_result, head_data, p_eye, p_head, state, calib)

            cv2.imshow("Drowsy Demo", frame)

            # 9) 키 입력
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                ear_calc.start_calibration()
                calib = {'active': True, 'start': now}
            elif key == ord('r'):
                perclos_e.reset()
                perclos_h.reset()
                print("[INFO] PERCLOS reset")

    finally:
        eye_ext.close()
        head_est.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
