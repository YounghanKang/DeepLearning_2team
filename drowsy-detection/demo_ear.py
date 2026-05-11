"""
demo_ear.py
===========
EAR 모듈 + eye_extractor 통합 데모.

웹캠을 열어 실시간으로:
- 양쪽 눈 EAR 수치 표시
- 눈 감김 여부 표시
- 5초간 baseline 측정으로 개인 맞춤 임계값 자동 설정

실행:
    python demo_ear.py

조작:
    c : 개인 baseline 측정 시작 (5초간 정면 응시)
    q : 종료
"""

import cv2
import time
import numpy as np
from eye_extractor import EyeExtractor
from ear import EARCalculator


def draw_status_bar(frame, ear_result, calibration_state):
    """화면 하단에 상태 정보 그리기."""
    h, w = frame.shape[:2]

    # 검은 배경
    cv2.rectangle(frame, (0, h - 120), (w, h), (0, 0, 0), -1)

    # 색상 결정
    if ear_result.is_closed:
        color = (0, 0, 255)        # 빨강 (감김)
        status = "CLOSED"
    else:
        color = (0, 255, 0)        # 초록 (뜸)
        status = "OPEN"

    # EAR 값 표시
    cv2.putText(frame, f"EAR: {ear_result.ear_smoothed:.3f}",
                (20, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Threshold: {ear_result.threshold:.3f}",
                (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(frame, f"Status: {status}",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 좌·우 EAR 분리 표시
    cv2.putText(frame, f"L: {ear_result.ear_left:.3f}",
                (w - 200, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"R: {ear_result.ear_right:.3f}",
                (w - 200, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # calibration 상태
    if calibration_state['active']:
        elapsed = time.time() - calibration_state['start_time']
        remaining = max(0, 5 - elapsed)
        cv2.putText(frame, f"CALIBRATING... {remaining:.1f}s",
                    (w - 380, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def draw_eye_landmarks(frame, eye_landmarks, color):
    """눈 6개 좌표를 다각형으로 그리기."""
    pts = eye_landmarks.astype(np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 2, color, -1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다.")
        return

    calibration_state = {'active': False, 'start_time': 0}

    print("=" * 50)
    print("EAR 데모 시작")
    print("=" * 50)
    print("'c' : 개인 baseline 측정 (5초 정면 응시)")
    print("'q' : 종료")
    print("=" * 50)

    with EyeExtractor() as extractor:
        ear_calc = EARCalculator(smoothing_window=5, threshold_ratio=0.7)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # 거울 모드

            # 1. 눈 좌표 추출
            result = extractor.extract(frame)

            if result is None:
                cv2.putText(frame, "Face not detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("EAR Demo", frame)
                if cv2.waitKey(1) == ord('q'):
                    break
                continue

            # 2. EAR 계산
            ear_result = ear_calc.process(
                result.left_eye_landmarks,
                result.right_eye_landmarks
            )

            # 3. 눈 다각형 그리기 (감김이면 빨강, 뜸이면 초록)
            line_color = (0, 0, 255) if ear_result.is_closed else (0, 255, 0)
            draw_eye_landmarks(frame, result.left_eye_landmarks, line_color)
            draw_eye_landmarks(frame, result.right_eye_landmarks, line_color)

            # 4. calibration 자동 종료 (5초 경과)
            if calibration_state['active']:
                if time.time() - calibration_state['start_time'] >= 5:
                    ear_calc.finish_calibration()
                    calibration_state['active'] = False

            # 5. 상태 표시
            draw_status_bar(frame, ear_result, calibration_state)

            cv2.imshow("EAR Demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                ear_calc.start_calibration()
                calibration_state = {'active': True, 'start_time': time.time()}

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
