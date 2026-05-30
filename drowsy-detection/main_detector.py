"""
main_detector.py
================
모든 모듈(EyeExtractor, CNNClassifier, StateTracker, AlertSystem)을 통합하여
졸음 운전을 실시간으로 탐지하는 최종 실행 파일입니다.

실행:
    python main_detector.py
"""

import cv2
from scipy.spatial.distance import euclidean

from eye_extractor import EyeExtractor
from classifier import CNNClassifier
from state_tracker import StateTracker
from alert_system import AlertSystem

def calculate_ear(eye_landmarks):
    """EAR (Eye Aspect Ratio) 수치 계산"""
    A = euclidean(eye_landmarks[1], eye_landmarks[5])
    B = euclidean(eye_landmarks[2], eye_landmarks[4])
    C = euclidean(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다.")
        return

    # 모듈 초기화
    cnn = CNNClassifier("best_eye_cnn.pth")
    # 30프레임(약 1초) 단위 판단 
    tracker = StateTracker(history_size=60, perclos_threshold=0.25, asleep_frames=20)
    alerter = AlertSystem()

    with EyeExtractor(eye_image_size=64) as extractor:
        print("===================================")
        print("졸음 탐지 시스템 시작 ('q'로 종료)")
        print("===================================")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1) # 거울 모드
            
            # 1. 눈 추출
            result = extractor.extract(frame)

            if result is not None:
                # 2-A. 수치적 계산 (EAR)
                ear_left = calculate_ear(result.left_eye_landmarks)
                ear_right = calculate_ear(result.right_eye_landmarks)
                ear_avg = (ear_left + ear_right) / 2.0
                
                # 2-B. 딥러닝 판단 (CNN)
                cnn_closed_prob = cnn.predict(result.left_eye_image, result.right_eye_image)
                
                # 앙상블 판단 로직: CNN의 확률이 0.5 이상이거나, EAR 수치가 너무 낮을 때 감은 것으로 판정
                # (CNN 가중치가 없을 때는 랜덤 확률이 나오므로, EAR이 Fallback 역할을 함)
                is_closed = (cnn_closed_prob > 0.6) or (ear_avg < 0.21)

                # 3. 상태 추적 및 PERCLOS 계산
                state = tracker.update(is_closed)
                
                # 4. 알람 발생 및 UI 업데이트
                if state != "Normal":
                    alerter.trigger_alert(level=state)
                frame = alerter.draw_warning(frame, level=state)

                # 화면에 정보 출력
                cv2.putText(frame, f"EAR: {ear_avg:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"CNN Closed Prob: {cnn_closed_prob:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"State: {state}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if state=="Normal" else (0,0,255), 2)
                            
                # 눈 영역에 박스 그리기
                h, w = frame.shape[:2]
                box_color = (0, 0, 255) if is_closed else (0, 255, 0)
                
                # 좌우 눈 미리보기를 우측 상단에 띄우기
                preview_size = 80
                left_preview = cv2.cvtColor(cv2.resize(result.left_eye_image, (preview_size, preview_size)), cv2.COLOR_GRAY2BGR)
                right_preview = cv2.cvtColor(cv2.resize(result.right_eye_image, (preview_size, preview_size)), cv2.COLOR_GRAY2BGR)
                
                # 박스 색상 입히기
                cv2.rectangle(left_preview, (0,0), (preview_size, preview_size), box_color, 2)
                cv2.rectangle(right_preview, (0,0), (preview_size, preview_size), box_color, 2)
                
                frame[10:10+preview_size, w-preview_size-10:w-10] = left_preview
                frame[10+preview_size+10:10+preview_size*2+10, w-preview_size-10:w-10] = right_preview

            else:
                cv2.putText(frame, "Face: NOT DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                # 얼굴을 놓치면 임시로 눈 뜬 걸로 처리하여 PERCLOS 왜곡 방지
                tracker.update(False) 

            cv2.imshow("Drowsy Driving Detection System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
