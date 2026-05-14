"""
demo_head_pose.py
=================
HeadPoseEstimator 동작 확인용 데모.

웹캠을 열어:
- pitch, yaw, roll 각도를 화면에 표시
- 머리 방향을 코끝에서 뻗어나가는 3D 축으로 시각화
- pitch가 일정 각도 이상이면 "HEAD DOWN" 경고

실행:
    python demo_head_pose.py

조작:
    q : 종료
"""

import cv2
import numpy as np
from head_pose import HeadPoseEstimator, LANDMARK_INDICES
import mediapipe as mp


# 졸음 감지 기준 (튜닝 가능)
PITCH_DOWN_THRESHOLD = 20.0  # 이 값 이상 떨구면 경고


def draw_axes(frame, head_pose, nose_tip_2d, axis_length=80):
    """
    머리 방향을 3D 축으로 시각화.
    코끝에서 X(빨강), Y(초록), Z(파랑) 축을 뻗어 그림.
    """
    pitch, yaw, roll = head_pose.pitch, head_pose.yaw, head_pose.roll

    # 라디안 변환
    pitch_rad = np.radians(-pitch)  # 부호 보정
    yaw_rad   = np.radians(yaw)
    roll_rad  = np.radians(roll)

    # 회전 행렬
    cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
    cos_y, sin_y = np.cos(yaw_rad),   np.sin(yaw_rad)
    cos_r, sin_r = np.cos(roll_rad),  np.sin(roll_rad)

    # X축 끝점 (빨강): 오른쪽
    x_end = (
        int(nose_tip_2d[0] + axis_length * (cos_y * cos_r)),
        int(nose_tip_2d[1] + axis_length * (cos_p * sin_r + cos_r * sin_p * sin_y))
    )
    # Y축 끝점 (초록): 아래
    y_end = (
        int(nose_tip_2d[0] + axis_length * (-cos_y * sin_r)),
        int(nose_tip_2d[1] + axis_length * (cos_p * cos_r - sin_p * sin_y * sin_r))
    )
    # Z축 끝점 (파랑): 정면
    z_end = (
        int(nose_tip_2d[0] + axis_length * sin_y),
        int(nose_tip_2d[1] - axis_length * cos_y * sin_p)
    )

    p0 = (int(nose_tip_2d[0]), int(nose_tip_2d[1]))
    cv2.line(frame, p0, x_end, (0, 0, 255), 3)    # X: 빨강
    cv2.line(frame, p0, y_end, (0, 255, 0), 3)    # Y: 초록
    cv2.line(frame, p0, z_end, (255, 0, 0), 3)    # Z: 파랑


def draw_info_panel(frame, head_pose):
    """화면 좌측 상단에 각도 정보 표시."""
    h, w = frame.shape[:2]

    # 검은 반투명 배경
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (320, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # 텍스트
    cv2.putText(frame, f"Pitch: {head_pose.pitch:+6.1f} (up/down)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Yaw:   {head_pose.yaw:+6.1f} (left/right)",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Roll:  {head_pose.roll:+6.1f} (tilt)",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Confidence: {head_pose.confidence:.2f}",
                (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 졸음 경고
    if head_pose.pitch >= PITCH_DOWN_THRESHOLD:
        cv2.putText(frame, "HEAD DOWN!", (20, 168),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "OK", (20, 168),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다.")
        return

    print("=" * 50)
    print("Head Pose 데모 시작")
    print("=" * 50)
    print("'q' : 종료")
    print("화면 색축: 빨강(X) 초록(Y) 파랑(Z)")
    print("=" * 50)

    # MediaPipe로 코끝 좌표를 별도 추출 (시각화용)
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5
    )

    with HeadPoseEstimator() as estimator:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # 머리 자세 추출
            head_pose = estimator.estimate(frame)

            if head_pose is None:
                cv2.putText(frame, "Face not detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Head Pose Demo", frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == 27:
                    break
                continue

            # 코끝 좌표 시각화용 (별도 추출)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_result = mp_face_mesh.process(rgb)
            if mp_result.multi_face_landmarks:
                nose_lm = mp_result.multi_face_landmarks[0].landmark[LANDMARK_INDICES["nose_tip"]]
                nose_2d = (nose_lm.x * w, nose_lm.y * h)

                # 3D 축 그리기
                draw_axes(frame, head_pose, nose_2d)

            # 정보 패널
            draw_info_panel(frame, head_pose)

            cv2.imshow("Head Pose Demo", frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                break

    mp_face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
