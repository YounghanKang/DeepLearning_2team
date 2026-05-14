"""
head_pose.py
============
얼굴 영상에서 머리 자세(pitch, yaw, roll) 3축 각도를 추출하는 모듈.

졸음 운전 탐지의 보조 신호로 사용됩니다.
EAR(눈 감김)이 못 잡는 "고개 떨굼" 신호를 보완합니다.

원리:
    1. MediaPipe Face Mesh로 얼굴 2D 좌표 추출
    2. 평균 얼굴의 3D 표준 모델 좌표와 매칭
    3. OpenCV solvePnP로 회전 행렬 추정
    4. 회전 행렬을 오일러 각도(pitch, yaw, roll)로 변환

사용 예시:
    from head_pose import HeadPoseEstimator

    estimator = HeadPoseEstimator()
    result = estimator.estimate(frame)

    if result is not None:
        pitch = result.pitch   # 위아래 (졸음 감지의 핵심)
        yaw   = result.yaw     # 좌우 회전
        roll  = result.roll    # 좌우 기울임
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional


# ============================================================
# MediaPipe Face Mesh에서 사용할 6개 랜드마크 인덱스
# solvePnP의 안정성을 위해 얼굴 전체에 골고루 분포된 점을 선택
# ============================================================
LANDMARK_INDICES = {
    "nose_tip":     1,      # 코끝
    "chin":         152,    # 턱끝
    "left_eye":     263,    # 왼쪽 눈 바깥쪽 (사용자 기준)
    "right_eye":    33,     # 오른쪽 눈 바깥쪽
    "left_mouth":   287,    # 왼쪽 입꼬리
    "right_mouth":  57,     # 오른쪽 입꼬리
}


# ============================================================
# 3D 표준 얼굴 모델 (mm 단위)
# 평균적인 사람 얼굴의 3D 좌표. 학계에서 널리 쓰이는 값을 사용.
# 코끝을 원점(0,0,0)으로 두고 다른 점의 상대 위치를 정의.
# ============================================================
FACE_3D_MODEL = np.array([
    [   0.0,    0.0,    0.0],   # 코끝 (원점)
    [   0.0, -330.0,  -65.0],   # 턱끝
    [-225.0,  170.0, -135.0],   # 왼쪽 눈 바깥
    [ 225.0,  170.0, -135.0],   # 오른쪽 눈 바깥
    [-150.0, -150.0, -125.0],   # 왼쪽 입꼬리
    [ 150.0, -150.0, -125.0],   # 오른쪽 입꼬리
], dtype=np.float64)


@dataclass
class HeadPoseResult:
    """
    머리 자세 추출 결과.

    Attributes:
        pitch:      위아래 각도 (degree)
                    + 값: 고개 떨굼 (졸음 신호)
                    0:    정면
                    - 값: 고개 들어올림
        yaw:        좌우 회전 각도 (degree)
                    + 값: 왼쪽 보기 (사용자 기준)
                    - 값: 오른쪽 보기
        roll:       좌우 기울임 (degree)
                    + 값: 왼쪽으로 기울임
                    - 값: 오른쪽으로 기울임
        confidence: 추정 신뢰도 (0~1)
                    측면 응시 시 낮아짐. 모델링 측에서 필터링 가능.
        face_detected: 얼굴 검출 성공 여부
    """
    pitch: float
    yaw: float
    roll: float
    confidence: float
    face_detected: bool = True


class HeadPoseEstimator:
    """
    웹캠 프레임에서 머리 자세를 추출하는 클래스.

    eye_extractor.py의 EyeExtractor와 동일한 인터페이스 패턴을 따름.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Args:
            min_detection_confidence: 얼굴 검출 최소 신뢰도 (0~1)
            min_tracking_confidence: 얼굴 추적 최소 신뢰도 (0~1)
        """
        # MediaPipe Face Mesh 초기화
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # 이전 프레임 결과 (얼굴 일시 미검출 시 사용 가능)
        self.last_result: Optional[HeadPoseResult] = None

    def estimate(self, frame: np.ndarray) -> Optional[HeadPoseResult]:
        """
        프레임 1장에서 머리 자세를 추출.

        Args:
            frame: BGR 이미지, shape (H, W, 3), dtype uint8

        Returns:
            HeadPoseResult 객체. 얼굴 미검출 시 None.
        """
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]

        # BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        result = self.face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if not result.multi_face_landmarks:
            return None

        face_landmarks = result.multi_face_landmarks[0]

        # 2D 좌표 추출
        landmarks_2d = self._extract_2d_landmarks(face_landmarks, w, h)

        # 카메라 내부 파라미터 (보정 안 된 일반 웹캠 가정)
        camera_matrix = self._get_camera_matrix(w, h)
        dist_coeffs = np.zeros((4, 1))  # 왜곡 무시

        # solvePnP로 회전 벡터 추정
        success, rotation_vector, translation_vector = cv2.solvePnP(
            FACE_3D_MODEL,
            landmarks_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        # 회전 벡터 → 회전 행렬 → 오일러 각도
        pitch, yaw, roll = self._rotation_to_euler(rotation_vector)

        # 신뢰도 계산
        confidence = self._calculate_confidence(yaw, pitch)

        head_pose = HeadPoseResult(
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            confidence=confidence,
            face_detected=True,
        )

        self.last_result = head_pose
        return head_pose

    def _extract_2d_landmarks(self, face_landmarks, width: int, height: int) -> np.ndarray:
        """선택된 6개 랜드마크의 2D 픽셀 좌표 추출."""
        coords = []
        for key in ["nose_tip", "chin", "left_eye", "right_eye", "left_mouth", "right_mouth"]:
            lm = face_landmarks.landmark[LANDMARK_INDICES[key]]
            coords.append([lm.x * width, lm.y * height])
        return np.array(coords, dtype=np.float64)

    def _get_camera_matrix(self, width: int, height: int) -> np.ndarray:
        """
        카메라 내부 파라미터 행렬.
        실제 카메라 보정값 없이 근사치 사용. 일반 웹캠에 충분.
        """
        focal_length = width  # 근사: 화면 폭과 같다고 가정
        center_x = width / 2
        center_y = height / 2

        return np.array([
            [focal_length, 0,            center_x],
            [0,            focal_length, center_y],
            [0,            0,            1       ],
        ], dtype=np.float64)

    def _rotation_to_euler(self, rotation_vector: np.ndarray) -> tuple:
        """
        회전 벡터 → 오일러 각도(pitch, yaw, roll) in degrees.

        OpenCV의 rotation vector를 회전 행렬로 변환 후,
        XYZ 축 회전 각도로 분해.
        """
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # 4x4 변환 행렬 (RQDecomp3x3 사용 위함)
        pose_matrix = cv2.hconcat((rotation_matrix, np.zeros((3, 1))))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)

        # euler_angles는 (3, 1) 배열, 단위는 도(degree)
        pitch = float(euler_angles[0][0])
        yaw   = float(euler_angles[1][0])
        roll  = float(euler_angles[2][0])

        # pitch 정규화: -180 ~ +180 → -90 ~ +90 (직관성 향상)
        if pitch > 90:
            pitch = 180 - pitch
        elif pitch < -90:
            pitch = -180 - pitch

        # 부호 보정 (직관에 맞게)
        # pitch: 고개 떨굼 = + 값으로 정의
        # yaw, roll: 그대로 사용
        pitch = -pitch  # 부호 뒤집기

        return pitch, yaw, roll

    def _calculate_confidence(self, yaw: float, pitch: float) -> float:
        """
        추정 신뢰도 계산.

        - 정면에 가까울수록 신뢰도 높음
        - 측면 응시(yaw 큼)에서 신뢰도 낮음
        - 극단적 pitch에서도 신뢰도 낮음
        """
        # yaw 절대값이 클수록 신뢰도 감소 (45도 이상에서 0.5 이하)
        yaw_score = max(0, 1 - abs(yaw) / 60.0)

        # pitch 절대값이 60도 이상이면 신뢰도 감소
        pitch_score = max(0, 1 - max(0, abs(pitch) - 30) / 60.0)

        return float(min(yaw_score, pitch_score))

    def close(self):
        """리소스 해제."""
        self.face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
