"""
eye_extractor.py
================
영상에서 눈 정보를 추출하는 모듈.

이 모듈은 졸음 운전 탐지 시스템의 첫 단계입니다.
이후 단계(EAR, CNN, PERCLOS)는 모두 이 모듈의 출력을 입력으로 사용합니다.

사용 예시:
    from eye_extractor import EyeExtractor

    extractor = EyeExtractor()
    result = extractor.extract(frame)

    if result is not None:
        left_landmarks = result['left_eye_landmarks']    # EAR 담당자용
        left_eye_image = result['left_eye_image']        # CNN 담당자용
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional


# MediaPipe Face Mesh의 눈 주변 랜드마크 인덱스
# Face Mesh는 얼굴 전체에 468개 점을 찍는데, 그중 눈 주변 6개 점만 사용
# 순서: [바깥쪽 끝, 위 안쪽, 위 바깥쪽, 안쪽 끝, 아래 바깥쪽, 아래 안쪽]
# 이 순서는 EAR 공식의 p1~p6 순서와 호환되도록 정의
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]


@dataclass
class EyeData:
    """
    눈 정보 추출 결과를 담는 데이터 클래스.

    Attributes:
        left_eye_landmarks:  왼쪽 눈 6개 좌표, shape (6, 2), dtype float32
                             EAR 계산에 사용
        right_eye_landmarks: 오른쪽 눈 6개 좌표, shape (6, 2)
        left_eye_image:      왼쪽 눈 영역 crop, shape (64, 64), 흑백, dtype uint8
                             CNN 학습/추론에 사용
        right_eye_image:     오른쪽 눈 영역 crop, shape (64, 64), 흑백
        face_detected:       얼굴 검출 성공 여부
    """
    left_eye_landmarks: np.ndarray
    right_eye_landmarks: np.ndarray
    left_eye_image: np.ndarray
    right_eye_image: np.ndarray
    face_detected: bool = True


class EyeExtractor:
    """
    웹캠 프레임에서 눈 정보를 추출하는 클래스.

    내부적으로 MediaPipe Face Mesh를 사용해 얼굴 랜드마크 468개를 추출한 뒤,
    그중 눈 주변 좌표 6개씩과 눈 영역 crop 이미지를 반환합니다.
    """

    def __init__(
        self,
        eye_image_size: int = 64,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Args:
            eye_image_size: 출력할 눈 이미지의 한 변 크기 (정사각형, 픽셀)
                           CNN 입력 크기와 맞춰야 함. 기본 64x64
            min_detection_confidence: 얼굴 검출 최소 신뢰도 (0~1)
            min_tracking_confidence: 얼굴 추적 최소 신뢰도 (0~1)
        """
        self.eye_image_size = eye_image_size

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.mp_drawing = mp.solutions.drawing_utils

    def extract(self, frame: np.ndarray) -> Optional[EyeData]:
        """
        프레임 1장에서 눈 정보를 추출.

        Args:
            frame: BGR 이미지, shape (H, W, 3), dtype uint8
                   OpenCV의 cap.read()가 반환하는 형태 그대로 입력

        Returns:
            EyeData 객체. 얼굴이 검출되지 않으면 None.

        Note:
            - 입력 이미지는 수정하지 않음 (안전성)
            - 좌표는 픽셀 단위 (0 ~ W, 0 ~ H)
            - 얼굴이 일시적으로 사라지면 None 반환 → 호출 측에서 처리 필요
        """
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]

        # MediaPipe는 RGB 입력을 기대 (OpenCV는 BGR이므로 변환 필요)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 성능 최적화: writeable=False로 설정하면 내부에서 복사 안 함
        rgb_frame.flags.writeable = False
        result = self.face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # 얼굴 검출 실패 처리
        if not result.multi_face_landmarks:
            return None

        face_landmarks = result.multi_face_landmarks[0]

        # 눈 주변 좌표 6개씩 추출
        left_landmarks = self._get_eye_landmarks(face_landmarks, LEFT_EYE_INDICES, w, h)
        right_landmarks = self._get_eye_landmarks(face_landmarks, RIGHT_EYE_INDICES, w, h)

        # 눈 영역 crop (CNN 입력용)
        left_image = self._crop_eye_region(frame, left_landmarks)
        right_image = self._crop_eye_region(frame, right_landmarks)

        return EyeData(
            left_eye_landmarks=left_landmarks,
            right_eye_landmarks=right_landmarks,
            left_eye_image=left_image,
            right_eye_image=right_image,
            face_detected=True,
        )

    def _get_eye_landmarks(
        self, face_landmarks, indices: list, width: int, height: int
    ) -> np.ndarray:
        """
        Face Mesh 결과에서 지정 인덱스의 좌표만 픽셀 단위로 변환.

        MediaPipe의 landmark는 [0, 1] 정규화 좌표라서
        실제 픽셀 좌표로 변환하려면 width, height를 곱해야 함.
        """
        coords = []
        for idx in indices:
            lm = face_landmarks.landmark[idx]
            x = lm.x * width
            y = lm.y * height
            coords.append([x, y])
        return np.array(coords, dtype=np.float32)

    def _crop_eye_region(
        self, frame: np.ndarray, eye_landmarks: np.ndarray
    ) -> np.ndarray:
        """
        눈 6개 좌표를 둘러싸는 사각형 영역을 crop.

        - 좌표의 min/max로 bounding box 계산
        - 약간의 여유(padding)를 줘서 눈썹·눈가까지 포함
        - 정사각형으로 맞추고 지정 크기로 resize
        - 흑백 변환 (CNN 입력 채널 1)

        Returns:
            shape (eye_image_size, eye_image_size), dtype uint8, 흑백
        """
        h, w = frame.shape[:2]

        # 좌표의 bounding box
        x_min = int(np.min(eye_landmarks[:, 0]))
        x_max = int(np.max(eye_landmarks[:, 0]))
        y_min = int(np.min(eye_landmarks[:, 1]))
        y_max = int(np.max(eye_landmarks[:, 1]))

        # padding (눈 너비의 30%)
        eye_width = x_max - x_min
        eye_height = y_max - y_min
        padding = int(max(eye_width, eye_height) * 0.3)

        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)

        # 정사각형으로 맞추기 (긴 변 기준)
        box_w = x_max - x_min
        box_h = y_max - y_min
        if box_w > box_h:
            diff = box_w - box_h
            y_min = max(0, y_min - diff // 2)
            y_max = min(h, y_max + diff // 2)
        else:
            diff = box_h - box_w
            x_min = max(0, x_min - diff // 2)
            x_max = min(w, x_max + diff // 2)

        # crop
        eye_region = frame[y_min:y_max, x_min:x_max]

        # 너무 작으면 검은 이미지 반환 (얼굴이 너무 멀거나 가려짐)
        if eye_region.size == 0 or eye_region.shape[0] < 5 or eye_region.shape[1] < 5:
            return np.zeros((self.eye_image_size, self.eye_image_size), dtype=np.uint8)

        # 흑백 변환 + 크기 조정
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.eye_image_size, self.eye_image_size))

        return resized

    def close(self):
        """리소스 해제. 프로그램 종료 시 호출 권장."""
        self.face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
