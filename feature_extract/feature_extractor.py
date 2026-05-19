"""
feature_extractor.py
====================
영상 파일에서 졸음 운전 탐지용 특징(feature)을 추출해 CSV로 저장하는 모듈.

출력 컬럼:
    video_id  : 영상 파일 이름
    frame_id  : 프레임 번호 (0부터)
    ear       : 눈 개폐 정도 (0.0 ~ 0.5)
    pitch     : 고개 상하 각도 (degree, 고개 떨굼 = +)
    yaw       : 고개 좌우 각도 (degree)
    roll      : 고개 좌우 기울기 (degree)
    label     : 졸음 상태 정답 (0, 1, 2)

사용 예시:
    from feature_extractor import FeatureExtractor

    with FeatureExtractor() as extractor:
        extractor.process_video(
            video_path="data/raw/driver_01.mp4",
            output_csv="data/features/driver_01.csv",
            label=1,
        )
"""
# feature_extract/feature_extractor.py 맨 위에 이걸 넣어주세요!

# ... 이 아래로 원석이가 원래 짰던 import cv2 등이 쭉 이어지면 됩니다 ...
import mediapipe as mp
import os
import cv2
import csv
import numpy as np
from typing import Optional
from pathlib import Path


# ============================================================
# MediaPipe Face Mesh 랜드마크 인덱스
# ============================================================

# 눈 6개 (EAR 계산용)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Head Pose 6개
HEAD_POSE_INDICES = {
    "nose_tip":    1,
    "chin":        152,
    "left_eye":    263,
    "right_eye":   33,
    "left_mouth":  287,
    "right_mouth": 57,
}

# 3D 표준 얼굴 모델 (mm 단위, solvePnP 입력)
FACE_3D_MODEL = np.array([
    [   0.0,    0.0,    0.0],   # 코끝
    [   0.0, -330.0,  -65.0],   # 턱끝
    [-225.0,  170.0, -135.0],   # 왼쪽 눈 바깥
    [ 225.0,  170.0, -135.0],   # 오른쪽 눈 바깥
    [-150.0, -150.0, -125.0],   # 왼쪽 입꼬리
    [ 150.0, -150.0, -125.0],   # 오른쪽 입꼬리
], dtype=np.float64)


# ============================================================
# EAR 계산
# ============================================================

def calculate_ear(eye_landmarks: np.ndarray) -> float:
    """
    눈 6점 좌표로 EAR(Eye Aspect Ratio) 계산.
    0에 가까우면 감김, 0.25~0.35면 정상적으로 뜬 상태.

    Args:
        eye_landmarks: shape (6, 2) float, 6개 점의 (x, y)
    Returns:
        EAR 값 (0.0 ~ 0.5 정도)
    """
    if eye_landmarks is None or eye_landmarks.shape != (6, 2):
        return 0.0
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    if C < 1e-6:
        return 0.0
    return float((A + B) / (2.0 * C))


# ============================================================
# 통합 추출기
# ============================================================

class FeatureExtractor:
    """
    영상 파일에서 EAR와 Head Pose를 한 번에 추출.
    MediaPipe Face Mesh를 프레임당 1회만 호출하므로 효율적.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # ─── 단일 프레임 처리 ─────────────────────────────────────

    def extract_from_frame(self, frame: np.ndarray) -> Optional[dict]:
        """
        한 프레임에서 ear, pitch, yaw, roll을 추출.

        Args:
            frame: BGR 이미지, shape (H, W, 3), dtype uint8
        Returns:
            {"ear":..., "pitch":..., "yaw":..., "roll":...} 또는 None
        """
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not result.multi_face_landmarks:
            return None

        face_landmarks = result.multi_face_landmarks[0]

        # 1. EAR (양쪽 평균)
        left_eye = self._pick(face_landmarks, LEFT_EYE_INDICES, w, h)
        right_eye = self._pick(face_landmarks, RIGHT_EYE_INDICES, w, h)
        ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0

        # 2. Head Pose
        head_pts_2d = np.array([
            self._pick_one(face_landmarks, HEAD_POSE_INDICES["nose_tip"], w, h),
            self._pick_one(face_landmarks, HEAD_POSE_INDICES["chin"], w, h),
            self._pick_one(face_landmarks, HEAD_POSE_INDICES["left_eye"], w, h),
            self._pick_one(face_landmarks, HEAD_POSE_INDICES["right_eye"], w, h),
            self._pick_one(face_landmarks, HEAD_POSE_INDICES["left_mouth"], w, h),
            self._pick_one(face_landmarks, HEAD_POSE_INDICES["right_mouth"], w, h),
        ], dtype=np.float64)

        pitch, yaw, roll = self._estimate_head_pose(head_pts_2d, w, h)

        return {
            "ear": ear,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
        }

    # ─── 영상 파일 처리 ──────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        output_csv: str,
        label: int = 0,
        video_id: Optional[str] = None,
        verbose: bool = True,
    ) -> int:
        """
        영상 파일 전체를 처리해 CSV로 저장.

        Args:
            video_path: 입력 영상 경로
            output_csv: 출력 CSV 경로 (없는 폴더는 자동 생성)
            label: 이 영상의 정답 레이블 (0: 정상, 1: 주의, 2: 위험)
            video_id: 영상 식별자. None이면 파일명(확장자 제외) 사용
            verbose: 진행 상황 출력 여부

        Returns:
            얼굴이 검출된 유효 프레임 수
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"영상 파일 없음: {video_path}")

        if video_id is None:
            video_id = Path(video_path).stem

        # 출력 폴더 자동 생성
        out_dir = os.path.dirname(output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"영상 열기 실패: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if verbose:
            print(f"[처리 시작] {video_id}")
            print(f"  총 프레임: {total}, FPS: {fps:.1f}")
            print(f"  레이블: {label}")

        valid_count = 0
        miss_count = 0

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "video_id", "frame_id",
                "ear", "pitch", "yaw", "roll",
                "label",
            ])

            frame_id = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                features = self.extract_from_frame(frame)

                if features is None:
                    # 얼굴 미검출 → 결측 표시 (-1)
                    writer.writerow([
                        video_id, frame_id,
                        -1, -1, -1, -1,
                        label,
                    ])
                    miss_count += 1
                else:
                    writer.writerow([
                        video_id, frame_id,
                        round(features["ear"], 5),
                        round(features["pitch"], 3),
                        round(features["yaw"], 3),
                        round(features["roll"], 3),
                        label,
                    ])
                    valid_count += 1

                frame_id += 1

                if verbose and frame_id % 100 == 0 and total > 0:
                    pct = frame_id / total * 100
                    print(f"  진행: {frame_id}/{total} ({pct:.1f}%)")

        cap.release()

        if verbose:
            print(f"[완료] 저장: {output_csv}")
            print(f"  유효 프레임: {valid_count}")
            print(f"  미검출 프레임: {miss_count}")
            if (valid_count + miss_count) > 0:
                rate = valid_count / (valid_count + miss_count) * 100
                print(f"  얼굴 검출률: {rate:.1f}%")

        return valid_count

    def process_folder(
        self,
        video_folder: str,
        output_folder: str,
        label_map: Optional[dict] = None,
    ):
        """
        폴더 안 모든 영상을 일괄 처리.

        Args:
            video_folder: 영상 파일들이 있는 폴더
            output_folder: CSV 저장 폴더
            label_map: {파일명: 레이블} 딕셔너리
                       예: {"driver_01.mp4": 0, "driver_02.mp4": 2}
                       None이면 모두 0
        """
        exts = {".mp4", ".avi", ".mov", ".mkv"}
        video_files = [
            f for f in os.listdir(video_folder)
            if Path(f).suffix.lower() in exts
        ]

        if not video_files:
            print(f"[경고] {video_folder} 에 영상 파일 없음")
            return

        print(f"[일괄 처리 시작] {len(video_files)}개 영상")

        for video_file in video_files:
            video_path = os.path.join(video_folder, video_file)
            csv_name = Path(video_file).stem + ".csv"
            output_csv = os.path.join(output_folder, csv_name)

            label = (label_map or {}).get(video_file, 0)

            try:
                self.process_video(
                    video_path=video_path,
                    output_csv=output_csv,
                    label=label,
                    verbose=True,
                )
            except Exception as e:
                print(f"[에러] {video_file}: {e}")

        print(f"[일괄 처리 완료]")

    # ─── 내부 helper ─────────────────────────────────────────

    def _pick(self, face_landmarks, indices: list, w: int, h: int) -> np.ndarray:
        coords = []
        for idx in indices:
            lm = face_landmarks.landmark[idx]
            coords.append([lm.x * w, lm.y * h])
        return np.array(coords, dtype=np.float32)

    def _pick_one(self, face_landmarks, idx: int, w: int, h: int) -> list:
        lm = face_landmarks.landmark[idx]
        return [lm.x * w, lm.y * h]

    def _estimate_head_pose(
        self, landmarks_2d: np.ndarray, width: int, height: int
    ) -> tuple:
        """solvePnP로 pitch, yaw, roll 추정 (degree)."""
        focal = width
        camera_matrix = np.array([
            [focal, 0, width / 2],
            [0, focal, height / 2],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rvec, _ = cv2.solvePnP(
            FACE_3D_MODEL, landmarks_2d,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        pose_mat = cv2.hconcat((rmat, np.zeros((3, 1))))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)

        pitch = float(euler[0][0])
        yaw   = float(euler[1][0])
        roll  = float(euler[2][0])

        # pitch 정규화 (-90 ~ +90)
        if pitch > 90:
            pitch = 180 - pitch
        elif pitch < -90:
            pitch = -180 - pitch

        # 부호 보정 (고개 떨굼 = +)
        pitch = -pitch
        return pitch, yaw, roll

    def close(self):
        self.face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
