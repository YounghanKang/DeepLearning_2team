"""
ear.py
======
눈 랜드마크 좌표로부터 EAR(Eye Aspect Ratio)을 계산하는 모듈.

EAR이란?
    눈의 가로/세로 비율을 수치화한 값.
    눈을 뜨면 값이 크고(보통 0.25~0.35), 감으면 작아짐(0.15 이하).

수식:
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

사용 예시:
    from ear import EARCalculator

    calc = EARCalculator()

    # 매 프레임마다
    ear, is_closed = calc.process(left_landmarks, right_landmarks)
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple


def calculate_ear(eye_landmarks: np.ndarray) -> float:
    """
    눈 6개 좌표로 EAR 값을 계산.

    Args:
        eye_landmarks: shape (6, 2), float, 눈 주변 6개 점의 (x, y) 좌표
                       순서는 eye_extractor.py의 LEFT_EYE_INDICES 순서와 동일
                       [0] 바깥쪽 끝, [1] 위 안쪽, [2] 위 바깥쪽,
                       [3] 안쪽 끝,  [4] 아래 바깥쪽, [5] 아래 안쪽

    Returns:
        EAR 값 (float). 일반적으로 0.0 ~ 0.4 범위.
        좌표가 비정상이면 0.0 반환.
    """
    if eye_landmarks is None or eye_landmarks.shape != (6, 2):
        return 0.0

    # 세로 거리 (눈 위·아래)
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])

    # 가로 거리 (눈 양 끝)
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

    # 0으로 나누기 방지
    if C < 1e-6:
        return 0.0

    ear = (A + B) / (2.0 * C)
    return float(ear)


@dataclass
class EARResult:
    """
    EAR 계산 결과.

    Attributes:
        ear_left:      왼쪽 눈 EAR
        ear_right:     오른쪽 눈 EAR
        ear_avg:       양쪽 평균 EAR (이동평균 적용 전 원본)
        ear_smoothed:  이동평균이 적용된 EAR (떨림 제거됨)
        is_closed:     눈 감김 여부 (PERCLOS로 전달할 핵심 출력)
        threshold:     현재 사용 중인 임계값 (디버깅용)
    """
    ear_left: float
    ear_right: float
    ear_avg: float
    ear_smoothed: float
    is_closed: bool
    threshold: float


class EARCalculator:
    """
    EAR 기반 눈 감김 판정 클래스.

    핵심 기능:
    1. 양쪽 눈 EAR 계산
    2. 개인별 baseline 자동 측정 (calibrate 모드)
    3. 이동평균으로 떨림(flickering) 제거
    4. 임계값 비교로 눈 감김 판정
    """

    def __init__(
        self,
        smoothing_window: int = 5,
        threshold_ratio: float = 0.7,
        default_threshold: float = 0.22,
    ):
        """
        Args:
            smoothing_window: 이동평균 윈도우 크기 (프레임 단위)
                              5 = 최근 5프레임의 평균을 사용 (30fps에서 약 0.17초)
            threshold_ratio: baseline 대비 임계값 비율
                             0.7 = 평소 EAR의 70% 미만으로 떨어지면 감김 판정
            default_threshold: baseline 측정 전 사용할 기본 임계값
        """
        self.smoothing_window = smoothing_window
        self.threshold_ratio = threshold_ratio
        self.threshold = default_threshold

        # 이동평균용 버퍼
        self.ear_history = deque(maxlen=smoothing_window)

        # baseline 측정용
        self.baseline_samples = []
        self.is_calibrating = False
        self.baseline_ear: Optional[float] = None

    def process(
        self,
        left_landmarks: np.ndarray,
        right_landmarks: np.ndarray,
    ) -> EARResult:
        """
        한 프레임의 양쪽 눈 좌표로 EAR과 감김 여부를 계산.

        Args:
            left_landmarks: shape (6, 2), 왼쪽 눈 좌표
            right_landmarks: shape (6, 2), 오른쪽 눈 좌표

        Returns:
            EARResult 객체. 모든 정보가 담겨있음.
        """
        # 1. 양쪽 눈 EAR 계산
        ear_left = calculate_ear(left_landmarks)
        ear_right = calculate_ear(right_landmarks)
        ear_avg = (ear_left + ear_right) / 2.0

        # 2. baseline 측정 모드면 샘플 누적
        if self.is_calibrating:
            self.baseline_samples.append(ear_avg)

        # 3. 이동평균 적용 (떨림 제거)
        self.ear_history.append(ear_avg)
        ear_smoothed = float(np.mean(self.ear_history))

        # 4. 눈 감김 판정
        is_closed = ear_smoothed < self.threshold

        return EARResult(
            ear_left=ear_left,
            ear_right=ear_right,
            ear_avg=ear_avg,
            ear_smoothed=ear_smoothed,
            is_closed=is_closed,
            threshold=self.threshold,
        )

    def start_calibration(self):
        """
        baseline 측정 시작.

        사용 흐름:
            1. start_calibration() 호출
            2. 사용자가 5초간 정상 응시 (눈 뜨고 정면)
            3. process()를 매 프레임 호출 → 자동으로 샘플 누적
            4. finish_calibration() 호출 → 개인 맞춤 임계값 설정 완료
        """
        self.baseline_samples = []
        self.is_calibrating = True
        print("[Calibration] 시작. 정면을 응시하세요.")

    def finish_calibration(self) -> Optional[float]:
        """
        baseline 측정 완료. 개인 맞춤 임계값 설정.

        Returns:
            계산된 baseline EAR. 샘플이 부족하면 None.
        """
        self.is_calibrating = False

        if len(self.baseline_samples) < 30:
            print(f"[Calibration] 실패: 샘플 수 부족 ({len(self.baseline_samples)}개, 최소 30개 필요)")
            return None

        # 이상치 제거 (눈 깜빡임 등): 상위 75%만 사용
        samples = np.array(self.baseline_samples)
        upper_quartile = np.percentile(samples, 25)
        clean_samples = samples[samples > upper_quartile]

        self.baseline_ear = float(np.mean(clean_samples))
        self.threshold = self.baseline_ear * self.threshold_ratio

        print(f"[Calibration] 완료!")
        print(f"  baseline EAR: {self.baseline_ear:.4f}")
        print(f"  임계값: {self.threshold:.4f} (baseline의 {self.threshold_ratio*100:.0f}%)")
        return self.baseline_ear

    def reset(self):
        """이동평균 버퍼 초기화. 사용자 교체 시 호출."""
        self.ear_history.clear()
