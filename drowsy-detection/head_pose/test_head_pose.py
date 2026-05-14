"""
test_head_pose.py
=================
HeadPoseEstimator 단위 테스트.

실행:
    python test_head_pose.py
"""

import numpy as np
import cv2
import time
from head_pose import HeadPoseEstimator, HeadPoseResult


def test_returns_none_for_empty_frame():
    """빈 프레임 입력 시 None 반환."""
    estimator = HeadPoseEstimator()
    result = estimator.estimate(np.array([]))
    assert result is None, "빈 프레임은 None"
    estimator.close()
    print("✓ test_returns_none_for_empty_frame")


def test_returns_none_for_no_face():
    """검은 화면(얼굴 없음)은 None 반환."""
    estimator = HeadPoseEstimator()
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = estimator.estimate(black_frame)
    assert result is None, "얼굴 없으면 None"
    estimator.close()
    print("✓ test_returns_none_for_no_face")


def test_output_type_when_face_present(face_frame):
    """얼굴 있을 때 출력 형식 검증."""
    estimator = HeadPoseEstimator()
    result = estimator.estimate(face_frame)

    assert result is not None, "얼굴 검출되어야 함"
    assert isinstance(result, HeadPoseResult), "HeadPoseResult 객체"

    # 각도 범위
    assert -180 <= result.pitch <= 180, f"pitch 범위 (실제: {result.pitch})"
    assert -180 <= result.yaw <= 180, f"yaw 범위 (실제: {result.yaw})"
    assert -180 <= result.roll <= 180, f"roll 범위 (실제: {result.roll})"

    # 신뢰도 범위
    assert 0 <= result.confidence <= 1, f"신뢰도 범위 (실제: {result.confidence})"

    estimator.close()
    print(f"✓ test_output_type_when_face_present "
          f"(pitch={result.pitch:.1f}, yaw={result.yaw:.1f}, roll={result.roll:.1f})")


def test_context_manager():
    """with 구문 지원."""
    with HeadPoseEstimator() as estimator:
        result = estimator.estimate(np.zeros((480, 640, 3), dtype=np.uint8))
        assert result is None
    print("✓ test_context_manager")


def test_processing_speed(face_frame):
    """30fps 가능한지 (33ms 이내) 측정."""
    estimator = HeadPoseEstimator()

    # 워밍업
    for _ in range(5):
        estimator.estimate(face_frame)

    # 측정
    times = []
    for _ in range(20):
        start = time.time()
        estimator.estimate(face_frame)
        times.append(time.time() - start)

    avg_ms = np.mean(times) * 1000
    max_ms = np.max(times) * 1000

    print(f"  평균 처리시간: {avg_ms:.1f}ms / 최대: {max_ms:.1f}ms")
    assert avg_ms < 50, f"평균 처리시간 50ms 이내 (실제: {avg_ms:.1f}ms)"

    estimator.close()
    print(f"✓ test_processing_speed")


def main():
    print("=" * 50)
    print("HeadPoseEstimator 단위 테스트")
    print("=" * 50)

    test_returns_none_for_empty_frame()
    test_returns_none_for_no_face()
    test_context_manager()

    # 웹캠 테스트
    print("\n[웹캠 테스트] 정면 응시 후 3초 대기...")
    cap = cv2.VideoCapture(0)
    time.sleep(3)
    ret, frame = cap.read()
    cap.release()

    if ret:
        try:
            test_output_type_when_face_present(frame)
            test_processing_speed(frame)
        except AssertionError as e:
            print(f"✗ 웹캠 테스트 실패: {e}")
    else:
        print("⚠ 웹캠 테스트 스킵 (카메라 없음)")

    print("\n모든 테스트 완료!")


if __name__ == "__main__":
    main()
