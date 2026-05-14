"""
test_eye_extractor.py
=====================
EyeExtractor의 인터페이스를 자동으로 검증하는 간단한 테스트.

실행:
    python test_eye_extractor.py

다른 팀원이 이 모듈을 사용하기 전에 이 테스트가 모두 통과하는지 확인하세요.
"""

import numpy as np
from eye_extractor import EyeExtractor, EyeData


def test_returns_none_for_empty_frame():
    """빈 프레임 입력 시 None 반환"""
    extractor = EyeExtractor()
    result = extractor.extract(np.array([]))
    assert result is None, "빈 프레임은 None 반환해야 함"
    extractor.close()
    print("✓ test_returns_none_for_empty_frame")


def test_returns_none_for_no_face():
    """얼굴이 없는 이미지(검은 화면)는 None 반환"""
    extractor = EyeExtractor()
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = extractor.extract(black_frame)
    assert result is None, "얼굴 없으면 None 반환해야 함"
    extractor.close()
    print("✓ test_returns_none_for_no_face")


def test_output_shape_when_face_present(face_image):
    """
    얼굴이 있는 이미지 입력 시 출력 형태 검증.

    Args:
        face_image: 실제 얼굴이 있는 BGR 이미지 (테스트 용)
    """
    extractor = EyeExtractor(eye_image_size=64)
    result = extractor.extract(face_image)

    assert result is not None, "얼굴 검출되어야 함"
    assert isinstance(result, EyeData), "EyeData 객체 반환해야 함"

    # 좌표 형태
    assert result.left_eye_landmarks.shape == (6, 2), "왼쪽 눈 좌표는 (6, 2)"
    assert result.right_eye_landmarks.shape == (6, 2), "오른쪽 눈 좌표는 (6, 2)"
    assert result.left_eye_landmarks.dtype == np.float32, "좌표는 float32"

    # 이미지 형태
    assert result.left_eye_image.shape == (64, 64), "왼쪽 눈 이미지는 64x64"
    assert result.right_eye_image.shape == (64, 64), "오른쪽 눈 이미지는 64x64"
    assert result.left_eye_image.dtype == np.uint8, "이미지는 uint8"

    # 좌표가 합리적 범위인지
    h, w = face_image.shape[:2]
    assert (result.left_eye_landmarks[:, 0] >= 0).all() and (result.left_eye_landmarks[:, 0] <= w).all(), "x 좌표 범위"
    assert (result.left_eye_landmarks[:, 1] >= 0).all() and (result.left_eye_landmarks[:, 1] <= h).all(), "y 좌표 범위"

    extractor.close()
    print("✓ test_output_shape_when_face_present")


def test_context_manager():
    """with 구문 사용 가능 여부"""
    with EyeExtractor() as extractor:
        result = extractor.extract(np.zeros((480, 640, 3), dtype=np.uint8))
        assert result is None
    print("✓ test_context_manager")


def main():
    """모든 테스트 실행."""
    print("=" * 50)
    print("EyeExtractor 단위 테스트")
    print("=" * 50)

    test_returns_none_for_empty_frame()
    test_returns_none_for_no_face()
    test_context_manager()

    # 얼굴이 있는 이미지 테스트는 웹캠으로 1프레임 캡처해서 진행
    print("\n[웹캠 테스트] 얼굴을 카메라에 보여주세요 (3초 대기)...")
    import cv2, time
    cap = cv2.VideoCapture(0)
    time.sleep(3)
    ret, frame = cap.read()
    cap.release()

    if ret:
        try:
            test_output_shape_when_face_present(frame)
        except AssertionError as e:
            print(f"✗ test_output_shape_when_face_present 실패: {e}")
            print("  → 얼굴이 카메라에 잘 보이는 상태에서 다시 테스트하세요.")
    else:
        print("⚠ 웹캠 테스트 스킵 (카메라 없음)")

    print("\n모든 테스트 완료!")


if __name__ == "__main__":
    main()
