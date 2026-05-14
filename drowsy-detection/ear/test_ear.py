"""
test_ear.py
===========
EAR 모듈 단위 테스트.

실행:
    python test_ear.py
"""

import numpy as np
from ear import calculate_ear, EARCalculator, EARResult


def test_calculate_ear_open_eye():
    """뜬 눈(세로 길이 큼)은 EAR이 크게 나와야 함."""
    # 가짜 좌표: 가로 100, 세로 30 (눈 뜸)
    open_eye = np.array([
        [0,  50],    # p1: 바깥쪽 끝
        [25, 35],    # p2: 위 안쪽
        [75, 35],    # p3: 위 바깥쪽
        [100, 50],   # p4: 안쪽 끝
        [75, 65],    # p5: 아래 바깥쪽
        [25, 65],    # p6: 아래 안쪽
    ], dtype=np.float32)

    ear = calculate_ear(open_eye)
    assert ear > 0.25, f"뜬 눈은 EAR > 0.25 (실제: {ear:.3f})"
    print(f"✓ test_calculate_ear_open_eye (EAR = {ear:.3f})")


def test_calculate_ear_closed_eye():
    """감은 눈(세로 길이 작음)은 EAR이 작게 나와야 함."""
    # 가짜 좌표: 가로 100, 세로 5 (눈 감음)
    closed_eye = np.array([
        [0,  50],
        [25, 48],
        [75, 48],
        [100, 50],
        [75, 52],
        [25, 52],
    ], dtype=np.float32)

    ear = calculate_ear(closed_eye)
    assert ear < 0.15, f"감은 눈은 EAR < 0.15 (실제: {ear:.3f})"
    print(f"✓ test_calculate_ear_closed_eye (EAR = {ear:.3f})")


def test_calculate_ear_invalid_input():
    """잘못된 입력은 0.0 반환."""
    assert calculate_ear(None) == 0.0
    assert calculate_ear(np.array([])) == 0.0
    assert calculate_ear(np.zeros((5, 2))) == 0.0  # shape 불일치
    print("✓ test_calculate_ear_invalid_input")


def test_ear_calculator_returns_result():
    """EARCalculator가 EARResult 객체를 반환."""
    calc = EARCalculator()

    open_eye = np.array([
        [0, 50], [25, 35], [75, 35],
        [100, 50], [75, 65], [25, 65]
    ], dtype=np.float32)

    result = calc.process(open_eye, open_eye)

    assert isinstance(result, EARResult)
    assert result.ear_left > 0
    assert result.ear_right > 0
    assert isinstance(result.is_closed, bool)
    print(f"✓ test_ear_calculator_returns_result (EAR = {result.ear_smoothed:.3f})")


def test_ear_smoothing():
    """이동평균이 적용되어 결과가 안정화되는지."""
    calc = EARCalculator(smoothing_window=5)

    open_eye = np.array([
        [0, 50], [25, 35], [75, 35],
        [100, 50], [75, 65], [25, 65]
    ], dtype=np.float32)

    # 5번 호출 후 smoothed EAR이 안정화
    for _ in range(5):
        result = calc.process(open_eye, open_eye)

    assert abs(result.ear_smoothed - result.ear_avg) < 1e-5, \
        "동일한 입력 5회 시 smoothed와 avg 일치해야 함"
    print("✓ test_ear_smoothing")


def test_calibration_with_few_samples():
    """샘플 30개 미만이면 calibration 실패."""
    calc = EARCalculator()
    calc.start_calibration()

    eye = np.array([[0, 50], [25, 35], [75, 35],
                    [100, 50], [75, 65], [25, 65]], dtype=np.float32)

    # 10번만 process → 30개 미만
    for _ in range(10):
        calc.process(eye, eye)

    baseline = calc.finish_calibration()
    assert baseline is None, "샘플 부족 시 None 반환"
    print("✓ test_calibration_with_few_samples")


def test_calibration_success():
    """50번 process 후 calibration 성공."""
    calc = EARCalculator()
    calc.start_calibration()

    eye = np.array([[0, 50], [25, 35], [75, 35],
                    [100, 50], [75, 65], [25, 65]], dtype=np.float32)

    for _ in range(50):
        calc.process(eye, eye)

    baseline = calc.finish_calibration()
    assert baseline is not None and baseline > 0, "calibration 성공해야 함"
    assert calc.threshold == baseline * calc.threshold_ratio, "임계값 자동 설정"
    print(f"✓ test_calibration_success (baseline = {baseline:.3f})")


def main():
    print("=" * 50)
    print("EAR 모듈 단위 테스트")
    print("=" * 50)

    test_calculate_ear_open_eye()
    test_calculate_ear_closed_eye()
    test_calculate_ear_invalid_input()
    test_ear_calculator_returns_result()
    test_ear_smoothing()
    test_calibration_with_few_samples()
    test_calibration_success()

    print("\n모든 테스트 통과!")


if __name__ == "__main__":
    main()
