"""
test_feature_extractor.py
=========================
FeatureExtractor 단위 테스트.

실행:
    python test_feature_extractor.py

웹캠 테스트는 자동으로 수행되며, 웹캠이 없으면 스킵됩니다.
"""

import os
import csv
import time
import cv2
import numpy as np

from feature_extractor import FeatureExtractor, calculate_ear


# ─── 순수 함수 테스트 ─────────────────────────────────────

def test_calculate_ear_open_eye():
    """뜬 눈은 EAR > 0.25"""
    open_eye = np.array([
        [0,  50], [25, 35], [75, 35],
        [100, 50], [75, 65], [25, 65],
    ], dtype=np.float32)
    ear = calculate_ear(open_eye)
    assert ear > 0.25, f"뜬 눈 EAR > 0.25 (실제: {ear:.3f})"
    print(f"✓ test_calculate_ear_open_eye (EAR = {ear:.3f})")


def test_calculate_ear_closed_eye():
    """감은 눈은 EAR < 0.15"""
    closed_eye = np.array([
        [0,  50], [25, 48], [75, 48],
        [100, 50], [75, 52], [25, 52],
    ], dtype=np.float32)
    ear = calculate_ear(closed_eye)
    assert ear < 0.15, f"감은 눈 EAR < 0.15 (실제: {ear:.3f})"
    print(f"✓ test_calculate_ear_closed_eye (EAR = {ear:.3f})")


def test_calculate_ear_invalid():
    """비정상 입력은 0.0 반환"""
    assert calculate_ear(None) == 0.0
    assert calculate_ear(np.array([])) == 0.0
    assert calculate_ear(np.zeros((5, 2))) == 0.0
    print("✓ test_calculate_ear_invalid")


# ─── extract_from_frame 테스트 ────────────────────────────

def test_extract_returns_none_for_no_face():
    """검은 화면은 None 반환"""
    with FeatureExtractor() as ext:
        result = ext.extract_from_frame(
            np.zeros((480, 640, 3), dtype=np.uint8)
        )
        assert result is None
    print("✓ test_extract_returns_none_for_no_face")


def test_extract_from_real_frame(face_frame):
    """실제 얼굴 프레임에서 4가지 특징 모두 추출 확인"""
    with FeatureExtractor() as ext:
        result = ext.extract_from_frame(face_frame)
        assert result is not None, "얼굴 검출되어야 함"

        # 필수 키
        for key in ["ear", "pitch", "yaw", "roll"]:
            assert key in result, f"키 누락: {key}"
            assert isinstance(result[key], float), f"{key} 타입 오류"

        # 값 범위
        assert 0.0 <= result["ear"] <= 0.6, f"ear 범위: {result['ear']}"
        assert -90 <= result["pitch"] <= 90, f"pitch 범위: {result['pitch']}"
        assert -180 <= result["yaw"] <= 180, f"yaw 범위: {result['yaw']}"
        assert -180 <= result["roll"] <= 180, f"roll 범위: {result['roll']}"

    print(f"✓ test_extract_from_real_frame "
          f"(ear={result['ear']:.3f}, pitch={result['pitch']:.1f}, "
          f"yaw={result['yaw']:.1f}, roll={result['roll']:.1f})")


# ─── 영상 처리 + CSV 출력 테스트 ─────────────────────────

def test_process_video_and_csv_output(face_frame):
    """짧은 임시 영상 만들고 process_video → CSV 검증"""
    # 1. 5프레임짜리 임시 영상 만들기
    tmp_video = "tmp_test_video.mp4"
    tmp_csv = "tmp_test_output.csv"

    h, w = face_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, 30, (w, h))
    for _ in range(5):
        writer.write(face_frame)
    writer.release()

    # 2. process_video 실행
    with FeatureExtractor() as ext:
        valid_count = ext.process_video(
            video_path=tmp_video,
            output_csv=tmp_csv,
            label=1,
            video_id="test_video",
            verbose=False,
        )

    assert valid_count > 0, "유효 프레임 1개 이상"

    # 3. CSV 검증
    with open(tmp_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # 헤더 확인
    expected_headers = ["video_id", "frame_id", "ear", "pitch", "yaw", "roll", "label"]
    assert rows[0] == expected_headers, f"헤더 오류: {rows[0]}"

    # 데이터 행 수
    assert len(rows) == 6, f"행 수 = 1 헤더 + 5 데이터 (실제: {len(rows)})"

    # 첫 데이터 행 확인
    first_row = rows[1]
    assert first_row[0] == "test_video"
    assert first_row[1] == "0"  # frame_id
    assert first_row[-1] == "1"  # label

    # 정리
    os.remove(tmp_video)
    os.remove(tmp_csv)

    print(f"✓ test_process_video_and_csv_output (5/5 프레임 처리)")


# ─── 속도 측정 ────────────────────────────────────────────

def test_processing_speed(face_frame):
    """30fps(33ms) 안에 처리 가능한지 측정"""
    with FeatureExtractor() as ext:
        # 워밍업
        for _ in range(5):
            ext.extract_from_frame(face_frame)

        # 측정
        times = []
        for _ in range(20):
            start = time.time()
            ext.extract_from_frame(face_frame)
            times.append(time.time() - start)

    avg_ms = np.mean(times) * 1000
    max_ms = np.max(times) * 1000

    print(f"  평균: {avg_ms:.1f}ms / 최대: {max_ms:.1f}ms")
    assert avg_ms < 50, f"평균 < 50ms (실제: {avg_ms:.1f}ms)"
    print(f"✓ test_processing_speed")


# ─── 실행 ─────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("FeatureExtractor 단위 테스트")
    print("=" * 50)

    # 순수 함수 (웹캠 불필요)
    test_calculate_ear_open_eye()
    test_calculate_ear_closed_eye()
    test_calculate_ear_invalid()
    test_extract_returns_none_for_no_face()

    # 웹캠 필요한 테스트
    print("\n[웹캠 테스트] 정면 응시 후 3초 대기...")
    cap = cv2.VideoCapture(0)
    time.sleep(3)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("⚠ 웹캠 테스트 스킵 (카메라 없음)")
    else:
        try:
            test_extract_from_real_frame(frame)
            test_process_video_and_csv_output(frame)
            test_processing_speed(frame)
        except AssertionError as e:
            print(f"✗ 웹캠 테스트 실패: {e}")
            print("  → 얼굴이 카메라에 잘 보이는 상태에서 다시 실행하세요.")

    print("\n모든 테스트 완료!")


if __name__ == "__main__":
    main()
