"""
run_extractor.py
================
FeatureExtractor 실행 예시 스크립트.

세 가지 시나리오를 제공:
    1. 영상 파일 한 개 → CSV
    2. 폴더 안 영상 일괄 → CSV들
    3. 웹캠 5초 녹화 → 즉시 추출 (개발 검증용)

사용:
    main() 안에서 원하는 함수만 호출
"""

from feature_extractor import FeatureExtractor


def example_single_video():
    """영상 한 개를 처리해 CSV로 저장."""
    with FeatureExtractor() as extractor:
        extractor.process_video(
            video_path="data/raw/driver_01.mp4",
            output_csv="data/features/driver_01.csv",
            label=1,
        )


def example_batch_processing():
    """폴더 안 모든 영상 일괄 처리."""
    # 영상 파일별 정답 레이블
    # 0 = 정상, 1 = 주의, 2 = 위험
    label_map = {
        "normal_01.mp4":  0,
        "normal_02.mp4":  0,
        "drowsy_01.mp4":  1,
        "drowsy_02.mp4":  1,
        "danger_01.mp4":  2,
        "danger_02.mp4":  2,
    }

    with FeatureExtractor() as extractor:
        extractor.process_folder(
            video_folder="data/raw",
            output_folder="data/features",
            label_map=label_map,
        )


def example_webcam_test():
    """
    웹캠으로 5초 영상 녹화 후 즉시 추출.
    개발 중 빠른 동작 확인용.
    """
    import cv2
    import os

    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/features", exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 웹캠 열기 실패")
        return

    fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = "data/raw/webcam_test.mp4"
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print("5초간 녹화합니다. 정면을 응시하세요.")
    for _ in range(fps * 5):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow("Recording (q to stop)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"녹화 완료: {out_path}")

    with FeatureExtractor() as extractor:
        extractor.process_video(
            video_path=out_path,
            output_csv="data/features/webcam_test.csv",
            label=0,
        )


def main():
    # 처음 실행 시 추천: 웹캠 테스트로 동작 확인
    example_webcam_test()

    # 영상 파일 준비되면 아래 함수 사용
    # example_single_video()
    # example_batch_processing()


if __name__ == "__main__":
    main()
