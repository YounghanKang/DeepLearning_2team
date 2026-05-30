"""
dataset_collector.py
====================
CNN 모델 학습을 위한 데이터 수집기 스크립트.
demo.py와 비슷하지만, 키보드 입력에 따라 눈 뜬 사진과 감은 사진을 분류하여 저장합니다.

사용법:
    1. python dataset_collector.py 실행
    2. 눈을 뜨고 있는 상태에서 'o' (알파벳 오) 키를 연타하거나 꾹 누르기
    3. 눈을 감은 상태에서 'c' 키를 연타하거나 꾹 누르기
    4. 'q'를 눌러 종료
"""

import cv2
import os
import uuid
from eye_extractor import EyeExtractor

# 저장할 데이터 폴더 경로 설정
DATASET_DIR = "dataset"
OPEN_DIR = os.path.join(DATASET_DIR, "open")
CLOSED_DIR = os.path.join(DATASET_DIR, "closed")

# 폴더가 없으면 생성
os.makedirs(OPEN_DIR, exist_ok=True)
os.makedirs(CLOSED_DIR, exist_ok=True)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다.")
        return

    open_count = len(os.listdir(OPEN_DIR))
    closed_count = len(os.listdir(CLOSED_DIR))

    with EyeExtractor(eye_image_size=64) as extractor:
        print("==================================================")
        print("데이터 수집 시작!")
        print("  - 눈 뜬 사진 저장 : 'o' 키 누르기 (Open)")
        print("  - 눈 감은 사진 저장 : 'c' 키 누르기 (Closed)")
        print("  - 프로그램 종료 : 'q' 키 누르기")
        print("==================================================")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 직관성을 위해 좌우 반전
            frame = cv2.flip(frame, 1)

            result = extractor.extract(frame)

            # 화면에 현재 저장된 데이터 개수 표시
            cv2.putText(frame, f"Open(O): {open_count} imgs", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Closed(C): {closed_count} imgs", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if result is not None:
                # 추출된 눈 영역을 화면 우측에 미리보기로 표시
                h, w = frame.shape[:2]
                preview_size = 128
                left_preview = cv2.cvtColor(cv2.resize(result.left_eye_image, (preview_size, preview_size)), cv2.COLOR_GRAY2BGR)
                right_preview = cv2.cvtColor(cv2.resize(result.right_eye_image, (preview_size, preview_size)), cv2.COLOR_GRAY2BGR)
                frame[10:10 + preview_size, w - preview_size - 10:w - 10] = left_preview
                frame[20 + preview_size:20 + preview_size * 2, w - preview_size - 10:w - 10] = right_preview

            cv2.imshow("Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
                
            elif result is not None:
                # 'o' 키를 누르면 눈 뜬 사진으로 저장
                if key == ord('o'):
                    unique_id = uuid.uuid4().hex[:8]
                    cv2.imwrite(os.path.join(OPEN_DIR, f"open_L_{unique_id}.png"), result.left_eye_image)
                    cv2.imwrite(os.path.join(OPEN_DIR, f"open_R_{unique_id}.png"), result.right_eye_image)
                    open_count += 2
                    print(f"저장됨: Open 데이터 (+2장) -> 총 {open_count}장")
                    
                # 'c' 키를 누르면 눈 감은 사진으로 저장
                elif key == ord('c'):
                    unique_id = uuid.uuid4().hex[:8]
                    cv2.imwrite(os.path.join(CLOSED_DIR, f"closed_L_{unique_id}.png"), result.left_eye_image)
                    cv2.imwrite(os.path.join(CLOSED_DIR, f"closed_R_{unique_id}.png"), result.right_eye_image)
                    closed_count += 2
                    print(f"저장됨: Closed 데이터 (+2장) -> 총 {closed_count}장")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
