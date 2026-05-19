# predict_video.py
# feature_extractor.py 파일 상단 import 구문에 이거 딱 한 줄만 추가해 주세요!
# predict_video.py의 상단 구조는 딱 이렇게만 되어 있으면 됩니다.
import cv2
import torch
import numpy as np
from collections import deque
# 팀원 원석이의 통합 추출기 가져오기
from feature_extract.feature_extractor import FeatureExtractor 
from model import DrowsyLSTM

# ... 아래는 동일 ...

def main():
    # 1. 하드웨어 세팅 및 모델 로드
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = DrowsyLSTM(input_dim=4, hidden_dim=64, num_layers=2, num_classes=3).to(device)
    
    # 영한님이 방금 학습시켜 저장한 따끈따끈한 가중치 파일 로드!
    model.load_state_dict(torch.load("models/best_lstm_model.pth", map_location=device))
    model.eval()
    print("[알림] 학습된 LSTM 모델 가중치 로드 완료!")

    # 2. 테스트할 영상 파일 경로 (웹캠으로 하려면 0 입력)
    # 예: 은지님이 준 원본 영상 중 하나를 'test.mp4'로 같은 폴더에 두고 테스트해보세요!
    video_path = 0
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[에러] 영상을 열 수 없습니다: {video_path} (웹캠으로 대체하려면 0으로 변경하세요)")
        return

    # 3. LSTM은 30프레임의 '시퀀스'가 필요하므로, 실시간 프레임을 담을 큐(Queue) 선언
    window_size = 30
    frame_queue = deque(maxlen=window_size)
    
    # 클래스 레이블 맵핑
    status_map = {0: "NORMAL", 1: "WARNING", 2: "DANGER"}
    colors = {0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 0, 255)} # 녹색, 노랑, 빨강

    print("🎬 실시간 영상 분석 데모를 시작합니다. 'q'를 누르면 종료됩니다.")
    
    with FeatureExtractor() as extractor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 원석이 코드로 현재 프레임에서 4개 특징 추출
            features = extractor.extract_from_frame(frame)
            
            if features is not None:
                # 얼굴 인식 성공 시 수치 저장
                ear = features["ear"]
                pitch = features["pitch"]
                yaw = features["yaw"]
                roll = features["roll"]
            else:
                # 얼굴 미검출 시 결측치(-1) 처리 로직 (dataset.py와 동일하게 0으로 임시 대체)
                ear, pitch, yaw, roll = 0.0, 0.0, 0.0, 0.0
                
            # 큐에 현재 프레임 특징 삽입
            frame_queue.append([ear, pitch, yaw, roll])
            
            # 4. 큐에 30프레임(1초치)이 가득 차면 LSTM 추론 시작!
            current_status = "CALIBRATING..."
            color = (255, 255, 255)
            
            if len(frame_queue) == window_size:
                # [30, 4] -> PyTorch 텐서 변환 및 배치를 위한 차원 확장 [1, 30, 4]
                input_tensor = torch.tensor(list(frame_queue), dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted = torch.max(outputs, 1)
                    pred_class = predicted.item()
                    
                    current_status = status_map[pred_class]
                    color = colors[pred_class]
            
            # 5. 영상 화면에 분석 결과 합성하기
            cv2.putText(frame, f"STATUS: {current_status}", (30, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
            
            # 실시간 수치도 화면에 작게 띄워주기
            cv2.putText(frame, f"EAR: {ear:.3f} | Pitch: {pitch:.1f}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Drowsiness Detection Real-time Demo", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()