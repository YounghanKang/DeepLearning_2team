import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import deque
import os
import urllib.request
from sklearn.preprocessing import StandardScaler
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 0. 구글 Face Landmarker 모델 자동 다운로드 함수
# ==========================================
def download_model_if_needed():
    model_name = os.path.join(BASE_DIR, "face_landmarker.task")
    if not os.path.exists(model_name):
        print(f"[안내] 구글 Face Landmarker 모델({model_name})이 없어 다운로드를 시작합니다... (약 5MB)")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        try:
            urllib.request.urlretrieve(url, model_name)
            print(" -> 다운로드 완료!")
        except Exception as e:
            print(f" -> 모델 다운로드 실패: {e}")
            raise e

# ==========================================
# 1. 모델 클래스 정의 (train_bilstm.py와 동일)
# ==========================================
class DrowsinessBiLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, num_classes=3):
        super(DrowsinessBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# 2. EAR(Eye Aspect Ratio) 및 각도 계산 함수
# ==========================================
def calculate_ear(eye_landmarks):
    # eye_landmarks는 (6, 2) 형태의 numpy 배열
    # 수직 거리 계산
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    # 수평 거리 계산
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_head_pose(landmarks, img_w, img_h):
    # MediaPipe Face Landmarker 기준 머리 포즈를 추정할 6개 주요 포인트
    # 33(왼눈바깥), 263(오른눈바깥), 1(코끝), 61(왼입꼬리), 291(오른입꼬리), 199(턱끝)
    face_3d = []
    face_2d = []
    
    for idx in [33, 263, 1, 61, 291, 199]:
        lm = landmarks[idx] # Tasks API는 직접 인덱스 참조
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        face_2d.append([x, y])
        face_3d.append([x, y, lm.z])
        
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    
    # 카메라 매트릭스 가정
    focal_length = 1 * img_w
    cam_matrix = np.array([
        [focal_length, 0, img_h / 2],
        [0, focal_length, img_w / 2],
        [0, 0, 1]
    ])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    
    # solvePnP로 회전 벡터 추출
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    
    # 회전 매트릭스로 변환 후 오일러 각도(Pitch, Yaw, Roll) 추출
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    
    # OpenCV 결과 각도 (x=pitch, y=yaw, z=roll)
    pitch = angles[0] * 360
    yaw = angles[1] * 360
    roll = angles[2] * 360
    
    return pitch, yaw, roll

# ==========================================
# 3. 메인 추론 스크립트
# ==========================================
def main():
    # 0. 모델 파일 체크 및 다운로드
    download_model_if_needed()

    print("1. 스케일러(Scaler) 훈련용 데이터 로드 중...")
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, 'rldd_features.csv'))
        scaler = StandardScaler()
        scaler.fit(df[['ear', 'pitch', 'yaw', 'roll']])
        print(" -> 스케일러 준비 완료!")
    except Exception as e:
        print(f"데이터셋을 불러올 수 없습니다. 경로를 확인해주세요: {e}")
        return

    print("2. Bi-LSTM 딥러닝 모델 로드 중...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DrowsinessBiLSTM(input_size=4, hidden_size=64, num_layers=2, num_classes=3).to(device)
    
    try:
        model.load_state_dict(torch.load(os.path.join(BASE_DIR, "bilstm_drowsiness_model.pth"), map_location=device))
        model.eval()
        print(" -> 모델 가중치 로드 완료!")
    except Exception as e:
        print(f"모델 가중치(bilstm_drowsiness_model.pth)를 찾을 수 없습니다: {e}")
        return

    # 3. 신규 MediaPipe Tasks Face Landmarker API 빌드
    base_options = python.BaseOptions(model_asset_path=os.path.join(BASE_DIR, "face_landmarker.task"))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    
    # 왼쪽/오른쪽 눈 랜드마크 인덱스 (기존과 동일)
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    # 실시간 30프레임 저장을 위한 Queue (슬라이딩 윈도우)
    SEQ_LEN = 30
    sequence_buffer = deque(maxlen=SEQ_LEN)
    
    # 클래스 레이블 매핑
    state_labels = {0: "Alert (정상)", 1: "Low Vigilant (주의)", 2: "Drowsy (졸음!)"}
    state_colors = {0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 0, 255)}

    print("4. 웹캠을 시작합니다...")
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        img_h, img_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # New API의 mp.Image 규격 변환
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        
        current_state = "Data Collecting..."
        color = (255, 255, 255)
        
        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            
            # 1. EAR 계산
            left_eye_pts = np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in LEFT_EYE])
            right_eye_pts = np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in RIGHT_EYE])
            
            ear_left = calculate_ear(left_eye_pts)
            ear_right = calculate_ear(right_eye_pts)
            avg_ear = (ear_left + ear_right) / 2.0
            
            # 2. Pitch, Yaw, Roll 계산
            pitch, yaw, roll = get_head_pose(landmarks, img_w, img_h)
            
            # 눈 시각화
            for pt in left_eye_pts:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)
            for pt in right_eye_pts:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)
                
            # 3. 큐에 추가 및 스케일링
            feature_vector = [avg_ear, pitch, yaw, roll]
            sequence_buffer.append(feature_vector)
            
            # 화면에 현재 추출된 특징 출력
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 4. 30프레임이 모이면 Bi-LSTM 추론
            if len(sequence_buffer) == SEQ_LEN:
                # 정규화(Standard Scaling)
                scaled_seq = scaler.transform(list(sequence_buffer))
                
                # Tensor로 변환 (batch_size=1, seq_len=30, input_size=4)
                input_tensor = torch.tensor([scaled_seq], dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted = torch.max(outputs.data, 1)
                    class_idx = predicted.item()
                    
                    current_state = state_labels.get(class_idx, "Unknown")
                    color = state_colors.get(class_idx, (255, 255, 255))
                    
                    # 만약 졸음(2)으로 판별되면 붉은색 테두리 경고
                    if class_idx == 2:
                        cv2.rectangle(frame, (0, 0), (img_w, img_h), (0, 0, 255), 10)
        else:
            # 얼굴이 안보일 땐 버퍼를 비움
            sequence_buffer.clear()
            current_state = "Face Not Detected"
            color = (100, 100, 100)
            
        # 결과 텍스트 출력
        cv2.putText(frame, current_state, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        cv2.imshow('Bi-LSTM Drowsiness Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
