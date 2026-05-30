"""
classifier.py
=============
학습된 모델(best_eye_cnn.pth)을 사용해 실시간으로 넘어오는 눈 영역이 
감겼는지(Closed) 떠져있는지(Open) 추론하는 모듈입니다.
"""

import os
import torch
import torchvision.transforms as transforms
from cnn_model import EyeClassifierCNN

class CNNClassifier:
    def __init__(self, model_path="best_eye_cnn.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EyeClassifierCNN().to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"[INFO] 모델 가중치 로드 완료 ({model_path})")
        else:
            print(f"[WARNING] 모델 가중치를 찾을 수 없습니다 ({model_path}). 랜덤 가중치로 추론합니다.")
            
        self.model.eval()
        
        # 이미지 전처리 (흑백, 64x64, 텐서 변환)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        
    def predict(self, left_eye_img, right_eye_img):
        """
        두 눈의 이미지를 받아 눈을 감았을 확률을 반환합니다.
        반환값: closed_probability (0.0 ~ 1.0)
        """
        # numpy 이미지를 PyTorch 텐서로 변환
        left_tensor = self.transform(left_eye_img).unsqueeze(0).to(self.device)
        right_tensor = self.transform(right_eye_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out_left = self.model(left_tensor)
            out_right = self.model(right_tensor)
            
            # Softmax를 적용하여 확률 값으로 변경
            prob_left = torch.softmax(out_left, dim=1)
            prob_right = torch.softmax(out_right, dim=1)
            
            # ImageFolder 기본 매핑: closed(0), open(1)
            # 확률 텐서의 0번 인덱스가 Closed일 확률
            closed_prob_left = prob_left[0][0].item()
            closed_prob_right = prob_right[0][0].item()
            
            # 두 눈의 평균 감김 확률 계산
            avg_closed_prob = (closed_prob_left + closed_prob_right) / 2.0
            
            return avg_closed_prob
