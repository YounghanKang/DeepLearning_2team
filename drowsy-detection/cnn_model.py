"""
cnn_model.py
============
눈 이미지(64x64, 흑백)를 입력받아 Open/Closed 상태를 분류하는 간단한 CNN PyTorch 모델.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EyeClassifierCNN(nn.Module):
    def __init__(self):
        super(EyeClassifierCNN, self).__init__()
        # 입력: (Batch, 1, 64, 64) 흑백 이미지
        
        # 첫 번째 합성곱 레이어
        # 1 채널 -> 16 채널, 커널 사이즈 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 64x64 -> 32x32
        
        # 두 번째 합성곱 레이어
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16
        
        # 세 번째 합성곱 레이어
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 16x16 -> 8x8
        
        # Fully Connected (완전연결) 레이어
        # 64채널 * 8 * 8 = 4096 피처
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5) # 과적합 방지
        
        # 출력: 2개의 클래스 (0: Open, 1: Closed)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # x shape: (B, 1, 64, 64)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    # 모델 구조 테스트
    model = EyeClassifierCNN()
    # 더미 데이터 생성 (배치사이즈 4, 1채널, 64x64)
    dummy_input = torch.randn(4, 1, 64, 64)
    output = model(dummy_input)
    print("모델 테스트 성공! 출력 형태:", output.shape) # 기대값: [4, 2]
