# dataset.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class RLDDDataset(Dataset):
    def __init__(self, csv_path, window_size=30):
        print(f"[알림] {csv_path} 데이터 로딩 중... 잠시만 기다려주세요.")
        df = pd.read_csv(csv_path)
        
        # 1. 결측치(-1) 처리 (직전 프레임 값으로 채우기)
        df[['ear', 'pitch', 'yaw', 'roll']] = df[['ear', 'pitch', 'yaw', 'roll']].replace(-1, np.nan)
        df[['ear', 'pitch', 'yaw', 'roll']] = df[['ear', 'pitch', 'yaw', 'roll']].ffill().fillna(0)
        
        self.window_size = window_size
        self.X_samples = []
        self.y_samples = []
        
        # 2. video_id 별로 그룹화하여 슬라이딩 윈도우 생성
        print("[알림] 영상별 30프레임 슬라이딩 윈도우 생성 중...")
        grouped = df.groupby('video_id')
        
        for video_id, group in grouped:
            features = group[['ear', 'pitch', 'yaw', 'roll']].values
            labels = group['label'].values
            
            if len(features) < window_size:
                continue
                
            for i in range(len(features) - window_size + 1):
                window_x = features[i : i + window_size]
                window_y = labels[i + window_size - 1] 
                
                self.X_samples.append(window_x)
                self.y_samples.append(window_y)
                
        self.X_samples = np.array(self.X_samples, dtype=np.float32)
        self.y_samples = np.array(self.y_samples, dtype=np.int64)
        print(f"[완료] 총 {len(self.X_samples)}개의 시계열 시퀀스 데이터 구축 완료!")

    def __len__(self):
        return len(self.X_samples)

    def __getitem__(self, idx):
        return torch.tensor(self.X_samples[idx]), torch.tensor(self.y_samples[idx])

# 🚨 실행 및 검증 파트
if __name__ == "__main__":
    # 경로가 올바른지 꼭 확인하세요!
    dataset = RLDDDataset(csv_path="data/rldd_features.csv", window_size=30)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    for batch_x, batch_y in dataloader:
        print("\n=== 배치가 고장 안 나고 잘 뽑히는지 검증 ===")
        print("입력 데이터 모양 [Batch, Window, Features]:", batch_x.shape) # [64, 30, 4]가 나와야 함
        print("정답 데이터 모양 [Batch]:", batch_y.shape)                    # [64]가 나와야 함
        break