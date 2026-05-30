import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb
import time
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ==========================================
# 1. 하이퍼파라미터 설정
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, 'rldd_features.csv')
SEQ_LEN = 30           # 30프레임(약 1초) 단위로 시퀀스 생성
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_CLASSES = 3        # 0: Alert, 1: Low Vigilant, 2: Drowsy (데이터 라벨에 맞게 자동 조정)

# ==========================================
# 2. 커스텀 Dataset 클래스 정의
# ==========================================
class DrowsinessDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 3. Bi-LSTM 딥러닝 모델 아키텍처
# ==========================================
class DrowsinessBiLSTMAttn(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, num_classes=3):
        super(DrowsinessBiLSTMAttn, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # bidirectional=True 로 설정하여 양방향 LSTM 구성
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        
        # Attention 메커니즘 레이어
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # Attention weights 계산
        attn_weights = self.attention(out) # shape: (batch_size, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Context vector (가중 합)
        context_vector = torch.sum(attn_weights * out, dim=1) # shape: (batch_size, hidden_size*2)
        
        out = self.fc(context_vector)
        return out

# ==========================================
# 4. 데이터 로드 및 전처리 파이프라인
# ==========================================
def load_and_preprocess_data(csv_file, seq_len=30):
    print(f"데이터셋 로딩 중: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # 정규화 (Standard Scaling): 수치 데이터 범위 맞추기
    scaler = StandardScaler()
    feature_cols = ['ear', 'pitch', 'yaw', 'roll']
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    X_list = []
    y_list = []
    
    print("시계열 시퀀스(Sliding Window) 생성 중...")
    # 비디오별로 묶어서 시계열 데이터 생성
    grouped = df.groupby('video_id')
    
    for video_id, group in tqdm(grouped, total=len(grouped)):
        # 프레임 순서대로 정렬 확인
        group = group.sort_values(by='frame_id')
        features = group[feature_cols].values
        labels = group['label'].values
        
        # 시퀀스 길이 단위로 잘라서 저장
        for i in range(len(features) - seq_len + 1):
            seq_x = features[i : i + seq_len]
            # 라벨은 시퀀스의 마지막 프레임 기준 (또는 가장 빈도 높은 라벨)
            seq_y = labels[i + seq_len - 1]
            
            X_list.append(seq_x)
            y_list.append(seq_y)
            
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"데이터셋 형태 - X: {X.shape}, y: {y.shape}")
    return X, y

# ==========================================
# 5. 메인 학습 스크립트
# ==========================================
def main():
    if not os.path.exists(CSV_FILE):
        print(f"오류: {CSV_FILE} 파일을 찾을 수 없습니다.")
        return

    # 1. 데이터 준비
    X, y = load_and_preprocess_data(CSV_FILE, seq_len=SEQ_LEN)
    
    # 2. Train / Validation 분할 (80:20)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"학습 데이터: {len(X_train)}개, 검증 데이터: {len(X_val)}개")
    
    train_dataset = DrowsinessDataset(X_train, y_train)
    val_dataset = DrowsinessDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. 모델 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습을 진행할 장치: {device}")
    
    # 라벨의 최대값을 기준으로 클래스 개수 유추
    num_classes = len(np.unique(y))
    print(f"감지 클래스 개수: {num_classes}개")
    
    model = DrowsinessBiLSTMAttn(input_size=4, hidden_size=HIDDEN_SIZE, 
                             num_layers=NUM_LAYERS, num_classes=num_classes).to(device)
    
    # 4. 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    save_path = os.path.join(BASE_DIR, "bilstm_attn_drowsiness_model.pth")
    
    wandb.init(project="drowsiness-detection", name="Bi-LSTM+Attention")
    
    # 5. Training Loop
    print("\n--- 본격적인 Bi-LSTM+Attention 학습을 시작합니다 ---")
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        # Training Phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()
            
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100 * correct_train / total_train
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        all_y_true = []
        all_y_pred = []
        
        infer_start_time = time.time()
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()
                
                all_y_true.extend(batch_y.cpu().numpy())
                all_y_pred.extend(predicted.cpu().numpy())
                
        infer_end_time = time.time()
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct_val / total_val
        
        epoch_time = time.time() - epoch_start_time
        infer_fps = len(val_loader.dataset) / (infer_end_time - infer_start_time) if (infer_end_time - infer_start_time) > 0 else 0
        
        labels = [0, 1, 2]
        val_precision_macro = precision_score(all_y_true, all_y_pred, average='macro', zero_division=0)
        val_recall_macro = recall_score(all_y_true, all_y_pred, average='macro', zero_division=0)
        val_f1_macro = f1_score(all_y_true, all_y_pred, average='macro', zero_division=0)
        
        val_precision_class = precision_score(all_y_true, all_y_pred, labels=labels, average=None, zero_division=0)
        val_recall_class = recall_score(all_y_true, all_y_pred, labels=labels, average=None, zero_division=0)
        val_f1_class = f1_score(all_y_true, all_y_pred, labels=labels, average=None, zero_division=0)
        
        cm = confusion_matrix(all_y_true, all_y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix (Epoch {epoch+1})')
        plt.tight_layout()
        cm_plot = wandb.Image(fig)
        plt.close(fig)
        
        wandb.log({
            "epoch":                        epoch + 1,
            "train/loss":                   train_loss,
            "train/accuracy":               train_acc,
            "val/loss":                     val_loss,
            "val/accuracy":                 val_acc,
            "val/precision_macro":          val_precision_macro,
            "val/recall_macro":             val_recall_macro,
            "val/f1_macro":                 val_f1_macro,
            "val/precision_alert":          val_precision_class[0] if len(val_precision_class)>0 else 0,
            "val/precision_low_vigilance":  val_precision_class[1] if len(val_precision_class)>1 else 0,
            "val/precision_drowsy":         val_precision_class[2] if len(val_precision_class)>2 else 0,
            "val/recall_alert":             val_recall_class[0] if len(val_recall_class)>0 else 0,
            "val/recall_low_vigilance":     val_recall_class[1] if len(val_recall_class)>1 else 0,
            "val/recall_drowsy":            val_recall_class[2] if len(val_recall_class)>2 else 0,
            "val/f1_alert":                 val_f1_class[0] if len(val_f1_class)>0 else 0,
            "val/f1_low_vigilance":         val_f1_class[1] if len(val_f1_class)>1 else 0,
            "val/f1_drowsy":                val_f1_class[2] if len(val_f1_class)>2 else 0,
            "epoch_time_sec":               epoch_time,
            "inference_fps":                infer_fps,
            "confusion_matrix":             cm_plot,
        })
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 검증 손실이 가장 낮을 때 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  --> 모델이 개선되어 '{save_path}'에 저장되었습니다!")

    wandb.finish()
    print("\n--- 학습 완료 ---")
    print(f"최종 성능이 가장 좋은 모델 가중치가 {save_path} 에 저장되었습니다.")

if __name__ == '__main__':
    main()
