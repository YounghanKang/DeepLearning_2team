# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import RLDDDataset
from model import DrowsyLSTM

def main():
    # 1. 하드웨어 가속기 설정 (맥북 M칩용 mps 설정, 없으면 cpu)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[알림] 현재 {device} 환경에서 학습을 진행합니다.")

    # 2. 하이퍼파라미터 세팅
    BATCH_SIZE = 512  # 데이터가 238만 개로 매우 많으므로 배치를 크게 잡아야 속도가 빠릅니다.
    EPOCHS = 10
    LEARNING_RATE = 0.001
    WINDOW_SIZE = 30

    # 3. 데이터셋 로드 및 분할 (Train 80% / Val 20%)
    # 실제 협업 시에는 video_id 기준으로 엄격히 쪼개야 하지만, 
    # 우선 baseline을 빠르게 돌려보기 위해 무작위 분할로 파이프라인을 검증합니다.
    full_dataset = RLDDDataset(csv_path="data/rldd_features.csv", window_size=WINDOW_SIZE)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. 모델, 손실함수, 옵티마이저 선언
    model = DrowsyLSTM(input_dim=4, hidden_dim=64, num_layers=2, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 모델 저장 폴더 생성
    os.makedirs("models", exist_ok=True)
    best_val_acc = 0.0

    # 5. 진짜 본격적인 학습 루프 시작
    print("\nLSTM 모델 학습을 시작합니다...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # 피드포워드 및 역전파
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100

        # 검증(Validation) 루프
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100

        print(f"Epoch [{epoch+1}/{EPOCHS}] ")
        print(f"  [Train] Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}%")
        print(f"  [Val]   Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.2f}%")

        # 가장 우수한 성능의 모델 가중치 저장
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), "models/best_lstm_model.pth")
            print(f"최고 성능 경신! 모델 저장 완료 -> models/best_lstm_model.pth")

    print("\n모든 학습이 완료되었습니다! 고생하셨습니다.")

if __name__ == "__main__":
    main()