# train.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
from sklearn.metrics import precision_recall_fscore_support

from dataset import RLDDDataset
from model import DrowsyLSTM

CLASS_NAMES = ["alert", "low_vigilance", "drowsy"]

def main():
    # 1. 하드웨어 가속기 설정 (맥북 M칩용 mps 설정, 없으면 cpu)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[알림] 현재 {device} 환경에서 학습을 진행합니다.")

    # 2. 하이퍼파라미터 세팅
    BATCH_SIZE = 512  # 데이터가 238만 개로 매우 많으므로 배치를 크게 잡아야 속도가 빠릅니다.
    EPOCHS = 10
    LEARNING_RATE = 0.001
    WINDOW_SIZE = 30

    # 3. wandb 초기화
    wandb.init(
        project="drowsy-detection-lstm",
        name="LSTM",
        config={
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "window_size": WINDOW_SIZE,
            "hidden_dim": 64,
            "num_layers": 2,
            "device": str(device),
        }
    )

    # 4. 데이터셋 로드 및 분할 (Train 80% / Val 20%)
    # 실제 협업 시에는 video_id 기준으로 엄격히 쪼개야 하지만,
    # 우선 baseline을 빠르게 돌려보기 위해 무작위 분할로 파이프라인을 검증합니다.
    full_dataset = RLDDDataset(csv_path="data/rldd_features.csv", window_size=WINDOW_SIZE)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    # 5. 모델, 손실함수, 옵티마이저 선언
    model     = DrowsyLSTM(input_dim=4, hidden_dim=64, num_layers=2, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs("models", exist_ok=True)
    best_val_acc = 0.0

    # 6. 학습 루프
    print("\nLSTM 모델 학습을 시작합니다...")
    for epoch in range(EPOCHS):
        # ── Train ──────────────────────────────────────────────
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        epoch_start = time.time()

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * batch_x.size(0)
            _, predicted   = torch.max(outputs, 1)
            total_train   += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()

        epoch_time       = time.time() - epoch_start
        epoch_train_loss = train_loss / total_train
        epoch_train_acc  = (correct_train / total_train) * 100

        # ── Validation ────────────────────────────────────────
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        all_preds, all_labels = [], []

        infer_start = time.time()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss    = criterion(outputs, batch_y)

                val_loss  += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val  += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        infer_time = time.time() - infer_start
        infer_fps  = total_val / infer_time if infer_time > 0 else 0.0

        epoch_val_loss = val_loss / total_val
        epoch_val_acc  = (correct_val / total_val) * 100

        # ── Precision / Recall / F1 ──────────────────────────
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, labels=[0, 1, 2], average=None, zero_division=0
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )

        # ── Confusion Matrix ──────────────────────────────────
        cm_plot = wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=CLASS_NAMES,
        )

        # ── wandb 로깅 ────────────────────────────────────────
        log_dict = {
            "epoch":            epoch + 1,
            "train/loss":       epoch_train_loss,
            "train/accuracy":   epoch_train_acc,
            "val/loss":         epoch_val_loss,
            "val/accuracy":     epoch_val_acc,
            "val/precision_macro": macro_p,
            "val/recall_macro":    macro_r,
            "val/f1_macro":        macro_f1,
            "epoch_time_sec":   epoch_time,
            "inference_fps":    infer_fps,
            "confusion_matrix": cm_plot,
        }
        # 클래스별 지표
        for i, name in enumerate(CLASS_NAMES):
            log_dict[f"val/precision_{name}"] = precision[i]
            log_dict[f"val/recall_{name}"]    = recall[i]
            log_dict[f"val/f1_{name}"]        = f1[i]

        wandb.log(log_dict)

        print(f"  [Val]   Precision: {macro_p:.4f} | Recall: {macro_r:.4f} | F1: {macro_f1:.4f}")

        print(f"Epoch [{epoch+1}/{EPOCHS}]  time: {epoch_time:.1f}s  infer_fps: {infer_fps:.0f}")
        print(f"  [Train] Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}%")
        print(f"  [Val]   Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.2f}%")

        # 가장 우수한 성능의 모델 가중치 저장
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), "models/best_lstm_model.pth")
            print(f"  최고 성능 경신! 모델 저장 완료 -> models/best_lstm_model.pth")
            wandb.summary["best_val_accuracy"] = best_val_acc

    wandb.finish()
    print("\n모든 학습이 완료되었습니다! 고생하셨습니다.")

if __name__ == "__main__":
    main()
