"""
train_cnn.py
============
dataset_collector.py 로 모은 이미지를 활용해 CNN 모델을 학습시키는 스크립트.

사전 요구사항:
    pip install torch torchvision

실행:
    python train_cnn.py
    (데이터셋이 `dataset/open`, `dataset/closed` 폴더에 각각 나뉘어 있어야 합니다.)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from cnn_model import EyeClassifierCNN

def main():
    # 데이터 경로
    data_dir = "dataset"
    
    if not os.path.exists(data_dir):
        print(f"[오류] '{data_dir}' 폴더가 없습니다. dataset_collector.py를 먼저 실행해 데이터를 모아주세요.")
        return

    # 1. 데이터 변환 (Transforms)
    # 이미지가 흑백(1채널)이므로 Grayscale을 적용하고 Tensor로 변환, 정규화(0~1)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.RandomRotation(10), # 약간의 회전 증강
        transforms.ToTensor()
    ])

    # 2. 데이터셋 로드 (ImageFolder는 하위 폴더명을 클래스로 사용함)
    try:
        full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    except Exception as e:
        print("[오류] 데이터 로드 실패. 'dataset/open' 과 'dataset/closed' 폴더에 이미지가 최소 1장씩 있어야 합니다.")
        print(e)
        return

    # 클래스 매핑 확인 (알파벳 순이므로 closed=0, open=1 로 매핑됨)
    print("클래스 매핑:", full_dataset.class_to_idx)
    # 나중에 사용할 수 있게 변수로 저장 (closed: 0, open: 1)
    
    # 3. 데이터셋 분할 (학습 80%, 검증 20%)
    total_size = len(full_dataset)
    if total_size == 0:
        print("[오류] 데이터가 없습니다.")
        return
        
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 4. DataLoader 설정
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"총 데이터: {total_size}장 (학습: {train_size}장, 검증: {val_size}장)")

    # 5. 모델, 손실함수, 최적화 함수 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 기기: {device}")
    
    model = EyeClassifierCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6. 모델 학습
    epochs = 15
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # 7. 모델 검증
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
              
        # 가장 성능이 좋은 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_eye_cnn.pth")
            
    print(f"학습 완료! 가장 높은 검증 정확도: {best_acc:.2f}%")
    print("가중치가 'best_eye_cnn.pth' 에 저장되었습니다.")

if __name__ == "__main__":
    main()
