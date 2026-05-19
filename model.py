# model.py
import torch
import torch.nn as nn

class DrowsyLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, num_classes=3):
        super(DrowsyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        last_time_step = out[:, -1, :] # 30번째 프레임의 결과만 사용
        logits = self.fc(last_time_step)
        return logits