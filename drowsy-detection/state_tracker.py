"""
state_tracker.py
================
눈 깜박임 상태의 히스토리를 관리하여 현재 상태(수면/졸음/정상)를 판단하는 모듈입니다.
"""

from collections import deque

class StateTracker:
    def __init__(self, history_size=90, perclos_threshold=0.2, asleep_frames=30):
        """
        history_size: 버퍼에 저장할 최근 프레임 수 (약 3초치, 30fps 기준 90프레임)
        perclos_threshold: 전체 프레임 중 눈을 감은 프레임의 비율 임계치 (0.2면 20%)
        asleep_frames: 이 프레임 수만큼 연속으로 눈을 감으면 '수면'으로 판단 (약 1초)
        """
        self.history = deque(maxlen=history_size)
        self.perclos_threshold = perclos_threshold
        self.asleep_frames = asleep_frames
        
        self.consecutive_closed = 0
        
    def update(self, is_closed: bool) -> str:
        """
        현재 프레임의 눈 감김 여부를 업데이트하고, 경고 수준을 반환합니다.
        반환값: "Normal", "Drowsy", "Asleep"
        """
        self.history.append(is_closed)
        
        if is_closed:
            self.consecutive_closed += 1
        else:
            self.consecutive_closed = 0
            
        # 1. Asleep (수면) 판단
        # 눈을 특정 프레임 이상 연속으로 감고 있는 경우 (가장 최우선 위험 상태)
        if self.consecutive_closed >= self.asleep_frames:
            return "Asleep"
            
        # 2. Drowsy (졸음) 판단
        # 최근 기록된 프레임 중에서 눈을 감고 있는 비율(PERCLOS) 계산
        if len(self.history) == self.history.maxlen:
            closed_count = sum(self.history)
            perclos = closed_count / len(self.history)
            
            if perclos >= self.perclos_threshold:
                return "Drowsy"
                
        # 3. Normal (정상)
        return "Normal"
