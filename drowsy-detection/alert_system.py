"""
alert_system.py
===============
위험 상태 감지 시 화면 효과와 소리를 재생하는 모듈입니다.
"""

import cv2
import threading
import time

try:
    # 윈도우 기본 비프음을 내기 위한 라이브러리
    import winsound
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False

class AlertSystem:
    def __init__(self):
        self.is_playing = False
        
    def _play_sound(self, frequency=1000, duration=500):
        if HAS_SOUND:
            winsound.Beep(frequency, duration)
        self.is_playing = False
        
    def trigger_alert(self, level="Drowsy"):
        """비동기적으로 알림음을 재생합니다 (영상이 멈추지 않도록)."""
        if not self.is_playing:
            self.is_playing = True
            
            if level == "Asleep":
                # 수면 상태: 높은 주파수, 길게
                freq, dur = 2000, 1000
            else:
                # 졸음 상태: 낮은 주파수, 짧게 두 번 (스레드 내부에서 한 번만 호출됨)
                freq, dur = 800, 400
                
            t = threading.Thread(target=self._play_sound, args=(freq, dur))
            t.daemon = True
            t.start()
            
    def draw_warning(self, frame, level):
        """화면에 붉은 테두리와 경고 문구를 렌더링합니다."""
        h, w = frame.shape[:2]
        
        if level == "Asleep":
            # 전체 붉은색 테두리 강하게
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 20)
            text = "WAKE UP!!! (ASLEEP)"
            color = (0, 0, 255)
            font_scale = 1.5
            thickness = 4
        elif level == "Drowsy":
            # 주황색 얇은 테두리
            cv2.rectangle(frame, (0, 0), (w, h), (0, 165, 255), 10)
            text = "WARNING (DROWSY)"
            color = (0, 165, 255)
            font_scale = 1.2
            thickness = 3
        else:
            return frame
            
        # 텍스트 중앙 정렬 계산
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        
        # 그림자 효과
        cv2.putText(frame, text, (text_x+2, text_y+2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness+2)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        return frame
