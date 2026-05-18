# 종합 파이프라인 설계 문서

> 졸음 운전 탐지 시스템의 전체 아키텍처와 모듈 간 연결 규약을 정의합니다.
> 각 담당자는 본 문서의 인터페이스 명세에 맞춰 개발합니다.

---

## 목차

1. [전체 아키텍처](#1-전체-아키텍처)
2. [두 가지 처리 흐름](#2-두-가지-처리-흐름)
3. [모듈 현황](#3-모듈-현황)
4. [모듈별 인터페이스 명세](#4-모듈별-인터페이스-명세)
5. [신호 융합 전략](#5-신호-융합-전략-ear--cnn--head-pose)
6. [에러 처리 규칙](#6-에러-처리-규칙)
7. [성능 및 스레딩](#7-성능-및-스레딩)
8. [폴더 구조](#8-폴더-구조)
9. [개발 로드맵](#9-개발-로드맵)

---

## 1. 전체 아키텍처

본 시스템은 **두 가지 독립적인 처리 흐름**으로 구성됩니다.

```
┌──────────────────────────────────────────────────────────────┐
│  A. 배치 처리 (Offline) — 학습 데이터 준비                       │
│                                                              │
│   영상 파일 (.mp4)                                           │
│       ↓                                                      │
│   [feature_extractor] ✅                                     │
│       ↓                                                      │
│   CSV (ear, pitch, yaw, roll, label)                         │
│       ↓                                                      │
│   [CNN / LSTM 학습] 🔲                                       │
│       ↓                                                      │
│   학습된 모델 (.pth)                                         │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  B. 실시간 처리 (Online) — 운전자 모니터링                       │
│                                                              │
│   웹캠 (30fps BGR)                                           │
│       ↓                                                      │
│   ┌───────────────┬─────────────────┐                       │
│   ↓               ↓                 ↓                        │
│ [eye_extractor]  [head_pose]   (CNN용 이미지)                │
│   ✅              ✅                                          │
│   ↓               ↓                 ↓                        │
│ [ear.py] ✅    pitch/yaw/roll    [cnn_model] 🔲              │
│   ↓               ↓                 ↓                        │
│ is_closed     head_down 신호      is_closed_cnn              │
│   └───────────────┴─────────────────┘                       │
│                   ↓                                          │
│            [신호 융합 로직] 🔲                                 │
│                   ↓                                          │
│            [perclos.py] 🔲                                   │
│                   ↓                                          │
│            state ('NORMAL' / 'WARNING' / 'DANGER')           │
│                   ↓                                          │
│            ┌──────┴──────┐                                   │
│         [alert] 🔲     [dashboard] 🔲                        │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 두 가지 처리 흐름

### A. 배치 처리 (학습 단계)

| 단계 | 모듈 | 입력 | 출력 |
|------|------|------|------|
| 1 | 영상 녹화 | 웹캠 | `data/raw/*.mp4` |
| 2 | `feature_extractor` | mp4 | `data/features/*.csv` |
| 3 | 모델 학습 | CSV | `models/*.pth` |

**특징:**
- 오프라인. 시간 제약 없음.
- 데이터 수집 담당이 영상 만들고, 모델링 담당이 CSV로 학습.
- 학습 결과물(`.pth`)을 실시간 처리 단계에서 로드.

### B. 실시간 처리 (추론·시연 단계)

| 단계 | 모듈 | 입력 | 출력 |
|------|------|------|------|
| 1 | 웹캠 캡처 | OpenCV | BGR 프레임 |
| 2 | `eye_extractor` | 프레임 | `EyeData` |
| 3 | `head_pose` | 프레임 | `HeadPoseResult` |
| 4 | `ear` | 눈 좌표 | `EARResult` (`is_closed`) |
| 5 | `cnn_model` | 눈 이미지 | Closed 확률 |
| 6 | 신호 융합 | bool/float 다수 | 통합 `is_closed`, `is_head_down` |
| 7 | `perclos` | bool 시퀀스 | PERCLOS 값 + 상태 |
| 8 | `alert` / `dashboard` | 상태 | UI 표시 |

**특징:**
- 실시간. **30 FPS 이상** 유지가 목표.
- MediaPipe Face Mesh가 가장 무거움 → 중복 호출 주의.

---

## 3. 모듈 현황

| 모듈 | 상태 | 담당 | 위치 |
|------|------|------|------|
| `eye_extractor` | ✅ 완성 | CV1 | `eye_extract/` |
| `ear` | ✅ 완성 | CV2 | `ear/` |
| `head_pose` | ✅ 완성 | CV3 | `head_pose/` |
| `feature_extractor` | ✅ 완성 | CV1 | `feature_extract/` |
| `cnn_model` | 🔲 미완성 | ML | `src/` (예정) |
| `perclos` | 🔲 미완성 | ML | `src/` (예정) |
| `state_classifier` | 🔲 미완성 | ML | `src/` (예정) |
| `alert` | 🔲 미완성 | UI | `src/` (예정) |
| `04_demo.py` | 🔲 미완성 | **파이프라인** | 루트 |
| `05_dashboard.py` | 🔲 미완성 | **파이프라인 + UI** | 루트 |
| `01_collect_data.py` | 🔲 미완성 | 데이터 | 루트 |
| `02_train_cnn.py` | 🔲 미완성 | ML | 루트 |
| `03_evaluate.py` | 🔲 미완성 | ML | 루트 |

---

## 4. 모듈별 인터페이스 명세

### 4-1. `eye_extractor.EyeExtractor` ✅

```python
class EyeExtractor:
    def __init__(eye_image_size=64,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5)

    def extract(frame: np.ndarray) -> Optional[EyeData]
```

**입력**: `frame: (H, W, 3) uint8 BGR`
**출력**: `EyeData` 또는 `None` (얼굴 미검출)

**EyeData 필드:**

| 필드 | 타입 | 용도 |
|------|------|------|
| `left_eye_landmarks` | `(6, 2) float32` | EAR 계산 입력 |
| `right_eye_landmarks` | `(6, 2) float32` | EAR 계산 입력 |
| `left_eye_image` | `(64, 64) uint8` | CNN 입력 (흑백) |
| `right_eye_image` | `(64, 64) uint8` | CNN 입력 (흑백) |
| `face_detected` | `bool` | 디버깅용 |

---

### 4-2. `ear.EARCalculator` ✅

```python
class EARCalculator:
    def __init__(smoothing_window=5,
                 threshold_ratio=0.7,
                 default_threshold=0.22)

    def process(left_landmarks, right_landmarks) -> EARResult
    def start_calibration()
    def finish_calibration() -> Optional[float]
```

**EARResult 필드:**

| 필드 | 타입 | 다음 단계 전달 |
|------|------|---------------|
| `ear_left`, `ear_right` | `float` | 분석용 |
| `ear_avg` | `float` | CSV 저장 |
| `ear_smoothed` | `float` | 시각화 |
| **`is_closed`** | **`bool`** | **PERCLOS 입력** |
| `threshold` | `float` | 디버깅 |

---

### 4-3. `head_pose.HeadPoseEstimator` ✅

```python
class HeadPoseEstimator:
    def __init__(min_detection_confidence=0.5,
                 min_tracking_confidence=0.5)

    def estimate(frame: np.ndarray) -> Optional[HeadPoseResult]
```

**HeadPoseResult 필드:**

| 필드 | 타입 | 의미 |
|------|------|------|
| `pitch` | `float` | 상하 각도 (양수=떨굼, 졸음 신호) |
| `yaw` | `float` | 좌우 회전 (한눈팔이) |
| `roll` | `float` | 좌우 기울임 |
| `confidence` | `float` | 추정 신뢰도 (0~1) |
| `face_detected` | `bool` | 검출 여부 |

**연결 규약:**
- `pitch >= 20도` → `is_head_down = True`
- `confidence < 0.5` → 해당 프레임 신호 무시

---

### 4-4. `feature_extractor.FeatureExtractor` ✅

```python
class FeatureExtractor:
    def __init__(min_detection_confidence=0.5,
                 min_tracking_confidence=0.5)

    def extract_from_frame(frame) -> Optional[dict]  # 실시간 가능
    def process_video(video_path, output_csv, label, ...) -> int
    def process_folder(video_folder, output_folder, label_map)
```

**CSV 출력 형식:**

```csv
video_id, frame_id, ear, pitch, yaw, roll, label
```

- 미검출 프레임은 `-1`로 표기
- label: 0=정상, 1=주의, 2=위험
- 저장 위치: `data/features/*.csv`

---

### 4-5. `cnn_model.EyeStateCNN` 🔲 (ML 담당이 만들 것)

**약속한 인터페이스:**

```python
class EyeStateCNN(nn.Module):
    def forward(x: Tensor) -> Tensor
        """
        Input:  shape (B, 1, 64, 64) float32 [0, 1]
        Output: shape (B, 2) logits (Open=0, Closed=1)
        """
```

**사용 측 호출 패턴:**

```python
# 학습된 모델 로드
model = EyeStateCNN()
model.load_state_dict(torch.load("models/eye_cnn.pth"))
model.eval()

# 추론
with torch.no_grad():
    # uint8 [0,255] (64,64) → float32 [0,1] (1,1,64,64)
    tensor = torch.from_numpy(eye_image).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    logits = model(tensor)
    closed_prob = torch.softmax(logits, dim=1)[0, 1].item()

is_closed_cnn = closed_prob > 0.5  # bool
```

---

### 4-6. `perclos.PerclosCalculator` 🔲 (ML 담당이 만들 것)

**약속한 인터페이스:**

```python
class PerclosCalculator:
    def __init__(fps=30, window_seconds=60)

    def update(eye_closed: bool) -> float
        """
        프레임 1개의 눈 감김 여부를 추가하고 현재 PERCLOS 값을 반환.
        """

    def get_perclos() -> float
    def get_state() -> str   # 'NORMAL' / 'WARNING' / 'DANGER'
    def reset()
```

**상태 판정 기준:**

| PERCLOS | 상태 | 색상 | 경보음 |
|---------|------|------|--------|
| `< 0.10` | `NORMAL` | 초록 | 없음 |
| `0.10 ~ 0.30` | `WARNING` | 노랑 | 없음 |
| `>= 0.30` | `DANGER` | 빨강 | 재생 |

---

### 4-7. `alert.AlertSystem` 🔲 (UI 담당이 만들 것)

**약속한 인터페이스:**

```python
class AlertSystem:
    def __init__(sound_file="assets/alarm.mp3")

    def trigger(state: str)
        """
        state에 따라 화면 색상 + 경보음 처리.
        DANGER 상태에서만 소리 재생.
        """
    def get_overlay_color(state: str) -> tuple  # BGR
```

---

### 4-8. `04_demo.py` 🔲 (파이프라인 담당이 작성)

**실행 흐름 (의사 코드):**

```python
def main():
    cap = cv2.VideoCapture(0)

    # 1) 모듈 초기화
    eye_ext   = EyeExtractor()
    head_est  = HeadPoseEstimator()
    ear_calc  = EARCalculator()
    cnn       = load_cnn_model("models/eye_cnn.pth")
    perclos_e = PerclosCalculator(window_seconds=60)  # eye
    perclos_h = PerclosCalculator(window_seconds=60)  # head
    alert     = AlertSystem()

    while True:
        ok, frame = cap.read()
        if not ok: break

        # 2) 특징 추출 (eye_extractor 1회 호출로 두 모듈에 공유 권장)
        eye_data  = eye_ext.extract(frame)
        head_data = head_est.estimate(frame)

        # 3) 얼굴 미검출 처리
        if eye_data is None:
            display_no_face(frame); continue

        # 4) EAR
        ear_result = ear_calc.process(eye_data.left_eye_landmarks,
                                      eye_data.right_eye_landmarks)

        # 5) CNN (학습 완료 후)
        is_closed_cnn = cnn_predict(cnn, eye_data.left_eye_image)

        # 6) 신호 융합 (5번 섹션 참고)
        is_closed   = fuse_eye(ear_result.is_closed, is_closed_cnn)
        is_head_dn  = fuse_head(head_data)

        # 7) PERCLOS
        p_eye  = perclos_e.update(is_closed)
        p_head = perclos_h.update(is_head_dn)
        state  = decide_state(p_eye, p_head)

        # 8) UI
        alert.trigger(state)
        draw_overlay(frame, ear_result, head_data, state)
        cv2.imshow("Drowsy Detection", frame)
        if cv2.waitKey(1) == ord('q'): break
```

---

### 4-9. `05_dashboard.py` 🔲 (파이프라인 + UI)

**Streamlit 구성:**

| 영역 | 내용 |
|------|------|
| 좌측 | 실시간 웹캠 영상 + 눈 다각형 오버레이 |
| 우측 상단 | 현재 상태 큰 글씨 (NORMAL/WARNING/DANGER) |
| 우측 중단 | EAR 시계열 그래프 (최근 60초) |
| 우측 하단 | PERCLOS 게이지 + pitch 시계열 |
| 하단 | 통계 (오늘 누적 위험 시간, 경보 횟수) |

**구현 노트:**
- 영상 갱신은 `st.empty()`에 프레임 덮어쓰기
- 그래프는 `plotly` 권장 (재렌더 비용 낮음)
- 별도 스레드에서 캡처/추론, 메인 스레드에서 렌더

---

## 5. 신호 융합 전략 (EAR + CNN + Head Pose)

세 가지 신호를 어떻게 합칠지 정의합니다.

### 5-1. 눈 감김 신호 융합 (EAR + CNN)

**우선순위 기반:**

```python
def fuse_eye(ear_closed: bool, cnn_closed: bool, cnn_available: bool) -> bool:
    if cnn_available:
        return cnn_closed         # CNN 결과 우선
    return ear_closed             # 미완성 시 EAR 사용
```

**대안: 다수결 / 가중치**

```python
# 둘 다 닫혔다고 해야 닫힘 → 거짓 양성 감소
return ear_closed AND cnn_closed
# 또는 하나라도 닫히면 닫힘 → 거짓 음성 감소
return ear_closed OR cnn_closed
```

### 5-2. 고개 떨굼 신호

```python
HEAD_DOWN_PITCH = 20.0  # degree

def fuse_head(head: HeadPoseResult) -> bool:
    if head is None or head.confidence < 0.5:
        return False
    return head.pitch >= HEAD_DOWN_PITCH
```

### 5-3. 최종 상태 결정 (두 PERCLOS 종합)

```python
def decide_state(p_eye: float, p_head: float) -> str:
    # 두 PERCLOS 중 큰 값 기준 판정
    combined = max(p_eye, p_head)
    if combined >= 0.30: return 'DANGER'
    if combined >= 0.10: return 'WARNING'
    return 'NORMAL'
```

또는 가중합 (`0.7 * p_eye + 0.3 * p_head`) 방식 채택 가능.

---

## 6. 에러 처리 규칙

| 상황 | 처리 |
|------|------|
| `eye_extractor.extract() is None` | 해당 프레임 스킵, PERCLOS 누적 멈춤 (미검출이 길어지면 reset) |
| `head_pose.estimate() is None` | head 신호는 무시 (직전 값 유지하지 않음) |
| `head_pose.confidence < 0.5` | 해당 head 신호 무시 |
| EAR landmarks 좌표 비정상 | `ear=0.0`로 반환 (이미 처리됨) |
| CNN 모델 파일 없음 | EAR만 사용 (fallback) |
| 웹캠 끊김 | 루프 종료, 자원 해제 |
| 얼굴 미검출 5초 이상 | PERCLOS 버퍼 초기화 + UI에 "운전자 감지 실패" 표시 |

---

## 7. 성능 및 스레딩

### 7-1. 병목 분석

| 모듈 | 1 프레임당 | 비고 |
|------|-----------|------|
| MediaPipe Face Mesh | 10~15ms | 가장 무거움 |
| EAR 계산 | < 1ms | 무시 가능 |
| Head Pose (solvePnP) | 2~3ms | |
| CNN 추론 | 5~15ms | GPU 시 1ms |
| **합계** | **20~35ms** | **약 28~50 FPS** |

### 7-2. MediaPipe 중복 호출 문제

현재 `eye_extractor`와 `head_pose`가 각자 MediaPipe를 호출 → **MediaPipe가 2번 실행됨.**

**해결안:**

A) **통합 모듈 신설** (권장)

```python
class FaceExtractor:
    """MediaPipe 1회 호출로 모든 정보 추출."""
    def extract_all(frame) -> dict:
        return {
            "eye_data": ...,
            "head_pose": ...,
        }
```

B) **MediaPipe 결과 캐싱**: 같은 프레임에 대해 한 번만 처리 (임시 방편)

### 7-3. 스레딩 구조 (대시보드용)

```
[Thread 1: Capture]     웹캠에서 프레임 캡처 → Queue
[Thread 2: Inference]   Queue에서 꺼내 추론 → Result Queue
[Thread 3: UI (Main)]   Result Queue에서 꺼내 렌더링
```

- Queue 최대 크기 1~2 (지연 방지)
- `threading.Lock`으로 모델 동시 접근 차단

---

## 8. 폴더 구조

```
drowsy-detection/
├── PIPELINE.md              ← 본 문서
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/                 ← 원본 영상 (mp4)
│   ├── eye_crops/           ← CNN 학습용 이미지
│   │   ├── open/
│   │   └── closed/
│   ├── features/            ← feature_extractor 출력 CSV
│   └── ear_logs/            ← 디버깅용 EAR 시계열
│
├── models/
│   └── eye_cnn.pth          ← 학습된 CNN 가중치
│
├── assets/
│   └── alarm.mp3            ← 경보음
│
├── ear/                     ← ✅ 기존 모듈
├── eye_extract/             ← ✅ 기존 모듈
├── head_pose/               ← ✅ 기존 모듈
├── feature_extract/         ← ✅ 기존 모듈
│
├── src/                     ← 🔲 새 모듈 (ML, UI 담당)
│   ├── cnn_model.py
│   ├── perclos.py
│   ├── state_classifier.py
│   ├── alert.py
│   └── utils.py
│
├── 01_collect_data.py       ← 데이터 담당
├── 02_train_cnn.py          ← ML 담당
├── 03_evaluate.py           ← ML 담당
├── 04_demo.py               ← 파이프라인 담당
└── 05_dashboard.py          ← 파이프라인 담당
```

**규칙:**
- 각 모듈 폴더 안에는 `모듈.py`, `test_*.py`, `README.md` 함께 위치
- 공통 유틸은 `src/utils.py`
- 모듈 import 경로 통일 (`from ear.ear import EARCalculator` 등)

---

## 9. 개발 로드맵

| 주차 | 마일스톤 | 파이프라인 담당 작업 |
|------|---------|-------------------|
| 1 | EAR 베이스라인 ✅ | (현재) PIPELINE.md 작성 + 모듈 인터페이스 합의 |
| 2 | CNN 학습 시작 | CNN 인터페이스 검증 |
| 3 | 실시간 데모 | **`04_demo.py` 작성**, 신호 융합 로직 구현 |
| 4 | 대시보드 | **`05_dashboard.py` 작성**, 스레딩 최적화 |

---

## 부록. 용어 정리

- **EAR** (Eye Aspect Ratio): 눈 종횡비. 감을수록 작아짐.
- **PERCLOS** (Percentage of Eye Closure): 일정 시간 윈도우 내 눈 감김 비율. 자동차 업계 표준.
- **landmark**: MediaPipe Face Mesh가 추출하는 얼굴 위 468개 좌표점.
- **solvePnP**: OpenCV의 3D 자세 추정 함수.
- **baseline**: 사용자의 평상시 EAR 또는 자세 기준값.

---

**문서 버전**: v2.0 (2026-05-16)
