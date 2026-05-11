# 눈 정보 추출 모듈 (Eye Extractor)

졸음 운전 탐지 시스템의 첫 단계 모듈입니다.
웹캠 프레임에서 눈 좌표(EAR용)와 눈 영역 이미지(CNN용)를 추출해 다음 단계로 전달합니다.

## 파일 구성

| 파일 | 역할 |
|------|------|
| `eye_extractor.py` | 핵심 모듈. 다른 팀원은 이 파일만 import해서 사용 |
| `demo.py` | 시각 데모 (웹캠 띄우고 눈 표시) |
| `test_eye_extractor.py` | 자동 테스트 |
| `requirements.txt` | 설치할 패키지 목록 |

---

## 환경 (중요)

아래 조합이 가장 안정적입니다. 다른 조합은 권장하지 않습니다.

- **Python 3.11** (3.13에서는 mediapipe 호환성 문제 발생)
- **mediapipe 0.10.9** (최신 버전에서는 `mp.solutions` API 호출 방식이 다름)
- macOS / Windows / Linux 모두 동작 확인됨

---

## 설치

### macOS / Linux

```bash
# 1. Python 3.11 설치 (없는 경우만)
brew install python@3.11

# 2. 프로젝트 폴더로 이동
cd drowsy-detection

# 3. 가상환경 생성 및 활성화
python3.11 -m venv venv
source venv/bin/activate

# 4. 패키지 설치
pip install -r requirements.txt
```

### Windows

```bash
# 1. https://www.python.org/downloads 에서 Python 3.11 설치

# 2. 프로젝트 폴더로 이동
cd drowsy-detection

# 3. 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate

# 4. 패키지 설치
pip install -r requirements.txt
```

---

## 실행

### 1. 시각 데모 (먼저 이걸 해보세요)

```bash
python demo.py
```

웹캠이 켜지고 눈 주변에 초록색 다각형이 그려지면 성공입니다.
우측 상단에 추출된 눈 영역 미리보기가 함께 표시됩니다.

**조작**
- `q` : 종료
- `s` : 현재 눈 영역 이미지를 파일로 저장 (테스트용)

### 2. 자동 테스트

```bash
python test_eye_extractor.py
```

모든 항목이 `✓` 로 표시되면 모듈이 정상입니다.

---

## 다른 모듈에서 사용하는 법

```python
import cv2
from eye_extractor import EyeExtractor

# 1. 추출기 생성 (한 번만)
extractor = EyeExtractor(eye_image_size=64)

# 2. 웹캠 열기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. 핵심 호출
    result = extractor.extract(frame)

    if result is None:
        # 얼굴이 안 보이는 상황
        continue

    # 4. 결과 사용
    # EAR 담당자는 좌표 사용
    left_landmarks = result.left_eye_landmarks    # shape (6, 2)
    right_landmarks = result.right_eye_landmarks  # shape (6, 2)

    # CNN 담당자는 이미지 사용
    left_image = result.left_eye_image    # shape (64, 64), 흑백
    right_image = result.right_eye_image  # shape (64, 64), 흑백

    # ... 다음 단계 처리 ...

cap.release()
extractor.close()
```

### with 구문 사용 (권장)

```python
with EyeExtractor() as extractor:
    result = extractor.extract(frame)
    # 사용 후 자동으로 close() 호출됨
```

---

## 출력 데이터 구조

`extract()` 메서드는 `EyeData` 객체를 반환합니다.

| 필드 | 타입 | 설명 | 사용처 |
|------|------|------|--------|
| `left_eye_landmarks` | np.ndarray (6, 2) float32 | 왼쪽 눈 좌표 6개 (픽셀 단위) | EAR 계산 |
| `right_eye_landmarks` | np.ndarray (6, 2) float32 | 오른쪽 눈 좌표 6개 | EAR 계산 |
| `left_eye_image` | np.ndarray (64, 64) uint8 | 왼쪽 눈 영역 흑백 이미지 | CNN 입력 |
| `right_eye_image` | np.ndarray (64, 64) uint8 | 오른쪽 눈 영역 흑백 이미지 | CNN 입력 |
| `face_detected` | bool | 얼굴 검출 성공 여부 | 디버깅 |

얼굴이 검출되지 않으면 `extract()`는 `None`을 반환합니다.
호출하는 쪽에서 반드시 `if result is None:` 체크를 해야 합니다.

---

## EAR 계산 담당자에게

`left_eye_landmarks`와 `right_eye_landmarks`의 6개 점은 다음 순서로 정렬되어 있습니다.

```
[0] 바깥쪽 끝       (p1)
[1] 위 안쪽         (p2)
[2] 위 바깥쪽       (p3)
[3] 안쪽 끝         (p4)
[4] 아래 바깥쪽     (p5)
[5] 아래 안쪽       (p6)
```

EAR 공식의 p1~p6에 그대로 매핑됩니다.

```python
# EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
from scipy.spatial.distance import euclidean

def calculate_ear(eye_landmarks):
    A = euclidean(eye_landmarks[1], eye_landmarks[5])
    B = euclidean(eye_landmarks[2], eye_landmarks[4])
    C = euclidean(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)

ear_left = calculate_ear(result.left_eye_landmarks)
ear_right = calculate_ear(result.right_eye_landmarks)
ear_avg = (ear_left + ear_right) / 2
```

---

## CNN 학습 담당자에게

`left_eye_image`와 `right_eye_image`는 다음과 같이 전처리되어 있습니다.

- 64×64 크기 (생성자 인자로 변경 가능)
- 흑백 (1채널)
- uint8 (0~255)

PyTorch에 입력할 때는 정규화가 필요합니다.

```python
import torch

# numpy uint8 [0, 255] → torch float32 [0, 1]
tensor = torch.from_numpy(left_image).float() / 255.0
tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64) batch + channel
```

---

## 자주 발생하는 문제 해결

| 증상 | 원인 | 해결 방법 |
|------|------|----------|
| `ModuleNotFoundError: mediapipe` | 가상환경 미활성화 | `source venv/bin/activate` |
| `externally-managed-environment` | 시스템 Python에 직접 설치 시도 | 가상환경 사용 (위 설치 절차) |
| `AttributeError: ... 'solutions'` | mediapipe 0.10.35 이상 설치됨 | `pip install mediapipe==0.10.9` |
| M1/M2 Mac에서 설치 실패 | 아키텍처 호환성 | `pip install mediapipe-silicon` 대체 시도 |
| `extract()`가 항상 `None` | 얼굴이 잘 안 보임 | 조명 확인, 정면 응시 |
| 눈 영역이 너무 작게 잘림 | 카메라가 너무 멀음 | 50~70cm 거리 권장 |
| 안경 반사로 좌표 흔들림 | 알려진 한계 | 안경 데이터 별도 수집 권장 |

---

## 다음 단계 인터페이스

이 모듈의 출력은 다음 단계로 전달됩니다.

```
[EyeExtractor] → result.landmarks → [EAR Calculator] → ear_value
                ↘ result.images   → [CNN Classifier] → open_or_closed
                                                          ↓
                                          [PERCLOS] → [State Judgment]
```

**약속**: `EyeData` 인터페이스는 한 달 동안 유지됩니다.
내부 구현 변경이 있어도 출력 형식은 동일합니다.

---

## 테스트 환경 (검증 완료)

- macOS Sonoma (Apple Silicon M1)
- Python 3.11
- mediapipe 0.10.9
- opencv-python 4.10.0
- numpy 1.26.4

다른 OS / 버전에서 문제 발생 시 이슈 등록 또는 담당자에게 연락해주세요.
