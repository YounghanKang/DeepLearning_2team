# EAR 계산 모듈

눈 랜드마크 6개 좌표로부터 **EAR(Eye Aspect Ratio)**을 계산하고 눈 감김 여부를 판정하는 모듈입니다.

## 파일 구성

| 파일 | 역할 |
|------|------|
| `ear.py` | 핵심 모듈 (`EARCalculator` 클래스) |
| `demo_ear.py` | 시각 데모 (`eye_extractor`와 통합) |
| `test_ear.py` | 자동 테스트 |

---

## EAR이란?

눈의 가로/세로 비율을 수치화한 값입니다.

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```

- 눈을 뜨면 값이 **큼** (보통 0.25~0.35)
- 눈을 감으면 값이 **작음** (0.15 이하)

이 값을 임계값과 비교해서 눈 감김 여부를 판정합니다.

---

## 본 모듈의 차별점

단순 EAR 공식을 넘어 **실사용에서 발생하는 3가지 문제**를 해결합니다.

### 1. 개인별 baseline 자동 측정
사람마다 눈 크기가 달라 절대 임계값(예: 0.25)을 일괄 적용하면 오인식이 많아집니다. 
사용 시작 시 5초간 정면 응시로 본인 평소 EAR을 측정해 **개인 맞춤 임계값**(평소의 70%)을 자동 설정합니다.

### 2. 이동평균으로 떨림 제거
매 프레임 결과가 흔들리는 현상(flickering)을 방지하기 위해 최근 5프레임 평균을 사용합니다.

### 3. 좌·우 눈 분리 처리
한쪽 눈만 윙크하거나 가려지는 경우를 위해 양쪽 EAR을 분리해 보고합니다.

---

## 사용법

### 빠른 시작

```python
import numpy as np
from ear import EARCalculator

calc = EARCalculator()

# 매 프레임마다
result = calc.process(left_landmarks, right_landmarks)

print(f"EAR: {result.ear_smoothed:.3f}")
print(f"눈 감김: {result.is_closed}")
```

### `eye_extractor`와 통합

```python
from eye_extractor import EyeExtractor
from ear import EARCalculator

with EyeExtractor() as extractor:
    calc = EARCalculator()

    while True:
        ret, frame = cap.read()
        eye_data = extractor.extract(frame)
        if eye_data is None:
            continue

        # 핵심 호출
        ear_result = calc.process(
            eye_data.left_eye_landmarks,
            eye_data.right_eye_landmarks
        )

        # PERCLOS 담당자에게 전달할 값
        is_closed = ear_result.is_closed  # True/False
```

### 개인 맞춤 임계값 설정 (권장)

```python
calc = EARCalculator()

# 1. calibration 시작
calc.start_calibration()

# 2. 사용자가 5초간 정면 응시
for _ in range(150):  # 30fps × 5초
    result = calc.process(left_lm, right_lm)

# 3. calibration 완료
baseline = calc.finish_calibration()
# baseline EAR과 임계값이 자동 설정됨
```

---

## 출력 구조: `EARResult`

| 필드 | 타입 | 설명 | 누가 쓰나 |
|------|------|------|----------|
| `ear_left` | float | 왼쪽 눈 EAR | 디버깅 |
| `ear_right` | float | 오른쪽 눈 EAR | 디버깅 |
| `ear_avg` | float | 양쪽 평균 (원본) | 분석 |
| `ear_smoothed` | float | 이동평균 적용 EAR | 시각화·분석 |
| **`is_closed`** | **bool** | **눈 감김 여부** | **PERCLOS 담당자** |
| `threshold` | float | 현재 임계값 | 디버깅 |

---

## 실행

### 자동 테스트

```bash
python test_ear.py
```

모든 항목이 `✓`로 표시되면 정상.

### 시각 데모

`eye_extractor.py`가 같은 폴더에 있어야 합니다.

```bash
python demo_ear.py
```

조작:
- `c`: baseline 측정 시작 (5초간 정면 응시)
- `q`: 종료

화면에서 확인할 것:
- 눈 다각형 색이 **초록(뜸) → 빨강(감음)**으로 바뀌는지
- 우측 하단 EAR 수치가 눈 깜빡임에 따라 변하는지
- `c` 누르고 5초 후 임계값이 자동 설정되는지

---

## PERCLOS 담당자에게 (다음 단계 인터페이스)

이 모듈의 출력 `is_closed`가 PERCLOS의 입력입니다.

```python
# EAR → PERCLOS 연결 예시
ear_result = ear_calc.process(left_lm, right_lm)
perclos_calc.update(ear_result.is_closed)  # True/False 전달
```

PERCLOS는 최근 60초간 `is_closed`가 `True`였던 비율을 계산합니다.

---

## 임계값 튜닝 가이드

기본 설정으로 시작하되, 필요 시 조정 가능합니다.

```python
calc = EARCalculator(
    smoothing_window=5,      # 이동평균 윈도우 (크면 안정적, 작으면 반응 빠름)
    threshold_ratio=0.7,     # baseline의 70%
    default_threshold=0.22,  # calibration 안 했을 때 기본값
)
```

**튜닝 팁:**
- 오경보가 많다면 → `threshold_ratio`를 0.65로 낮춤
- 졸음을 놓친다면 → `threshold_ratio`를 0.75로 올림
- 결과가 너무 흔들리면 → `smoothing_window`를 10으로 늘림

---

## 알려진 한계

- **점멸과 졸음 구분 불가**: 0.3초 깜빡임도 `is_closed=True` 발생. PERCLOS에서 시간 누적으로 해결.
- **안경 반사**: 좌표 자체가 흔들리면 EAR도 흔들림. (`eye_extractor`의 한계)
- **고개 기울임**: 정면 가정. 좌우 30도 이상 기울면 정확도 저하.

---

## 환경

- Python 3.11
- numpy >= 1.24
- (선택) opencv-python, mediapipe — demo_ear.py 실행 시
