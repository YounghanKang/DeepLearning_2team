# Feature Extractor (EAR + Head Pose 통합 추출기)

영상 파일에서 졸음 운전 탐지에 필요한 특징을 한 번에 추출해 **CSV로 저장**하는 모듈입니다.

모델링 담당자가 학습 데이터로 바로 사용할 수 있는 형식으로 출력합니다.

## 파일 구성

| 파일 | 역할 |
|------|------|
| `feature_extractor.py` | 핵심 모듈 (`FeatureExtractor` 클래스) |
| `run_extractor.py` | 사용 예시 스크립트 (영상 단일/일괄/웹캠 녹화) |
| `test_feature_extractor.py` | 자동 테스트 (CSV 형식 검증 포함) |

---

## CSV 출력 형식

| 컬럼 | 데이터 | 단위/범위 |
|------|--------|----------|
| `video_id` | 영상 파일 이름 | String |
| `frame_id` | 프레임 번호 | Integer (0부터) |
| `ear` | 눈 개폐 정도 | 0.0 ~ 0.5 |
| `pitch` | 고개 상하 각도 | Degree (-90 ~ +90, 떨굼 = +) |
| `yaw` | 고개 좌우 각도 | Degree (-90 ~ +90) |
| `roll` | 고개 좌우 기울기 | Degree (-90 ~ +90) |
| `label` | 졸음 상태 정답 | 0, 1, 2 |

**얼굴 미검출 프레임**은 모든 특징값이 `-1`로 기록됩니다. 모델링 측에서 결측 처리하세요.

---

## 빠른 시작

### 1. 자동 테스트 (먼저)

```bash
python test_feature_extractor.py
```

CSV 출력 형식까지 검증합니다. 모든 ✓ 통과되어야 합니다.

### 2. 웹캠 녹화 → 즉시 추출 (개발 검증)

```bash
python run_extractor.py
```

기본 설정으로 5초간 웹캠 녹화 후 자동으로 CSV 생성됩니다.

결과 파일:
```
data/raw/webcam_test.mp4         ← 녹화 영상
data/features/webcam_test.csv    ← 추출된 특징
```

### 3. 실제 영상 파일 처리

`run_extractor.py`에서 `example_single_video()`를 호출:

```python
with FeatureExtractor() as extractor:
    extractor.process_video(
        video_path="data/raw/driver_01.mp4",
        output_csv="data/features/driver_01.csv",
        label=1,  # 0: 정상, 1: 주의, 2: 위험
    )
```

### 4. 폴더 일괄 처리

```python
label_map = {
    "normal_01.mp4":  0,
    "drowsy_01.mp4":  1,
    "danger_01.mp4":  2,
}

with FeatureExtractor() as extractor:
    extractor.process_folder(
        video_folder="data/raw",
        output_folder="data/features",
        label_map=label_map,
    )
```

---

## 환경

이전 모듈과 동일합니다. 추가 패키지 불필요.

- Python 3.11
- mediapipe 0.10.9
- opencv-python
- numpy

`csv`는 Python 표준 라이브러리라 별도 설치 불필요합니다.

---

## API 상세

### `FeatureExtractor.extract_from_frame(frame)`

**단일 프레임 처리**. 실시간 데모용이나 디버깅용으로 쓸 수 있습니다.

```python
result = extractor.extract_from_frame(frame)
# {"ear": 0.31, "pitch": -1.2, "yaw": 2.5, "roll": 0.3}
# 또는 None (얼굴 미검출)
```

### `FeatureExtractor.process_video(...)`

**영상 → CSV** 변환의 핵심 메서드.

| 인자 | 설명 | 예시 |
|------|------|------|
| `video_path` | 입력 영상 경로 | `"data/raw/driver_01.mp4"` |
| `output_csv` | 출력 CSV 경로 | `"data/features/driver_01.csv"` |
| `label` | 정답 레이블 | `0`, `1`, `2` |
| `video_id` | 식별자 (선택) | `None`이면 파일명 사용 |
| `verbose` | 진행 출력 여부 | `True`/`False` |

### `FeatureExtractor.process_folder(...)`

폴더 안 모든 영상(`.mp4`, `.avi`, `.mov`, `.mkv`)을 자동 처리.

---

## 출력 예시

```csv
video_id,frame_id,ear,pitch,yaw,roll,label
driver_01,0,0.31254,1.234,-2.567,0.123,1
driver_01,1,0.30987,1.456,-2.341,0.234,1
driver_01,2,-1,-1,-1,-1,1
driver_01,3,0.28534,5.678,-1.234,0.345,1
...
```

3번 행은 얼굴 미검출 프레임의 예시입니다.

---

## 모델링 담당자를 위한 가이드

### pandas로 불러오기

```python
import pandas as pd

df = pd.read_csv("data/features/driver_01.csv")

# 미검출 프레임 제거
df = df[df["ear"] != -1]

print(df.head())
print(f"유효 프레임: {len(df)}")
print(f"평균 EAR: {df['ear'].mean():.3f}")
```

### 여러 CSV 합치기

```python
import glob

csv_files = glob.glob("data/features/*.csv")
df_all = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
df_all = df_all[df_all["ear"] != -1]
```

### 사용 예시: 시계열 특징 생성

각 프레임의 개별 값보다 **시간 누적 특징**이 졸음 판정에 효과적입니다.

```python
# 1초 윈도우(30프레임) 평균
df["ear_rolling"] = df["ear"].rolling(window=30, min_periods=1).mean()
df["pitch_rolling"] = df["pitch"].rolling(window=30, min_periods=1).mean()
```

---

## 알려진 한계

| 항목 | 영향 | 대응 |
|------|------|------|
| 얼굴 미검출 프레임 | `-1`로 기록 | 모델 학습 시 필터링 또는 보간 |
| pitch 부호 | 고개 떨굼 = + (직관화) | 모델링 측에서 일관되게 사용 |
| 카메라 보정 없음 | 절대 각도 약간 오차 | 상대 변화 위주로 분석 |
| 측면 응시 시 정확도 ↓ | yaw 30도 이상에서 pitch 오차 증가 | 모델링 측에서 yaw 필터링 가능 |

---

## 처리 속도 (참고)

M1 Mac 기준:
- 한 프레임 처리: 약 10~15ms
- 1분짜리 30fps 영상(1800프레임): 약 20~30초

CPU에서도 충분히 빠릅니다. GPU 불필요.

---

## 다음 단계 인터페이스

```
[FeatureExtractor]
   ↓ (CSV)
[모델링 측]
   - pandas로 CSV 로드
   - 시계열 특징 생성
   - LSTM / Transformer 학습
   ↓
[졸음 상태 분류]
```

**약속**: CSV 컬럼 형식은 한 달 동안 유지됩니다.
변경 필요 시 사전 합의 후 진행합니다.
