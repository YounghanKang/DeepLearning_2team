## 실행 방법

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 앱 실행
```bash
streamlit run app.py
```

### 3. 브라우저 접속
기본: http://localhost:8501

---

## 파일 구조

```
drowsiness_app/
├── app.py            # 메인 Streamlit 앱
├── requirements.txt  # 의존 패키지
└── README.md         # 이 파일
```

---

## 코드 구조 상세 설명

### 1. 라이브러리 임포트 및 페이지 설정

앱 구동에 필요한 라이브러리를 불러오고, Streamlit 페이지의 기본 속성을 설정하는 영역입니다.

- `streamlit` — UI 전체를 구성하는 웹 프레임워크
- `cv2 (OpenCV)` — 웹캠 영상 캡처 및 프레임 위에 텍스트·도형 오버레이 그리기
- `mediapipe` — Google의 얼굴 랜드마크 감지 모델 (눈 좌표 추출에 사용)
- `numpy` — 로그 점수들의 중앙값 계산 (`adapt_thresholds`에서 사용)
- `time` — 루프 FPS 제어(`sleep`), 로그 기록 간격 측정, 긴급 단계 지속 시간 측정
- `math` — 눈 랜드마크 좌표 간 유클리드 거리 계산
- `deque` — 고정 길이 EAR 버퍼 (오래된 값 자동 제거)
- `datetime` — 로그 및 활동 메시지에 현재 시각 표시

`st.set_page_config`로 브라우저 탭 제목, 아이콘, 레이아웃(wide), 사이드바 기본 상태(접힘)를 지정합니다.


---


### 2. CSS 스타일 정의

`st.markdown`으로 전역 CSS를 주입하는 영역입니다. 앱 전체의 시각적 디자인을 담당하며, 아래의 색상 팔레트를 기준으로 구성됩니다.

| 색상 | 코드 | 용도 |
|------|------|------|
| BLUE | `#2563EB` | 헤더, 섹션 레이블, 버튼, 현황 패널 좌측 선 |
| VIOLET | `#7C3AED` | 로그 패널 좌측 선, 스크롤바 |
| SLATE | `#1E293B` | 텍스트, 탑바 배경 |
| WHITE | `#FFFFFF` | 페이지 배경 |

정의된 CSS 컴포넌트는 다음과 같습니다.

- **기본 초기화** — 모든 요소의 `margin`, `padding`을 0으로 초기화하고 배경을 흰색으로 고정. Streamlit 기본 메뉴·푸터·헤더를 숨김
- **`.topbar`** — 상단 다크 네이비 바. 좌측에 시스템 제목과 배지, 우측에 현재 시각 표시
- **`.section-label`** — 각 패널 상단의 파란색 소제목. `::before` 가상요소로 좌측에 파란 세로선 장식
- **`.log-panel`** — 왼쪽 측정 로그 영역. 보라색 좌측 선, 고정 높이 454px, 세로 스크롤, 보라색 스크롤바
- **`.log-entry`** — 로그 한 줄의 레이아웃. 항목 진입 시 왼쪽에서 슬라이드되는 `fadeSlide` 애니메이션 적용
- **`.alert-panel`** — 오른쪽 위험 경보 패널. 단계별 배경·테두리 색상이 동적으로 변경됨
- **`.stage-number` / `.stage-label`** — 경보 패널 중앙의 단계 숫자(대형)와 단계명 텍스트
- **`.activity-panel`** — 오른쪽 하단 실시간 현황 패널. 파란색 좌측 선
- **`.gauge-wrap` / `.gauge-track` / `.gauge-segments` / `.gauge-fill` / `.gauge-siren`** — 중앙 하단 위험 점수 게이지 바. 5색 세그먼트 배경 위에 그라디언트 채움 바가 올라가고, 🚨 이모지가 현재 점수 위치로 이동
- **`.cam-standby`** — 웹캠이 꺼진 상태의 회색 점선 박스 (대기 화면)
- **`div[data-testid="stButton"] button`** — 모든 Streamlit 버튼을 파란색 스타일로 통일
- **`.rec-indicator` / `.rec-dot`** — 녹화 중일 때 표시되는 빨간색 깜빡이는 REC 표시
- **`.alert-danger` / `.alert-warn`** — 위험(4단계)·경고(2단계 이상) 시 경보 패널에 적용되는 펄스 애니메이션
- **`.log-index`** — 로그 항목의 순번 표시 (`#1`, `#2` ...)


---


### 3. MediaPipe 설정 및 눈 감지 함수

웹캠 프레임에서 눈의 개폐 정도를 수치로 변환하는 핵심 감지 로직입니다.

**랜드마크 인덱스 정의**

```python
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
```

MediaPipe FaceMesh가 반환하는 478개 얼굴 랜드마크 중 왼쪽·오른쪽 눈 주변 각 6개 점의 인덱스입니다. 이 6개 점으로 EAR을 계산합니다.

**`euclidean(p1, p2)`**

두 랜드마크 점 사이의 유클리드 거리를 계산합니다. EAR 공식에서 눈의 세로·가로 길이를 구할 때 사용됩니다.

**`calc_ear(lm, eye_idx)`**

EAR(Eye Aspect Ratio)을 계산합니다. 눈 6개 점 중 세로 방향 2쌍의 거리 평균을 가로 방향 거리로 나눈 값입니다. 눈이 완전히 열려 있으면 약 0.3~0.4, 감기면 0.2 이하로 떨어집니다. 분모(C)가 0이면 0을 반환해 ZeroDivisionError를 방지합니다.

```
EAR = (A + B) / (2.0 × C)
A = pts[1] ↔ pts[5] 거리 (세로)
B = pts[2] ↔ pts[4] 거리 (세로)
C = pts[0] ↔ pts[3] 거리 (가로)
```

**`calc_perclos(ear_buf)`**

EAR 버퍼(최근 150프레임) 중 EAR이 0.21 미만인 프레임의 비율을 반환합니다. 이 값이 PERCLOS(눈 감김 비율)이며, 0.0~1.0 범위입니다. 버퍼가 비어있으면 0.0을 반환합니다.

**`perclos_to_score(perclos)`**

PERCLOS(0.0~1.0)에 250을 곱해 0~100 범위의 졸음 점수로 변환합니다. 최대값은 100으로 제한됩니다. PERCLOS가 0.4 이상이면 점수가 100으로 고정됩니다.


---


### 4. 단계 설정 (STAGE_CFG / score_to_stage)

졸음 점수를 5단계로 분류하고, 각 단계의 시각적 스타일을 정의하는 영역입니다.

**`STAGE_CFG`**

5개 딕셔너리 리스트로, 각 단계(0~4)의 라벨·색상·배경색·테두리색·텍스트색을 보관합니다. 경보 패널, 로그 뱃지, 프레임 오버레이 색상 등 UI 전반에서 단계 번호를 인덱스로 참조합니다.

| 단계 | 라벨 | 의미 |
|------|------|------|
| 0 | 정상 | 졸음 없음 |
| 1 | 주의 | 경미한 졸음 징후 |
| 2 | 경고 | 졸음 감지 |
| 3 | 위험 | 즉시 휴식 필요 |
| 4 | 긴급 | 즉각 정차 필요, 자동 신고 대기 |

**`score_to_stage(score, thresholds)`**

현재 졸음 점수와 임계값 리스트를 받아 0~4 단계를 반환합니다. 임계값은 기본적으로 `[0, 20, 40, 60, 80, 100]`이며, 적응형 임계값 기능으로 사용자에 맞게 조정될 수 있습니다.


---


### 5. 세션 상태 초기화 (init_state)

Streamlit은 UI 이벤트마다 스크립트 전체를 재실행하기 때문에, 상태값을 `st.session_state`에 저장해야 값이 유지됩니다. `init_state()`는 앱 최초 실행 시 아래 키들을 초기화합니다. 이미 존재하는 키는 덮어쓰지 않습니다.

| 키 | 초기값 | 용도 |
|----|--------|------|
| `running` | `False` | 웹캠 루프 실행 여부 |
| `score` | `0.0` | 현재 졸음 점수 (지수 평활 적용값) |
| `ear_buffer` | `deque(maxlen=150)` | 최근 150프레임 EAR 값 저장 |
| `log_entries` | `[]` | 3초마다 기록되는 로그 목록 |
| `thresholds` | `[0,20,40,60,80,100]` | 단계 경계값 |
| `last_log_time` | `0` | 마지막 로그 기록 시각 |
| `activity_messages` | `[]` | 실시간 현황 메시지 목록 |
| `alert_stage` | `0` | 현재 경보 단계 |
| `stage_start_time` | `None` | 위험·긴급 단계 진입 시각 |
| `auto_called` | `False` | 자동 119 신고 여부 (중복 방지) |
| `cap` | `None` | OpenCV VideoCapture 객체 |


---


### 6. 헬퍼 함수 (adapt_thresholds / reset_thresholds / add_activity)

버튼 클릭이나 상태 변화 시 호출되는 보조 함수들입니다.

**`adapt_thresholds(log_entries)`**

로그가 20개 미만이면 경고 메시지를 출력하고 종료합니다. 20개 이상이면 기록된 점수들의 중앙값을 기준으로 임계값을 재계산합니다. 중앙값이 40보다 높으면 그만큼 전체 임계값을 위로 이동시켜, 해당 사용자의 평소 졸음 수준에 맞게 단계 경계를 자동 조정합니다. 재조정 후 점수를 0으로 리셋합니다.

**`reset_thresholds()`**

임계값을 기본값 `[0, 20, 40, 60, 80, 100]`으로 복원하고, 점수·로그·자동신고 여부를 모두 초기화합니다.

**`add_activity(msg)`**

현재 시각과 함께 메시지를 `activity_messages` 리스트에 추가합니다. 최근 10개만 유지하며, 이 메시지들이 오른쪽 '실시간 현황' 패널에 표시됩니다.


---


### 7. 렌더 함수 (normalize_score / render_gauge / render_log / render_alert / render_activity)

각 UI 패널의 HTML을 문자열로 생성해 반환하는 함수들입니다. Streamlit의 `st.empty().markdown()`과 결합해 실시간으로 업데이트됩니다.

**`normalize_score(score, thresholds)`**

임계값 구간이 가변적(적응형)이기 때문에, 게이지 바의 시각적 위치를 계산하려면 점수를 화면 비율(0~100%)로 변환해야 합니다. 현재 점수가 어느 구간에 속하는지 찾고, 그 구간 내 위치를 0~100% 게이지 위치로 선형 변환합니다. 이 값이 게이지 채움 바와 🚨 이모지의 `left` CSS 속성에 사용됩니다.

**`render_gauge(score, thresholds)`**

졸음 위험 점수 게이지 바 전체를 HTML로 생성합니다. 5개 세그먼트(초록→노랑→주황→빨강→진빨강)의 너비는 임계값 구간 크기에 비례하며, 그 위에 그라디언트 채움 바와 🚨 이모지가 현재 점수 위치에 표시됩니다. 하단에는 임계값 숫자 눈금이 표시됩니다.

**`render_log(log_entries, thresholds)`**

로그 항목이 없으면 안내 문구를, 있으면 최근 80개 항목을 최신순(역순)으로 나열합니다. 각 항목은 순번(`#N`), 시각, 점수, 단계 뱃지로 구성되며 단계별 색상이 적용됩니다.

**`render_alert(stage)`**

현재 단계 번호와 라벨을 중앙에 크게 표시하는 경보 패널 HTML을 생성합니다. 2단계 이상이면 주황 펄스(`alert-warn`), 4단계이면 빨간 펄스(`alert-danger`) 애니메이션 클래스가 추가됩니다. 우측 상단에 단계별 아이콘(🔔 / ⚠️ / 🚨)이 표시됩니다.

**`render_activity(messages)`**

메시지가 없으면 '시스템 대기 중...'을, 있으면 최근 6개를 최신순으로 나열합니다. 각 메시지는 시각과 내용으로 구성됩니다.


---


### 8. 상단 바 및 3단 레이아웃 구성

**상단 바**

`st.markdown`으로 `.topbar` HTML을 직접 렌더링합니다. 시스템 제목, DROWSY-DETECTION 배지, 앱 시작 시각이 표시됩니다.

**3단 컬럼 레이아웃 (`col_left : col_mid : col_right = 1 : 2.2 : 1`)**

- **왼쪽 (`col_left`)** — '측정 로그' 레이블, 로그 패널(`log_placeholder`), 그 아래 ⚙ 변경 / 🔄 초기화 버튼 2개
- **가운데 (`col_mid`)** — '실시간 모니터링' 레이블 + REC 표시, ▶/■ 토글 버튼, 웹캠 프레임 영역(`frame_placeholder`), 게이지 바(`gauge_placeholder`). 웹캠이 꺼져 있으면 `frame_placeholder`에 대기 화면을 표시
- **오른쪽 (`col_right`)** — '위험 경보' 레이블, 경보 패널(`alert_placeholder`), '실시간 현황' 패널(`activity_placeholder`)

각 `placeholder`는 `st.empty()`로 선언되어, 웹캠 루프에서 `update_ui()`가 호출될 때마다 내용이 덮어써지며 실시간 업데이트됩니다.

**토글 버튼 동작**

- 시작 클릭 → `running=True`, EAR 버퍼 초기화, `auto_called=False` 리셋 후 `st.rerun()`
- 중지 클릭 → `running=False`, VideoCapture 객체 해제(`cap.release()`), `session_state["cap"] = None` 후 `st.rerun()`


---


### 9. UI 업데이트 함수 (update_ui)

웹캠 루프에서 매 프레임마다 호출되어 모든 UI 패널을 갱신하는 중앙 함수입니다.

**단계 계산 및 경보 갱신**

현재 점수로 `stage`, 이전 점수로 `prev_stage`를 계산합니다. 경보 패널과 게이지 바를 즉시 갱신하고, `alert_stage`를 세션에 저장합니다.

**단계 변화 감지 및 활동 메시지**

`stage != prev_stage`일 때만 해당 단계의 안내 메시지를 현황 패널에 추가합니다. 단계가 3 이상으로 올라가면 `stage_start_time`에 현재 시각을 기록합니다.

**자동 119 신고 로직**

4단계(긴급) 상태에서 `auto_called`가 False이고 `stage_start_time`으로부터 5초가 지나면, 신고 메시지를 현황 패널에 추가하고 `auto_called=True`로 설정합니다. 단계가 3 미만으로 내려가면 `auto_called`와 `stage_start_time`을 초기화합니다.

**로그 기록**

`LOG_INTERVAL = 1` 상수에 따라 마지막 로그 기록 시각으로부터 1초 이상 경과했을 때만 현재 시각·점수·단계를 `log_entries`에 추가하고 로그 패널을 갱신합니다.


---


### 10. 웹캠 루프

`st.session_state["running"]`이 True일 때만 진입하는 메인 감지 루프입니다.

**VideoCapture 초기화**

`cap` 객체가 없거나 열려있지 않으면 새로 생성합니다. 해상도 640×480, FPS 30으로 설정합니다.

**MediaPipe FaceMesh 초기화**

최대 1개 얼굴, 정밀 랜드마크 모드(`refine_landmarks=True`), 감지·추적 신뢰도 각 0.5로 설정합니다.

**프레임 처리 루프 (`while st.session_state["running"]`)**

매 반복마다 아래 순서로 처리됩니다.

1. `cap.read()`로 프레임 획득. 실패 시 활동 메시지 추가 후 루프 종료
2. `cv2.flip(frame, 1)`으로 좌우 반전 (거울 모드)
3. BGR→RGB 변환 후 MediaPipe에 전달해 얼굴 랜드마크 추출
4. 얼굴이 감지되면 왼쪽·오른쪽 EAR 평균을 EAR 버퍼에 추가. 눈 랜드마크 12개 위치에 파란 점 오버레이
5. EAR 버퍼로 PERCLOS 계산 → 졸음 점수(raw) 계산 → 지수 평활(`score = 0.85 × prev + 0.15 × raw`)로 급격한 변동 완화
6. 현재 단계·색상 결정, 단계에 따른 두께로 프레임 테두리 사각형 그리기 (3단계 이상이면 더 두꺼운 선)
7. 프레임 좌상단에 EAR, PERCLOS, SCORE, STAGE 텍스트 오버레이
8. 3단계 이상이면 `frame_count % 15 < 8` 조건으로 빨간 경고 테두리 깜빡임 효과 적용
9. `update_ui()` 호출로 전체 UI 갱신
10. `time.sleep(1/30)`으로 약 30FPS 유지

루프 종료 후 `face_mesh.close()`로 MediaPipe 리소스를 해제합니다.


---


## 기능 요약

| 항목 | 설명 |
|------|------|
| **EAR** | Eye Aspect Ratio — 눈 개폐 정도 (0.0 ~ 0.5) |
| **PERCLOS** | 최근 150프레임 중 EAR < 0.21인 프레임 비율 |
| **졸음 점수** | PERCLOS × 250 → 0~100점, 지수 평활 적용 |
| **5단계 위험** | 0(정상) / 1(주의) / 2(경고) / 3(위험) / 4(긴급) |
| **적응형 임계값** | 로그 중앙값 기반으로 단계 경계 자동 재조정 |
| **자동 긴급 신고** | 4단계 5초 지속 시 119 자동 신고 알림 발생 |

## 설정 변경

`app.py` 내 상수 수정:
```python
LOG_INTERVAL = 3     # 로그 기록 간격 (초)
maxlen=150           # EAR 버퍼 길이 (150프레임 ≈ 5초 @ 30fps)
0.21                 # EAR 기준 임계값 (눈 감김 판정, calc_perclos 내)
```

## 자동 긴급 신고 연동

실제 신고 기능 연동 시 `app.py`의 `auto_called` 블록에 API 호출을 추가합니다.

```python
if t_start and time.time() - t_start > 5:
    add_activity("📞 긴급 단계 5초 지속! — 119 자동 신고합니다.")
    # 여기에 실제 신고 API 호출 추가
    st.session_state["auto_called"] = True
```
