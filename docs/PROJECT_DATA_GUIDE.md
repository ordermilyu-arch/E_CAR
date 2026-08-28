# EV 물류 운행 지원 시스템 - 데이터 구성 가이드

## 1. 이 문서의 목적

이 프로젝트는 Volvo FH Electric 기반의 전기 물류 트럭 운행 지원 시스템이다.

사용자가 8자리 사원번호를 입력하면 가상의 차량/화물/운행환경을 배정하고,
머신러닝으로 전력 소비량을 예측한 뒤 부산항 신선대 컨테이너터미널에서
한진인천컨테이너터미널(HJIT)까지의 고정 노선을 기준으로 충전 휴게소와 예상 도착시간을 계산한다.

---

## 2. VS Code 프로젝트에 넣을 데이터 파일

아래 4개 데이터 파일은 모두 프로젝트에 포함한다.

```text
data/
├─ vehicles_volvo_fh_electric_10.csv
├─ busan_port_hjit_service_route.csv
├─ route_config_busan_port_hjit.json
└─ model_adapter_config.json
```

이 MD 파일은 프로젝트 루트 또는 `docs/` 폴더에 둔다.

```text
docs/
└─ PROJECT_DATA_GUIDE.md
```

---

## 3. 각 파일의 역할

### `vehicles_volvo_fh_electric_10.csv`

Volvo FH Electric 10대의 가상 차량 데이터.

주요 역할:

- 사원번호 입력 후 1~10호차 중 기본 차량 배정
- 사용자가 차량을 변경할 때 선택 목록으로 사용
- 차량별 `tire_pressure_bar` 값을 ML 입력에 사용
- 배터리 용량 / 충전 최대 출력 / 기본 SOC 제공

차량 10대는 같은 Volvo FH Electric을 사용하되,
프로젝트에서는 타이어 공기압을 서로 다르게 설정한다.

---

### `busan_port_hjit_service_route.csv`

부산 → 인천 고정 물류 노선의 휴게소 데이터.

고정 노선:

```text
부산항 신선대 컨테이너터미널
→ 경부고속도로 부산기점
→ 경부고속도로 서울방향
→ 신갈JC
→ 영동고속도로 인천방향
→ 월곶JC
→ 제3경인고속화도로
→ 인천신항
→ 한진인천컨테이너터미널(HJIT)
```

CSV에는 다음 정보가 들어 있다.

- 휴게소 순서
- 누적거리
- 이전 휴게소에서의 거리
- EV 충전 가능 여부
- 확인된 충전기 출력
- 서비스에서 사용할 충전기 상태

프로젝트 규칙:

```text
충전기 출력 정보가 확인되지 않음
→ OUT_OF_SERVICE
→ 충전 추천 후보에서 제외
```

원본 데이터의 NULL 의미는 유지하고,
서비스용 CSV에서만 고장 상태로 해석한다.

---

### `route_config_busan_port_hjit.json`

노선 전체에서 공통으로 사용하는 설정값.

주요 값:

```text
출발지:
부산항 신선대 컨테이너터미널

도착지:
한진인천컨테이너터미널(HJIT)

프로젝트 고정 총거리:
465.6 km

기본 출발 SOC:
90%

최소 안전 SOC:
10%

평균 주행속도:
80 km/h

최대 충전 목표 SOC:
80%

충전 효율:
92%

충전 1회 추가 정차시간:
7분
```

이 값들은 Python 코드에 직접 숫자를 반복해서 적지 말고
JSON에서 불러와 사용한다.

---

### `model_adapter_config.json`

현재 머신러닝 모델과 실제 Volvo FH Electric 서비스 입력값 사이를 연결한다.

현재 ML 학습 데이터의:

- 화물량
- 타이어 공기압
- 전력소비량

스케일이 실제 대형 전기트럭과 다르기 때문에
서비스 입력값을 기존 모델의 범위에 맞게 변환하기 위한 설정 파일이다.

주의:

이 보정은 미니 프로젝트 시연용이다.

실제 물류 서비스 수준의 정확도를 주장하려면
향후 대형 전기트럭 실측 데이터로 다시 학습해야 한다.

---

## 4. 권장 프로젝트 폴더 구조

```text
ev-logistics-project/
│
├─ data/
│   ├─ ev_energy_consumption.csv
│   ├─ vehicles_volvo_fh_electric_10.csv
│   ├─ busan_port_hjit_service_route.csv
│   ├─ route_config_busan_port_hjit.json
│   └─ model_adapter_config.json
│
├─ models/
│   └─ energy_model.joblib
│
├─ src/
│   ├─ assignment.py
│   ├─ vehicle.py
│   ├─ energy_predictor.py
│   ├─ route.py
│   ├─ soc_simulator.py
│   ├─ charging_optimizer.py
│   ├─ eta.py
│   └─ simulation.py
│
├─ app/
│   └─ app.py
│
├─ docs/
│   └─ PROJECT_DATA_GUIDE.md
│
└─ README.md
```

---

## 5. 서비스 흐름

```text
8자리 사원번호 입력
        ↓
assignment.py
        ↓
오늘의 가상 배차 생성
        ↓
차량 / 화물 / 날씨 / HVAC
        ↓
사용자가 차량 변경 가능
사용자가 화물량 변경 가능
        ↓
energy_predictor.py
        ↓
예상 kWh/100km
        ↓
route.py
        ↓
고정 노선 및 휴게소 로드
        ↓
soc_simulator.py
        ↓
구간별 SOC 계산
        ↓
charging_optimizer.py
        ↓
가장 빠른 충전 계획 계산
        ↓
eta.py
        ↓
예상 도착시간 계산
        ↓
simulation.py
        ↓
전체 결과 반환
        ↓
Streamlit / Gradio
```

---

## 6. 중요한 설계 규칙

### 사원번호

```text
8자리 숫자
```

사원번호 + 날짜를 seed로 사용하여:

- 차량
- 화물량
- 날씨
- HVAC

를 가상 생성한다.

같은 날짜에 같은 사원번호를 입력하면 같은 배차가 나오도록 한다.

---

### 차량

기본 차량은 자동 배정하지만 사용자가 1~10호차 중 변경 가능하다.

차량을 변경하면:

```text
tire_pressure_bar 변경
→ ML 입력 변경
→ 예상 소비전력 변경
→ SOC 계산 변경
→ 충전 계획 변경 가능
```

화물량은 차량을 변경해도 유지한다.

---

### 화물

초기 화물량은 가상 배정한다.

사용자는 서비스 화면에서 화물 무게를 직접 수정할 수 있다.

---

### 날씨 / HVAC

가상 환경은 다음 세 종류를 사용한다.

```text
추운 날
→ 난방 ON

기본
→ HVAC OFF

더운 날
→ 냉방 ON
```

이 값은 단순 표시용이 아니라 ML 입력값에 사용한다.

---

### 충전 휴게소

충전 후보가 되려면:

```text
ev_charger_available == 1
AND
service_usable_for_charging == 1
```

이어야 한다.

충전기 출력 정보가 없는 휴게소는 프로젝트에서 고장으로 취급한다.

---

### 안전 SOC

```text
minimum_reserve_soc_pct = 10
```

다음 구간 도착 예상 SOC가 10% 미만이라면
그 구간을 안전하게 도달할 수 없는 것으로 판단한다.

---

### 충전 목표

충전이 필요한 경우 100%까지 충전한다.

주행 중 충전 횟수는 목적지 도착 가능 조건을 만족하는 최소 횟수로 계획한다.

---

### 최적화 목표

이 프로젝트의 목표는:

```text
충전 횟수 최소화
```

만이 아니라

```text
총 예상 도착시간 최소화
```

이다.

따라서:

- 이동시간
- 충전시간
- 충전소 진입/연결/출차시간

을 모두 포함하여 후보 충전 계획을 비교한다.

---

## 7. Codex 구현 순서

ML 모델 학습은 별도 진행 중이므로 서비스 코드는 아래 순서로 구현한다.

```text
1. assignment.py
2. vehicle.py
3. route.py
4. energy_predictor.py
5. soc_simulator.py
6. charging_optimizer.py
7. eta.py
8. simulation.py
9. app.py
```

### 1단계 `assignment.py`

구현:

```text
8자리 사원번호 검증
사원번호 + 날짜 기반 seed 생성
1~10호차 기본 배정
화물량 생성
날씨 생성
HVAC 생성
```

### 2단계 `vehicle.py`

구현:

```text
vehicles CSV 로드
차량 1대 조회
전체 차량 목록 조회
사용자 차량 변경 처리
```

### 3단계 `route.py`

구현:

```text
서비스 노선 CSV 로드
route_config JSON 로드
충전 가능한 휴게소 필터링
거리 데이터 반환
```

### 4단계 `energy_predictor.py`

구현:

```text
ML 모델 로드
model_adapter_config 로드
서비스 입력값 → 모델 입력값 변환
예측
서비스용 전력소비량으로 반환
```

### 5단계 `soc_simulator.py`

구현:

```text
거리
예상 kWh/100km
배터리 용량
현재 SOC
```

를 이용하여 구간별 SOC 계산.

### 6단계 `charging_optimizer.py`

구현:

```text
도달 가능한 휴게소 판단
고장 충전기 제외
충전출력 비교
필요 충전량 계산
예상 충전시간 계산
가장 빠른 충전 계획 선택
```

### 7단계 `eta.py`

구현:

```text
주행시간
+ 충전시간
+ 정차 부가시간
= 예상 도착시간
```

### 8단계 `simulation.py`

모든 기능을 연결한다.

### 9단계 `app.py`

Streamlit 또는 Gradio UI를 구현한다.

---

## 8. Codex에게 가장 먼저 시킬 작업

다음 작업부터 시작한다.

```text
PROJECT_DATA_GUIDE.md를 읽고 프로젝트 구조를 이해한다.

우선 src/assignment.py를 작성한다.

조건:
- 사원번호는 숫자 8자리만 허용
- 사원번호와 날짜를 이용하여 deterministic seed 생성
- vehicles_volvo_fh_electric_10.csv의 10대 중 하나를 기본 배정
- 프로젝트 최대 화물량 범위 안에서 화물량 생성
- 추운 날 / 기본 / 더운 날 중 하나 생성
- 추운 날이면 HEATING
- 기본이면 OFF
- 더운 날이면 COOLING
- 결과는 dict 형태로 반환
```

이 단계가 정상 동작한 뒤 다음 파일로 진행한다.
