# EV 에너지 소비 예측 정확도 개선 계획

## 1. 문서 목적

이 문서는 `ev_energy_consumption.csv`를 이용한 전기차 에너지 소비 예측 프로젝트에서
머신러닝 예측 정확도와 데이터의 현실성을 높이기 위한 개선 계획을 정리한 작업 지침서이다.

VS Code에서 Codex가 이 문서를 읽고 바로 구현할 수 있도록 작성하였다.

---

# 2. 현재 프로젝트 목표

현재 CSV 데이터를 기반으로 전기차의 주행 조건에 따른

- 예상 전력 소비량
- 100km당 에너지 소비량
- 고속도로 장거리 주행 시 예상 배터리 사용량
- 충전 필요 시점
- 충전 휴게소 추천

등을 예측하는 머신러닝 기반 서비스로 확장하는 것이 목표이다.

현재 주요 타깃 컬럼:

```text
energy_consumption_kwhper100km
```

---

# 3. 현재 CSV 입력 피처

현재 `ev_energy_consumption.csv`에는 다음과 같은 주요 피처가 존재한다.

| 컬럼명 | 의미 |
|---|---|
| `speed_kmh` | 차량 주행 속도 |
| `payload_kg` | 적재 중량 |
| `ambient_temp_C` | 외부 기온 |
| `hvac_power_kw` | 냉난방장치 소비 전력 |
| `road_grade_pct` | 도로 경사도 |
| `battery_temp_C` | 배터리 온도 |
| `driving_style_index` | 운전 성향 지수 |
| `tire_pressure_bar` | 타이어 공기압 |
| `trip_distance_km` | 주행 거리 |
| `energy_consumption_kwhper100km` | 100km당 에너지 소비량 / 예측 타깃 |

---

# 4. 현재 문제점

현재 데이터는 입력값을 그대로 머신러닝 모델에 넣어 학습하기에는 충분하지만,
실제 전기차의 물리적 특성을 완전히 표현하지 못한다.

특히 다음 관계들은 단순 선형 관계가 아니다.

```text
외기온도 ↔ 에너지 소비량
배터리온도 ↔ 에너지 효율
속도 ↔ 공기저항
차량 중량 × 경사도
외기온도 × HVAC 사용량
속도 × 운전 성향
```

따라서 단순히 `StandardScaler` 또는 `MinMaxScaler`만 적용하는 것보다
물리적으로 의미 있는 파생변수를 먼저 생성하는 것이 중요하다.

---

# 5. 개선 방향

## 핵심 전략

다음 네 가지 실험 결과를 비교한다.

```text
1. Baseline
   기존 피처 사용

2. Scaling
   기존 피처 + StandardScaler

3. Feature Engineering
   기존 피처 + 물리 기반 파생변수

4. Feature Engineering + Scaling
   기존 피처 + 파생변수 + StandardScaler
```

각 모델의 성능을 아래 지표로 비교한다.

```text
R²
MAE
RMSE
```

---

# 6. 추가할 파생변수

## 6.1 속도 제곱

공기저항은 속도 증가에 따라 선형이 아니라 급격히 증가한다.

물리식:

```text
Fd ∝ v²
```

추가 피처:

```python
df["speed_squared"] = df["speed_kmh"] ** 2
```

목적:

```text
고속 주행에서 에너지 소비 증가 패턴 표현
```

---

## 6.2 외기온도 편차

전기차 에너지 효율은 특정 적정 온도에서 가장 높으며
너무 춥거나 너무 더우면 소비전력이 증가한다.

기준 온도:

```text
25°C
```

추가 피처:

```python
df["ambient_temp_deviation"] = (
    df["ambient_temp_C"] - 25
).abs()
```

예:

```text
25°C → 0
20°C → 5
10°C → 15
0°C  → 25
-10°C → 35
40°C → 15
```

목적:

```text
단순 기온이 아니라 EV 최적 온도에서 얼마나 벗어났는지 학습
```

---

# 6.3 배터리 온도 편차

배터리는 일정한 온도 범위에서 가장 효율적이다.

임시 기준값:

```text
30°C
```

추가 피처:

```python
df["battery_temp_deviation"] = (
    df["battery_temp_C"] - 30
).abs()
```

목적:

```text
저온 및 고온 환경에서 배터리 효율 변화 표현
```

주의:

기준값 30°C는 프로젝트 실험용 초기값이며
추후 실제 차량 또는 논문 데이터를 기준으로 변경 가능하다.

---

# 6.4 타이어 공기압 편차

타이어 공기압이 적정값에서 벗어나면 구름저항이 달라진다.

임시 기준값:

```text
2.4 bar
```

추가 피처:

```python
df["tire_pressure_deviation"] = (
    df["tire_pressure_bar"] - 2.4
).abs()
```

목적:

```text
적정 타이어 공기압에서 벗어나는 정도 표현
```

---

# 6.5 적재중량 × 도로 경사

차량 중량이 무거운 상태에서 오르막을 주행하면
더 많은 에너지가 필요하다.

기본 물리 개념:

```text
E = mgh
```

추가 피처:

```python
df["payload_grade"] = (
    df["payload_kg"] *
    df["road_grade_pct"]
)
```

목적:

```text
적재량과 경사도의 상호작용 표현
```

---

# 6.6 HVAC × 외기온도 편차

같은 HVAC 출력이라도 외기온도가 극단적인 환경에서는
배터리 에너지 소비에 미치는 영향이 더 크다.

추가 피처:

```python
df["hvac_temp_interaction"] = (
    df["hvac_power_kw"] *
    df["ambient_temp_deviation"]
)
```

목적:

```text
냉난방 전력과 외부 온도 환경의 상호작용 표현
```

---

# 6.7 속도 × 운전성향

같은 평균속도라도 급가속/급감속이 많은 운전자는
에너지 소비량이 달라질 수 있다.

추가 피처:

```python
df["speed_driving_interaction"] = (
    df["speed_kmh"] *
    df["driving_style_index"]
)
```

목적:

```text
속도와 운전성향의 복합 영향을 모델에 반영
```

---

# 7. 최종 예상 피처 구성

기존 피처:

```text
speed_kmh
payload_kg
ambient_temp_C
hvac_power_kw
road_grade_pct
battery_temp_C
driving_style_index
tire_pressure_bar
trip_distance_km
```

추가 피처:

```text
speed_squared
ambient_temp_deviation
battery_temp_deviation
tire_pressure_deviation
payload_grade
hvac_temp_interaction
speed_driving_interaction
```

예상 총 입력 피처:

```text
16개
```

타깃:

```text
energy_consumption_kwhper100km
```

---

# 8. 스케일링 전략

## StandardScaler 사용 대상

다음 모델에서는 스케일링 효과를 비교한다.

```text
LinearRegression
Ridge
Lasso
SVR
KNN
MLP / Neural Network
```

사용 예:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

중요:

다음 방식은 사용하지 않는다.

```python
X_scaled = scaler.fit_transform(X)
```

이후 `train_test_split()`을 하면 테스트 데이터 정보가 학습 과정에 포함될 수 있다.

즉 Data Leakage가 발생한다.

반드시:

```text
train_test_split
↓
X_train
X_test
↓
X_train에 fit
↓
X_test는 transform만 수행
```

순서로 처리한다.

---

# 9. 트리 모델 스케일링

다음 모델은 스케일링이 필수는 아니다.

```text
DecisionTree
RandomForest
XGBoost
LightGBM
GradientBoosting
```

트리 기반 모델은 값의 크기 자체보다
분기 기준을 이용하기 때문이다.

따라서 아래 두 버전을 별도로 비교할 필요는 없다.

```text
XGBoost + Scaling
XGBoost without Scaling
```

단, 실험 목적으로 비교하는 것은 가능하다.

---

# 10. 추천 모델

우선 다음 모델을 비교한다.

## Baseline

```text
LinearRegression
RandomForestRegressor
```

## 주요 비교 모델

```text
XGBoostRegressor
LightGBMRegressor
GradientBoostingRegressor
RandomForestRegressor
SVR
```

추천 우선순위:

```text
1. XGBoost
2. LightGBM
3. RandomForest
4. GradientBoosting
5. SVR
6. LinearRegression
```

---

# 11. 평가 지표

회귀 문제이므로 Accuracy 또는 F1 Score를 사용하지 않는다.

사용 지표:

## R²

```text
1에 가까울수록 좋음
```

## MAE

```text
실제값과 예측값 차이의 평균
낮을수록 좋음
```

## RMSE

```text
큰 예측 오류에 더 큰 패널티
낮을수록 좋음
```

예시 코드:

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(
    y_test,
    y_pred
)
```

---

# 12. 권장 실험 구조

다음 네 가지 실험을 동일한 Train/Test 데이터로 수행한다.

## Experiment A

```text
Original Features
```

## Experiment B

```text
Original Features
+
StandardScaler
```

## Experiment C

```text
Original Features
+
Engineered Features
```

## Experiment D

```text
Original Features
+
Engineered Features
+
StandardScaler
```

결과 예시:

| Experiment | R² | MAE | RMSE |
|---|---:|---:|---:|
| Baseline | | | |
| Scaling | | | |
| Feature Engineering | | | |
| Feature Engineering + Scaling | | | |

---

# 13. Codex 구현 작업 순서

Codex는 다음 순서대로 구현한다.

## STEP 1

`ev_energy_consumption.csv` 로드

```python
import pandas as pd

df = pd.read_csv("ev_energy_consumption.csv")
```

---

## STEP 2

기존 데이터 확인

```python
df.head()
df.info()
df.describe()
df.isnull().sum()
```

---

## STEP 3

파생변수 생성 함수 작성

함수명:

```python
create_ev_features()
```

예:

```python
def create_ev_features(df):

    df = df.copy()

    df["speed_squared"] = (
        df["speed_kmh"] ** 2
    )

    df["ambient_temp_deviation"] = (
        df["ambient_temp_C"] - 25
    ).abs()

    df["battery_temp_deviation"] = (
        df["battery_temp_C"] - 30
    ).abs()

    df["tire_pressure_deviation"] = (
        df["tire_pressure_bar"] - 2.4
    ).abs()

    df["payload_grade"] = (
        df["payload_kg"] *
        df["road_grade_pct"]
    )

    df["hvac_temp_interaction"] = (
        df["hvac_power_kw"] *
        df["ambient_temp_deviation"]
    )

    df["speed_driving_interaction"] = (
        df["speed_kmh"] *
        df["driving_style_index"]
    )

    return df
```

---

# 14. Train/Test 분리

```python
from sklearn.model_selection import train_test_split

TARGET = "energy_consumption_kwhper100km"

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

모든 모델 실험에서 동일한 `random_state`를 사용한다.

---

# 15. 성능 비교 함수 생성

추천 함수명:

```python
evaluate_regression_model()
```

반환값:

```text
model_name
r2
mae
rmse
```

최종적으로 pandas DataFrame으로 정리한다.

예:

```text
model_results
```

---

# 16. Feature Importance 분석

트리 모델 학습 후 다음을 확인한다.

```text
feature_importances_
```

특히 다음 파생변수가 실제로 유효한지 확인한다.

```text
speed_squared
ambient_temp_deviation
payload_grade
hvac_temp_interaction
speed_driving_interaction
```

---

# 17. SHAP 분석

가능하다면 최종 XGBoost 또는 LightGBM 모델에 SHAP을 적용한다.

확인할 내용:

```text
어떤 피처가 에너지 소비량에 가장 큰 영향을 주는가?
```

특히 다음 관계를 확인한다.

```text
speed
road_grade
payload
HVAC
ambient temperature
battery temperature
driving style
```

---

# 18. 프로젝트에서 논문 기반으로 강조할 내용

단순히 다음과 같이 설명하지 않는다.

```text
정확도를 높이기 위해 StandardScaler를 사용했다.
```

대신 다음과 같이 설명한다.

```text
최근 EV 에너지 소비 연구에서 확인되는
비선형 환경·차량 요인을 기반으로
물리적 의미를 갖는 파생변수를 생성하였다.

주요 파생변수는

- 외기온도 편차
- 배터리 온도 편차
- 속도 제곱
- 적재량 × 경사도
- HVAC × 외기온도
- 속도 × 운전성향

이며 기존 데이터와 비교하여
예측 정확도 향상 여부를 검증하였다.
```

---

# 19. 실제 서비스 확장 방향

머신러닝 모델 완성 이후 다음 기능으로 확장한다.

```text
사용자 출발지
↓
목적지
↓
예상 주행거리
↓
도로 환경
↓
차량 상태
↓
예상 kWh/100km
↓
예상 총 소비전력
↓
현재 SOC
↓
충전 필요 시점 계산
↓
경로상의 EV 충전 휴게소 추천
```

---

# 20. 추후 데이터셋에 추가하면 좋은 피처

실제 차량 데이터를 추가할 수 있다면 아래 변수를 우선 고려한다.

```text
SOC
tire_temperature_C
wind_speed
wind_direction
vehicle_weight
acceleration_mean
acceleration_std
speed_std
stop_ratio
regen_braking_ratio
traffic_density
road_surface
precipitation
```

특히 중요도가 높은 후보:

```text
SOC
acceleration_std
speed_std
tire_temperature_C
wind_speed
```

---

# 21. 파일 및 코드 네이밍 권장

프로젝트 폴더:

```text
ev_energy_prediction/
```

추천 구조:

```text
ev_energy_prediction/

├── data/
│   └── ev_energy_consumption.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_model_analysis.ipynb
│
├── src/
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── ev_energy_model.pkl
│
├── README.md
│
└── EV_ENERGY_MODEL_IMPROVEMENT_PLAN.md
```

---

# 22. Codex에게 줄 핵심 지시

Codex는 이 프로젝트에서 다음 원칙을 따른다.

1. 기존 CSV 원본 데이터는 수정하지 않는다.

2. 파생변수는 별도의 함수에서 생성한다.

3. Data Leakage가 발생하지 않도록 Train/Test Split 이후 Scaling을 적용한다.

4. 회귀 평가 지표는 `R²`, `MAE`, `RMSE`를 사용한다.

5. Baseline 모델 결과를 반드시 저장한다.

6. Feature Engineering 전후 성능을 비교한다.

7. 모델별 결과를 하나의 DataFrame으로 정리한다.

8. 트리 모델에서는 Feature Importance를 출력한다.

9. 가능하면 SHAP 분석을 추가한다.

10. 최종 모델은 추후 웹 서비스에서 사용할 수 있도록 저장한다.

---

# 23. 최종 목표

최종적으로 다음 질문에 답할 수 있어야 한다.

```text
현재 차량 상태와 도로 환경에서
이 전기차는 100km 주행 시
몇 kWh를 사용할 것으로 예상되는가?
```

그리고 이 값을 이용하여:

```text
예상 총 주행 에너지
배터리 잔량
주행 가능 거리
충전 필요 시점
추천 충전 휴게소
```

를 계산할 수 있는 서비스로 확장한다.
