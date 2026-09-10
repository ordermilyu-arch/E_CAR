# EV 물류 운행 지원 시스템 — 일일 물류 배송 조회 서비스

Volvo FH Electric 전기 트럭을 기준으로, **사원번호와 배차일만 입력하면**
차량 배정 → 에너지 소비량 예측 → 구간별 SOC 시뮬레이션 → 충전 계획 →
예상 도착시간까지 한 번에 계산하는 Streamlit 기반 물류 운영 서비스다.

부산항 신선대 컨테이너터미널에서 한진인천컨테이너터미널(HJIT)까지의
고정 노선(465.6km, 17개 구간)을 기준으로 차량·화물·운행환경·배터리 상태와
구간별 경사도를 반영해 운행 가능 여부를 판정한다.

---

## 핵심 기능

| 단계 | 설명 |
|---|---|
| 결정적 배차 생성 | 사원번호 + 배차일을 SHA-256 seed로 변환해 동일 입력에 동일한 차량·SOC·운전성향·날씨를 생성 |
| 에너지 소비 예측 | 물리 기반 파생변수 + StandardScaler + Linear Regression 모델로 구간별 kWh/100km 예측 (R² 0.94) |
| SOC 시뮬레이션 | 17개 구간의 거리·평균경사를 개별 입력해 누적 소비에너지와 도착 SOC 계산 |
| 충전 계획 최적화 | 운영 가능한 휴게소만 후보로 두고, 필요한 경우에만 100%까지 충전해 정차 횟수 최소화 |
| ETA 계산 | 배차일 10:00 출발 기준으로 주행시간 + 충전시간 + 정차시간을 합산 |
| 일일 배차기록 저장 | 사원번호·차량번호·배차 허가 여부를 CSV / Excel로 다운로드 |

## 시스템 흐름

```text
8자리 사원번호 + 배차일
        │  SHA-256 seed
        ▼
배차 생성 (차량 / 화물 / 날씨·HVAC / 출발 SOC / 운전성향)
        │
        ▼
노선 로드 (17개 구간 거리 + Copernicus DEM 기반 구간 평균경사)
        │
        ▼
모델 어댑터 (실제 트럭 화물량·타이어 공기압 → 학습 데이터 범위로 변환)
        │
        ▼
구간별 에너지 예측  →  구간별 SOC  →  충전 계획  →  ETA
        │
        ▼
Streamlit 결과 화면 + 일일 배차기록 CSV/Excel
```

## 머신러닝 모델

- **파이프라인**: Feature Engineering + `StandardScaler` + `LinearRegression`
- **학습 데이터**: `datas/ev_energy_consumption.csv` (8,000행), 타깃 `energy_consumption_kwhper100km`
- **성능** (test 20%, `random_state=42`)

  | 지표 | 값 | 화면 표기 |
  |---|---|---|
  | R² | 0.9417 | 모델 설명력 94.2% |
  | MAE | 0.705 kWh/100km | — |
  | RMSE | 0.887 kWh/100km | 예상 오차 ±0.89 kWh/100km |

- **실험 설계**: Baseline / Scaling / Feature Engineering / Feature Engineering + Scaling
  네 조합을 동일 분할로 비교해 최종 조합을 선택
- **물리 기반 파생변수 7개**: 속도 제곱(공기저항), 외기온도 편차, 배터리 온도 편차,
  타이어 공기압 편차, 적재량 × 경사도, HVAC × 외기온도 편차, 속도 × 운전성향
- 학습된 모델은 `models/energy_model.joblib`로 저장해 서비스 실행 시 재학습 없이 재사용

## 노선 경사도 데이터 방법론

기존 노선 CSV에는 거리·충전기 정보만 있고 경사/고도 정보가 없어, 서비스에서
`road_grade_pct = 0`을 고정 입력하던 문제가 있었다. 이를 다음과 같이 보완했다.

1. [OpenStreetMap Nominatim](https://nominatim.org/release-docs/latest/api/Search/)으로
   출발지·도착지·휴게소 16곳의 좌표 조회 (일부는 `서울방향` 명칭으로 재조회해 보정)
2. [Open-Meteo Elevation API](https://open-meteo.com/en/docs/elevation-api)
   (Copernicus DEM, 약 90m 해상도)로 각 지점 고도 조회
3. `구간 평균경사(%) = (도착 고도 − 출발 고도) / 구간 거리 × 100` 으로 계산 →
   `datas/busan_port_hjit_route_grade.csv`에 저장
4. 모델에 17개 구간의 거리와 평균경사를 **개별 입력**해 구간별 소비율·소비에너지를
   예측하고 누적 (노선 전체를 하나의 평균값으로 처리하지 않음)

> **한계**: 구간 시작·끝 고도 차이 기반 평균경사이므로 구간 내부의 오르막·내리막이
> 상쇄될 수 있고, DEM 지표면 고도는 교량·터널의 실제 도로 높이와 다를 수 있다.
> 상용 정확도를 위해서는 실제 주행 GPS·고도 로그로 교체해야 한다.

## 운영 기준

| 항목 | 값 |
|---|---|
| 출발 시각 | 배차일 오전 10:00 |
| 노선 총거리 | 465.6 km |
| 최소 안전 SOC | 10% |
| 충전 목표 SOC | 100% (충전이 필요한 경우) |
| 충전 효율 | 92% |
| 충전 1회 추가 정차시간 | 7분 |
| 충전 후보 제외 | 충전 출력 미확인 / 운영 불가 휴게소 |

## 기술 스택

Python 3.12 · Streamlit · scikit-learn · pandas · NumPy · joblib · matplotlib ·
python-pptx · XlsxWriter · [uv](https://github.com/astral-sh/uv)

## 프로젝트 구조

```text
E_car/
├─ app/
│  └─ app.py                       # Streamlit 서비스 (배차·예측·SOC·충전·ETA·다운로드)
├─ 0828.ipynb                      # 전체 분석 + 서비스 구현 노트북 (EDA → 모델 비교 → 통합 서비스)
├─ model_training.ipynb            # 모델 학습 후 energy_model.joblib 저장
├─ service_simulation.ipynb        # 저장된 모델을 불러와 배차·시뮬레이션 실행
├─ datas/
│  ├─ ev_energy_consumption.csv            # ML 학습 데이터 (8,000행)
│  ├─ vehicles_volvo_fh_electric_10.csv    # Volvo FH Electric 10대 가상 차량 사양
│  ├─ busan_port_hjit_service_route.csv    # 부산→인천 노선 휴게소·충전기 정보
│  ├─ busan_port_hjit_route_grade.csv      # 구간별 고도·평균경사 (DEM 유도)
│  └─ service_simulation_result.json       # 서비스 시뮬레이션 결과 예시
├─ docs/
│  ├─ EV_LOGISTICS_SERVICE_PRESENTATION.md / .pptx / .pdf   # 서비스 발표자료
│  ├─ PROJECT_DATA_GUIDE.md                # 데이터 구성·설계 규칙 가이드
│  ├─ route_config_busan_port_hjit.json    # 노선 공통 설정값 (거리·SOC·충전 상수)
│  ├─ model_adapter_config.json            # 학습 스케일 ↔ 실제 트럭 스케일 보정 (데모용)
│  └─ presentation_assets/*.png            # 발표자료용 분석 차트
├─ scripts/
│  ├─ export_presentation_assets.py        # 분석 차트 PNG 생성
│  └─ create_presentation.py               # PPTX 생성
├─ models/                          # energy_model.joblib (학습 시 생성, git 제외)
├─ src/e_car/                       # 패키지 스켈레톤
├─ pyproject.toml / uv.lock / .python-version
└─ .streamlit/config.toml
```

> `BOX/` 폴더는 이전 버전·실행 산출물·구현 이전 작업 메모를 담은 로컬 보관함으로,
> `.gitignore`에 등록되어 포트폴리오에는 포함되지 않는다.

## 실행 방법

```bash
# 1. 의존성 설치
uv sync

# 2. 서비스 실행 (반드시 streamlit run - python app/app.py 아님)
uv run streamlit run app/app.py
```

pip를 쓴다면:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
streamlit run app/app.py
```

**VS Code에서 실행**: `Terminal → New Terminal` 에서 위 명령을 입력한다. `F5`(디버그
실행)로는 Streamlit 앱이 뜨지 않는다.

### 실행 시 주의

- **반드시 `streamlit run app/app.py`** 로 실행한다. `python app/app.py` 로 직접 실행하면
  웹 서버가 뜨지 않고 `missing ScriptRunContext! ... running in bare mode` 경고만 출력되고 끝난다.
- `.streamlit/config.toml`에 `headless = false`, 포트 `8501`을 설정해 두어 실행하면
  브라우저가 `http://localhost:8501`을 자동으로 연다. 안 열리면 그 주소를 직접 입력한다.
  종료는 `Ctrl+C`.
- `app/app.py`는 `models/energy_model.joblib`가 없으면 최초 실행 시 직접 학습해 저장한다.
  `model_training.ipynb`로도 만들 수 있다(Jupyter 필요).

## 실행 화면

`docs/EV_LOGISTICS_SERVICE_PRESENTATION.md`의 "실제 사용화면" 절과
`docs/presentation_assets/`의 분석 차트를 참고한다. (앱 스크린샷은 추후 추가)

## 참고 논문

- [Electric vehicle routing problem with machine learning for energy prediction](https://www.sciencedirect.com/science/article/pii/S0191261520304549) — 에너지 예측·SOC·충전 판단 흐름
- [Energy consumption analysis and prediction of electric vehicles based on real-world driving data](https://www.sciencedirect.com/science/article/pii/S030626192030920X) — 속도·온도·운전조건을 모델 입력으로 구성
- [Adaptive Routing and Recharging Policies for Electric Vehicles](https://pubsonline.informs.org/doi/10.1287/trsc.2016.0724) — 휴게소 필터링과 안전 SOC 기준

## 한계 및 향후 개선

- `model_adapter_config.json`의 승용 EV ↔ 대형 전기트럭 스케일 보정은 **데모 시연용**이다.
  실제 서비스 정확도를 주장하려면 대형 전기트럭 실측 데이터로 재학습해야 한다.
- 구간 평균경사는 DEM 유도값으로, 실제 도로 종단선형과 차이가 있다.
- 트리 기반 모델(XGBoost/LightGBM)과 SHAP 분석은 향후 실험 대상으로 남겨 두었다.
