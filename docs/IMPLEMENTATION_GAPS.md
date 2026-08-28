# EV 물류 운행 지원 시스템 구현 현황

이 문서는 `PROJECT_DATA_GUIDE.md`와 현재 구현을 비교하여, 구현된 기능과 앞으로 보완할 코드의 위치를 정리한 작업 목록이다. 이번 작업에서 `app/app.py`에 노선, 모델 어댑터, SOC, 충전 계획, ETA 계산을 연결했다.

## 현재 구현된 기능

- `app/app.py`
  - Streamlit 화면
  - 8자리 사원번호 검증
  - 사원번호와 날짜 기반 결정적 배차 생성
  - 차량 CSV 로드 및 차량 선택
  - 화물 세트 수와 10kg 단위 화물중량 계산
  - 날씨와 HVAC 조건 생성
  - 일일 배정 CSV 저장
  - 차량 부족 시 덮어쓰기 팝업과 예비차량 저장
  - 기존 ML 모델 학습 및 소비전력 예측
  - 노선 설정 JSON의 총거리 표시

## 누락 기능과 작성 위치

### 1. 차량 데이터 전용 처리

- 대상 파일: `datas/vehicles_volvo_fh_electric_10.csv`
- 작성 위치: `vehicle.py` 역할의 노트북 서비스 단계 또는 추후 모듈
- 누락 내용:
  - 차량 전체 목록 조회 API
  - `vehicle_id`로 특정 차량 조회
  - 사용자가 차량을 변경할 때 화물량은 유지하는 처리
  - 차량 상태(`vehicle_status`)와 적재 한계 검증

### 2. 노선과 휴게소 처리

- 대상 파일:
  - `datas/busan_port_hjit_service_route.csv`
  - `docs/route_config_busan_port_hjit.json`
- 작성 위치: `route.py` 역할의 서비스 단계
- 누락 내용:
  - 노선 CSV 로드
  - 순서와 누적거리 기준 정렬
  - `ev_charger_available == 1` 필터
  - `service_usable_for_charging == 1` 필터
  - `max_verified_power_kw`가 비어 있거나 확인되지 않은 충전기 제외
  - `OUT_OF_SERVICE` 휴게소를 추천 후보에서 제외
  - 목적지까지의 구간 거리 계산

### 3. ML 서비스 어댑터

- 대상 파일:
  - `datas/ev_energy_consumption.csv`
  - `docs/model_adapter_config.json`
- 작성 위치: `energy_predictor.py` 역할의 서비스 단계
- 누락 내용:
  - 학습 모델 파일 저장 및 재사용
  - 실제 트럭 화물량을 학습 데이터 범위로 변환
  - 실제 트럭 타이어 공기압을 학습 데이터 범위로 변환
  - `output_calibration_factor`를 사용한 서비스 표시값 보정
  - 보정값이 데모용이라는 경고 표시
  - 모델 입력 피처 순서와 누락값 검증

### 4. SOC 시뮬레이션

- 대상 설정: `docs/route_config_busan_port_hjit.json`
- 작성 위치: `soc_simulator.py` 역할의 서비스 단계
- 누락 내용:
  - 배터리 usable capacity와 현재 SOC 계산
  - 구간별 소비에너지 계산
  - 휴게소 도착 예상 SOC 계산
  - `minimum_reserve_soc_pct = 10` 기준 안전 여부 판단
  - 목적지 도착 가능 여부 판단

### 5. 충전 계획 최적화

- 대상 설정: 노선 CSV와 route config의 충전 설정
- 작성 위치: `charging_optimizer.py` 역할의 서비스 단계
- 누락 내용:
  - 도달 가능한 충전소 후보 생성
  - 고장 및 출력 미확인 충전소 제외
  - 충전 출력별 예상 충전시간 계산
  - 충전 효율 92% 적용
  - 최대 충전 목표 SOC 100% 적용
  - 충전 1회 추가 정차시간 7분 적용
  - 충전 횟수만이 아니라 총 예상 도착시간 기준 후보 비교

### 6. 예상 도착시간

- 대상 설정: `simulation_average_speed_kmh = 80`
- 작성 위치: `eta.py` 역할의 서비스 단계
- 누락 내용:
  - 구간별 주행시간 계산
  - 충전시간 합산
  - 충전소 진입·연결·출차 부가시간 합산
  - 출발 시각을 기준으로 예상 도착 시각 반환

### 7. 전체 시뮬레이션 연결

- 작성 위치: `simulation.py` 역할의 서비스 단계
- 누락 내용:
  - 배차 결과와 차량 변경값 결합
  - ML 소비전력 예측 호출
  - 노선 구간 생성
  - SOC 시뮬레이션 호출
  - 최적 충전 계획 호출
  - ETA 계산 호출
  - 화면에서 사용할 하나의 결과 딕셔너리 반환

### 8. Streamlit 화면 확장

- 대상 파일: `app/app.py`
- 누락 내용:
  - 휴게소별 충전 가능 여부와 제외 사유 표시
  - 구간별 SOC 표와 충전 계획 표시
  - 예상 출발·도착 시각 표시
  - 현재 차량, 화물, SOC를 변경했을 때 전체 결과 갱신
  - 모델 보정 경고와 예측 신뢰 범위 표시
  - 배차 결과와 시뮬레이션 결과 다운로드

## 이번 작업에서 구현한 서비스 계산

- `load_service_route()`로 노선 휴게소 CSV 로드 및 누적거리 정렬
- 충전 가능 여부, 서비스 사용 가능 여부, 확인된 충전 출력 기준 후보 필터링
- `model_adapter_config.json` 기반 화물량·타이어 공기압 입력 변환
- 어댑터의 출력 보정계수를 적용한 서비스용 kWh/100km 계산
- usable battery와 출발 SOC를 이용한 구간별 SOC 계산
- 최소 안전 SOC 10% 미만 도착 여부 판단
- 충전 효율, 충전기 출력, 충전 정차시간을 반영한 충전 계획 계산
- 평균 주행속도와 충전·정차 시간을 합산한 예상 도착시각 계산
- Streamlit 화면에 충전 후보, SOC 구간, 충전 계획, ETA 표시

현재 충전 계획은 도착 가능 조건을 유지하면서 필요한 경우에만 충전하는 운영 방식이다.

## 권장 구현 순서

```text
1. vehicle 처리
2. route 처리
3. energy predictor 어댑터
4. SOC simulator
5. charging optimizer
6. ETA 계산
7. 전체 simulation 연결
8. Streamlit 화면 확장
```

## 실행 방법

프로젝트 루트에서 다음 명령을 실행한다.

```bash
streamlit run app/app.py
```

`.streamlit/config.toml`에서 `headless = false`와 포트 `8501`을 설정했기 때문에, `--server.headless true` 옵션을 붙이지 않고 실행하면 브라우저가 `http://localhost:8501`을 자동으로 열도록 동작한다.
