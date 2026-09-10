# BOX — 로컬 보관함

이 폴더는 **GitHub 포트폴리오에 올리지 않는 자료**를 모아둔 곳이다.
`.gitignore`에 `BOX/`가 등록되어 있어 커밋 대상에서 제외된다.

프로젝트 결과물(코드·데이터·문서)은 상위 폴더에 정리되어 있고, 여기에는
이전 버전, 실행 중 자동 생성된 파일, 구현 이전 단계의 작업 메모만 남긴다.

## 구성

### `old_notebooks/`

- `backup.ipynb` — `0828.ipynb`의 이전 버전. 셀 23개로, 최종본(`0828.ipynb`, 셀 26개)에
  있는 "6. 노선·에너지·SOC·충전·ETA 통합 서비스" 장이 빠져 있다. 최종 분석/서비스
  노트북은 상위 폴더의 `0828.ipynb`를 사용한다.

### `runtime_assignment_dumps/`

- `daily_vehicle_assignments*.csv` — Streamlit 앱(`app/app.py`)이 실행 세션마다
  `datas/`에 생성하는 일일 배차 기록 파일. 소스 데이터가 아니라 앱 실행 산출물이라
  포트폴리오에서 제외한다. 앞으로 생성되는 같은 이름의 파일도 `.gitignore` 규칙
  (`datas/daily_vehicle_assignments*.csv`)으로 자동 제외된다.

### `working_docs/`

구현을 시작하기 전이나 진행 중에 작성한 내부 작업 지침서. 실제 구현 결과와 설명은
상위 `docs/EV_LOGISTICS_SERVICE_PRESENTATION.md`와 `0828.ipynb`에 반영되어 있다.

- `EV_ENERGY_MODEL_IMPROVEMENT_PLAN.md` — 에너지 모델 정확도 개선을 위한 사전 계획서
  (파생변수 설계, 실험 구조, 결과표는 비어 있는 상태).
- `IMPLEMENTATION_GAPS.md` — 데이터 가이드와 초기 구현을 비교한 미구현 항목 점검표.
