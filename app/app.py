from __future__ import annotations

import hashlib
import json
import random
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'datas'
ENERGY_PATH = DATA_DIR / 'ev_energy_consumption.csv'
VEHICLE_PATH = DATA_DIR / 'vehicles_volvo_fh_electric_10.csv'
ASSIGNMENT_PATH = DATA_DIR / 'daily_vehicle_assignments.csv'
ROUTE_PATH = ROOT / 'docs' / 'route_config_busan_port_hjit.json'
SERVICE_ROUTE_PATH = DATA_DIR / 'busan_port_hjit_service_route.csv'
ROUTE_GRADE_PATH = DATA_DIR / 'busan_port_hjit_route_grade.csv'
ADAPTER_PATH = ROOT / 'docs' / 'model_adapter_config.json'
MODEL_PATH = ROOT / 'models' / 'energy_model.joblib'
CARGO_SET_WEIGHT_KG = 10.0
DEPARTURE_SOC_OPTIONS = tuple(range(50, 91, 5))
CARGO_MIN_SETS = 3
CARGO_MAX_SETS = 10
TARGET = 'energy_consumption_kwhper100km'
BASE_FEATURES = [
    'speed_kmh', 'payload_kg', 'ambient_temp_C', 'hvac_power_kw',
    'road_grade_pct', 'battery_temp_C', 'driving_style_index',
    'tire_pressure_bar', 'trip_distance_km',
]
ASSIGNMENT_COLUMNS = [
    'assignment_date', 'employee_id', 'assignment_status', 'vehicle_id',
    'vehicle_display_name', 'cargo_set_count', 'cargo_set_weight_kg',
    'cargo_weight_kg', 'vehicle_empty_weight_kg', 'total_vehicle_weight_kg',
    'battery_soc_pct',
    'payload_kg', 'project_max_payload_kg',
    'tire_pressure_bar', 'battery_usable_kwh', 'max_dc_charge_kw',
    'default_soc_pct', 'weather_type', 'ambient_temp_C', 'hvac_mode',
    'hvac_power_kw',
]


def create_ev_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result['speed_squared'] = result['speed_kmh'] ** 2
    result['ambient_temp_deviation'] = (result['ambient_temp_C'] - 25.0).abs()
    result['battery_temp_deviation'] = (result['battery_temp_C'] - 30.0).abs()
    result['tire_pressure_deviation'] = (result['tire_pressure_bar'] - 2.4).abs()
    result['payload_grade'] = result['payload_kg'] * result['road_grade_pct']
    result['hvac_temp_interaction'] = (
        result['hvac_power_kw'] * result['ambient_temp_deviation']
    )
    result['speed_driving_interaction'] = (
        result['speed_kmh'] * result['driving_style_index']
    )
    return result


@st.cache_data
def load_vehicles() -> pd.DataFrame:
    vehicles = pd.read_csv(VEHICLE_PATH, dtype={'vehicle_id': str})
    vehicles['display_name'] = vehicles['vehicle_id'].str.extract(r'(\d+)$')[0].astype(int).map(
        lambda number: f'{number}호차 (Volvo FH)'
    )
    return vehicles


@st.cache_data
def load_route() -> dict:
    return json.loads(ROUTE_PATH.read_text(encoding='utf-8'))


@st.cache_data
def load_service_route() -> pd.DataFrame:
    route = pd.read_csv(SERVICE_ROUTE_PATH)
    grade = pd.read_csv(ROUTE_GRADE_PATH)
    route = route.merge(
        grade[[
            'point_name', 'latitude', 'longitude', 'elevation_m',
            'segment_average_grade_pct', 'grade_data_quality',
        ]],
        left_on='rest_area_name_ko', right_on='point_name', how='left',
        validate='one_to_one',
    ).drop(columns='point_name')
    route['trip_distance_from_start_or_prev_km'] = pd.to_numeric(
        route['trip_distance_from_start_km'], errors='coerce'
    )
    route['max_verified_power_kw'] = pd.to_numeric(
        route['max_verified_power_kw'], errors='coerce'
    )
    return route.sort_values(['sequence', 'trip_distance_from_start_km']).reset_index(drop=True)


@st.cache_data
def load_adapter() -> dict:
    return json.loads(ADAPTER_PATH.read_text(encoding='utf-8'))


@st.cache_data
def load_route_grade() -> pd.DataFrame:
    return pd.read_csv(ROUTE_GRADE_PATH)


def build_route_grade_view(
    grade_rows: pd.DataFrame,
    segment_predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """노선 순서에 맞춘 구간별 거리·고도·경사 표시표를 생성한다."""
    ordered = grade_rows.sort_values('sequence').reset_index(drop=True).copy()
    view = ordered.loc[ordered['sequence'] > 0].copy()
    view['출발지'] = ordered['point_name'].shift(1).loc[view.index]
    view['출발 고도(m)'] = ordered['elevation_m'].shift(1).loc[view.index]
    view['고도차(m)'] = view['elevation_m'] - view['출발 고도(m)']
    view['경사각도(°)'] = np.degrees(
        np.arctan(view['segment_average_grade_pct'] / 100.0)
    )
    view['구간 유형'] = np.select(
        [
            view['segment_average_grade_pct'] > 0.01,
            view['segment_average_grade_pct'] < -0.01,
        ],
        ['오르막', '내리막'],
        default='평지',
    )
    if segment_predictions is not None:
        view = view.merge(
            segment_predictions[[
                'sequence', 'predicted_kwh_per_100km', 'predicted_energy_kwh',
            ]],
            on='sequence', how='left', validate='one_to_one',
        )
    view = view.rename(columns={
        'sequence': '순서',
        'point_name': '도착지',
        'segment_distance_km': '구간거리(km)',
        'elevation_m': '도착 고도(m)',
        'segment_average_grade_pct': '평균경사(%)',
        'predicted_kwh_per_100km': '예상 소비율(kWh/100km)',
        'predicted_energy_kwh': '구간 소비에너지(kWh)',
    })
    columns = [
        '순서', '출발지', '도착지', '구간거리(km)',
        '출발 고도(m)', '도착 고도(m)', '고도차(m)',
        '평균경사(%)', '경사각도(°)', '구간 유형',
    ]
    if segment_predictions is not None:
        columns.extend(['예상 소비율(kWh/100km)', '구간 소비에너지(kWh)'])
    numeric_columns = [
        '구간거리(km)', '출발 고도(m)', '도착 고도(m)',
        '고도차(m)', '평균경사(%)', '경사각도(°)',
    ]
    if segment_predictions is not None:
        numeric_columns.extend(['예상 소비율(kWh/100km)', '구간 소비에너지(kWh)'])
    view[numeric_columns] = view[numeric_columns].round(3)
    return view[columns]


def adapt_model_input(
    payload_kg: float,
    tire_pressure_bar: float,
    adapter: dict,
) -> tuple[float, float]:
    """실제 트럭 입력을 현재 학습 데이터의 승용 EV 범위로 변환한다."""
    service = adapter['service_domain']
    training = adapter['training_domain']
    model_payload = np.interp(
        payload_kg,
        [service['payload_kg']['min'], service['payload_kg']['max']],
        [training['payload_kg']['min'], training['payload_kg']['max']],
    )
    model_tire = np.interp(
        tire_pressure_bar,
        [service['tire_pressure_bar']['min'], service['tire_pressure_bar']['max']],
        [training['tire_pressure_bar']['min'], training['tire_pressure_bar']['max']],
    )
    return float(model_payload), float(model_tire)


def predict_route_segments(
    model: object,
    grade_rows: pd.DataFrame,
    payload_kg: float,
    tire_pressure_bar: float,
    speed_kmh: float,
    dispatch: dict,
    adapter: dict,
) -> pd.DataFrame:
    """각 노선 구간의 거리와 경사를 반영해 소비전력을 개별 예측한다."""
    model_payload, model_tire = adapt_model_input(
        payload_kg, tire_pressure_bar, adapter
    )
    segments = grade_rows.loc[grade_rows['segment_distance_km'] > 0].copy()
    segments['start_distance_km'] = (
        segments['trip_distance_from_start_km'] - segments['segment_distance_km']
    )
    model_input = pd.DataFrame({
        'speed_kmh': float(speed_kmh),
        'payload_kg': model_payload,
        'ambient_temp_C': float(dispatch['ambient_temp_C']),
        'hvac_power_kw': float(dispatch['hvac_power_kw']),
        'road_grade_pct': segments['segment_average_grade_pct'].to_numpy(),
        'battery_temp_C': 30.0,
        'driving_style_index': float(dispatch['driving_style_index']),
        'tire_pressure_bar': model_tire,
        'trip_distance_km': segments['segment_distance_km'].to_numpy(),
    })
    raw_predictions = model.predict(create_ev_features(model_input))
    calibration_factor = float(adapter['output_mapping']['output_calibration_factor'])
    segments['predicted_kwh_per_100km'] = raw_predictions * calibration_factor
    segments['predicted_energy_kwh'] = (
        segments['predicted_kwh_per_100km']
        * segments['segment_distance_km'] / 100.0
    )
    return segments.reset_index(drop=True)


def available_chargers(route_rows: pd.DataFrame) -> pd.DataFrame:
    """프로젝트 규칙에 맞는 실제 충전 후보만 반환한다."""
    usable = route_rows.loc[
        (route_rows['ev_charger_available'] == 1)
        & (route_rows['service_usable_for_charging'] == 1)
        & route_rows['max_verified_power_kw'].notna()
    ].copy()
    return usable.sort_values('trip_distance_from_start_or_prev_km').reset_index(drop=True)


def charger_status_view(route_rows: pd.DataFrame) -> pd.DataFrame:
    """모든 휴게소와 충전 후보 제외 사유를 화면에 표시한다."""
    view = route_rows.copy()

    def reason(row: pd.Series) -> str:
        if row['ev_charger_available'] != 1:
            return 'EV 충전기 없음'
        if row['service_usable_for_charging'] != 1:
            return '서비스 사용 불가'
        if pd.isna(row['max_verified_power_kw']):
            return '충전 출력 미확인'
        return '추천 가능'

    view['추천 상태'] = view.apply(reason, axis=1)
    return view


def simulate_route(
    segment_predictions: pd.DataFrame,
    battery_usable_kwh: float,
    departure_soc_pct: float,
    route_config: dict,
    route_rows: pd.DataFrame,
    departure_time: datetime,
) -> dict[str, object]:
    """충전 후보를 순서대로 검토하며 SOC와 ETA를 계산한다."""
    constants = route_config['simulation_constants']
    reserve_kwh = battery_usable_kwh * constants['minimum_reserve_soc_pct'] / 100
    max_charge_kwh = battery_usable_kwh * constants['maximum_charge_target_soc_pct'] / 100
    efficiency = constants['charging_efficiency']
    speed = constants['simulation_average_speed_kmh']
    overhead = constants['charging_stop_overhead_minutes']
    chargers = available_chargers(route_rows)
    points = [
        {'name': route_config['start']['name'], 'distance_km': 0.0, 'power_kw': None},
        *[
            {
                'name': row['rest_area_name_ko'],
                'distance_km': float(row['trip_distance_from_start_or_prev_km']),
                'power_kw': float(row['max_verified_power_kw']),
            }
            for _, row in chargers.iterrows()
        ],
        {
            'name': route_config['destination']['name'],
            'distance_km': float(route_config['total_route_distance_km']),
            'power_kw': None,
        },
    ]
    current_soc_kwh = battery_usable_kwh * departure_soc_pct / 100
    elapsed = timedelta(0)
    legs = []
    charges = []

    def energy_between(start_km: float, end_km: float) -> float:
        """경계가 겹치는 모든 노선 구간의 예측 소비에너지를 합산한다."""
        energy_kwh = 0.0
        for _, segment in segment_predictions.iterrows():
            overlap_km = max(
                0.0,
                min(end_km, float(segment['trip_distance_from_start_km']))
                - max(start_km, float(segment['start_distance_km'])),
            )
            energy_kwh += (
                overlap_km * float(segment['predicted_kwh_per_100km']) / 100.0
            )
        return energy_kwh

    for point in points[1:]:
        previous = points[points.index(point) - 1]
        distance = point['distance_km'] - previous['distance_km']
        energy = energy_between(previous['distance_km'], point['distance_km'])
        if current_soc_kwh - energy < reserve_kwh:
            if previous['power_kw'] is None:
                return {'feasible': False, 'legs': legs, 'charges': charges, 'arrival_time': None}
            # 충전 정차가 발생하면 다음 운행을 위해 100%까지 충전한다.
            target_soc_kwh = max_charge_kwh
            if target_soc_kwh - energy < reserve_kwh:
                return {'feasible': False, 'legs': legs, 'charges': charges, 'arrival_time': None}
            charge_from_grid_kwh = max(0.0, (target_soc_kwh - current_soc_kwh) / efficiency)
            charge_minutes = charge_from_grid_kwh / previous['power_kw'] * 60
            current_soc_kwh = target_soc_kwh
            elapsed += timedelta(minutes=charge_minutes + overhead)
            charges.append({
                '휴게소': previous['name'],
                '충전량(kWh)': charge_from_grid_kwh,
                '충전출력(kW)': previous['power_kw'],
                '충전시간(분)': int(round(charge_minutes)),
                '총 정차시간(분)': int(round(charge_minutes + overhead)),
            })
        arrival_soc_kwh = current_soc_kwh - energy
        elapsed += timedelta(hours=distance / speed)
        legs.append({
            '구간': f'{previous["name"]} → {point["name"]}',
            '거리(km)': distance,
            '소비에너지(kWh)': energy,
            '도착 SOC(%)': arrival_soc_kwh / battery_usable_kwh * 100,
        })
        current_soc_kwh = arrival_soc_kwh

    return {
        'feasible': True,
        'legs': legs,
        'charges': charges,
        'arrival_time': departure_time + elapsed,
        'total_minutes': elapsed.total_seconds() / 60,
    }


@st.cache_resource
def train_model() -> tuple[object, dict[str, float]]:
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        return bundle['model'], bundle['metrics']

    data = pd.read_csv(ENERGY_PATH).dropna(subset=BASE_FEATURES + [TARGET])
    engineered = create_ev_features(data)
    feature_names = [column for column in engineered.columns if column != TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        engineered[feature_names], data[TARGET], test_size=0.2, random_state=42
    )
    # 노트북 실험에서 가장 좋은 Feature Engineering + Scaling 조합을 사용한다.
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        'R²': r2_score(y_test, predictions),
        'MAE': mean_absolute_error(y_test, predictions),
        'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'metrics': metrics}, MODEL_PATH)
    return model, metrics


def validate_employee_id(employee_id: str) -> str:
    normalized = str(employee_id).strip()
    if not normalized:
        raise ValueError('사원번호를 입력해주세요.')
    if not normalized.isdigit():
        raise ValueError('사원번호는 숫자만 입력할 수 있습니다.')
    if len(normalized) != 8:
        raise ValueError('사원번호는 숫자 8자리여야 합니다.')
    return normalized


def create_dispatch(employee_id: str, dispatch_date: str) -> dict:
    seed = int.from_bytes(
        hashlib.sha256(f'{employee_id}:{dispatch_date}'.encode()).digest()[:8],
        byteorder='big',
    )
    rng = random.Random(seed)
    battery_soc_pct = float(rng.choice(DEPARTURE_SOC_OPTIONS))
    driving_style_index = round(rng.random(), 2)
    weather_type = rng.choice(['cold', 'normal', 'hot'])
    if weather_type == 'cold':
        ambient_temp, hvac_mode = round(rng.uniform(-10, 5), 1), 'HEATING'
        hvac_power = round(rng.uniform(2, 5), 2)
    elif weather_type == 'hot':
        ambient_temp, hvac_mode = round(rng.uniform(30, 40), 1), 'COOLING'
        hvac_power = round(rng.uniform(2, 5), 2)
    else:
        ambient_temp, hvac_mode, hvac_power = round(rng.uniform(15, 25), 1), 'OFF', 0.0
    return {
        'employee_id': employee_id,
        'assignment_date': dispatch_date,
        'seed': seed,
        'battery_soc_pct': battery_soc_pct,
        'driving_style_index': driving_style_index,
        'weather_type': weather_type,
        'ambient_temp_C': ambient_temp,
        'hvac_mode': hvac_mode,
        'hvac_power_kw': hvac_power,
    }


def apply_hvac_control(dispatch: dict, control: str) -> dict:
    """날씨 기본값을 유지하거나 사용자가 HVAC를 강제로 켜고 끈다."""
    controlled = dispatch.copy()
    if control == '강제 OFF':
        controlled['hvac_mode'] = 'OFF'
        controlled['hvac_power_kw'] = 0.0
    elif control == '강제 ON' and controlled['hvac_mode'] == 'OFF':
        controlled['hvac_mode'] = 'HEATING' if controlled['ambient_temp_C'] < 25 else 'COOLING'
        controlled['hvac_power_kw'] = 2.5
    return controlled


def driving_style_label(index: float) -> str:
    if index < 0.34:
        return 'Eco / Smooth Driving'
    if index < 0.67:
        return 'Normal Driving'
    return 'Aggressive Driving'


def reset_to_main_screen() -> None:
    """조회 결과와 사원번호 입력값을 함께 초기화한다."""
    for key in [
        'dispatch', 'saved_row', 'message', 'pending_overwrite',
        'employee_id_input', 'locked_employee_id', 'locked_dispatch_date',
    ]:
        st.session_state.pop(key, None)


def load_assignments(assignment_path: Path = ASSIGNMENT_PATH) -> pd.DataFrame:
    if not assignment_path.exists():
        return pd.DataFrame(columns=ASSIGNMENT_COLUMNS)
    saved = pd.read_csv(
        assignment_path,
        dtype={'assignment_date': str, 'employee_id': str, 'vehicle_id': str},
    )
    # 이전 형식의 CSV도 읽을 수 있도록 새 컬럼은 빈 값으로 보완한다.
    for column in ASSIGNMENT_COLUMNS:
        if column not in saved.columns:
            saved[column] = np.nan
    return saved[ASSIGNMENT_COLUMNS]


def save_dispatch(
    dispatch: dict,
    requested_vehicle_id: str,
    cargo_set_count: int,
    overwrite_confirmed: bool,
    assignment_path: Path = ASSIGNMENT_PATH,
) -> tuple[str, pd.Series]:
    if cargo_set_count < 0:
        raise ValueError('화물 박스 수는 0 이상이어야 합니다.')
    saved = load_assignments(assignment_path)
    same_employee = (
        (saved['assignment_date'] == dispatch['assignment_date'])
        & (saved['employee_id'] == dispatch['employee_id'])
    )
    saved = saved.loc[~same_employee].copy()
    vehicles = load_vehicles()
    requested = vehicles.loc[vehicles['vehicle_id'].eq(requested_vehicle_id)]
    if requested.empty:
        raise ValueError('선택한 차량을 찾을 수 없습니다.')
    max_payload = float(requested.iloc[0]['project_max_payload_kg'])
    if cargo_set_count * CARGO_SET_WEIGHT_KG > max_payload:
        raise ValueError(
            f'화물량은 선택 차량 적재 한도({max_payload:g}kg) 이하여야 합니다.'
        )
    used = set(saved.loc[
        (saved['assignment_date'] == dispatch['assignment_date'])
        & saved['vehicle_id'].isin(set(vehicles['vehicle_id'])), 'vehicle_id'
    ])
    available = vehicles.loc[~vehicles['vehicle_id'].isin(used)]

    if available.empty:
        st.warning('모든 차량이 사용 중입니다. 이미 배정된 차량입니다.')
        if overwrite_confirmed:
            selected = vehicles.iloc[dispatch['seed'] % len(vehicles)]
            saved = saved.loc[
                ~(
                    (saved['assignment_date'] == dispatch['assignment_date'])
                    & (saved['vehicle_id'] == selected['vehicle_id'])
                )
            ].copy()
            status, message = 'OVERWRITTEN', f'기존 배정 덮어쓰기: {selected["display_name"]}'
        else:
            selected, status = None, 'RESERVE'
            message = '차량 배정 취소: 예비차량으로 저장했습니다.'
    else:
        matches = available.loc[available['vehicle_id'].eq(requested_vehicle_id)]
        selected = matches.iloc[0] if not matches.empty else available.iloc[dispatch['seed'] % len(available)]
        status, message = 'ASSIGNED', f'차량 배정 완료: {selected["display_name"]}'

    row = {
        'assignment_date': dispatch['assignment_date'],
        'employee_id': dispatch['employee_id'],
        'assignment_status': status,
        'vehicle_id': selected['vehicle_id'] if selected is not None else 'RESERVE_VEHICLE',
        'vehicle_display_name': selected['display_name'] if selected is not None else '예비차량',
        'cargo_set_count': cargo_set_count,
        'cargo_set_weight_kg': CARGO_SET_WEIGHT_KG,
        'cargo_weight_kg': cargo_set_count * CARGO_SET_WEIGHT_KG,
        'vehicle_empty_weight_kg': float(selected['project_combination_empty_weight_kg']) if selected is not None else np.nan,
        'total_vehicle_weight_kg': (
            float(selected['project_combination_empty_weight_kg']) + cargo_set_count * CARGO_SET_WEIGHT_KG
            if selected is not None else np.nan
        ),
        'payload_kg': cargo_set_count * CARGO_SET_WEIGHT_KG,
        'project_max_payload_kg': selected['project_max_payload_kg'] if selected is not None else np.nan,
        'tire_pressure_bar': selected['tire_pressure_bar'] if selected is not None else np.nan,
        'battery_usable_kwh': selected['battery_usable_kwh'] if selected is not None else np.nan,
        'max_dc_charge_kw': selected['max_dc_charge_kw'] if selected is not None else np.nan,
        'default_soc_pct': selected['default_soc_pct'] if selected is not None else np.nan,
        'battery_soc_pct': dispatch['battery_soc_pct'],
        'weather_type': dispatch['weather_type'],
        'ambient_temp_C': dispatch['ambient_temp_C'],
        'hvac_mode': dispatch['hvac_mode'],
        'hvac_power_kw': dispatch['hvac_power_kw'],
    }
    updated = pd.concat([saved, pd.DataFrame([row])], ignore_index=True)[ASSIGNMENT_COLUMNS]
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(assignment_path, index=False, encoding='utf-8-sig')
    return message, updated.iloc[-1]


def has_available_vehicle(dispatch: dict, assignment_path: Path) -> bool:
    """현재 사원이 차량을 점유하지 않는다는 전제로 빈 차량이 있는지 확인한다."""
    saved = load_assignments(assignment_path)
    saved = saved.loc[~(
        (saved['assignment_date'] == dispatch['assignment_date'])
        & (saved['employee_id'] == dispatch['employee_id'])
    )]
    used = set(saved.loc[saved['assignment_date'] == dispatch['assignment_date'], 'vehicle_id'])
    return not load_vehicles()['vehicle_id'].isin(used).all()


st.set_page_config(page_title='EV 물류 운행 지원', page_icon='⚡', layout='wide')

# Streamlit은 입력마다 스크립트를 다시 실행하므로 세션 시작 시각을 한 번만 기록한다.
if 'session_assignment_path' not in st.session_state:
    session_stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    session_path = DATA_DIR / f'daily_vehicle_assignments_{session_stamp}.csv'
    pd.DataFrame(columns=ASSIGNMENT_COLUMNS).to_csv(
        session_path, index=False, encoding='utf-8-sig'
    )
    st.session_state['session_assignment_path'] = str(session_path)
SESSION_ASSIGNMENT_PATH = Path(st.session_state['session_assignment_path'])

st.title('일일 물류 배송 조회')

try:
    vehicles = load_vehicles()
    route = load_route()
    route_rows = load_service_route()
    route_grade_rows = load_route_grade()
    adapter = load_adapter()
    model, metrics = train_model()
except Exception as error:
    st.error(f'데이터 또는 모델을 불러오지 못했습니다: {error}')
    st.stop()

lookup_locked = 'dispatch' in st.session_state

with st.sidebar:
    st.header('사원번호 조회')
    with st.form('employee_lookup_form', clear_on_submit=False):
        if lookup_locked:
            locked_dispatch = st.session_state['dispatch']
            employee_id_input = st.text_input(
                '8자리 사원번호', value=locked_dispatch['employee_id'],
                max_chars=8, key='locked_employee_id', disabled=True,
            )
            dispatch_date = st.date_input(
                '배차일',
                value=date.fromisoformat(locked_dispatch['assignment_date']),
                key='locked_dispatch_date', disabled=True,
            )
        else:
            employee_id_input = st.text_input(
                '8자리 사원번호', value='', max_chars=8,
                key='employee_id_input',
            )
            dispatch_date = st.date_input('배차일', value=date.today())
        # form_submit_button은 입력창에서 Enter를 눌러도 폼을 제출한다.
        lookup_button = st.form_submit_button(
            '사원번호 조회', type='primary', use_container_width=True,
            disabled=lookup_locked,
        )
    if lookup_locked:
        st.button(
            '메인화면으로 돌아가기',
            use_container_width=True,
            on_click=reset_to_main_screen,
        )

if lookup_button:
    try:
        employee_id = validate_employee_id(employee_id_input)
        dispatch = create_dispatch(employee_id, dispatch_date.isoformat())
        default_vehicle = vehicles.iloc[dispatch['seed'] % len(vehicles)]
        default_cargo_sets = CARGO_MIN_SETS + dispatch['seed'] % (CARGO_MAX_SETS - CARGO_MIN_SETS + 1)
        message, saved_row = save_dispatch(
            dispatch,
            default_vehicle['vehicle_id'],
            default_cargo_sets,
            False,
            assignment_path=SESSION_ASSIGNMENT_PATH,
        )
        st.session_state['dispatch'] = dispatch
        st.session_state['saved_row'] = saved_row.to_dict()
        st.session_state['message'] = message
        st.session_state['cargo_count_input'] = default_cargo_sets
        st.session_state['speed_input'] = 100.0
        st.rerun()
    except ValueError as error:
        st.error(str(error))

if 'dispatch' in st.session_state:
    dispatch = st.session_state['dispatch']
    saved_row = pd.Series(st.session_state['saved_row'])
    saved_vehicle_id = saved_row.get('vehicle_id')
    default_index = (
        vehicles['vehicle_id'].tolist().index(saved_vehicle_id)
        if saved_vehicle_id in set(vehicles['vehicle_id']) else 0
    )

    with st.sidebar:
        st.header('운행 조건')
        vehicle_name = st.selectbox(
            '차량 변경', vehicles['display_name'].tolist(), index=default_index
        )
        cargo_set_count = st.number_input(
            f'화물 박스 수 (1박스 = {CARGO_SET_WEIGHT_KG:.0f}kg)',
            0, int(float(vehicles.iloc[default_index]['project_max_payload_kg']) // CARGO_SET_WEIGHT_KG),
            key='cargo_count_input', step=1,
        )
        speed_kmh = st.number_input(
            '평균 속도 (km/h)', 20, 130, key='speed_input', step=5, format='%d'
        )
        hvac_control = st.selectbox(
            'HVAC 제어', ['자동(날씨 기준)', '강제 ON', '강제 OFF']
        )
        update_button = st.button('조건 저장 및 다시 예측', use_container_width=True)

    if update_button:
        requested_vehicle_id = vehicles.loc[
            vehicles['display_name'].eq(vehicle_name), 'vehicle_id'
        ].iloc[0]
        controlled_dispatch = apply_hvac_control(dispatch, hvac_control)
        pending = {
            'dispatch': controlled_dispatch,
            'requested_vehicle_id': requested_vehicle_id,
            'cargo_set_count': int(cargo_set_count),
        }
        if has_available_vehicle(controlled_dispatch, SESSION_ASSIGNMENT_PATH):
            message, saved_row = save_dispatch(
            controlled_dispatch, requested_vehicle_id, int(cargo_set_count), False,
                assignment_path=SESSION_ASSIGNMENT_PATH,
            )
            st.session_state['dispatch'] = controlled_dispatch
            st.session_state['saved_row'] = saved_row.to_dict()
            st.session_state['message'] = message
            st.rerun()
        st.session_state['pending_overwrite'] = pending
        st.rerun()

if 'pending_overwrite' in st.session_state:
    pending = st.session_state['pending_overwrite']

    @st.dialog('이미 배정된 차량입니다')
    def confirm_overwrite_dialog():
        st.warning('현재 실행의 모든 차량이 사용 중입니다.')
        st.write('기존 배정값을 삭제하고 선택한 조건으로 덮어쓰시겠습니까?')
        confirm, cancel = st.columns(2)
        if confirm.button('확인: 덮어쓰기', type='primary', use_container_width=True):
            message, saved_row = save_dispatch(
                pending['dispatch'], pending['requested_vehicle_id'],
                pending['cargo_set_count'], True,
                assignment_path=SESSION_ASSIGNMENT_PATH,
            )
            st.session_state['saved_row'] = saved_row.to_dict()
            st.session_state['dispatch'] = pending['dispatch']
            st.session_state['message'] = message
            del st.session_state['pending_overwrite']
            st.rerun()
        if cancel.button('취소: 예비차량 저장', use_container_width=True):
            message, saved_row = save_dispatch(
                pending['dispatch'], pending['requested_vehicle_id'],
                pending['cargo_set_count'], False,
                assignment_path=SESSION_ASSIGNMENT_PATH,
            )
            st.session_state['saved_row'] = saved_row.to_dict()
            st.session_state['dispatch'] = pending['dispatch']
            st.session_state['message'] = message
            del st.session_state['pending_overwrite']
            st.rerun()

    confirm_overwrite_dialog()

if 'dispatch' in st.session_state:
    dispatch = st.session_state['dispatch']
    saved_row = pd.Series(st.session_state['saved_row'])
    cargo_set_count = int(saved_row.get('cargo_set_count', 0))
    selected_vehicle_id = saved_row.get('vehicle_id')
    selected_vehicle = vehicles.loc[
        vehicles['vehicle_id'].eq(selected_vehicle_id)
    ]
    vehicle_tire = 9.0 if selected_vehicle.empty else float(
        selected_vehicle.iloc[0]['tire_pressure_bar']
    )
    vehicle_empty_weight = 0.0 if selected_vehicle.empty else float(
        selected_vehicle.iloc[0]['project_combination_empty_weight_kg']
    )

    if saved_row['assignment_status'] == 'RESERVE':
        st.warning('모든 차량이 사용 중이어서 예비차량으로 저장되었습니다.')
    else:
        st.success(st.session_state['message'])

    segment_predictions = predict_route_segments(
        model=model,
        grade_rows=route_grade_rows,
        payload_kg=cargo_set_count * CARGO_SET_WEIGHT_KG,
        tire_pressure_bar=vehicle_tire,
        speed_kmh=st.session_state.get('speed_input', 100),
        dispatch=dispatch,
        adapter=adapter,
    )
    total_energy = float(segment_predictions['predicted_energy_kwh'].sum())
    prediction = total_energy / float(route['total_route_distance_km']) * 100.0
    route_average_grade = float(np.average(
        segment_predictions['segment_average_grade_pct'],
        weights=segment_predictions['segment_distance_km'],
    ))
    battery_usable = float(saved_row.get('battery_usable_kwh', 460.0))
    if pd.isna(battery_usable):
        battery_usable = 460.0
    departure_soc = float(saved_row.get('battery_soc_pct', dispatch['battery_soc_pct']))
    if pd.isna(departure_soc):
        departure_soc = dispatch['battery_soc_pct']
    simulation = simulate_route(
        segment_predictions,
        battery_usable,
        departure_soc,
        route,
        route_rows,
        datetime.combine(
            date.fromisoformat(dispatch['assignment_date']),
            time(10, 0),
        ),
    )
    date_text = datetime.strptime(dispatch['assignment_date'], '%Y-%m-%d').strftime('%y.%m.%d')
    st.subheader(f'{date_text} / 운행 정보')
    primary = st.columns(2)
    primary[0].markdown(
        f'**사원번호**<br><span style="font-size:1.15rem">'
        f'{dispatch["employee_id"]}</span>', unsafe_allow_html=True,
    )
    primary[1].markdown(
        f'**차량번호**<br><span style="font-size:1.15rem">'
        f'{saved_row["vehicle_display_name"]}</span>', unsafe_allow_html=True,
    )
    primary = st.columns(2)
    primary[0].markdown(
        f'**출발 배터리 SOC**<br><span style="font-size:1.15rem">'
        f'{dispatch["battery_soc_pct"]:.0f}%</span>', unsafe_allow_html=True,
    )
    primary[1].markdown(
        f'**화물 박스 / 총 화물량**<br><span style="font-size:1.15rem">'
        f'{cargo_set_count:,}박스 / {cargo_set_count * CARGO_SET_WEIGHT_KG:,.1f} kg</span>',
        unsafe_allow_html=True,
    )
    if simulation['charges']:
        recommendations = []
        for charge in simulation['charges']:
            matching_leg = next(
                (leg for leg in simulation['legs'] if str(leg['구간']).startswith(f"{charge['휴게소']} →")),
                None,
            )
            next_stop = matching_leg['구간'].split(' → ', 1)[1] if matching_leg else '다음 구간'
            recommendations.append(
                f'{charge["휴게소"]} ~ {next_stop}<br>사이의 휴게소를 이용하세요'
            )
        charger_recommendation = '<br>'.join(recommendations)
    else:
        charger_recommendation = '충전 불필요'
    primary = st.columns(2)
    primary[0].markdown(
        f'**충전 휴게소**<br><span style="font-size:1.05rem; line-height:1.3">'
        f'{charger_recommendation}</span>', unsafe_allow_html=True,
    )
    departure_display = f'{dispatch["assignment_date"]} 10:00'
    arrival_display = (
        simulation['arrival_time'].strftime('%Y-%m-%d %H:%M')
        if simulation['feasible'] else '도착 불가'
    )
    primary[1].markdown(
        f'**운행 시간**<br><span style="font-size:1.05rem; line-height:1.5">'
        f'출발: {departure_display}<br>도착: {arrival_display}</span>',
        unsafe_allow_html=True,
    )
    primary = st.columns(2)
    primary[0].markdown(
        f'**예상 에너지 소비량**<br><span style="font-size:1.05rem">'
        f'{total_energy:.1f} kWh<br>({prediction:.2f} kWh/100km)</span>',
        unsafe_allow_html=True,
    )
    model_explanatory_power = max(0.0, min(100.0, float(metrics['R²']) * 100))
    primary[1].markdown(
        f'**예측 신뢰 지표**<br><span style="font-size:1.05rem; line-height:1.5">'
        f'모델 설명력: {model_explanatory_power:.1f}%<br>'
        f'예상 오차: ±{float(metrics["RMSE"]):.2f} kWh/100km</span>',
        unsafe_allow_html=True,
    )

    with st.expander('배차 상세정보', expanded=False):
        detail_left, detail_right = st.columns(2)
        with detail_left:
            st.metric('총 운행중량', f'{vehicle_empty_weight + cargo_set_count * CARGO_SET_WEIGHT_KG:,.1f} kg')
            st.metric('배정 상태', saved_row['assignment_status'])
            driving_style = float(dispatch['driving_style_index'])
            st.metric('운전성향', f'{driving_style:.2f} · {driving_style_label(driving_style)}')
        with detail_right:
            st.metric('날씨', f'{dispatch["weather_type"]} / {dispatch["ambient_temp_C"]}°C')
            st.metric('HVAC', f'{dispatch["hvac_mode"]} / {dispatch["hvac_power_kw"]} kW')

    with st.expander('에너지·충전 정보', expanded=False):
        st.metric('모델 R²', f'{metrics["R²"]:.4f}')
        st.caption(f'MAE {metrics["MAE"]:.3f} kWh/100km · RMSE {metrics["RMSE"]:.3f} kWh/100km')
        st.info(adapter['warning'])
        if simulation['feasible']:
            st.success(f"총 소요시간: {simulation['total_minutes']:.0f}분")
            st.caption(
                '총 정차시간은 충전시간과 휴게소 진입·연결·출차 7분을 합한 값이며, '
                '총 소요시간은 전체 주행시간과 모든 정차시간을 합한 값입니다.'
            )
            if simulation['charges']:
                st.dataframe(pd.DataFrame(simulation['charges']), use_container_width=True, hide_index=True)
            else:
                st.info('예상 경로에서 충전이 필요하지 않습니다.')
            st.dataframe(pd.DataFrame(simulation['legs']), use_container_width=True, hide_index=True)
        else:
            st.error('현재 배차 조건으로는 최소 안전 SOC를 유지하며 목적지에 도착할 수 없습니다.')

    with st.expander('노선 정보', expanded=False):
        route_summary = st.columns(2)
        route_summary[0].metric(
            '전체 노선 거리', f'{float(route["total_route_distance_km"]):,.1f} km'
        )
        route_summary[1].metric(
            '노선 평균 경사', f'{route_average_grade:.3f}%'
        )
        st.caption(
            '경사각도는 구간 시작·끝 고도 차이로 계산한 평균값이며 '
            '구간 내부의 순간 최대경사와는 다를 수 있습니다.'
        )
        st.dataframe(
            build_route_grade_view(route_grade_rows, segment_predictions),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander('충전소 정보', expanded=False):
        charger_view = charger_status_view(route_rows)[[
            'sequence', 'rest_area_name_ko', 'trip_distance_from_start_km',
            'max_verified_power_kw', 'service_charger_status', '추천 상태',
        ]]
        st.dataframe(charger_view, use_container_width=True, hide_index=True)

record_date = st.session_state.get('dispatch', {}).get(
    'assignment_date', date.today().isoformat()
)
record_date_text = datetime.strptime(record_date, '%Y-%m-%d').strftime('%y.%m.%d')
st.subheader(f'{record_date_text} / 일일 배차기록')
st.caption(f'저장 파일: {SESSION_ASSIGNMENT_PATH.name}')
daily_records = load_assignments(SESSION_ASSIGNMENT_PATH)
st.dataframe(
    daily_records.tail(20),
    use_container_width=True,
    hide_index=True,
)
export_records = daily_records.rename(columns={
    'employee_id': '사원번호',
    'vehicle_display_name': '차량번호',
    'assignment_status': '배차 허가 여부',
})[['사원번호', '차량번호', '배차 허가 여부']].copy()
export_records['배차 허가 여부'] = export_records['배차 허가 여부'].map({
    'ASSIGNED': '허가',
    'OVERWRITTEN': '허가(덮어쓰기)',
    'RESERVE': '예비차량',
}).fillna(export_records['배차 허가 여부'])
assignment_date_for_file = st.session_state.get(
    'dispatch', {}
).get('assignment_date', date.today().isoformat())
download_csv, download_excel = st.columns(2)
with download_csv:
    st.download_button(
        'CSV로 저장',
        data=export_records.to_csv(index=False, encoding='utf-8-sig'),
        file_name=f'일일배차기록_{assignment_date_for_file}.csv',
        mime='text/csv',
        use_container_width=True,
    )
with download_excel:
    excel_buffer = __import__('io').BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        export_records.to_excel(writer, index=False, sheet_name='일일 배차기록')
    st.download_button(
        'Excel로 저장',
        data=excel_buffer.getvalue(),
        file_name=f'일일배차기록_{assignment_date_for_file}.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        use_container_width=True,
    )
