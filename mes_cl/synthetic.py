"""MES 입고 이력을 흉내 낸 합성 데이터 생성기 (실제 데이터는 저장소에 포함하지 않음).

각 행은 입고(발생) 시점의 입력 피처와, 며칠~몇 주 뒤에야 확정되는 처리 결과 라벨을 가진다.
- 라벨: PROCESS(정상 공정 투입) / RETURN(반송·폐기) / REPAIR(수리 요청)
- 제품마다 기본 위험도가 다르고, 사용 시간·누적 처리 횟수·텍스트 신호가 위험도를 높인다.
- 일부 행은 라벨이 비어 있거나(미확정) 확정일이 누락되어 point-in-time 필터의 제외 경로를 탄다.
- `drift_after` 이후에는 특정 라인의 위험도가 올라가 지속 학습이 필요한 상황을 만든다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Mapping

import numpy as np

CUSTOMERS = ("customer_a", "customer_b", "customer_c")
LINES = ("line_1", "line_2", "line_3")
TYPES = ("type_a", "type_b", "type_c")
DETAILS = ("", "외관 이상 없음", "표면 얼룩", "사용 횟수 기준 초과", "볼트 누락", "누설 의심", "고객 요청 재작업")


@dataclass(frozen=True, slots=True)
class Row:
    """temporal / domain_features 가 기대하는 행 프로토콜 구현."""
    row_id: str
    source_row: int
    date: str
    month: str
    customer_key: str
    product_key: str
    features: Mapping[str, Any] = field(default_factory=dict)


def generate(n_rows: int = 6000, *, start: str = "2026-01-01", days: int = 180,
             drift_after: str = "2026-05-01", seed: int = 0, dataset_id: str = "synthetic"):
    """(raw_rows, rows, truth) 반환.

    raw_rows: labels.LabelAvailabilityIndex.from_rows 입력 (원천 테이블 형태)
    rows:     Row 목록 (row_id 는 '<dataset_id>:<source_row>' 로 labels 모듈과 동일 규칙)
    truth:    row_id -> 실제 라벨 (미확정 행 포함, 평가용)
    """
    rng = np.random.default_rng(seed)
    start_day = date.fromisoformat(start)
    drift_day = date.fromisoformat(drift_after)
    products = [f"product_{i:02d}" for i in range(24)]
    product_risk = {p: float(rng.beta(2, 8)) for p in products}
    serial_pool = [f"SN{i:05d}" for i in range(n_rows // 3)]

    raw_rows, rows, truth = [], [], {}
    for k in range(n_rows):
        source_row = k + 1
        occurred = start_day + timedelta(days=int(rng.integers(0, days)))
        customer = CUSTOMERS[int(rng.integers(0, len(CUSTOMERS)))]
        line = LINES[int(rng.integers(0, len(LINES)))]
        product = products[int(rng.integers(0, len(products)))]
        type_value = TYPES[int(rng.integers(0, len(TYPES)))]
        detail = DETAILS[int(rng.choice(len(DETAILS), p=[.35, .3, .1, .08, .06, .06, .05]))]
        intake = int(rng.integers(1, 30))
        service = max(0, intake - int(rng.integers(0, 3)))
        treatment = max(0, service - int(rng.integers(0, 4)))
        usage = float(rng.gamma(2.0, 40.0))
        cum_usage = usage + float(rng.gamma(3.0, 120.0)) * (intake / 10)

        risk = product_risk[product] + 0.004 * intake + 0.0004 * usage
        risk += {"사용 횟수 기준 초과": .35, "누설 의심": .45, "볼트 누락": .25, "표면 얼룩": .15}.get(detail, 0.0)
        if occurred >= drift_day and line == "line_3":
            risk += 0.25  # 운영 중 발생한 분포 변화
        risk = min(risk, 0.95)
        if rng.random() < risk:
            label = "REPAIR" if rng.random() < 0.55 else "RETURN"
        else:
            label = "PROCESS"

        available = occurred + timedelta(days=int(rng.integers(2, 25)))
        raw = {"__sourceRowNumber": source_row, "occurrence_date": occurred.isoformat(),
               "registered_date": occurred.isoformat(), "action_result": label,
               "action_result_date": available.isoformat()}
        roll = rng.random()
        if roll < 0.03:
            raw["action_result"] = ""            # 아직 확정 안 됨
        elif roll < 0.04:
            raw["action_result_date"] = None     # 확정일 누락
        raw_rows.append(raw)

        row_id = f"{dataset_id}:{source_row}"
        truth[row_id] = label
        rows.append(Row(
            row_id=row_id, source_row=source_row, date=occurred.isoformat(), month=occurred.strftime("%Y-%m"),
            customer_key=customer, product_key=product,
            features={"line": line, "type": type_value, "symptom": "", "detail": detail,
                      "serial": serial_pool[int(rng.integers(0, len(serial_pool)))],
                      "intake_count": intake, "service_count": service, "treatment_count": treatment,
                      "usage_hours": round(usage, 1), "cumulative_usage_hours": round(cum_usage, 1)}))
    return raw_rows, rows, truth


def parse_date(value):
    """labels 모듈에 주입하는 날짜 파서 (ISO 문자열 외 값은 None)."""
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None
