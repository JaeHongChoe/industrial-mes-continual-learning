# MES 입고 처리 판정: V5 학습 파이프라인 + 지속 학습 안전장치

부품 입고 이력(MES)으로 처리 판정(PROCESS / RETURN / REPAIR)을 자동화한 **V5 학습 파이프라인 전체**(`v5_pipeline/`)와, 이를 운영 중에 안전하게 다시 학습하기 위한 **미래 정보 누수 방지 피처·재학습 안전 게이트**(`mes_cl/`)를 담은 저장소입니다.

## 한눈에 보기

- **문제**: 처리 결과(라벨)는 입고 후 며칠~몇 주 뒤에 확정됩니다. 이를 무시하고 학습하면 오프라인 점수만 좋고 운영에서는 무너지며, 평균 개선만 보고 모델을 바꾸면 일부 고객사에서 위험한 오판정이 늘 수 있습니다.
- **접근**: (1) 라벨 *확정일* 기준으로 "그 시점에 알 수 있던 정보"만 쓰는 피처·라벨 파이프라인, (2) 같은 평가 행에서 기준 모델과 후보 모델을 고객사 단위로 비교하는 안전 게이트.
- **결과물**: `v5_pipeline/` (입력 계약 → 룰 기준선 → 회사·라인 계층형 ML → Green → 위험 앙상블·guard → 동결·봉인·감사, 75개 모듈) + `mes_cl/` 패키지(라벨 인덱스, 시계열·도메인 피처, 게이트), 합성 데이터 생성기, end-to-end `demo.py`, pytest 테스트.
- **기술 스택**: Python 3.10+, numpy, scipy · V5 파이프라인: scikit-learn(TF-IDF, 선형 모델), LightGBM, openpyxl · 테스트: pytest
- **포함 / 미포함**: 학습·선택·안전장치·봉인 코드와 합성 데이터만 포함. 실제 데이터, 학습된 모델과 동결 산출물, 고객사별 룰 마이닝 엔진, 대시보드·API·DB·배포 설정은 포함하지 않습니다.

## 왜 필요한가

**1) 미래 정보 누수.** 어떤 부품이 1일에 입고됐고, "수리(REPAIR)"라는 처리 결과는 10일에 확정됐다고 합시다.
5일 시점에 판정하는 모델은 이 결과를 알 수 없습니다. 그런데 과거 데이터를 한꺼번에 놓고 학습하면 10일에야 알게 된 정보가
5일 시점의 피처·라벨에 섞이기 쉽고, 그러면 오프라인 평가는 실제보다 좋아 보입니다.

**2) 평균의 함정.** 재학습한 후보 모델이 전체 정확도는 올랐는데, 특정 고객사에서는 "수리해야 할 부품을 공정에 투입"하는
위험한 오판정이 늘었다면 교체해야 할까요? 이 저장소의 게이트는 이런 경우 **교체를 차단**합니다.

## 동작 방식

### (a) Point-in-time 피처: "그때 알 수 있던 것만"

```text
날짜     1일 ──────── 5일 ──────── 10일 ──────── 15일
부품 A   입고                       결과 확정(REPAIR)
                      ▲                           ▲
               cutoff=5일 모델:            cutoff=15일 모델:
               A의 라벨 사용 불가          A의 라벨 사용 가능
```

- **확정일 기준**: 행마다 (발생일, 라벨 확정일, 라벨)을 기록하고, `확정일 < cutoff`인 라벨만 학습에 씁니다 (`LabelAvailabilityIndex`).
- **같은 날은 score → label update**: 날짜순으로 이벤트를 재생하며 같은 날에는 "대상 행 피처 계산 → 발생 반영 → 라벨 반영" 순서로 처리해, 당일 확정된 라벨이 같은 날 행의 피처로 새지 않습니다.
- **Frozen history**: 추론용 피처는 cutoff 이전 이력으로 고정하고(`fit_history` + `transform_frozen`, `label_cutoff_exclusive`), cutoff 이전 날짜의 행이 들어오면 거부합니다. 학습 행은 `fit_transform_oof`로 자기 이후 라벨을 보지 않게 만듭니다.
- **Receipt**: cutoff마다 사용·제외 행 수와 제외 사유, 행 목록의 SHA-256 해시를 남겨 "어떤 데이터로 학습했는지" 사후 검증할 수 있습니다.

### (b) 안전 게이트: 같은 행, 고객사 단위, 사람 승인

```text
평가 행(확정 라벨 + 기준 모델 판정) + 후보 모델 예측
  → id 1:1 일치 확인 (불일치 시 ValueError)
  → 전체 / 고객사별 지표 비교
  → 고객사가 표본 부족?  ─ 예 → 판정이 하나라도 바뀌면 차단
                        └ 아니오 → 위험 방향 증가·정확도 하락 등 하나라도 있으면 차단
  → 통과해도 production_eligible = False (사람 승인 대기)
```

- **위험 방향**: 실제와 다르게 PROCESS로 보내거나(NG→공정 투입), PROCESS여야 할 것을 RETURN/REPAIR로 판정(공정 투입→NG)한 경우.
- **고객사 단위 차단**: 전체가 좋아져도 어느 고객사에서든 위험 방향 오판정 증가, 신규 위험 케이스, 잘못된 자동 판정 증가, 정확도 하락, 자동 처리율 2%p 초과 감소 중 하나라도 있으면 차단합니다.
- **표본 부족 = 판단 보류**: 행 수나 라벨별 건수가 기준 미만인 그룹은 "안전"으로 보지 않습니다.
- **승격은 사람 몫**: 게이트는 같은 cohort 검증일 뿐이며, 운영 적용에는 전향적 근거와 사람 승인이 필요합니다.

## V5 학습 파이프라인 (`v5_pipeline/`)

과거 처리 결과의 분포, 업무 규칙, 텍스트·구조 ML을 함께 쓰는 **하이브리드 판정 엔진**입니다.
과거 기간에서 만든 분포·모델·정책을 하나의 "세대(generation)"로 **동결**하고, 신규 행은 그 동결본으로만 예측한 뒤 결과를 봉인합니다.
실제 결과(라벨)는 예측이 끝난 뒤 별도 파일로만 감사(audit)에 쓰입니다.

```text
입력 계약 ─▶ 룰 기준선 ─▶ 회사·라인 계층형 ML ─▶ Green ─▶ 위험 앙상블 + 수리 guard ─▶ 무반례 회수 ─▶ 봉인 ─▶ 감사
 (A)          (Loop93,     (C·D)                   (E)       (E)                          (E)           (F)      (F)
              미포함)
```

| 단계 | 하는 일 | 주요 모듈 |
|---|---|---|
| **A. 입력 계약** | 입고 시점에 알 수 있는 필드만 허용. 사후 필드(처리 결과, 대기 여부, 고객 의견)가 채워진 입력은 지우지 않고 **입력 자체를 거부**(fail-closed). 기준일은 발생일 → 등록일 순 | `src/data_contract.py`, `src/shadow_validation/input_contract.py`, `src/runtime_contract.py` |
| **B. 피처** | 필드별 단어(1~2gram)·문자(3~5gram) TF-IDF, 의미 태그, 횟수 차이·비율, 결측, 과거 분포(학습은 월별 expanding OOF, 추론은 동결 이력) | `src/features.py`, `src/temporal_features.py`, `src/v4_domain_features.py`, `experiment_lib/` |
| **룰 기준선 (Loop93)** | 과거 분포 룰만으로 만든 깨끗한 기준 판정. **고객사별 마이닝 룰이라 공개본에서 제외**하고 연결부(`loop93.py`)만 남김 | `src/shadow_validation/loop93.py` |
| **C. 회사·라인 계층형 ML** | 전사 → 회사 → 회사×라인 순서로 적용할 후보를 선택. 전사 모델(공투 이진 2종, 다중분류 LightGBM, temporal signature)과 주요 고객사 전용 모델(선형 + LightGBM)을 함께 운용 | `src/hierarchy.py`, `src/hierarchical_*.py`, `src/company_*.py`, `src/robust_company_allocator.py` |
| **D. 후보 학습·선택** | 후보 학습(선형 OVR, LightGBM OVR, temporal signature), 임계값 적합, 선택 기간에서 정책 선택 후 lockbox로 검증, 위험 예산·정확도 하한 조건 | `src/candidates.py`, `src/binary_candidates.py`, `src/thresholds.py`, `src/select.py`, `src/power_selector.py`, `src/rolling_validation.py`, `src/risk_budget_optimizer.py` |
| **E. Green · 안전장치** | V4 아이디어를 재구성한 Green 판정(action별 독립 head) → 공투로 승인된 행만 **3-seed LightGBM 위험 앙상블**로 재검사 → 좁고 강한 수리 신호 **guard** → 과거 반례가 0건인 좁은 패턴만 공투로 **회수**(guard가 내린 행은 보호) | `src/v4_*.py`, `experiment_lib/point_in_time_danger_veto.py`(위험 head 학습·보정), `src/rank_based_danger_veto.py`, `src/prior_supported_repair_guard.py`, `src/prior_supported_process_recovery.py`, `src/hierarchical_veto.py` |
| **F. 동결·봉인·감사** | 정책·모델·이력을 SHA-256으로 묶어 세대로 동결, 예측 결과와 입력을 receipt로 봉인, 확정 라벨이 생긴 뒤 별도 감사 리포트(회사·월·오검 방향별) | `src/freeze.py`, `src/shadow_validation/{receipt,integrity,predictor,audit,report}.py` |
| **G. 실행 스크립트** | 후보 frontier 탐색, 계층 lockbox 감사, 정책 materialize, 세대 동결, shadow 예측·감사 | `scripts/*.py` |

**설계 원칙**

- **판정을 올리는 단계와 내리는 단계를 분리**: 안전장치(위험 앙상블, guard, semantic veto)는 PROCESS → REVIEW로 *내리기만* 하고, 회수 단계만 좁은 조건에서 PROCESS로 올립니다.
- **선택과 평가의 시간 분리**: 후보·임계값은 선택 기간에서 정하고 동결하며, 이후 lockbox·shadow 기간에서는 다시 탐색하지 않습니다.
- **재현성**: 세대의 모든 입력·모델·정책 파일을 해시로 고정하고, 해시가 다르면 실행을 거부합니다.

<details>
<summary>공개본에서 V5 코드에 적용한 익명화</summary>

- MES 컬럼명과 판정 라벨을 일반 이름으로 변경 (예: 처리 결과 → `action_result`, 횟수 필드 → `intake_count` / `service_count` / `treatment_count`, 사용 시간 → `usage_hours`, 라벨 → `PROCESS` / `RETURN` / `REPAIR` / `REVIEW`).
- 고객사 표기 정규화 규칙은 코드에서 삭제하고 빈 설정값(`CUSTOMER_ALIASES`)으로 교체. 고객사 키는 `customer_a` ~ `customer_e`.
- 의미 태그·guard·veto의 현장 키워드 정규식은 같은 group id를 유지한 채 예시 어휘로 교체 (원본과 매칭 결과가 다릅니다).
- 로컬 경로는 `artifacts/...` placeholder로 변경. 튜닝된 임계값·가중치 상수는 원본 그대로입니다.
- 제외: Loop93 룰 마이닝 엔진과 실행기, `frozen/` 산출물 전체(모델, 고객사별 이력, 행 단위 정책), 운영 결과와의 parity 검사.

</details>

## 데모 결과

합성 데이터 6,000행(2026-04-01 이후 `line_3`에 위험도 상승 주입)으로 재학습 한 사이클을 돌린 실제 출력입니다 (`python3 demo.py`, 기본값 `--rows 6000 --seed 0`, 일부 생략).

```text
label receipt @ 2026-05-15 {"sourceRows": 6000, "actionRows": 5826, "eligibleRows": 3871, "excludedRows": {..., "invalidOrMissingLabelAvailableDate": 54, "nonActionLabel": 174, "futureOccurrenceDate": 1476, "notAvailableBeforeCutoff": 425}}
train rows: baseline=1520 (cutoff 2026-03-01), candidate=3871 (cutoff 2026-05-15)
eval rows: 1476
  baseline  auto_rate=0.930 accuracy=0.489 ng_to_process=265 process_to_ng=270
  candidate auto_rate=0.618 accuracy=0.640 ng_to_process=251 process_to_ng=33
  customer_a: Δaccuracy=+0.124 Δng_to_process=-5 supported=True
  customer_b: Δaccuracy=+0.178 Δng_to_process=-12 supported=True
  customer_c: Δaccuracy=+0.147 Δng_to_process=+3 supported=True
gate passed: False | production_eligible: False
  - overall:increased_process_to_return
  - overall:new_dangerous_cases
  - overall:excessive_auto_loss
  - customer:customer_c:increased_ng_to_process
  - ... (총 15개 사유)
```

- cutoff 직전에 발생했지만 아직 확정되지 않은 425행은 학습 라벨에서 자동으로 빠졌습니다.
- 후보 모델은 자동 판정 정확도(0.489 → 0.640)와 공정 투입→NG 오판정(270 → 33)이 크게 좋아졌지만, 자동 처리율이 31%p 떨어졌고 `customer_c`에서는 NG→공정 투입이 3건 늘었습니다. 그래서 **게이트가 차단**합니다.
- 평균 지표만 보면 교체하고 싶어지는 상황에서 멈추게 하는 것이 목적입니다. **합성 데이터 수치이며 실제 성능 지표가 아닙니다.**

## 빠른 시작

```bash
pip install -r requirements.txt pytest
python3 demo.py                                # 약 2초
python3 -m pytest -q -p no:cacheprovider tests/test_*.py   # mes_cl: 12 passed

# V5 파이프라인 smoke 테스트 (scikit-learn, LightGBM 필요)
pip install -r requirements-v5.txt
python3 -m pytest -q -p no:cacheprovider       # 전체: 65 passed
```

`v5_pipeline` 모듈을 직접 import 할 때는 `v5_pipeline/`을 `sys.path` 앞에, `v5_pipeline/src`는 **맨 뒤에** 붙여야 합니다
(`src/select.py`가 표준 라이브러리 `select`를 가리지 않도록). `tests/v5/conftest.py`가 이 설정을 보여줍니다.

```python
from mes_cl.labels import LabelAvailabilityIndex
from mes_cl.synthetic import parse_date
from mes_cl.gate import compare_predictions

# 1) cutoff 시점에 이미 확정된 라벨만 꺼내기
raw_rows = [
    {"__sourceRowNumber": 1, "occurrence_date": "2026-01-01",
     "action_result": "REPAIR", "action_result_date": "2026-01-10"},
    {"__sourceRowNumber": 2, "occurrence_date": "2026-01-05",
     "action_result": "RETURN", "action_result_date": "2026-02-01"},
]
index = LabelAvailabilityIndex.from_rows(
    raw_rows, source_dataset_id="demo", parse_date_value=parse_date)
dict(index.eligible_labels("2026-01-20"))   # {'demo:1': 'REPAIR'}  (2번은 아직 미확정)

# 2) 같은 평가 행에서 기준 vs 후보 비교
labels = ("PROCESS", "RETURN", "REPAIR")
rows = [dict(id=f"r{i}", date="2026-05-20", customer="customer_a", line="line_1",
             product_key="p1", actual=labels[i % 3],
             baseline="REVIEW" if i < 3 else labels[i % 3]) for i in range(90)]
preds = [dict(id=r["id"], predicted=r["actual"]) for r in rows]
gate = compare_predictions(rows, preds)["gate"]
gate["passed"], gate["production_eligible"]   # (True, False)
```

## 폴더 구조

```text
industrial-mes-continual-learning/
├── v5_pipeline/             # V5 학습 파이프라인 (익명화 원본 코드, 75개 모듈)
│   ├── src/                 # A~F 단계 모듈 (shadow_validation/ = 동결 추론·봉인·감사)
│   ├── scripts/             # G 단계 실행 스크립트
│   └── experiment_lib/      # point-in-time 라벨·피처 원형 (mes_cl 과 같은 계열)
├── mes_cl/
│   ├── labels.py            # 라벨 가용성 인덱스, cutoff별 receipt
│   ├── temporal.py          # 범위·기간별 라벨 이력, serial 재입고 피처
│   ├── domain_features.py   # 수치·텍스트·smoothed 이력 피처 (OOF / frozen)
│   ├── gate.py              # 기준 vs 후보 같은-cohort 안전 게이트
│   └── synthetic.py         # 합성 MES 이력 생성기
├── demo.py                  # end-to-end 재학습 사이클 데모
├── requirements-v5.txt     # v5_pipeline 의존성 (scikit-learn, LightGBM, openpyxl)
└── tests/                   # mes_cl 테스트 12개 + v5/ 단계별 smoke 테스트 53개
```

## 기술 상세

<details>
<summary>피처, 게이트 임계값, receipt 필드</summary>

**피처** (데모에서는 두 행렬을 이어 붙여 사용)

- `temporal.py` (75개): 범위 4종(제품 / 고객사×제품 / 제품×상세내용 / 라인×제품)마다 전체·30/90/180일 라벨 건수·비율, 최근-전체 위험률 차이(30·90일) + 동일 serial 재입고 횟수·마지막 입고 후 경과일·평균 간격.
- `domain_features.py` (43개): 수치(입고·서비스·처리 횟수, 사용 시간과 차이·비율, 결측 플래그), 텍스트(상세내용 길이·항목 수, 의미 패턴 9종), cold-start 플래그, 4개 수준의 라벨 이력 비율(전역 분포로 smoothing, 기본 `smoothing=10.0`).

**게이트 임계값** (`mes_cl/gate.py`)

| 상수 | 값 | 의미 |
|---|---|---|
| `MIN_SUPPORT` | 30 | 그룹 판정에 필요한 최소 행 수 |
| `MIN_CLASS_SUPPORT` | 5 | PROCESS / RETURN / REPAIR 각각의 최소 행 수 |
| `MAX_AUTO_LOSS` | 0.02 | 허용되는 자동 처리율 감소폭 |

- 차단 사유 키: `increased_<방향>`(6개 방향), `new_dangerous_cases`, `increased_incorrect_auto`, `accuracy_regression`, `excessive_auto_loss`, `insufficient_support_changed`, `overall:insufficient_support`, `no_meaningful_improvement`.
- `REVIEW`(사람 검토로 보냄)는 자동 판정이 아니므로 위험 방향으로 세지 않습니다. 라인 단위 지표는 리포트에만 포함되고 차단 기준은 고객사 단위입니다.
- 반환값 `gate`에는 `production_blockers=["prospective_evidence_not_established", "human_approval_required"]`, `scope="same_cohort_validation_only"`가 함께 들어갑니다.

**Receipt 필드** (`LabelAvailabilityIndex.receipt_payload(cutoff)`)

- 메타: `schemaVersion`, `kind`, `sourceDatasetId`, `cutoffExclusive`
- 집계: `sourceRows`, `actionRows`, `eligibleRows`, `excludedRows`
- 제외 사유: `nonActionLabel`, `invalidOrMissingOccurrenceDate`, `invalidOrMissingLabelAvailableDate`, `actionBeforeOccurrence`, `futureOccurrenceDate`, `notAvailableBeforeCutoff`
- 해시: `eligibleRowIdsSha256`, `eligibleLabelsSha256`, `eligibleEvidenceSha256`, `recordMapSha256`, `excludedRowIdsSha256`, `contractSha256`

</details>

## 공개본에서 바꾼 점

- 원본은 판정 분석·모니터링 대시보드, PostgreSQL 이력 저장소, 학습 워커, 승인·롤백 관리를 포함한 내부 플랫폼이며, 여기에는 재현 가능한 ML 핵심만 떼어냈습니다.
- 필드명·판정 라벨을 일반화했습니다 (`action_result`, `intake_count`, `usage_hours`, `PROCESS / RETURN / REPAIR` 등).
- 텍스트 의미 패턴은 현장 용어 대신 예시 어휘로 바꿨습니다 (키 이름이 feature 이름이므로 현장 적용 시 어휘만 교체).
- V5 파이프라인 코드는 위 "익명화" 항목대로 바꿔 넣었고, 동결 룰 엔진(Loop93), 학습된 모델, 고객사별 이력·정책 산출물, 운영 이력은 포함하지 않습니다.

## 한계

- `v5_pipeline/`은 실제 데이터·설정 파일·동결 산출물이 있어야 전체가 돌아갑니다. 공개본에서는 합성 데이터로 A~F 단계별 대표 함수만 smoke 테스트합니다 (`tests/v5/`, 53개). 원본 성능 재현은 불가능합니다.
- 룰 기준선(Loop93)이 빠져 있어 `scripts/predict_shadow.py` 같은 동결 추론 경로는 그대로 실행되지 않습니다.
- 데모 수치는 모두 합성 데이터에서 나온 값이며, 실제 운영 성능을 나타내지 않습니다.
- 데모 모델은 의존성을 줄이기 위한 numpy 다항 로지스틱 회귀(최대 확률 0.55 미만은 REVIEW)이고, 데모의 기준 모델도 동결 룰이 아니라 오래된 cutoff로 학습한 같은 모델입니다.
- 게이트는 같은 cohort에서의 오프라인 비교만 합니다. 전향적(prospective) 검증, 승인 절차, 롤백은 이 저장소 범위 밖입니다.
- 임계값(`MIN_SUPPORT` 등)은 상수로 고정돼 있어, 실제 적용 시 데이터 규모에 맞춘 조정이 필요합니다.
