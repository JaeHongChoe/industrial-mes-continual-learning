# MES Continual Learning: Point-in-Time Features & Candidate Safety Gate

반도체 부품 입고 검수(MES) 데이터에서 **입고 부적합 처리 판정(공정 투입 / 반송 / 수리)** 을 자동화하는 모델을
운영 중에 안전하게 재학습하기 위해 만든 핵심 로직 정리본입니다.

실제 시스템은 판정 분석·모니터링 대시보드, PostgreSQL 이력 저장소, 학습 워커, 승인·롤백 관리까지 포함한 내부 플랫폼이지만,
이 저장소에는 그중 **재현 가능한 ML 핵심 두 가지**만 공개용으로 떼어냈습니다.

1. **Point-in-time 피처 / 라벨** — 학습·평가에 "그 시점에 알 수 있었던 정보"만 쓰도록 강제
2. **후보 vs 기준 안전 게이트** — 재학습 후보가 평균은 좋아도 특정 고객사에서 위험하게 나빠지면 차단

## Project Snapshot

- Domain: 제조 MES 입고 이력 기반 처리 판정 (tabular + 짧은 자유 텍스트)
- Problem: 처리 결과 라벨은 입고 후 며칠~몇 주 뒤에 확정되고 사후 정정도 생김. 이를 무시하고 학습하면 오프라인 성능은 좋아 보이지만 운영에서는 미래 정보 누수로 무너짐. 또 전체 평균 개선만 보고 모델을 바꾸면 소수 고객사·라인에서 위험 방향 오판정이 늘 수 있음
- Approach: 라벨 확정일 기준 가용성 인덱스 + 날짜 이벤트 재생(같은 날은 *score → label update*) 피처, 같은 평가 행에서 고객사 단위 비교·표본 부족 보류·사람 승인 고정
- Scope: 피처·게이트 로직, 합성 데이터 생성기, end-to-end 데모, 단위 테스트. 대시보드·API·DB·배포 설정·실제 데이터·동결 룰/모델은 포함하지 않음

## 전체 운영 흐름 (실제 시스템)

```text
MES 입고 이력 + 동결(기준) 모델 판정
  → 나중에 확정되는 처리 결과를 별도 보관 (확정일 기록)
  → 모델 / 고객사 / 라인 / 제품별 성능·입력 품질 모니터링
  → 학습 자료 준비: 학습 / 기준 선택 / 평가 기간 분리, 시점 고정       ← mes_cl.labels / temporal / domain_features
  → 보조 모델 재학습 → 후보 저장
  → 같은 평가 행에서 기준 vs 후보 비교, 그룹 악화·표본 부족 확인       ← mes_cl.gate
  → 사람 검토 → 제한된 적용 / 이전 기준으로 롤백
```

## Point-in-time 설계

| 장치 | 내용 | 막는 누수 |
|---|---|---|
| 라벨 가용성 인덱스 (`labels.py`) | 행마다 (발생일, 라벨 확정일, 라벨). cutoff 이전에 *확정된* 라벨만 학습 라벨로 사용 | 평가 기간 직전에 확정된 라벨이 학습에 섞이는 것 |
| 제외 사유 집계 + SHA-256 receipt | 미확정, 확정일 누락, 확정일<발생일, cutoff 이후 발생, cutoff 이전 발생이지만 미확정 | "어떤 행으로 학습했는지" 사후 검증 불가 |
| 날짜 이벤트 재생 (`temporal.py`, `domain_features.py`) | 같은 날은 대상 행 snapshot → 발생 반영 → 라벨 반영 순서 | 당일 확정 라벨이 같은 날 행의 피처로 새는 것 |
| 동결 이력 (`fit_history` + `transform_frozen`, `label_cutoff_exclusive`) | 추론 피처는 cutoff 이전 이력만 사용, cutoff 이전 행 입력은 거부 | 운영 시점보다 뒤의 이력으로 과거를 재평가하는 것 |
| OOF 학습 피처 (`fit_transform_oof`) | 학습 행도 자기 이후 라벨을 보지 않도록 순차 생성 | target encoding 계열 이력 피처의 자기 참조 |

피처 구성은 범위(제품 / 고객사×제품 / 제품×상세내용 / 라인×제품)별 전체·30/90/180일 라벨 비율과 최근-전체 위험률 차이,
동일 serial 재입고 간격, 수치(입고·처리 횟수, 사용 시간과 차이·비율, 결측), 텍스트 의미 패턴, smoothing 된 라벨 이력입니다.

## 안전 게이트 설계 (`gate.py`)

- **위험 방향 정의**: 실제와 다르게 PROCESS 로 보내거나(NG→공정투입), PROCESS 여야 할 것을 NG 처리(공정투입→NG)
- **같은 cohort 비교**: 후보 예측은 평가 행과 id 가 정확히 1:1 이어야 하며, 기준 모델과 같은 행에서만 비교
- **고객사 단위 차단**: 전체가 좋아져도 고객사별로 위험 방향 증가, 신규 위험 케이스, 오자동 증가, 정확도 하락, 자동처리율 2%p 초과 감소 중 하나라도 있으면 차단
- **표본 부족 = 판단 보류**: 최소 행·라벨별 최소 수에 못 미치는 그룹은 "안전"으로 보지 않고, 그 그룹에서 판정이 바뀌면 차단
- **승격은 사람 몫**: 게이트를 통과해도 `production_eligible=False` (전향적 근거·사람 승인 필요)

## 데모

```bash
pip install -r requirements.txt
python3 demo.py
python3 -m pytest -q
```

합성 데이터(6,000행, 운영 중 특정 라인에 분포 변화 주입)로 한 사이클을 재현합니다.

```text
label receipt @ 2026-05-15 {"sourceRows": 6000, "actionRows": 5826, "eligibleRows": 3871, "excludedRows": {..., "futureOccurrenceDate": 1476, "notAvailableBeforeCutoff": 425}}
train rows: baseline=1520 (cutoff 2026-03-01), candidate=3871 (cutoff 2026-05-15)
eval rows: 1476
  baseline  auto_rate=0.930 accuracy=0.489 ng_to_process=265 process_to_ng=270
  candidate auto_rate=0.618 accuracy=0.640 ng_to_process=251 process_to_ng=33
gate passed: False | production_eligible: False
  - customer:customer_c:increased_ng_to_process
  - ... (자동처리율 과다 감소, 신규 위험 케이스 등)
```

후보 모델은 전체 정확도가 크게 올랐지만 자동처리율이 기준보다 크게 떨어지고, 일부 고객사에서 위험 방향 오판정이 늘어 **게이트가 차단**합니다.
평균 지표만 보면 교체하고 싶어지는 상황에서 멈추게 하는 것이 이 게이트의 목적입니다. (합성 데이터 수치이며 실제 성능 지표가 아닙니다.)

## 폴더 구성

```text
industrial-mes-continual-learning/
├── mes_cl/
│   ├── labels.py            # 라벨 가용성 인덱스, cutoff 별 receipt
│   ├── temporal.py          # 범위·기간별 라벨 이력, serial 재입고 피처
│   ├── domain_features.py   # 수치·텍스트·smoothed 이력 피처 (OOF / frozen)
│   ├── gate.py              # 기준 vs 후보 같은-cohort 안전 게이트
│   └── synthetic.py         # 합성 MES 이력 생성기
├── demo.py                  # end-to-end 재학습 사이클 데모
└── tests/                   # point-in-time 누수 방지·게이트 차단 규칙 테스트
```

## 공개본에서 바꾼 점

- 필드명·판정 라벨을 일반화했습니다 (`action_result`, `intake_count`, `usage_hours`, `PROCESS / RETURN / REPAIR` 등).
- 텍스트 의미 패턴은 현장 용어 대신 예시 어휘로 교체했습니다 (키 이름이 feature 이름이므로 현장 적용 시 어휘만 교체).
- 실제 플랫폼의 동결 룰 엔진, 고객사·라인 보정, 학습된 모델, 운영 이력은 포함하지 않습니다.
