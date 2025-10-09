# DataScience
# KAMP ABH Sensor — 품질 이상탐지/진단 실습 (Colab)

센서 CSV(`Index, Lot, Time, pH, Temp, Current`)와 **Error Lot list**를 이용해

* QC 라벨이 생성되면 **지도학습(XGBoost 분류)**,
* QC 라벨이 없으면 비지도( IsolationForest )로 자동 전환하여 이상치 탐지합니다.
  모든 실습은 **Google Colab**에서 실행하도록 구성했습니다.

---

## 1. 폴더/파일 준비

Colab 노트북에서 다음 파일들을 **업로드**합니다.

* 센서 로그:
  `kemp-abh-sensor-YYYY.MM.DD.csv` (여러 개 가능)
  예) `kemp-abh-sensor-2021.09.06.csv`, …
* 에러 LOT 리스트:
  `Error Lot list.csv`

> 파일 인코딩은 혼합되어 있을 수 있어 `utf-8-sig / utf-8 / cp949 / euc-kr` 순으로 자동 시도합니다.

---

## 2. 환경(Colab)

```python
!pip -q install --upgrade pip
!pip -q install pandas numpy scikit-learn matplotlib seaborn xgboost
```

> AutoKeras는 Colab 기본 TF와 버전 충돌 가능성이 있어 **필수 아님**(백업 모델로 XGBoost 사용).

---

## 3. 파이프라인 개요

1. **CSV 로딩 & 정규화**

* 여러 CSV 자동 탐색, 인코딩 자동 시도
* 컬럼명 통일(소문자, 공백/특수문자 제거)

2. **Timestamp 생성**

* 파일명에서 날짜(`YYYY.MM.DD`) 추출 → `date`
* `Time`은 한글 표기(“오전/오후 4:29:15.0”) 지원: `오전/오후 → AM/PM` 치환 + `%p %I:%M:%S.%f` 파싱
* `timestamp = date + time_parsed` → 정렬, NaT 제거

3. **피처 표준화**

* `pH/Temp/Current` 문자열 섞임(단위, `--`, 콤마 소수점 등) → 숫자화하여 `ph_std/temp_std/current_std` 생성

4. **라벨 생성(QC)**

* `Error Lot list.csv`의 날짜/LOT와 센서 LOT 정규화(대문자/공백 제거, 선행 0 보존)
* 매칭 성공: `QC={불량:0, 정상:1}`
* 매칭 실패(한 클래스만 존재): **비지도 경로로 자동 전환**

5. **학습**

* **지도**: `XGBClassifier` (Accuracy/Precision/Recall/F1/ROC/Feature Importance)
* **비지도**: `IsolationForest` (anomaly score/flag, 분포, 상위 이상치)

6. **EDA**

* 히스토그램(분포), 상관관계 히트맵(변수 연동성)

7. **결과 저장**

* `artifacts/metrics.csv`, `test_predictions.csv`, `confusion_matrix.png`, `roc_curve.png`,
  `feature_importance.csv`, `model_xgb.joblib`(지도)
* `artifacts/anomalies_sorted.csv`, `anomaly_score_hist.png`(비지도)

---

## 4. 실행 순서 (노트북 셀)

1. **설치**
2. **임포트 & 환경설정**
3. **파일 업로드 / (선택) 드라이브 마운트**
4. **CSV 스캔 & 로딩** (정규화 포함)
5. **Timestamp 생성**(한글 AM/PM 파싱)
6. **피처 표준화**(`*_std` 생성)
7. **EDA**(히스토그램, 상관 히트맵)
8. **라벨링(강화)**: 날짜+LOT → 실패 시 날짜-only로 폴백
9. **모델 학습(자동 전환)**: 지도(XGBoost) ↔ 비지도(IsolationForest)
10. **예측 & 평가지표**(지도) / **이상치 요약**(비지도)
11. **ROC Curve**(지도)
12. **결과/아티팩트 저장**

> 각 단계 코드는 노트북에 포함되어 있으며, 위에서 아래로 실행하면 됩니다.

---

## 5. 모델 구성

### 지도학습(라벨 2클래스 존재 시)

* **XGBClassifier**

  ```python
  XGBClassifier(
      n_estimators=300, max_depth=5, learning_rate=0.05,
      subsample=0.8, colsample_bytree=0.8,
      reg_lambda=1.0, random_state=42, eval_metric="logloss"
  )
  ```
* 출력: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC/AUC, Feature Importance, 예측 CSV, 모델 파일

### 비지도(라벨 미존재/한 클래스일 때)

* **IsolationForest**

  * `contamination=0.05`(상위 5% 이상치) 기본
  * 출력: anomaly_score(높을수록 이상), anomaly_flag(1=이상), 분포, 상위 이상치, 테이블 CSV

---

## 6. 그래프 해석(제조업 관점)

* **히스토그램**: 변수 분포 확인 → 정상 범위/이탈 구간 식별
* **상관 히트맵**: 변수 간 연동성 확인 → 평소 상관이 깨지면 공정 이상 신호
* **Anomaly Score 분포**: 정상 집단 vs 이상 집단 분리 → 꼬리 부근이 핵심 후보

---

## 7. 주요 파라미터(튜닝 포인트)

* **비지도 임계치**: `IsolationForest(contamination=0.05)` → 0.01~0.10 사이로 조정
* **특징 컬럼**: `["ph_std","temp_std","current_std"]` 우선, 없으면 원본명
* **시간 파싱**: 오전/오후 한글 포맷 외에 다른 포맷 있으면 파서에 추가
* **LOT 정규화**: 대소문자/공백/하이픈/슬래시 제거, 문자열 유지(선행 0 보존)

---

## 8. 자주 나는 오류 & 해결

* `KeyError: 'Time'`
  → 실제 컬럼명이 다름(`time/timestamp/datetime`) → 후보 검색 후 일치 컬럼 사용
* `dd.head()`가 비어 있음
  → `timestamp` 파싱 실패로 모두 NaT → 한글 AM/PM 파서 적용/포맷 확인
* 모든 QC가 1(정상)
  → Error Lot 리스트 매칭 실패 → 날짜/LOT 포맷 정규화, 날짜 교집합 확인 → 실패 시 비지도 전환
* `cannot reindex on an axis with duplicate labels`
  → 동일 timestamp 중복 → `reset_index()` 후 `concat`으로 병합
* AutoKeras 속성 없음
  → 버전 차이 → XGBoost 경로 사용(권장)

---

## 9. 산출물(Artifacts)

* 지도:
  `artifacts/metrics.csv`, `test_predictions.csv`, `confusion_matrix.png`, `roc_curve.png`,
  `feature_importance.csv`, `model_xgb.joblib`
* 비지도:
  `artifacts/anomalies_sorted.csv`, `anomaly_score_hist.png`

---

## 10. 해석 가이드(현장 적용)

* **Anomaly 상위 리스트**를 LOT/공정 로그와 대조 → 설비/약품/레시피 변경 시점 확인
* 분포 꼬리에서 반복적으로 등장하는 **시간대/LOT** → 재발성 이슈 가능성
* 히트맵 패턴이 시간이 지나며 변하면 → 공정 drift 또는 센서 보정 필요 신호

---

## 11. 라이선스/인용

* 본 실습 코드는 교육/연구용 예시입니다.
* 데이터 소유권/보안 정책에 맞게 사용하세요.

---

### 문의/확장

* 라벨이 정상/불량 모두 나오도록 Error Lot 리스트 매칭 규칙을 더 강화할 수 있습니다(날짜 윈도우, LOT 패턴 추가 등).
* 필요 시, 알람 임계치(컨테미네이션) 최적화, 피처 엔지니어링, 계절성/주기성 제거, 설비별 모델 분리 등을 고려하세요.
