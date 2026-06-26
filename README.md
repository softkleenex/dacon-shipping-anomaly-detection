# 스마트 해운물류 x AI 미션 챌린지 : 이상신호 감지 기반 비정상 작동 진단

해양수산부 주최 장비 센서 데이터 기반 고장 진단 머신러닝 프로젝트

## 프로젝트 개요

장비에서 수집된 52개 센서 신호 데이터를 분석하여 21가지 정상/비정상 작동 유형을 분류하는 머신러닝 모델을 개발했습니다. 블랙박스 환경에서 도메인 지식 없이 순수 데이터 분석과 Feature Engineering만으로 성능을 최적화하는 것이 핵심 과제였습니다.

### 주요 성과 (Private Leaderboard)

- **최종 순위**: **236등** / 946 팀 (상위 24.9%)
- **최종 점수**: **0.75515** (1위 점수: 0.88997, 격차: 0.13482)
- **평가지표**: Macro-F1 Score
- **데이터**: 21,693개 학습 샘플, 15,004개 테스트 샘플
- **특징**: 52개 센서 신호 (X_01 ~ X_52)
- **목표**: 21개 클래스 분류

## 기술 스택

- **Python 3.12**
- **ML Framework**: scikit-learn, LightGBM, XGBoost, CatBoost
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **GPU Acceleration**: CUDA 12.6, RTX 4060

## 프로젝트 구조

```
Shipping_logistics_abnormal_signal/
├── notebooks/                          # 탐색적 분석 및 모델링
│   ├── 01_EDA.ipynb                   # 데이터 탐색 및 시각화
│   ├── 02_baseline_model.ipynb        # 베이스라인 모델 구축
│   └── 03_feature_engineering_ensemble.ipynb  # 고급 피처 엔지니어링
│
├── src/                                # 실행 가능한 스크립트
│   ├── advanced_feature_engineering.py # 149개 고급 피처 생성
│   ├── deep_analysis.py               # 성능 분석 및 개선점 도출
│   ├── train_improved_model.py        # GPU 최적화 모델 학습
│   └── multi_submit.py                # Dacon API 자동 제출
│
├── data/                               # 데이터 저장소
├── submissions/                        # 제출 파일
├── requirements.txt                    # 의존성 패키지
└── README.md
```

## 주요 기능

### 1. 탐색적 데이터 분석 (EDA)

- 52개 센서 신호의 분포 및 상관관계 분석
- 클래스 불균형 확인 (완벽하게 균형잡힌 21개 클래스)
- 이상치 및 결측치 탐지
- 저분산 피처 및 고상관 피처 식별

**주요 발견**:
- 29개 저분산 피처 발견
- 29쌍의 고상관 피처 쌍 (r > 0.95)
- 주요 판별 피처: X_19, X_37, X_40, X_11, X_28

### 2. Feature Engineering

기본 52개 피처에서 149개 고급 피처로 확장:

```python
# Interaction Features (30개)
- 주요 판별 피처 간 곱셈, 뺄셈, 나눗셈 조합

# Statistical Features (23개)
- 행 단위 평균, 표준편차, 최대/최소, 왜도, 첨도
- 분위수 (Q25, Q75, IQR)

# Polynomial Features (10개)
- 주요 피처의 2차 다항식

# Clustering Features (54개)
- KMeans (k=5, 10, 15, 21)
- 클러스터 레이블 및 거리

# Dimensionality Reduction (30개)
- PCA 30 components

# Feature Selection
- 고상관 피처 4개 제거 → 최종 149개
```

### 3. 모델 앙상블

**GPU 최적화 Voting Ensemble**:

```python
# LightGBM (GPU)
- n_estimators: 800
- max_depth: 12
- device: 'gpu'
- class_weight: 'balanced'

# XGBoost (GPU)
- n_estimators: 800
- max_depth: 12
- tree_method: 'hist'
- device: 'cuda'

# CatBoost (CPU)
- iterations: 1000
- depth: 10
- task_type: 'CPU'  # GPU OOM 방지
- auto_class_weights: 'Balanced'
```

**Soft Voting**: 3개 모델의 확률 평균

### 4. 성능 분석

```
Baseline F1 Score: 0.8004
Target F1 Score: 0.9000
Gap: 0.0996

Class-wise Performance:
- Best: Class 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 20 (F1 > 0.90)
- Worst: Class 9 (0.2056), 0 (0.2489), 15 (0.2511), 19 (0.4474), 3 (0.5105)
```

## 설치 및 실행

### 환경 설정

```bash
# 저장소 클론
git clone https://github.com/yourusername/shipping-anomaly-detection.git
cd shipping-anomaly-detection

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 패키지 설치
pip install -r requirements.txt
```

### 데이터 준비

```bash
# Dacon에서 데이터 다운로드
# https://dacon.io/competitions/official/236590/data

# 데이터 배치
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

### 실행 방법

```bash
# 1. 고급 피처 생성
python src/advanced_feature_engineering.py

# 2. 모델 학습 및 예측
python src/train_improved_model.py

# 3. Dacon 제출 (선택)
python src/multi_submit.py
```

### Jupyter 노트북 실행

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

## 💡 Lessons Learned & Failure Analysis (비판적 회고)

### 1. 블랙박스 데이터의 한계와 돌파구 (도메인 지식 부재)
- **문제**: 52개 센서 신호의 의미(도메인 지식)가 완전히 차단된 상태로 데이터가 주어졌습니다.
- **해결**: 통계적 패턴(분산, 왜도) 분석과 고상관 피처 쌍 탐색에 집중하여 숨겨진 클래스 간 차이를 클러스터링으로 유도했습니다.
- **교훈**: 도메인 지식 없이도 순수 데이터 기반 접근(Feature Engineering)만으로 유의미한 성능 향상이 가능함을 확인했습니다.

### 2. Feature Engineering의 함정 (차원의 저주)
- **현상**: 기본 52개 피처에서 다항식, 클러스터링 등 149개로 피처를 대폭 확장했을 때 예측 성능 향상이 정체되는 구간이 있었습니다.
- **교훈**: 피처 수의 무분별한 증가는 앙상블 모델에 노이즈를 더할 뿐입니다. 저분산 피처나 불필요한 고상관 피처를 식별하고 적극적으로 제거(Feature Selection)하는 정제 과정이 성능 최적화에 훨씬 중요했습니다.

### 3. OOM(Out of Memory) 문제와 하이브리드 리소스 할당
- **현상**: 늘어난 차원(149개)으로 인해 VRAM 8GB 환경에서 CatBoost 학습 시 지속적인 OOM 에러가 발생했습니다.
- **해결**: LightGBM과 XGBoost는 GPU 연산을 유지하되, 메모리 요구량이 큰 CatBoost는 CPU 모드로 할당하는 하이브리드 환경을 구축해 해결했습니다.
- **교훈**: 제한된 리소스 환경에서는 모델 프레임워크별 하드웨어 최적화 특성을 딥하게 이해하고 유연하게 리소스를 분배해야 합니다.

### 4. 외형적 클래스 균형의 함정
- **현상**: 전체 데이터셋은 21개 클래스가 완벽하게 동일한 비율(Balanced)로 존재했으나, 실제 모델의 특정 클래스(0, 3, 9, 15, 19)에 대한 예측 성능(F1 Score)은 0.20~0.50에 불과한 심각한 편차를 보였습니다.
- **교훈**: 단순한 '데이터 개수'의 균형이 '학습의 균형(난이도)'을 보장하지 않습니다. 클래스별 분류 난이도를 고려한 동적 가중치 조절이나 타겟 Pseudo Labeling 도입이 필수적이라는 인사이트를 얻었습니다.

## 라이선스

MIT License

## 참고 자료

- [대회 링크](https://dacon.io/competitions/official/236590/overview/description)
- [Macro-F1 Score](https://en.wikipedia.org/wiki/F-score)
- [LightGBM GPU 튜토리얼](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html)

---

**개발 기간**: 2025.09.08 - 2025.10.02
**주최**: 해양수산부, 울산항만공사, 한국정보산업연합회
**운영**: 데이콘

<!-- BLOG-URL:START -->

## Blog

- Blog note: [스마트 해운물류 x AI 미션 챌린지 : 이상신호 감지 기반 비정상 작동 진단](https://softkleenex.github.io/coding_training/dacon/dacon-shipping-anomaly-detection)

<!-- BLOG-URL:END -->
