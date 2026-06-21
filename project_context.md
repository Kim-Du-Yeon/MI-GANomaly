# MI-GANomaly 프로젝트 컨텍스트

## 목표
태양광 패널 결함 탐지 Semi-Supervised Anomaly Detection 포트폴리오.
정상 이미지만 학습 → 복원 품질(Anomaly Score)로 결함 판별.
딥러닝 엔지니어링 역량(실험설계, 손실함수 설계, 시각화, ablation) 증명이 핵심.

## 배경 & 접근법 선택 이유
- 태양광 패널은 정상 데이터 多, 결함 데이터 少 → 클래스 불균형
- 기존 GANomaly(CNN 기반): 로컬 특징엔 강하나 전역(Global) 문맥 파악 약함
- 해결: 입력 이미지 일부를 무작위 마스킹 → 주변 문맥으로 복원 유도
  → 잘 복원할수록 패널 전체 구조를 이해한다는 아이디어 (MI-GANomaly)

## 데이터셋
- 주: ELPV Dataset (github.com/zae-bayern/elpv-dataset)
  - 전체 2,624장 중 정상만 필터링 ~1,200장, train/test = 3:7
  - Mono / Poly 두 종류 존재
- 보조: DAGM 2007 (일반화 실험용, 산업 결함 벤치마크)
- 증강: RandomFlip, RandomRotation(±15°), ColorJitter, GaussianBlur

## 아키텍처: MI-GANomaly
GANomaly 기반, 핵심 변경점:
1. Masking 기법: MAE 스타일 비정형 랜덤 마스킹 (8×8 / 16×16 / 32×32, 비율 10~30%)
   - 기존 고정 패치 마스킹 → 위치 편향 학습 문제
   - MAE 스타일로 교체 → 전역 문맥 강제 학습
2. Loss 재설계:
   L_total = λ1*L_recon + λ2*L_contextual + λ3*L_encoder
   - L_recon: MSE + SSIM (MSE 단독 → 구조적 왜곡 무시 문제 보완)
   - L_contextual: perceptual feature similarity (원본↔복원 문맥 차이)
   - L_encoder: latent space consistency (특징 맵 차이)
3. λ 조합 ablation: (1:1:1), (2:1:0.5), (1:2:1) 등 실험
4. 정규화: BatchNorm → GroupNorm 교체
   - 소배치에서 BatchNorm 통계 불안정 문제 해결
5. Threshold: 고정 sweep → μ+kσ 통계적 자동화
   - 데이터 의존적 sweep 방식 → 재현 가능한 자동화로 일반화

## 핵심 하이퍼파라미터 (options.py)
nz=500, isize=32, nc=3, batchsize=64, dataset=ELPV

## 평가 전략 (삼성전자 멘토 피드백 반영)
- Metrics: Accuracy, Precision, Recall, F1, AUC 전부 기록
- Threshold sweep (0.3~0.7) + μ+kσ 자동화 비교
- PR Curve, ROC Curve, 혼동행렬
- CAM → Grad-CAM++ 업그레이드, 이상 케이스에서도 작동 검증
- 글로벌 패턴 근거: PCA or 공분산 행렬로 시각화
- Before(베이스 GANomaly) vs After(MI-GANomaly) 비교 테이블 필수

## 삼성전자 멘토 피드백 (강현규 책임연구원)
- Metric 다양화: 다양한 지표로 기존 연구 대비 우위 입증
- CAM 보완: 이상 케이스에서도 Grad-CAM++ 작동 검증
- 마스킹 다양화: 크기·수 변경 실험으로 글로벌 패턴 탐지 실증
- 글로벌 패턴 근거: PCA, 공분산 행렬로 논리적 근거 제시
- Threshold 분석: PR Curve, ROC Curve로 최적값 도출

## 기술 업데이트 이력 (포트폴리오 핵심)
| 기술 | 문제 | 해결 | 기록 포인트 |
|------|------|------|------------|
| SSIM 추가 | MSE는 구조적 왜곡 무시 | SSIM으로 밝기/대비/구조 동시 측정 | L_recon Before/After AUC 변화 |
| MAE 스타일 마스킹 | 고정 패치 → 위치 편향 학습 | 비정형 랜덤 마스킹으로 전역 문맥 강제 | 마스킹 방식별 F1 비교 테이블 |
| μ+kσ Threshold | sweep 방식 재현 어려움 | 통계적 자동화로 일반화 | threshold 방식별 Precision/Recall 변화 |
| GroupNorm 교체 | 소배치에서 BatchNorm 불안정 | GroupNorm으로 배치 크기 무관 안정화 | 학습 안정성 loss curve 비교 |

## 디렉토리 구조
mi_ganomaly/
├── data/train/normal/, test/normal/, test/anomaly/
├── models/ganomaly.py, networks.py, loss.py
├── utils/dataloader.py, visualize.py, metrics.py
├── train.py, evaluate.py, options.py
└── experiments_log.md  # 실험별 AUC/F1 기록

## Phase 진행 상태
- [ ] Phase 1: 베이스라인 GANomaly + 분포 그래프 (Before 기준점)
- [ ] Phase 2: 마스킹 기법 (MAE 스타일, 크기/비율 실험)
- [ ] Phase 3: 손실함수 재구성 (SSIM 추가) + ablation table
- [ ] Phase 4: Grad-CAM++ 시각화
- [ ] Phase 5: Metric & Threshold (μ+kσ) 분석 + 최종 Before/After 정리

## 포트폴리오 전략
각 Phase 완료 시 아래 4가지 기록 필수:
1. 문제 정의 - 왜 이 기법이 필요했나
2. 이슈 - 구현/실험 중 막힌 것
3. 해결 - 어떻게 극복했나 (코드/수식/실험 근거)
4. 결과 - Before/After 수치 비교

## GitHub 전략
- README.md: Phase 5 완료 후 수치/그래프 포함해 작성
- experiments_log.md: 실험 수치 누적 기록 (신뢰성 증명)
- 커밋 히스토리: 엔지니어링 과정 증명용으로 메시지 작성