## Phase 1 - 베이스라인 GANomaly (Before 기준점)
**구현 완료일**: 2026-06-21

### 아키텍처
- Encoder → Decoder → Encoder(E2) + Discriminator
- Loss: MSE only (recon_alpha=1.0)
- Threshold: sweep(0.1~0.9) + μ+kσ 자동화 두 방식 구현

### 이슈 & 해결
1. auto threshold(μ+2σ)가 더미 데이터에서 1.082로 튀어 전체 normal 분류
   → 실제 ELPV 데이터로 k값 재검증 필요 (k=1.5 또는 k=2 비교 예정)

### 다음 단계
- Phase 2: MAE 스타일 마스킹 적용 후 F1 비교
- Before 기준 AUC/F1: 실제 데이터 학습 후 기록 예정

### 실제 ELPV 데이터 결과 (Before 기준점)
- 데이터: train/normal=452, test/normal=1056, test/anomaly=1116
- 학습: 50 epoch, batchsize=64, MSE only, 마스킹 없음

| 방식 | threshold | Accuracy | Precision | Recall | F1 | AUC |
|------|-----------|----------|-----------|--------|----|-----|
| sweep | 0.100 | 0.4972 | 0.5203 | 0.2751 | 0.3599 | 0.5096 |
| auto(μ+2σ) | 0.237 | 0.5032 | 0.7467 | 0.0502 | 0.0940 | 0.5096 |

**결론**: 정상/이상 score 분포 미분리, AUC≈0.51 (랜덤 수준) → Phase 2 마스킹 도입 근거 확보

### 발견된 이슈
1. DataLoader persistent_workers 미설정 → 매 epoch 워커 재생성으로 학습 지연
   → persistent_workers=True, pin_memory=True 추가로 해결
2. train 중 AUC(latent L2)와 evaluate AUC(TotalLoss) 정의 불일치
   → Phase 5에서 score 정의 통일 예정

## Phase 2 - 마스킹 크기/비율 Ablation (더미 데이터)

| mask_size | mask_ratio | AUC | F1 |
|---|---|---|---|
| 8 | 0.1 | 0.4167 | 0.7059 |
| 8 | 0.2 | 0.5833 | 0.8000 |
| 8 | 0.3 | 0.5833 | 0.7500 |
| 16 | 0.1 | 0.5278 | 0.7059 |
| 16 | 0.2 | 0.8056 | 0.8571 |
| 16 | 0.3 | 0.5833 | 0.8000 |
| 32 | 0.1 | 0.4722 | 0.7059 |
| 32 | 0.2 | 0.5833 | 0.7692 |
| 32 | 0.3 | 0.7778 | 0.8571 |

**Best combo (F1 기준)**: mask_size=16, mask_ratio=0.2 (AUC=0.8056, F1=0.8571)

> 더미(랜덤 노이즈) 데이터 기준 동작 검증용 결과이며, 실제 ELPV 데이터로 재실험 필요.
