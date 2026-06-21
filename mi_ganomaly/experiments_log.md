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
