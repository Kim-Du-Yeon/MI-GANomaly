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

## Phase 2 - MAE 스타일 마스킹 실제 ELPV 결과
- 설정: mask_size=8, mask_ratio=0.2, 50 epoch

| 방식 | threshold | Accuracy | Precision | Recall | F1 | AUC |
|------|-----------|----------|-----------|--------|----|-----|
| sweep | 0.100 | 0.4222 | 0.4423 | 0.4776 | 0.4593 | 0.4203 |
| auto(μ+2σ) | 0.312 | 0.5000 | 0.8261 | 0.0341 | 0.0654 | 0.4203 |

### Before vs After (Phase 1 → Phase 2)
| 지표 | Phase 1 | Phase 2 | 변화 |
|------|---------|---------|------|
| AUC | 0.5096 | 0.4203 | -0.0893 |
| F1 (sweep) | 0.3599 | 0.4593 | +0.0994 |
| Recall | 0.2751 | 0.4776 | +0.2025 |

### 해석 & 포트폴리오 근거
- 마스킹 도입으로 Recall +0.20 개선 확인
- AUC 하락 → 마스킹 단독으로는 score 분리력 확보 불충분
- 결론: Loss 재설계(SSIM + contextual 강화) 필요성 실험적 증명 → Phase 3 근거

## Phase 3 - SSIM Loss 재설계 실제 ELPV 결과
- 설정: mask_size=8, mask_ratio=0.2, recon_alpha=0.5 (MSE+SSIM), 50 epoch

| 방식 | threshold | Accuracy | Precision | Recall | F1 | AUC |
|------|-----------|----------|-----------|--------|----|-----|
| sweep | 0.100 | 0.5677 | 0.5799 | 0.5753 | 0.5776 | 0.6230 |
| auto(μ+2σ) | 0.274 | 0.5120 | 0.9828 | 0.0511 | 0.0971 | 0.6230 |

### Phase 1 → 2 → 3 누적 비교
| 지표 | Phase 1 | Phase 2 | Phase 3 | 총 변화 |
|------|---------|---------|---------|---------|
| AUC | 0.5096 | 0.4203 | 0.6230 | +0.1134 |
| F1 | 0.3599 | 0.4593 | 0.5776 | +0.2177 |
| Precision | 0.5203 | 0.4423 | 0.5799 | +0.0596 |
| Recall | 0.2751 | 0.4776 | 0.5753 | +0.3002 |

### 해석 & 포트폴리오 근거
- 마스킹 단독(Phase 2)으로는 AUC 하락 → Loss 재설계 필요성 실험적 도출
- SSIM 추가(Phase 3)로 구조적 유사도 기반 복원 평가 → 정상/이상 분리력 회복
- AUC 0.51→0.62, F1 0.36→0.58 달성
- 결론: 마스킹 + SSIM 조합이 시너지 효과 발휘, 단일 기법보다 복합 설계가 유효함을 증명

## Phase 4 - Grad-CAM++ 시각화
- 대상: Phase 3 best_model.pth
- 타겟 레이어: Encoder 피라미드 마지막 Conv (출력 4×4×256)
  - 설계 이유: 최종 Conv(4×4→1×1)는 공간 정보 소실 → CAM 단색 출력
  - 공간 해상도 보존된 직전 레이어로 변경 (Grad-CAM 표준 관례)

### 결과 요약
| 구분 | score 범위 | CAM 활성화 |
|------|-----------|-----------|
| 정상 5장 | 0.1090~0.1174 | 분산된 그라디언트, hotspot 없음 |
| 이상 5장 | 1.4722~1.9886 | 중앙 집중 hotspot 형성 |

### 이슈 & 해결
- 이슈: anomaly_score()가 @torch.no_grad()라 backward 불가
  → TotalLoss를 grad 추적 가능하게 재계산해 CAM score로 사용
- 이슈: 4×4 해상도 업샘플로 블록형 heatmap 한계
  → 고해상도 레이어 비교는 Phase 5 보완 예정

### 멘토 피드백 충족 여부
- [x] 이상 케이스에서도 Grad-CAM++ 작동 검증
- [x] 구조적 손상 셀(score 1.99)에서 결함 위치 정렬 확인

## Phase 5 - 최종 분석 (Before/After 종합)

### 전체 Phase 흐름 요약
1. Phase 1: 베이스라인 GANomaly (마스킹 없음, MSE only) — Before 기준점, AUC 0.5096 (거의 랜덤)
2. Phase 2: MAE 스타일 패치 마스킹(mask_size=8, ratio=0.2) 추가 — Recall +0.20, but AUC 0.4203로 오히려 하락
3. Phase 3: Loss 재설계(MSE 0.5 + SSIM 0.5) — AUC 0.6230, F1 0.5776로 전 지표 개선 (After 기준점)
4. Phase 4: Grad-CAM++로 이상 케이스 활성화 시각적 검증 (Phase3 모델 기준)
5. Phase 5: Phase 1/2/3 score 분포 종합 비교 + threshold 방식(sweep vs μ+kσ) 비교 + 최종 Before/After 정리

### 최종 Before/After 수치 (sweep threshold 기준)
| 지표 | Phase1 (Before) | Phase3 (After) | Delta |
|------|-----------------|-----------------|-------|
| Accuracy | 0.4972 | 0.5677 | +0.0704 |
| Precision | 0.5203 | 0.5799 | +0.0596 |
| Recall | 0.2751 | 0.5753 | +0.3002 |
| F1 | 0.3599 | 0.5776 | +0.2177 |
| AUC | 0.5096 | 0.6230 | +0.1134 |

### Threshold 방식 비교 (Phase3 기준)
| 방식 | threshold | Precision | Recall | F1 |
|------|-----------|-----------|--------|-----|
| sweep | 0.100 | 0.5799 | 0.5753 | 0.5776 |
| auto(k=1.5) | 0.233 | 0.9796 | 0.0860 | 0.1582 |
| auto(k=2.0) | 0.274 | 0.9828 | 0.0511 | 0.0971 |
| auto(k=2.5) | 0.315 | 1.0000 | 0.0287 | 0.0557 |

→ μ+kσ는 k가 커질수록 Precision은 1.0에 근접하지만 Recall이 급격히 무너져 F1이 sweep 대비 크게 낮음. 실제 운영 기준값은 sweep(best F1) 방식이 더 합리적.

### 포트폴리오 핵심 3줄 요약
1. 마스킹만 단독 적용 시 AUC가 하락(0.51→0.42)하는 것을 실험으로 확인 — 단일 기법 추가가 항상 성능을 개선하지 않음을 정량적으로 입증
2. SSIM 기반 Loss 재설계 결합 시 AUC 0.51→0.62, F1 0.36→0.58로 전 지표 개선 — 마스킹+구조적 유사도 손실의 복합 설계가 핵심 기여
3. Grad-CAM++로 이상 케이스의 모델 판단 근거(활성화 영역)를 시각적으로 검증해 "정확도 수치"를 넘어 "왜 이상으로 판단했는지" 설명 가능성까지 확보

## 증강 전략 설계 (도메인 특화)
### 도메인 분석
- 태양광 패널: 직사각형 격자 구조, 색상이 결함 판단 기준
- 정상 이미지만 학습하는 Semi-Supervised 구조

### 채택 기법 & 이유
| 기법 | 이유 |
|------|------|
| HorizontalFlip | 패널 좌우 대칭성 |
| VerticalFlip | 패널 상하 대칭성 |
| Rotation(±15°) | 촬영 각도 미세 변동 |
| GaussianBlur(약) | 카메라 초점 변동 모사 |
| ColorJitter(밝기/대비만) | 조명 조건 변동, 색조 제외 |

### 금지 기법 & 이유
| 기법 | 금지 이유 |
|------|----------|
| RandomErasing | 인위적 결함 생성 → 정상 데이터 오염 |
| CutMix/MixUp | 존재하지 않는 패턴 생성 |
| ElasticTransform | 패널 격자 구조 파괴 |
| ColorJitter(saturation) | 색조 변경 → 결함 판단 기준 훼손 |
| Rotation(>15°) | 비현실적 구조 왜곡 |

### 실행 결과
- 원본 452장 → 증강 후 2,712장 (mi_ganomaly/data/train/normal_augmented/, 원본 폴더는 변경하지 않고 별도 폴더에 생성)
- 기법별 452장씩 균등 배분 (original 452 + 5기법 × 452 = 2,712)

### 이슈
- pixel_dist_after.png에서 intensity=0 부근에 스파이크 발생 → RandomRotation의 회전 후 빈 모서리가 검은색(fill=0)으로 채워지는 부작용. 실제 결함 판단에 영향 줄 수 있어 Phase 6 이후 fill 값 조정 검토 필요

### 재현성
- mi_ganomaly/utils/reproducibility.py의 set_seed(42)를 train.py, evaluate.py에 적용 (torch/cuda/numpy/random/cudnn 전부 고정)
