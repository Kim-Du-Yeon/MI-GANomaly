# MI-GANomaly 프로젝트

## 작업 루트
D:\AX\MI-GANomaly

## 디렉토리 구조
mi_ganomaly/
├── data/
│   ├── train/normal/
│   └── test/normal/, anomaly/
├── models/
│   ├── ganomaly.py     # GAN 학습 루프
│   ├── networks.py     # Generator, Discriminator 아키텍처
│   └── loss.py         # L_recon, L_contextual, L_encoder
├── utils/
│   ├── dataloader.py   # ELPV 데이터 로드 + 증강
│   ├── visualize.py    # Anomaly Score 분포, Grad-CAM++
│   └── metrics.py      # Accuracy, F1, AUC, threshold sweep
├── train.py
├── evaluate.py
├── options.py          # nz=500, isize=32, nc=3, batchsize=64
└── experiments_log.md

## 핵심 규칙
- Python, PyTorch만 사용
- LLM 호출 필요 시 OpenAI API만 사용 (anthropic import 금지)
- 전체 코드 출력 금지, 수정은 지정 파일만
- 완료 후 보고: 수정위치 / 변경전후 핵심코드 / 테스트방법

## 아키텍처 요약
- 베이스: GANomaly (github.com/samet-akcay/ganomaly)
- 추가: 랜덤 마스킹 (8x8 / 16x16 / 32x32, 비율 10~30%)
- Loss: L_total = λ1*L_recon + λ2*L_contextual + λ3*L_encoder

## 데이터셋
- ELPV Dataset: 정상 ~1,200장, train/test = 3:7
- 증강: RandomFlip, RandomRotation(±15°), ColorJitter, GaussianBlur