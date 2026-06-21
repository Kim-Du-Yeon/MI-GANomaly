import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ganomaly import GANomaly
from models.loss import TotalLoss
from options import get_options
from utils.dataloader import get_dataloader
from utils.gradcam import GradCAMPlusPlus
from utils.visualize import save_gradcam

CKPT_DIR = 'output/phase3_ssim'
OUT_DIR = 'output/phase4_gradcam'
N_SAMPLES = 5


def collect_scores_with_images(model, dataset, device):
    model.eval()
    images, scores, labels = [], [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            img, label = dataset[i]
            score = model.anomaly_score(img.unsqueeze(0).to(device)).item()
            images.append(img)
            scores.append(score)
            labels.append(label)
    return images, np.array(scores), np.array(labels)


def main():
    opt = get_options()
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

    model = GANomaly(opt)
    model.criterion = TotalLoss(opt, recon_alpha=0.5)  # Phase 3 setting
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'best_model.pth'), map_location=device))
    model.eval()

    test_loader, _ = get_dataloader(opt, 'test')
    dataset = test_loader.dataset

    images, scores, labels = collect_scores_with_images(model, dataset, device)

    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]

    normal_top = normal_idx[np.argsort(scores[normal_idx])][:N_SAMPLES]
    anomaly_top = anomaly_idx[np.argsort(-scores[anomaly_idx])][:N_SAMPLES]

    conv_layers = [m for m in model.e1.main if isinstance(m, nn.Conv2d)]
    target_layer = conv_layers[-2]  # last spatial conv (final-conv collapses to 1x1)
    cam_engine = GradCAMPlusPlus(model, target_layer)

    os.makedirs(OUT_DIR, exist_ok=True)

    for rank, idx in enumerate(normal_top, start=1):
        img = images[idx]
        heatmap = cam_engine.generate(img.unsqueeze(0).to(device))
        save_path = os.path.join(OUT_DIR, f'normal_{rank:02d}_score{scores[idx]:.4f}.png')
        save_gradcam(img, heatmap, save_path)
        print(f'normal  rank={rank} idx={idx} score={scores[idx]:.4f} -> {save_path}')

    for rank, idx in enumerate(anomaly_top, start=1):
        img = images[idx]
        heatmap = cam_engine.generate(img.unsqueeze(0).to(device))
        save_path = os.path.join(OUT_DIR, f'anomaly_{rank:02d}_score{scores[idx]:.4f}.png')
        save_gradcam(img, heatmap, save_path)
        print(f'anomaly rank={rank} idx={idx} score={scores[idx]:.4f} -> {save_path}')

    cam_engine.remove_hooks()


if __name__ == '__main__':
    main()
