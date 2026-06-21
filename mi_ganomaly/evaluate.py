import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_recall_curve, precision_score,
                              recall_score, roc_auc_score, roc_curve)

from models.ganomaly import GANomaly
from models.loss import TotalLoss
from options import get_options
from utils.dataloader import get_dataloader
from utils.reproducibility import set_seed


def collect_scores(model, loader, device):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            for j in range(x.size(0)):
                scores.append(model.anomaly_score(x[j:j + 1]).item())
            labels.extend(y.tolist())
    return np.array(scores), np.array(labels)


def normalize_scores(scores):
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-8:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def sweep_threshold(scores, labels):
    best_th, best_f1 = 0.5, -1.0
    for th in np.arange(0.1, 0.91, 0.05):
        pred = (scores >= th).astype(int)
        f1 = f1_score(labels, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return best_th


def auto_threshold(scores, k):
    return float(scores.mean() + k * scores.std())


def compute_metrics(scores, labels, threshold):
    pred = (scores >= threshold).astype(int)
    return {
        'threshold': threshold,
        'accuracy': accuracy_score(labels, pred),
        'precision': precision_score(labels, pred, zero_division=0),
        'recall': recall_score(labels, pred, zero_division=0),
        'f1': f1_score(labels, pred, zero_division=0),
    }


def plot_roc_curve(labels, scores, auc, save_dir):
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC={auc:.4f}')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()


def plot_pr_curve(labels, scores, save_dir):
    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'))
    plt.close()


def plot_confusion_matrices(labels, scores, sweep_th, auto_th, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, th, name in zip(axes, [sweep_th, auto_th], ['sweep (best F1)', 'mu+k*sigma']):
        pred = (scores >= th).astype(int)
        cm = confusion_matrix(labels, pred, labels=[0, 1])
        ax.imshow(cm, cmap='Blues')
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha='center', va='center')
        ax.set_title(f'{name}\nth={th:.3f}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()


def plot_score_distribution(scores, labels, save_dir):
    plt.figure()
    plt.hist(scores[labels == 0], bins=20, alpha=0.6, label='normal')
    plt.hist(scores[labels == 1], bins=20, alpha=0.6, label='anomaly')
    plt.xlabel('normalized anomaly score')
    plt.ylabel('count')
    plt.title('Score Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'score_distribution.png'))
    plt.close()


def print_metrics(name, metrics, auc):
    print(f'--- {name} ---')
    print(f"threshold={metrics['threshold']:.3f} "
          f"accuracy={metrics['accuracy']:.4f} "
          f"precision={metrics['precision']:.4f} "
          f"recall={metrics['recall']:.4f} "
          f"f1={metrics['f1']:.4f} "
          f"auc={auc:.4f}")


def main():
    opt = get_options()
    set_seed(42)
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

    model = GANomaly(opt)
    model.criterion = TotalLoss(opt, recon_alpha=0.5)  # Phase 3: MSE 0.5 + SSIM 0.5
    model = model.to(device)

    ckpt_path = os.path.join(opt.save_dir, 'best_model.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    test_loader, _ = get_dataloader(opt, 'test')
    raw_scores, labels = collect_scores(model, test_loader, device)
    scores = normalize_scores(raw_scores)

    auc = roc_auc_score(labels, scores)
    sweep_th = sweep_threshold(scores, labels)
    auto_th = auto_threshold(scores, opt.k)

    sweep_metrics = compute_metrics(scores, labels, sweep_th)
    auto_metrics = compute_metrics(scores, labels, auto_th)

    print_metrics('sweep (best F1)', sweep_metrics, auc)
    print_metrics(f'auto (mu + {opt.k}*sigma)', auto_metrics, auc)

    plot_roc_curve(labels, scores, auc, opt.save_dir)
    plot_pr_curve(labels, scores, opt.save_dir)
    plot_confusion_matrices(labels, scores, sweep_th, auto_th, opt.save_dir)
    plot_score_distribution(scores, labels, opt.save_dir)


if __name__ == '__main__':
    main()
