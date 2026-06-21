import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import collect_scores, compute_metrics, normalize_scores, sweep_threshold
from models.ganomaly import GANomaly
from models.loss import TotalLoss
from options import get_options
from utils.dataloader import get_dataloader

OUT_DIR = 'output/final'

# (표시 이름, 체크포인트 경로, 해당 phase의 recon_alpha)
PHASES = [
    ('Phase1 (Before)', 'output/phase1_baseline', 1.0),
    ('Phase2 (Masking)', 'output/phase2_masking', 1.0),
    ('Phase3 (After)', 'output/phase3_ssim', 0.5),
]


def load_phase_scores(opt, device, ckpt_dir, recon_alpha):
    model = GANomaly(opt)
    model.criterion = TotalLoss(opt, recon_alpha=recon_alpha)
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best_model.pth'), map_location=device))
    model.eval()

    test_loader, _ = get_dataloader(opt, 'test')
    raw_scores, labels = collect_scores(model, test_loader, device)
    scores = normalize_scores(raw_scores)
    return scores, labels


def plot_phase_comparison(phase_data, save_path):
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    plt.figure(figsize=(8, 5))
    for color, (name, (scores, labels)) in zip(colors, phase_data.items()):
        plt.hist(scores[labels == 0], bins=20, histtype='step', linewidth=1.6,
                  linestyle='-', color=color, density=True, label=f'{name} - normal')
        plt.hist(scores[labels == 1], bins=20, histtype='step', linewidth=1.6,
                  linestyle='--', color=color, density=True, label=f'{name} - anomaly')
    plt.xlabel('normalized anomaly score')
    plt.ylabel('density')
    plt.title('Phase 1/2/3 Score Distribution Comparison')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def threshold_comparison(scores, labels):
    results = []
    sweep_th = sweep_threshold(scores, labels)
    results.append(('sweep', compute_metrics(scores, labels, sweep_th)))
    for k in (1.5, 2.0, 2.5):
        th = float(scores.mean() + k * scores.std())
        results.append((f'auto(k={k})', compute_metrics(scores, labels, th)))
    return results


def print_threshold_table(results, auc):
    print(f"{'method':>12} {'threshold':>10} {'accuracy':>9} {'precision':>10} {'recall':>8} {'f1':>8} {'auc':>8}")
    for name, m in results:
        print(f"{name:>12} {m['threshold']:>10.3f} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>8.4f} {auc:>8.4f}")


def plot_threshold_comparison(results, save_path):
    names = [r[0] for r in results]
    precisions = [r[1]['precision'] for r in results]
    recalls = [r[1]['recall'] for r in results]
    f1s = [r[1]['f1'] for r in results]

    x = np.arange(len(names))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, precisions, width, label='Precision')
    plt.bar(x, recalls, width, label='Recall')
    plt.bar(x + width, f1s, width, label='F1')
    plt.xticks(x, names)
    plt.ylabel('score')
    plt.title('Threshold Method Comparison (Phase 3)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_final_summary(phase1_metrics, phase1_auc, phase3_metrics, phase3_auc, save_path):
    rows = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    before = [phase1_metrics['accuracy'], phase1_metrics['precision'],
              phase1_metrics['recall'], phase1_metrics['f1'], phase1_auc]
    after = [phase3_metrics['accuracy'], phase3_metrics['precision'],
             phase3_metrics['recall'], phase3_metrics['f1'], phase3_auc]
    cell_text = [[f'{b:.4f}', f'{a:.4f}', f'{(a - b):+.4f}'] for b, a in zip(before, after)]

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.axis('off')
    table = ax.table(cellText=cell_text, rowLabels=rows,
                      colLabels=['Phase1 (Before)', 'Phase3 (After)', 'Delta'],
                      loc='center', cellLoc='center')
    table.scale(1, 1.6)
    ax.set_title('Final Before/After Summary (sweep threshold)', pad=20)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    opt = get_options()
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUT_DIR, exist_ok=True)

    phase_data = {}
    for name, ckpt_dir, recon_alpha in PHASES:
        scores, labels = load_phase_scores(opt, device, ckpt_dir, recon_alpha)
        phase_data[name] = (scores, labels)
        print(f'{name}: scores loaded (n={len(scores)})')

    # 1. Phase 1/2/3 score distribution overlay
    plot_phase_comparison(phase_data, os.path.join(OUT_DIR, 'phase_comparison.png'))

    # 2. Threshold method comparison (Phase 3)
    phase3_scores, phase3_labels = phase_data['Phase3 (After)']
    phase3_auc = roc_auc_score(phase3_labels, phase3_scores)
    results = threshold_comparison(phase3_scores, phase3_labels)
    print()
    print('--- Threshold method comparison (Phase 3) ---')
    print_threshold_table(results, phase3_auc)
    plot_threshold_comparison(results, os.path.join(OUT_DIR, 'threshold_comparison.png'))

    # 3. Final Before/After table (Phase1 vs Phase3, sweep threshold)
    phase1_scores, phase1_labels = phase_data['Phase1 (Before)']
    phase1_auc = roc_auc_score(phase1_labels, phase1_scores)
    phase1_th = sweep_threshold(phase1_scores, phase1_labels)
    phase1_metrics = compute_metrics(phase1_scores, phase1_labels, phase1_th)
    phase3_metrics = results[0][1]  # sweep result computed above

    plot_final_summary(phase1_metrics, phase1_auc, phase3_metrics, phase3_auc,
                        os.path.join(OUT_DIR, 'final_summary.png'))

    print()
    print('--- Final Before/After (sweep threshold) ---')
    print(f"Phase1: acc={phase1_metrics['accuracy']:.4f} precision={phase1_metrics['precision']:.4f} "
          f"recall={phase1_metrics['recall']:.4f} f1={phase1_metrics['f1']:.4f} auc={phase1_auc:.4f}")
    print(f"Phase3: acc={phase3_metrics['accuracy']:.4f} precision={phase3_metrics['precision']:.4f} "
          f"recall={phase3_metrics['recall']:.4f} f1={phase3_metrics['f1']:.4f} auc={phase3_auc:.4f}")


if __name__ == '__main__':
    main()
