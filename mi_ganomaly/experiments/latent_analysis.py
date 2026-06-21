import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ganomaly import GANomaly
from options import get_options
from utils.dataloader import get_dataloader
from utils.reproducibility import set_seed

DPI = 300


def extract_latents(model, loader, device):
    model.eval()
    latents, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = model.e1(x)
            latents.append(z.view(z.size(0), -1).cpu())
            labels.append(y)
    return torch.cat(latents).numpy(), torch.cat(labels).numpy()


def plot_pca(latents, labels, save_path):
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(latents)
    var_ratio = pca.explained_variance_ratio_

    plt.figure(figsize=(6, 5))
    plt.scatter(proj[labels == 0, 0], proj[labels == 0, 1], c='blue', s=8, alpha=0.5, label='normal')
    plt.scatter(proj[labels == 1, 0], proj[labels == 1, 1], c='red', s=8, alpha=0.5, label='anomaly')
    plt.xlabel(f'PC1 ({var_ratio[0] * 100:.2f}%)')
    plt.ylabel(f'PC2 ({var_ratio[1] * 100:.2f}%)')
    plt.title('PCA of Encoder Latent Space')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI)
    plt.close()

    print(f'[PCA] explained variance ratio: PC1={var_ratio[0]:.4f}, PC2={var_ratio[1]:.4f}')
    return var_ratio


def plot_tsne(latents, labels, save_path):
    # sklearn>=1.5: TSNE의 n_iter는 max_iter로 대체됨 (구버전 n_iter는 1.7+에서 제거)
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, init='pca', random_state=42)
    proj = tsne.fit_transform(latents)

    plt.figure(figsize=(6, 5))
    plt.scatter(proj[labels == 0, 0], proj[labels == 0, 1], c='blue', s=8, alpha=0.5, label='normal')
    plt.scatter(proj[labels == 1, 0], proj[labels == 1, 1], c='red', s=8, alpha=0.5, label='anomaly')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE of Encoder Latent Space')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI)
    plt.close()


def plot_covariance(latents, labels, save_path):
    cov_normal = np.cov(latents[labels == 0], rowvar=False)
    cov_anomaly = np.cov(latents[labels == 1], rowvar=False)
    vmax = max(np.abs(cov_normal).max(), np.abs(cov_anomaly).max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(cov_normal, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    axes[0].set_title('Normal latent covariance')
    im1 = axes[1].imshow(cov_anomaly, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    axes[1].set_title('Anomaly latent covariance')
    fig.colorbar(im1, ax=axes, fraction=0.025)
    plt.savefig(save_path, dpi=DPI)
    plt.close()


def append_to_log(latents, labels, score):
    log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments_log.md'))
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('\n## Latent Space 분석 (PCA / t-SNE / 공분산 / Silhouette)\n')
        f.write(f'- latent shape: {latents.shape}, normal={int(np.sum(labels == 0))}, '
                f'anomaly={int(np.sum(labels == 1))}\n')
        f.write(f'- Silhouette Score (정상 vs 이상): {score:.4f}\n')


def main():
    opt = get_options()
    set_seed(42)
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

    model = GANomaly(opt).to(device)
    model.load_state_dict(torch.load(os.path.join(opt.save_dir, 'best_model.pth'), map_location=device))

    test_loader, _ = get_dataloader(opt, 'test')
    latents, labels = extract_latents(model, test_loader, device)
    print(f'latents: {latents.shape}, normal={np.sum(labels == 0)}, anomaly={np.sum(labels == 1)}')

    plot_pca(latents, labels, os.path.join(opt.save_dir, 'pca_latent.png'))
    plot_tsne(latents, labels, os.path.join(opt.save_dir, 'tsne_latent.png'))
    plot_covariance(latents, labels, os.path.join(opt.save_dir, 'covariance_matrix.png'))

    score = silhouette_score(latents, labels)
    print(f'[Silhouette] score={score:.4f}')

    append_to_log(latents, labels, score)


if __name__ == '__main__':
    main()
